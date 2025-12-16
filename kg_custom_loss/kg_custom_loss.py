import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.triples.weights import LossWeighter
from torch.optim import Adam

# ============================================================================
# Config
# ============================================================================
HOOM_CSV = Path("./hoom_orphanet_2.4.csv")
OUT_DIR = Path("./weighted_runs")
RANDOM_SEED = 42

MODEL_KWARGS = dict(embedding_dim=128)
TRAINING_KWARGS = dict(num_epochs=200, batch_size=2048)  # GPU recommended
OPTIMIZER_KWARGS = dict(lr=1e-3)
LOSS_NAME = "BCEWithLogitsLoss"
TRAINING_LOOP = "slcwa"

# Choose device externally via CUDA_VISIBLE_DEVICES, but still set 'cuda' here.
# If you want cpu fallback, set DEVICE="cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

freq_to_multiplier_1 = {
    "FREQ:O": 3.0,
    "FREQ:VF": 2.5,
    "FREQ:F": 2.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.5,
    "CRIT:E": 5.0,
}

freq_to_multiplier_2 = {
    "FREQ:O": 2.0,
    "FREQ:VF": 1.5,
    "FREQ:F": 1.2,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.8,
    "CRIT:E": 10.0,
}


# ============================================================================
# KG + Weighting
# ============================================================================
def build_triples_factory(hoom_df: pd.DataFrame, kg_type: str) -> TriplesFactory:
    """
    kg_type:
      - "assoc_only": only (hpo, is_associated_with, disease), skips CRIT:E entirely
      - "assoc_excl": uses relation "excludes" for CRIT:E, and "is_associated_with" otherwise
    """
    triples = []
    for _, row in hoom_df.iterrows():
        hpo_id = row["hpo_id"]
        disease = row["orpha_code"]
        freq = str(row["frequency"]).upper()

        if freq == "CRIT:E":
            if kg_type == "assoc_only":
                continue
            relation = "excludes"
        else:
            relation = "is_associated_with"

        triples.append((hpo_id, relation, disease))

    df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    tf = TriplesFactory.from_labeled_triples(df[["head", "relation", "tail"]].values)
    return tf


class FrequencyLossWeighter(LossWeighter):
    def __init__(self, freq_weight_lookup: dict[tuple[int, int, int], float]):
        super().__init__()
        self.freq_weight_lookup = freq_weight_lookup

    def __call__(self, h, r, t) -> torch.FloatTensor:
        # For sLCWA, all three should be provided
        if h is None or r is None or t is None:
            non_none = h if h is not None else (r if r is not None else t)
            if non_none is None:
                return torch.tensor([1.0])
            return torch.ones(non_none.shape[0], dtype=torch.float32, device=non_none.device)

        h = h.view(-1)
        r = r.view(-1)
        t = t.view(-1)

        weights = []
        for h_id, r_id, t_id in zip(h.tolist(), r.tolist(), t.tolist()):
            weights.append(self.freq_weight_lookup.get((h_id, r_id, t_id), 1.0))

        return torch.tensor(weights, dtype=torch.float32, device=h.device)


def build_weight_lookup(
        hoom_df: pd.DataFrame,
        tf: TriplesFactory,
        kg_type: str,
        freq_to_multiplier: dict[str, float],
) -> dict[tuple[int, int, int], float]:
    """
    Build lookup keyed by (h_id, r_id, t_id) using tf's IDs.
    IMPORTANT: tf must match kg_type (assoc_only vs assoc_excl), otherwise relation IDs won't match.
    """
    lookup = {}

    # Relation IDs that exist in THIS tf
    assoc_rel_id = tf.relation_to_id.get("is_associated_with")
    excl_rel_id = tf.relation_to_id.get("excludes")  # only exists for assoc_excl

    for _, row in hoom_df.iterrows():
        hpo_id = row["hpo_id"]
        disease = row["orpha_code"]
        freq = str(row["frequency"]).upper()

        if hpo_id not in tf.entity_to_id or disease not in tf.entity_to_id:
            continue

        # Determine relation according to kg_type
        if freq == "CRIT:E":
            if kg_type == "assoc_only":
                continue
            if excl_rel_id is None:
                continue
            r_id = excl_rel_id
        else:
            if assoc_rel_id is None:
                continue
            r_id = assoc_rel_id

        h_id = tf.entity_to_id[hpo_id]
        t_id = tf.entity_to_id[disease]

        w = float(freq_to_multiplier.get(freq, 1.0))
        lookup[(h_id, r_id, t_id)] = w

    return lookup


def export_embeddings_dict(result, tf: TriplesFactory) -> dict[str, object]:
    """Export embeddings as dict[str, np.ndarray] incl. relations, like your project expects."""
    model = result.model

    entity_emb_matrix = model.entity_representations[0]().detach().cpu().numpy()
    id_to_entity = {v: k for k, v in tf.entity_to_id.items()}
    embeddings = {entity_id: entity_emb_matrix[idx] for idx, entity_id in id_to_entity.items()}

    rel_emb_matrix = model.relation_representations[0]().detach().cpu().numpy()
    id_to_rel = {v: k for k, v in tf.relation_to_id.items()}
    for idx, rel_id in id_to_rel.items():
        embeddings[rel_id] = rel_emb_matrix[idx]

    return embeddings


# ============================================================================
# Run definition
# ============================================================================
@dataclass
class RunSpec:
    name: str
    kg_type: str  # "assoc_only" or "assoc_excl"
    weights_name: str  # "w1" or "w2"
    freq_to_multiplier: dict  # actual mapping


def run_one(spec: RunSpec, hoom_df: pd.DataFrame) -> None:
    run_dir = OUT_DIR / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build TF
    tf = build_triples_factory(hoom_df, kg_type=spec.kg_type)

    # Split (fixed seed so comparable per kg_type)
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=RANDOM_SEED)

    # Build weights for THIS tf
    weight_lookup = build_weight_lookup(
        hoom_df=hoom_df,
        tf=tf,
        kg_type=spec.kg_type,
        freq_to_multiplier=spec.freq_to_multiplier,
    )
    loss_weighter = FrequencyLossWeighter(freq_weight_lookup=weight_lookup)

    # Save config
    cfg = {
        "run": asdict(spec) | {"freq_to_multiplier": spec.freq_to_multiplier},
        "seed": RANDOM_SEED,
        "device": DEVICE,
        "model": "TransE",
        "model_kwargs": MODEL_KWARGS,
        "training_kwargs": TRAINING_KWARGS,
        "optimizer_kwargs": OPTIMIZER_KWARGS,
        "loss": LOSS_NAME,
        "training_loop": TRAINING_LOOP,
        "tf_stats": {
            "num_triples": int(tf.num_triples),
            "num_entities": int(tf.num_entities),
            "num_relations": int(tf.num_relations),
        },
        "weight_lookup_size": int(len(weight_lookup)),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Train
    result = pipeline(
        model="TransE",
        model_kwargs=MODEL_KWARGS,
        training=training,
        validation=validation,
        testing=testing,
        training_loop=TRAINING_LOOP,
        loss=LOSS_NAME,
        training_loop_kwargs=dict(loss_weighter=loss_weighter),
        training_kwargs=TRAINING_KWARGS,
        optimizer=Adam,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        device=DEVICE,
        random_seed=RANDOM_SEED,
    )

    # Save pipeline outputs
    result.save_to_directory(run_dir / "pykeen_result")

    # Export + save embeddings dict
    embeddings = export_embeddings_dict(result, tf)
    with open(run_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # Save quick test metrics
    test_metrics = {
        "hits@1": float(result.metric_results.get_metric("hits@1")),
        "hits@3": float(result.metric_results.get_metric("hits@3")),
        "hits@10": float(result.metric_results.get_metric("hits@10")),
        "mrr": float(result.metric_results.get_metric("mrr")),
        "mr": float(result.metric_results.get_metric("mr")),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    print(f"[OK] {spec.name}  hits@10={test_metrics['hits@10']:.4f}  saved to {run_dir}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hoom_df = pd.read_csv(HOOM_CSV)

    runs = [
        RunSpec(name="transe_assoc_only_w1", kg_type="assoc_only", weights_name="w1",
                freq_to_multiplier=freq_to_multiplier_1),
        RunSpec(name="transe_assoc_only_w2", kg_type="assoc_only", weights_name="w2",
                freq_to_multiplier=freq_to_multiplier_2),
        RunSpec(name="transe_assoc_excl_w1", kg_type="assoc_excl", weights_name="w1",
                freq_to_multiplier=freq_to_multiplier_1),
        RunSpec(name="transe_assoc_excl_w2", kg_type="assoc_excl", weights_name="w2",
                freq_to_multiplier=freq_to_multiplier_2),
    ]

    for spec in runs:
        run_one(spec, hoom_df)


if __name__ == "__main__":
    main()
