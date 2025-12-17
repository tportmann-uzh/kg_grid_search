import pickle
from pathlib import Path

import pandas as pd
import torch
from pykeen.losses import UnsupportedLabelSmoothingError, PairwiseLoss
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.triples.weights import LossWeighter
from torch import FloatTensor, BoolTensor, nn
from torch.optim import Adam

# ============================================================================
# Config
# ============================================================================
HOOM_CSV = Path("./hoom_orphanet_2.4.csv")
OUT_DIR = Path("./weighted_runs")

# Choose device externally via CUDA_VISIBLE_DEVICES, but still set 'cuda' here.
# If you want cpu fallback, set DEVICE="cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

freq_to_multiplier_4 = {
    "FREQ:O": 8.0,
    "FREQ:VF": 6.0,
    "FREQ:F": 3.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.2,
    "CRIT:E": 10.0,
}

freq_to_multiplier_5 = {
    "FREQ:O": 10.0,
    "FREQ:VF": 7.0,
    "FREQ:F": 4.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.1,
    "CRIT:E": 10.0,
}

freq_to_multiplier_6 = {
    "FREQ:O": 12.0,
    "FREQ:VF": 8.0,
    "FREQ:F": 4.5,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.1,
    "CRIT:E": 10.0,
}

freq_to_multiplier_7 = {
    "FREQ:O": 15.0,
    "FREQ:VF": 10.0,
    "FREQ:F": 5.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.05,
    "CRIT:E": 10.0,
}

freq_to_multiplier_8 = {
    "FREQ:O": 5.0,
    "FREQ:VF": 4.0,
    "FREQ:F": 2.5,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.3,
    "CRIT:E": 10.0,
}

freq_to_multiplier_9 = {
    "FREQ:O": 20.0,
    "FREQ:VF": 12.0,
    "FREQ:F": 6.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.05,
    "CRIT:E": 10.0,
}

freq_to_multiplier_10 = {
    "FREQ:O": 30.0,
    "FREQ:VF": 15.0,
    "FREQ:F": 7.5,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.03,
    "CRIT:E": 10.0,
}

freq_to_multiplier_11 = {
    "FREQ:O": 50.0,
    "FREQ:VF": 25.0,
    "FREQ:F": 10.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.02,
    "CRIT:E": 10.0,
}

freq_to_multiplier_12 = {
    "FREQ:O": 100.0,
    "FREQ:VF": 40.0,
    "FREQ:F": 15.0,
    "FREQ:OC": 1.0,
    "FREQ:VR": 0.01,
    "CRIT:E": 10.0,
}


# ============================================================================
# KG + Weighting
# ============================================================================
def build_triples_factory(hoom_df: pd.DataFrame) -> TriplesFactory:
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
        relation = "is_associated_with"
        if freq == "CRIT:E":
            continue
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


class WeightedSoftMarginPairwiseLoss(PairwiseLoss):

    def __init__(self, reduction: str = "mean", margin: float = 1.0):
        super().__init__(reduction=reduction)
        self.margin = margin
        # self.activation = nn.Softplus()  # soft-margin style
        self.activation = nn.ReLU()  # margin-ranking style

    def forward(
            self,
            pos_scores,
            neg_scores,
            pos_weights=None,
            neg_weights=None,
    ):
        # typical pairwise margin-style loss: softplus(margin + neg - pos)
        diff = neg_scores - pos_scores + self.margin
        raw_loss = self.activation(diff)  # shape: [batch_size]

        if pos_weights is not None:
            # ensure broadcast / shape alignment
            raw_loss = raw_loss * pos_weights

        # use the built-in reduction to get scalar loss
        return self._reduction_method(raw_loss)

    def process_slcwa_scores(
            self,
            positive_scores: FloatTensor,
            negative_scores: FloatTensor,
            label_smoothing: float | None = None,
            batch_filter: BoolTensor | None = None,
            num_entities: int | None = None,
            pos_weights: FloatTensor | None = None,
            neg_weights: FloatTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        if batch_filter is not None:
            # negative_scores have already been filtered in the sampler!
            num_neg_per_pos = batch_filter.shape[1]
            positive_scores = positive_scores.repeat(1, num_neg_per_pos)[batch_filter]
            # shape: (nnz,)

        return self(
            pos_scores=positive_scores, neg_scores=negative_scores, pos_weights=pos_weights, neg_weights=neg_weights
        )

    # docstr-coverage: inherited
    def process_lcwa_scores(
            # TODO: Pass down weights for lcwa (see slcwa above).
            self,
            predictions: FloatTensor,
            labels: FloatTensor,
            label_smoothing: float | None = None,
            num_entities: int | None = None,
            weights: FloatTensor | None = None,
    ) -> FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)
        # self._raise_on_weights(weights)

        # for LCWA scores, we consider all pairs of positive and negative scores for a single batch element.
        # note: this leads to non-uniform memory requirements for different batches, depending on the total number of
        # positive entries in the labels tensor.

        # This shows how often one row has to be repeated,
        # shape: (batch_num_positives,), if row i has k positive entries, this tensor will have k entries with i
        repeat_rows = (labels == 1).nonzero(as_tuple=False)[:, 0]
        # Create boolean indices for negative labels in the repeated rows, shape: (batch_num_positives, num_entities)
        labels_negative = labels[repeat_rows] == 0
        # Repeat the predictions and filter for negative labels, shape: (batch_num_pos_neg_pairs,)
        negative_scores = predictions[repeat_rows][labels_negative]

        # This tells us how often each true label should be repeated
        repeat_true_labels = (labels[repeat_rows] == 0).nonzero(as_tuple=False)[:, 0]
        # First filter the predictions for true labels and then repeat them based on the repeat vector
        positive_scores = predictions[labels == 1][repeat_true_labels]

        return self(pos_scores=positive_scores, neg_scores=negative_scores)


def build_weight_lookup(
        hoom_df: pd.DataFrame,
        tf: TriplesFactory,
        freq_to_multiplier: dict[str, float],
) -> dict[tuple[int, int, int], float]:
    """
    Build lookup keyed by (h_id, r_id, t_id) using tf's IDs.
    IMPORTANT: tf must match kg_type (assoc_only vs assoc_excl), otherwise relation IDs won't match.
    """
    lookup = {}

    # Relation IDs that exist in THIS tf
    assoc_rel_id = tf.relation_to_id.get("is_associated_with")

    for _, row in hoom_df.iterrows():
        hpo_id = row["hpo_id"]
        disease = row["orpha_code"]
        freq = str(row["frequency"]).upper()

        if hpo_id not in tf.entity_to_id or disease not in tf.entity_to_id:
            continue

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

if __name__ == "__main__":
    hoom_df = pd.read_csv("hoom_orphanet_2.4.csv")

    tf = build_triples_factory(hoom_df)

    weight_lookups = [
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_4),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_5),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_6),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_7),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_8),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_9),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_10),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_11),
        build_weight_lookup(hoom_df, tf, freq_to_multiplier_12),
    ]

    for idx, weight_lookup in enumerate(weight_lookups):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_weighter = FrequencyLossWeighter(freq_weight_lookup=weight_lookup)

        result = pipeline(
            model="TransE",
            model_kwargs=dict(embedding_dim=128),
            training=tf,
            validation=None,
            testing=tf,
            training_loop="slcwa",
            loss=WeightedSoftMarginPairwiseLoss,
            loss_kwargs=dict(margin=1.0),
            training_loop_kwargs=dict(loss_weighter=loss_weighter),
            training_kwargs=dict(num_epochs=200, batch_size=2048),
            optimizer=Adam,
            optimizer_kwargs=dict(lr=1e-3),
            device=device,
            random_seed=42
        )
        embedings = export_embeddings_dict(result, tf)
        pickle.dump(embedings, open(f"w{idx + 4}_embedings.emb", "wb"))
