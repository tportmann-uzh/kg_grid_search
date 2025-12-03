from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import obonet
import pandas as pd


def get_obsolete_term_mappings() -> dict[str, str]:
    graph = obonet.read_obo("./data/hpo_v2025_09_01.obo", ignore_obsolete=False)
    obsolete_map = {}

    for node_id, data in graph.nodes(data=True):
        if data.get("is_obsolete") == "true" and "replaced_by" in data:
            replaced_by = data["replaced_by"]
            if isinstance(replaced_by, list):
                # Use first replacement if multiple exist
                obsolete_map[node_id] = replaced_by[0]
            else:
                obsolete_map[node_id] = replaced_by

    return obsolete_map


def fix_obsolete_phenotypes(hoom_df: pd.DataFrame, obsolete_map: dict[str, str]) -> pd.DataFrame:
    hoom_df = hoom_df.copy()
    hoom_df["hpo_id"] = hoom_df["hpo_id"].apply(
        lambda x: obsolete_map.get(x, x)  # replace if obsolete
    )
    return hoom_df


def load_hoom_disease_phenotype_associations() -> pd.DataFrame:
    path = Path("./data/hoom_orphanet_2.4.csv")
    hoom_df = pd.read_csv(path)
    obsolete_map = get_obsolete_term_mappings()
    return fix_obsolete_phenotypes(hoom_df, obsolete_map)


def get_disease_phenotype_associations() -> Dict[str, List[str]]:
    hoom_df = load_hoom_disease_phenotype_associations()
    disease_phenotype_associations = defaultdict(list)

    for disease, group in hoom_df.groupby("orpha_code"):
        for _, row in group.iterrows():
            hpo = row["hpo_id"]
            disease_phenotype_associations[disease].append(hpo)

    # {"ORPHA:2485": ["HP:0001", "HP:0002", ...], ...}
    return disease_phenotype_associations


def get_graph():
    directed = obonet.read_obo("./data/hpo_v2025_09_01.obo")
    directed_reversed = directed.reverse(copy=True)
    # prune to Phenotypic abnormality subtree
    keep = nx.descendants(directed_reversed, "HP:0000118")
    keep.add("HP:0000118")
    directed_reversed_subgraph = directed_reversed.subgraph(keep).copy()
    return directed_reversed_subgraph


def build_kg(include_is_a: bool):
    dpa = get_disease_phenotype_associations()
    hpo_graph = get_graph()

    rows = []

    # Include is_a hpo structure
    if include_is_a:
        for node in hpo_graph.nodes:
            parents = hpo_graph.predecessors(node)
            rows.extend((node, "is_a", parent) for parent in parents)

    for disease, phenotypes in dpa.items():
        for phenotype in phenotypes:
            if not hpo_graph.has_node(phenotype):
                continue

            rows.append((phenotype, "is_associated_with", disease))

    return pd.DataFrame(rows, columns=["head", "relation", "tail"])
