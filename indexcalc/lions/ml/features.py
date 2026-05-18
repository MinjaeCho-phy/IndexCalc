"""Vocab tables for ``EncodedGraph`` → integer features.

Each categorical attribute (node kind, name, statistics; edge kind, space,
position) gets a fixed vocabulary that yields a stable int ID per token.
Unseen tokens get the ``<unk>`` ID 0 — a mild safety net for v1.1+ when
new fields/spaces appear.

The vocab is hand-curated to the v1 SM-lite world (`presets/sm_lite.py`)
plus headroom for future presets. Keeping it explicit (rather than
auto-built from the dataset) lets ``num_relations`` and embedding sizes
stay constant across train/val/test splits and across reruns.
"""

from __future__ import annotations
from typing import Sequence


# ─── Node-side vocab ─────────────────────────────────────


NODE_KIND = {"<unk>": 0, "field": 1, "invariant": 2, "operator": 3}

NODE_NAME = {
    "<unk>": 0,
    # B0/B1/B2 fields (SM-lite)
    "H": 1, "Hdag": 2,
    "L": 3, "Lbar": 4,
    "eR": 5, "eRbar": 6,
    "W": 7, "F": 8,
    # invariants
    "eta": 10, "delta": 11, "gamma": 12, "epsilon": 13,
    "Sigma": 14, "P_L": 15, "P_R": 16, "gamma5": 17,
    # operators
    "partial": 20, "covariant_deriv": 21,
}

STATISTICS = {"<unk>": 0, "bosonic": 1, "fermionic": 2}


# ─── Edge-side vocab ─────────────────────────────────────


EDGE_KIND = {"<unk>": 0, "contraction": 1, "acts_on": 2}

EDGE_SPACE = {
    "<unk>": 0, "": 1,
    "spacetime": 2, "su2_adj": 3, "su2_fund": 4, "su3_fund": 5,
    "su3_adj": 6, "dirac": 7, "frame": 8,
}

POSITION = {"<unk>": 0, "": 1, "upper": 2, "lower": 3}


# ─── Group label order (Task 1 y vector) ─────────────────


GROUP_ORDER: tuple[str, ...] = ("SU(2)", "U(1)_Y", "Lorentz")


# ─── Per-group rep vocab (for k-hot node feature) ────────


REP_VOCAB: dict[str, dict[str, int]] = {
    "SU(2)": {
        "<unk>": 0, "<none>": 1,
        "singlet": 2, "fund": 3, "antifund": 4, "adj": 5,
    },
    "U(1)_Y": {
        "<unk>": 0, "<none>": 1,
        "0": 2, "+1/2": 3, "-1/2": 4, "+1": 5, "-1": 6,
    },
    "Lorentz": {
        "<unk>": 0, "<none>": 1,
        "singlet": 2, "vector": 3,
        "spinor": 4, "conj_spinor": 5,
        "L_spinor": 6, "R_spinor": 7,
        "conj_L_spinor": 8, "conj_R_spinor": 9,
    },
}


def _lookup(table: dict[str, int], token: str) -> int:
    return table.get(token, table["<unk>"])


# ─── Public API ──────────────────────────────────────────


def node_feature_ids(
    kind: str, name: str, rank: int, statistics: str,
    reps: dict[str, str],
) -> list[int]:
    """Return a fixed-length list of int IDs for one node.

    Order: [kind, name, rank, statistics, rep_SU(2), rep_U(1)_Y, rep_Lorentz].
    Length always 7.
    """
    out = [
        _lookup(NODE_KIND, kind),
        _lookup(NODE_NAME, name),
        int(rank),
        _lookup(STATISTICS, statistics),
    ]
    for g in GROUP_ORDER:
        rep = reps.get(g, "<none>")
        out.append(_lookup(REP_VOCAB[g], rep))
    return out


def edge_feature_ids(
    kind: str, space: str, src_pos: str, dst_pos: str,
) -> tuple[int, list[int]]:
    """Return ``(edge_type, edge_attr_ids)`` for one edge.

    ``edge_type`` packs only ``(kind, space)`` so PyG ``RGCNConv`` learns
    one relation matrix per typed contraction (contraction-in-spacetime,
    contraction-in-su2_fund, acts_on, …). Position info travels in
    ``edge_attr_ids`` for richer-feature models; keeping it out of
    ``edge_type`` drops the relation matrix count from 1600 → 64 and
    matches the observation that v1-toy has only ~10 distinct edge types.
    """
    k = _lookup(EDGE_KIND, kind)
    s = _lookup(EDGE_SPACE, space)
    sp = _lookup(POSITION, src_pos)
    dp = _lookup(POSITION, dst_pos)
    SP_BASE = 16
    edge_type = k * SP_BASE + s
    return edge_type, [k, s, sp, dp]


def num_relations() -> int:
    """Upper bound used when sizing ``RGCNConv``."""
    return 4 * 16  # 64
