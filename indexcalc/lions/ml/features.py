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
    # v2 NR mechanics fields
    "Phi": 30, "Psi": 31, "A": 32, "B": 33, "C": 34,
    "TimeDeriv": 40, "ScalarFunction": 41,
}

STATISTICS = {"<unk>": 0, "bosonic": 1, "fermionic": 2}


# ─── Edge-side vocab ─────────────────────────────────────


EDGE_KIND = {"<unk>": 0, "contraction": 1, "acts_on": 2}

EDGE_SPACE = {
    "<unk>": 0, "": 1,
    "spacetime": 2, "su2_adj": 3, "su2_fund": 4, "su3_fund": 5,
    "su3_adj": 6, "dirac": 7, "frame": 8,
    # v2: NR mechanics vector spaces
    "so3_vec": 9, "so2_vec": 10, "so4_vec": 11,
}

POSITION = {"<unk>": 0, "": 1, "upper": 2, "lower": 3}


# ─── Group label order (Task 1 y vector) ─────────────────
#
# v1: ("SU(2)", "U(1)_Y", "Lorentz") — SM-lite gauge + Lorentz
# v2: + ("O(3)", "SO(3)") — NR mechanics orthogonal groups
#
# Append-only so v1 dataset/model encodings remain length-prefix compatible
# (existing 3 bits still align). New rows default missing reps to "<none>".


GROUP_ORDER: tuple[str, ...] = (
    "SU(2)", "U(1)_Y", "Lorentz",          # v1 (SM-lite)
    "O(3)", "SO(3)",                       # v2 (NR mechanics)
)


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
    "O(3)": {
        "<unk>": 0, "<none>": 1,
        "singlet": 2, "vector": 3,
    },
    "SO(3)": {
        "<unk>": 0, "<none>": 1,
        "singlet": 2, "vector": 3,
    },
}


# ─── Numeric scalar features (I1: charge as numeric, 2026-05-18) ────


# U(1) charge tag → numeric value. Used as a *numeric* node feature
# alongside the categorical embedding so the GNN can compute charge
# sums via message passing (rather than learning charge arithmetic
# from one-hot rep IDs alone). Default for unmapped / missing tags is
# 0.0 (consistent with "<none>" being charge-neutral).
U1Y_CHARGE_VALUE: dict[str, float] = {
    "<unk>": 0.0, "<none>": 0.0,
    "0": 0.0, "+1/2": 0.5, "-1/2": -0.5, "+1": 1.0, "-1": -1.0,
}


def node_charge_features(reps: dict[str, str]) -> list[float]:
    """Return numeric scalar features per node — currently U(1)_Y only.

    Order: [u1y_charge]. Length 1. Extend here if other numeric reps
    join later (e.g. another U(1) factor).
    """
    rep = reps.get("U(1)_Y", "<none>")
    return [U1Y_CHARGE_VALUE.get(rep, 0.0)]


def num_charge_features() -> int:
    return 1


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
