"""LIONS v2.5 — features for the catalog-driven pattern matcher.

Different from v1.x ``features.py``:
- NODE_NAME vocab is anonymized: F1..F8 instead of physics-specific
  names (Phi, H, L, ...). Field naming bias removed per the v2.5
  redirect — random rename augment applies these slots in a fresh
  permutation each epoch (see ``LionsV25Dataset.__getitem__``).
- ``LABEL_ORDER`` = ``indexcalc.lions.catalog.all_labels()``, 19 entries.
- Node feature includes 2 extra ints encoding field-property hints
  (statistics, antisym-pair flag). Blank variants set both to "unknown".

Other vocab tables (NODE_KIND, EDGE_KIND, EDGE_SPACE, POSITION) are
imported unchanged from v1.x features to keep the graph encoder /
serializer / pyg_bridge surface aligned.
"""

from __future__ import annotations
from typing import Sequence

from indexcalc.lions.catalog import all_labels
from indexcalc.lions.ml.features import (
    NODE_KIND, STATISTICS, EDGE_KIND, EDGE_SPACE, POSITION,
)


# ─── v2.5 catalog label order ─────────────────────────────


LABEL_ORDER: tuple[str, ...] = tuple(all_labels())   # 19 entries


# ─── Anonymized field name vocab ──────────────────────────
#
# F1..F8 cover the n_fields=3 setup (plus headroom). Invariant tensor
# names and operator labels match v1.x for cross-encoder consistency.

NODE_NAME = {
    "<unk>": 0,
    # anonymized fields
    "F1": 1, "F2": 2, "F3": 3, "F4": 4, "F5": 5, "F6": 6, "F7": 7, "F8": 8,
    # invariants
    "eta": 10, "delta": 11, "gamma": 12, "epsilon": 13,
    # operators
    "partial": 20, "TimeDeriv": 21, "ScalarFunction": 22,
}


# ─── Property hint vocab ─────────────────────────────────
#
# "unknown" handles the blank-variant case (field_properties == {}).

PROP_STATISTICS = {"unknown": 0, "bosonic": 1, "fermionic": 2}
PROP_ANTISYM = {"unknown": 0, "none": 1, "has_antisym": 2}


# ─── Helpers ─────────────────────────────────────────────


def _lookup(table, token: str) -> int:
    return table.get(token, table["<unk>"]) if "<unk>" in table else table.get(token, 0)


def node_feature_ids_v25(
    kind: str, name: str, rank: int, statistics: str,
    stats_hint: str = "unknown", antisym_hint: str = "unknown",
) -> list[int]:
    """Return a fixed-length int list for one node.

    Layout (length 6): [kind, name, rank, statistics, stats_hint, antisym_hint].
    The reps slot used in v1.x is dropped — v2.5 fields are anonymized so
    rep labels are not a stable signal across catalog entries.
    """
    return [
        _lookup(NODE_KIND, kind),
        _lookup(NODE_NAME, name),
        int(rank),
        _lookup(STATISTICS, statistics),
        _lookup(PROP_STATISTICS, stats_hint),
        _lookup(PROP_ANTISYM, antisym_hint),
    ]


def num_relations() -> int:
    """Same edge-type packing as v1.x."""
    return 4 * 16  # kind ∈ [0,4) × space ∈ [0,16)


# Field-name token id pool — used by the random rename augment.
FIELD_TOKEN_IDS: tuple[int, ...] = tuple(
    NODE_NAME[k] for k in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
)
