"""LIONS v2.5 — features for the catalog-driven pattern matcher.

Different from v1.x ``features.py``:
- NODE_NAME vocab is anonymized: F1..F8 instead of physics-specific
  names (Phi, H, L, ...). Field naming bias removed per the v2.5
  redirect — random rename augment applies these slots in a fresh
  permutation each epoch (see ``LionsV25Dataset.__getitem__``).
- ``LABEL_ORDER`` = ``indexcalc.lions.catalog.all_labels()``, 23 entries (v3.0).
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


LABEL_ORDER: tuple[str, ...] = tuple(all_labels())   # 23 entries (v3.0: +Sp(4..10))


# ─── Anonymized field name vocab ──────────────────────────
#
# F1..F8 cover the n_fields=3 setup (plus headroom). Invariant tensor
# names and operator labels match v1.x for cross-encoder consistency.

NODE_NAME = {
    "<unk>": 0,
    # anonymized fields
    "F1": 1, "F2": 2, "F3": 3, "F4": 4, "F5": 5, "F6": 6, "F7": 7, "F8": 8,
    # invariants
    "eta": 10, "delta": 11, "gamma": 12, "epsilon": 13, "omega": 14,
    "eta_conf": 15, "eta_dd": 16,
    # operators
    "partial": 20, "TimeDeriv": 21, "ScalarFunction": 22,
}


# ─── Anonymize policy: kind decided by name, not by reps ────
# Used in datasets_v25._encode_to_pyg to override graph_encode's
# reps-based kind so user input with empty/dummy reps still classifies
# correctly.

INVARIANT_TENSOR_NAMES = frozenset({
    "eta", "delta", "gamma", "epsilon", "omega", "eta_conf", "eta_dd",
    # v1.x spinor projectors / Lorentz invariants — included for legacy
    # compatibility even though v2.5 doesn't enumerate them yet.
    "P_L", "P_R", "gamma5", "Sigma",
})

OPERATOR_NAMES = frozenset({"partial", "TimeDeriv", "ScalarFunction"})


def kind_from_name(name: str, fallback: str = "field") -> str:
    """Decide a node's kind purely from its name, ignoring ``reps``.

    Operator names → "operator", invariant tensor names → "invariant",
    anything else → ``fallback`` (default "field"). The fallback lets
    a user input with an empty reps dict (the anonymize ideal) still
    end up as a field rather than being demoted to "invariant".
    """
    if name in OPERATOR_NAMES:
        return "operator"
    if name in INVARIANT_TENSOR_NAMES:
        return "invariant"
    return fallback


# ─── Property hint vocab ─────────────────────────────────
#
# "unknown" handles the blank-variant case (field_properties == {}).

PROP_STATISTICS = {"unknown": 0, "bosonic": 1, "fermionic": 2}
PROP_ANTISYM = {"unknown": 0, "none": 1, "has_antisym": 2}


# ─── M4: IndexSpace primary metric vocab ─────────────────
#
# Lets the model see whether a node's indices live in an orthogonal
# (metric="delta"), Lorentz-like (metric="eta"), or unitary fund
# (metric="") space. Crucial for separating SU(N) ε from SO(N) ε.

PRIMARY_METRIC = {"unknown": 0, "none": 1, "delta": 2, "eta": 3, "conf": 4,
                  "dd": 5}


# ─── v3.1 tuning: discrete primary-dim vocab ─────────────
#
# Catalog index-space dims: U(1)=1, N∈{2..5} (U/SU/O/SO), Sp dims {4,6,8,10},
# Lorentz/Poincaré=4. A *discrete* embedding (vs a /5 scalar) lets the model
# resolve close high dims — Sp(8) (dim 8) vs Sp(10) (dim 10) were confused
# under the scalar encoding. 0 = no-index / unknown.
CATALOG_DIMS = (1, 2, 3, 4, 5, 6, 8, 10)
PRIMARY_DIM_VOCAB = {0: 0, **{d: i + 1 for i, d in enumerate(CATALOG_DIMS)}}


# ─── HS1.0: ScalarFunction potential-class vocab ─────────
#
# A ScalarFunction node wraps a non-polynomial central potential f(r²).
# Hidden (dynamical) symmetry depends on the *shape* of f: only 1/r gives
# SO(4) (Kepler/LRL), only r² gives SU(N) (isotropic HO). The graph encoder
# previously collapsed every f to one "ScalarFunction" node, so the model
# could not separate Kepler from a generic V(r) — teaching SO(4) would have
# leaked onto all central potentials. This vocab tags the node by physical
# potential class so SO(4) keys on inv_sqrt alone.
#   "none"     — node is not a ScalarFunction (the common case)
#   "generic"  — a ScalarFunction whose name we don't have a class for
SCALAR_FUNC_CLASS = {
    "none": 0,
    "generic": 1,
    "inv_sqrt": 2,   # 1/r        → Kepler/Coulomb, hidden SO(4)
    "inverse": 3,    # 1/r²
    "quadratic": 4,  # r²         (also expressible as a δ mass term)
    "exp": 5,        # e^{...}
    "log": 6,
}


def scalar_func_class_id(func_name: str) -> int:
    """Map a node's ScalarFunction name to its potential-class id.

    "" (not a ScalarFunction node) → "none"; a recognized function name →
    its class; any other name → "generic".
    """
    if not func_name:
        return SCALAR_FUNC_CLASS["none"]
    return SCALAR_FUNC_CLASS.get(func_name, SCALAR_FUNC_CLASS["generic"])


# ─── Helpers ─────────────────────────────────────────────


def _lookup(table, token: str) -> int:
    return table.get(token, table["<unk>"]) if "<unk>" in table else table.get(token, 0)


def node_feature_ids_v25(
    kind: str, name: str, rank: int, statistics: str,
    stats_hint: str = "unknown", antisym_hint: str = "unknown",
    primary_dim: int = 0, primary_metric: str = "unknown",
    secondary_dim: int = 0, secondary_metric: str = "unknown",
    func_name: str = "",
) -> list[int]:
    """Return a fixed-length int list for one node.

    Layout (length 11):
      [kind, name, rank, statistics, stats_hint, antisym_hint,
       primary_dim, primary_metric_id, secondary_dim, secondary_metric_id,
       scalar_func_class_id].

    ``primary_dim``/``primary_metric`` describe the node's first (canonically
    sorted) index space; ``secondary_dim``/``secondary_metric`` describe its
    second distinct index space, or (0, "unknown") when the node lives in a
    single space. The secondary slot is what makes a multi-index field
    ψ^{iα} — charged under two groups at once — expose *both* sectors' (dim,
    metric); without it only the first sector was visible and the model could
    not tell a dim-3 SU(3) internal index from a dim-8 Sp one (v3.3 Sp fix).

    ``func_name`` (HS1.0) is a ScalarFunction node's function name; it maps to
    a potential-class id so the model can tell a 1/r potential (hidden SO(4))
    from a generic central potential. "" for every non-ScalarFunction node.
    """
    return [
        _lookup(NODE_KIND, kind),
        _lookup(NODE_NAME, name),
        int(rank),
        _lookup(STATISTICS, statistics),
        _lookup(PROP_STATISTICS, stats_hint),
        _lookup(PROP_ANTISYM, antisym_hint),
        int(primary_dim),
        _lookup(PRIMARY_METRIC, primary_metric if primary_metric else "none"),
        int(secondary_dim),
        _lookup(PRIMARY_METRIC, secondary_metric if secondary_metric else "none"),
        scalar_func_class_id(func_name),
    ]


def num_relations() -> int:
    """Same edge-type packing as v1.x. v3.4: SP_BASE 16→32 (edge "space" is now
    a (dim,metric) token vocab of ~20 entries, so kind*32+space needs 4*32)."""
    return 4 * 32  # kind ∈ [0,4) × space-token ∈ [0,32)


# Field-name token id pool — used by the random rename augment.
FIELD_TOKEN_IDS: tuple[int, ...] = tuple(
    NODE_NAME[k] for k in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
)
