"""D25 — TimeDeriv / ScalarFunction graph encoder + serializer.

v3+ backend extension flagged in v2-nr build. Lifts the encoder/serializer
gap that previously excluded kinetic and inverse-sqrt potential terms
(Kepler) from auto-enumeration and ML inference.

Coverage:
- ``TimeDeriv(Phi)`` → operator node "TimeDeriv" + acts_on edge to Phi.
- ``ScalarFunction("inv_sqrt", δ_kl Φ^k Φ^l)`` → operator node
  "ScalarFunction" + acts_on edges to inner factors; inner contraction
  edges still present.
- Kepler L (kinetic + 1/r) encodes without raising and partitions into
  two terms (TensorSum semantics carry through new operators).
- Serializer round-trips both TimeDeriv and ScalarFunction; nested
  TimeDeriv/ScalarFunction inside TensorSum/TensorProduct round-trips.
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorSum, ScalarMul
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv

from indexcalc.lions.graph import graph_encode
from indexcalc.lions.dataset import LabeledSample
from indexcalc.lions.serializer import (
    expr_to_dict, expr_from_dict, collect_spaces,
    save_dataset, load_dataset,
)


@pytest.fixture
def vec_space():
    return IndexSpace("so3_vec", dim=3, indices="ijkl", metric="delta")


def _phi(vec, ix, pos="upper"):
    if pos == "upper":
        idx = vec.upper(ix)
    else:
        idx = vec.lower(ix)
    return Tensor("Phi", [idx], reps={"SO(3)": "vector", "O(3)": "vector"})


def _delta(vec, i, j):
    return Tensor(
        "delta", [vec.lower(i), vec.lower(j)],
        symmetric_pairs=[(0, 1)],
        reps={"SO(3)": "singlet", "O(3)": "singlet"},
    )


# ─── graph_encode: TimeDeriv ────────────────────────────


def test_timederiv_emits_operator_and_acts_on(vec_space):
    """δ_ij ̇Φ^i ̇Φ^j → 2 TimeDeriv ops + 2 Phi + 1 δ + 2 acts_on + 1 contract."""
    Phi_i = _phi(vec_space, "i")
    Phi_j = _phi(vec_space, "j")
    expr = _delta(vec_space, "i", "j") * TimeDeriv(Phi_i) * TimeDeriv(Phi_j)

    g = graph_encode(expr)
    op_names = [n.name for n in g.nodes if n.kind == "operator"]
    assert op_names.count("TimeDeriv") == 2

    acts_on = [e for e in g.edges if e.kind == "acts_on"]
    contract = [e for e in g.edges if e.kind == "contraction"]
    # TimeDeriv → Phi (×2), and δ_ij is contracted via i (Phi^i via TimeDeriv)
    # and j (Phi^j via TimeDeriv) — both contractions are with the TimeDeriv
    # operator since the operator's free indices == inner expr's.
    assert len(acts_on) == 2
    # Two contraction edges (one per i, j).
    assert len(contract) == 2
    for e in contract:
        # v3.4: edge.space is the contraction space's "{dim}:{metric}" token.
        assert e.space == "3:delta"


# ─── graph_encode: ScalarFunction ───────────────────────


def test_scalarfunction_emits_operator_and_acts_on(vec_space):
    """inv_sqrt(δ_kl Φ^k Φ^l) → ScalarFunction op + acts_on to (δ, Φ_k, Φ_l)."""
    Phi_k = _phi(vec_space, "k")
    Phi_l = _phi(vec_space, "l")
    r_sq = _delta(vec_space, "k", "l") * Phi_k * Phi_l
    expr = ScalarFunction("inv_sqrt", r_sq)

    g = graph_encode(expr)
    op_names = [n.name for n in g.nodes if n.kind == "operator"]
    assert op_names == ["ScalarFunction"]

    # acts_on edges from ScalarFunction to inner 3 factors.
    acts_on = [e for e in g.edges if e.kind == "acts_on"]
    assert len(acts_on) == 3

    # Inner contraction edges still emitted.
    contract = [e for e in g.edges if e.kind == "contraction"]
    assert len(contract) == 2
    assert all(e.space == "3:delta" for e in contract)  # v3.4 (dim,metric) token


# ─── HS1.0: ScalarFunction carries its potential class ──


def test_scalarfunction_node_carries_func_name(vec_space):
    """The ScalarFunction graph node records its function name (HS1.0)."""
    Phi_k = _phi(vec_space, "k")
    Phi_l = _phi(vec_space, "l")
    r_sq = _delta(vec_space, "k", "l") * Phi_k * Phi_l
    g = graph_encode(ScalarFunction("inv_sqrt", r_sq))
    sf = [n for n in g.nodes if n.name == "ScalarFunction"]
    assert len(sf) == 1
    assert sf[0].func_name == "inv_sqrt"


def test_potential_class_distinguishes_shapes():
    """1/r and a generic potential map to *different* feature ids — the whole
    point of HS1.0: SO(4) must key on inv_sqrt, not on any ScalarFunction."""
    from indexcalc.lions.ml.features_v25 import (
        node_feature_ids_v25, scalar_func_class_id,
    )
    assert scalar_func_class_id("inv_sqrt") == 2
    assert scalar_func_class_id("weird_pot") == 1   # generic fallback
    assert scalar_func_class_id("") == 0            # not a ScalarFunction

    f_kepler = node_feature_ids_v25(
        "operator", "ScalarFunction", 0, "bosonic", func_name="inv_sqrt")
    f_generic = node_feature_ids_v25(
        "operator", "ScalarFunction", 0, "bosonic", func_name="weird_pot")
    assert len(f_kepler) == 11 and len(f_generic) == 11
    # Identical everywhere except the potential-class slot (index 10).
    assert f_kepler[:10] == f_generic[:10]
    assert f_kepler[10] != f_generic[10]


def test_pyg_encoding_separates_kepler_from_generic_potential(tmp_path, vec_space):
    """Two Lagrangians identical but for the ScalarFunction name encode to
    *different* node-feature tensors (so the model can learn the split)."""
    import json
    from indexcalc.lions.serializer import (
        expr_to_dict, collect_spaces, space_to_dict,
    )
    from indexcalc.lions.ml.datasets_v25 import LionsV25Dataset

    def _make(name):
        Phi_k = _phi(vec_space, "k")
        Phi_l = _phi(vec_space, "l")
        r_sq = _delta(vec_space, "k", "l") * Phi_k * Phi_l
        L = ScalarFunction(name, r_sq)
        spaces = collect_spaces(L)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({
            "schema_version": "v2.5-catalog",
            "spaces": {nm: space_to_dict(sp) for nm, sp in spaces.items()},
            "rows": [{"expr": expr_to_dict(L), "primary": "SO(3)",
                      "labels": {}, "field_properties": {}, "provenance": "t"}],
        }))
        return LionsV25Dataset(p)[0].x

    x_kepler = _make("inv_sqrt")
    x_generic = _make("inverse")
    assert x_kepler.shape == x_generic.shape
    assert not bool((x_kepler == x_generic).all())


# ─── Kepler L composite ────────────────────────────────


def _kepler_expr(vec_space):
    Phi_i = _phi(vec_space, "i")
    Phi_j = _phi(vec_space, "j")
    kinetic = ScalarMul(
        0.5,
        _delta(vec_space, "i", "j")
        * TimeDeriv(Phi_i) * TimeDeriv(Phi_j),
    )
    Phi_k = _phi(vec_space, "k")
    Phi_l = _phi(vec_space, "l")
    r_sq = _delta(vec_space, "k", "l") * Phi_k * Phi_l
    potential = ScalarFunction("inv_sqrt", r_sq)
    return TensorSum(kinetic, potential)


def test_kepler_lagrangian_encodes(vec_space):
    expr = _kepler_expr(vec_space)
    g = graph_encode(expr)
    assert g is not None
    # Two TensorSum terms ⇒ num_terms == 2.
    assert g.num_terms == 2
    # ScalarMul(0.5, ...) at the kinetic branch flows into graph.scalar.
    assert g.scalar == 0.5 + 0j
    # All operators present.
    op_names = [n.name for n in g.nodes if n.kind == "operator"]
    assert op_names.count("TimeDeriv") == 2
    assert op_names.count("ScalarFunction") == 1


# ─── Serializer round-trip ──────────────────────────────


def test_serializer_roundtrip_timederiv(vec_space):
    Phi = _phi(vec_space, "i")
    expr = TimeDeriv(Phi)
    d = expr_to_dict(expr)
    spaces = collect_spaces(expr)
    back = expr_from_dict(d, spaces)
    assert isinstance(back, TimeDeriv)
    assert isinstance(back.expr, Tensor)
    assert back.expr.name == "Phi"


def test_serializer_roundtrip_scalarfunction(vec_space):
    Phi_k = _phi(vec_space, "k")
    Phi_l = _phi(vec_space, "l")
    expr = ScalarFunction("inv_sqrt",
                          _delta(vec_space, "k", "l") * Phi_k * Phi_l)
    d = expr_to_dict(expr)
    spaces = collect_spaces(expr)
    back = expr_from_dict(d, spaces)
    assert isinstance(back, ScalarFunction)
    assert back.name == "inv_sqrt"


def test_save_load_kepler_dataset(tmp_path, vec_space):
    expr = _kepler_expr(vec_space)
    sample = LabeledSample(
        expr=expr,
        labels={"O(3)": True, "SO(3)": True},
        mass_dim=0.0, field_counts={"Phi": 4},
        partial_count=0, invariant_counts={},
        provenance="test_kepler",
    )
    p = tmp_path / "kepler.json"
    save_dataset([sample], p)
    loaded = load_dataset(p)
    assert len(loaded) == 1
    s = loaded[0]
    assert s.labels == {"O(3)": True, "SO(3)": True}
    # Re-encode succeeds end-to-end.
    g = graph_encode(s.expr)
    assert g is not None
    assert g.num_terms == 2
