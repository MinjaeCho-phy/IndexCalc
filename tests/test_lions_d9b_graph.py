"""D9b — graph encoder acceptance.

Coverage per ``LIONS/notes/graph_encoding_spec.md §6``:
- |H|² = H · Hdag → 2 nodes, 1 contraction edge in su2_fund.
- F · F (M9.6 post-absorption form) → 2 F nodes + 3 contraction edges
  (1 in su2_adj + 2 in spacetime).
- ∂_μ H → 2 nodes (partial operator + H field) + 1 acts_on edge.
- ScalarMul(-0.5, ScalarMul(2, expr)) → graph.scalar == -1.
- ZeroTensor → None.
- B0/B1/B2/SM-lite at scale: every sample encodes without error and the
  reported node/edge counts make sense (no orphan free indices).
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import partial
from indexcalc.core.variation import ZeroTensor

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
    graph_encode,
    encode_sample,
    encode_dataset,
)
from indexcalc.lions.presets.su2_higgs_scalar import build_b0
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1
from indexcalc.lions.presets.su2_gauge import build_b2
from indexcalc.lions.presets.sm_lite import build_sm_lite


# ─── Spec §6.4: |H|² ────────────────────────────────────


def test_h_norm_squared_topology():
    """H^i · Hdag_i  →  2 field nodes, 1 su2_fund contraction edge."""
    from indexcalc.core.index import IndexSpace
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    g = graph_encode(TensorProduct(H, Hd))
    assert g is not None
    assert len(g.nodes) == 2
    assert all(n.kind == "field" for n in g.nodes)
    contractions = [e for e in g.edges if e.kind == "contraction"]
    assert len(contractions) == 1
    e = contractions[0]
    assert e.space == "su2_fund"
    # One upper + one lower (F1: position as edge attribute)
    assert {e.src_pos, e.dst_pos} == {"upper", "lower"}


# ─── Spec §6.5: F · F ───────────────────────────────────


def test_ff_topology():
    """F^A_{μν} F_A^{μν} → 2 field nodes + 3 contraction edges
    (1 in su2_adj, 2 in spacetime)."""
    from indexcalc.core.index import IndexSpace
    adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    F1 = Tensor(
        "F", [adj.upper("A"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    F2 = Tensor(
        "F", [adj.lower("A"), st.upper("μ"), st.upper("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    g = graph_encode(TensorProduct(F1, F2))
    assert g is not None
    assert len(g.nodes) == 2
    contractions = [e for e in g.edges if e.kind == "contraction"]
    spaces = sorted(e.space for e in contractions)
    assert spaces == ["spacetime", "spacetime", "su2_adj"]


# ─── Spec §6.6: PartialDeriv ────────────────────────────


def test_partial_h_topology():
    """∂_μ H → 2 nodes (partial operator + H) + 1 acts_on edge.
    μ is free (count==1 in expr), no contraction edge."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    dH = partial(H, st.lower("μ"))
    g = graph_encode(dH)
    assert g is not None
    assert len(g.nodes) == 2
    kinds = sorted(n.kind for n in g.nodes)
    assert kinds == ["field", "operator"]
    acts_on = [e for e in g.edges if e.kind == "acts_on"]
    assert len(acts_on) == 1
    contractions = [e for e in g.edges if e.kind == "contraction"]
    # i (in H) is free; μ (in partial) is free — no contractions.
    assert contractions == []


def test_kinetic_term_topology():
    """∂_μ H · ∂^μ Hdag — 4 nodes (2 partials + 2 fields), 2 acts_on
    + at least one contraction (μ)."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    dH = partial(H, st.lower("μ"))
    dHd = partial(Hd, st.upper("μ"))
    g = graph_encode(TensorProduct(dH, dHd))
    assert g is not None
    assert len(g.nodes) == 4
    acts_on = [e for e in g.edges if e.kind == "acts_on"]
    assert len(acts_on) == 2
    # μ contracted between two partial nodes; i contracted between H/Hdag.
    contractions = [e for e in g.edges if e.kind == "contraction"]
    spaces = sorted(e.space for e in contractions)
    assert spaces == ["spacetime", "su2_fund"]


# ─── Spec §6.3: scalar accumulation ────────────────────


def test_scalar_accumulation():
    """ScalarMul nesting collapses into one graph.scalar."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    T = Tensor("T", [st.upper("μ")], reps={})
    inner = ScalarMul(2.0, T)
    outer = ScalarMul(-0.5, inner)
    g = graph_encode(outer)
    assert g is not None
    assert g.scalar == -1
    # The Tensor node is still emitted (scalar lives outside the graph).
    assert len(g.nodes) == 1


# ─── F8: ZeroTensor → None ─────────────────────────────


def test_zero_tensor_returns_none():
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    z = ZeroTensor([st.upper("μ")])
    assert graph_encode(z) is None


# ─── Preset coverage ────────────────────────────────────


@pytest.fixture(params=[
    ("b0", build_b0, lambda b: {"SU(2)": b.su2_gen, "U(1)_Y": b.u1y_gen}),
    ("b1", build_b1, lambda b: {"SU(2)": b.su2_gen, "U(1)_Y": b.u1y_gen,
                                "Lorentz": b.lorentz_gen}),
    ("b2", build_b2, lambda b: {"SU(2)": b.su2_gen, "Lorentz": b.lorentz_gen}),
])
def preset_setup(request):
    name, build, gens = request.param
    setup = build()
    return name, setup, gens(setup)


def test_preset_dataset_encodes(preset_setup):
    """Every preset sample encodes without error and yields ≥1 node."""
    name, setup, generators = preset_setup
    caps = EnumeratorCaps(
        max_field_total=3, max_per_field=2,
        max_partials_total=1, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        setup.fields, spacetime=setup.spacetime, caps=caps,
        invariant_alphabet=getattr(setup, "invariant_alphabet", None),
        forbid_like_position_spaces=(
            {setup.dirac} if hasattr(setup, "dirac") else None
        ),
    )
    labeled = label_samples(samples, generators)
    graphs = encode_dataset(labeled)
    assert len(graphs) == len(labeled), f"{name}: drop count mismatch"
    for g in graphs:
        assert len(g.nodes) >= 1
        # No free index left as orphan in invariant samples — every
        # contraction edge has a matched space + positions.
        for e in g.edges:
            # Self-loops are valid (a tensor with two contracted indices,
            # e.g. γ^μ_α^α trace inside an invariant).
            assert e.src <= e.dst
            if e.kind == "contraction":
                assert e.space != ""


# ─── SM-lite at scale ───────────────────────────────────


@pytest.fixture(scope="module")
def sm_dataset():
    sm = build_sm_lite()
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=1, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        sm.fields, spacetime=sm.spacetime, caps=caps,
        invariant_alphabet=sm.invariant_alphabet,
        forbid_like_position_spaces={sm.dirac},
    )
    generators = {
        "SU(2)": sm.su2_gen, "U(1)_Y": sm.u1y_gen, "Lorentz": sm.lorentz_gen,
    }
    return sm, label_samples(samples, generators)


def test_sm_lite_full_encode(sm_dataset):
    _sm, labeled = sm_dataset
    graphs = encode_dataset(labeled)
    assert len(graphs) >= 500
    # Sanity: at least one graph carries labels (proves encode_sample
    # propagates them).
    assert any(g.labels for g in graphs)
    # Average node count is reasonable (not blowing up).
    avg_nodes = sum(len(g.nodes) for g in graphs) / len(graphs)
    assert 1 < avg_nodes < 20
