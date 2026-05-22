"""LIONS M9.6 acceptance — Einstein metric absorption.

η^{αβ} T_β = T^α (and the dual lower-form) normalization, so that
``f^A_{Pa} W^a W^B η_{AB}`` style expressions get reduced to
``f^A_{Pa} W^a W_A``, which the existing ``is_zero_by_antisym_swap`` then
recognizes as antisym × symmetric = 0.

Use cases:
- M9.6-A: bare η^{μν} T_μ U_ν → T^ν U_ν (one absorption).
- M9.6-B: bare η_{AB} W^A W^B contracted form → W·W (rank-1 trace).
- M9.6-C: end-to-end — apply_generator(W·W, su2_adj) → ZeroTensor.
- M9.6-D: end-to-end F·F under SU(2) adjoint → ZeroTensor. (Lorentz
  invariance on the **hand-built** η-explicit form spawns 4 terms from
  F's two frame slots × two F instances; cancelling all four needs
  antisym-sign-aware ``collect_scalar_terms`` — deferred to M9.7. On the
  enumerator's post-simplified form the cancellation does succeed.)
- M9.6-E: safety — η appears with a single host (self-trace) → no
  change (deferred to a later milestone).
- M9.6-F: safety — different-tensor X·Y·η must NOT be zeroed (X≠Y means
  the resulting X^a Y_A is not symmetric in (a,A)).
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.group import Group
from indexcalc.core.generator import make_su_n_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify, absorb_einstein_metric, canonical_form
from indexcalc.core.variation import ZeroTensor


@pytest.fixture
def st():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")


@pytest.fixture
def adj():
    return IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")


# ─── M9.6-A: bare η absorption (spacetime) ────────────────


def test_a_bare_eta_absorbs_into_lower(st):
    """η^{μν} T_μ U_ν → T^ν U_ν (free indices preserved)."""
    T = Tensor("T", [st.lower("μ")], reps={})
    U = Tensor("U", [st.lower("ν")], reps={})
    eta = Tensor("eta", [st.upper("μ"), st.upper("ν")],
                 symmetric_pairs=[(0, 1)], reps={})
    expr = TensorProduct(TensorProduct(T, U), eta)
    out = absorb_einstein_metric(expr)
    # Expect a TensorProduct of T^ν · U_ν (η gone), so just check eta absent.
    s = str(out)
    assert "eta" not in s, f"η not absorbed in {s!r}"
    # canonical_form should match a hand-built T^ν · U_ν
    Tnu = Tensor("T", [st.upper("ν")], reps={})
    Unu = Tensor("U", [st.lower("ν")], reps={})
    expected = TensorProduct(Tnu, Unu)
    assert canonical_form(out) == canonical_form(expected)


# ─── M9.6-B: bare η_{AB} W^A W^B ───────────────────────────


def test_b_bare_lower_eta_absorbs(adj):
    """η_{AB} W^A W^B → W^A W_A (one absorption)."""
    W1 = Tensor("W", [adj.upper("A")], reps={})
    W2 = Tensor("W", [adj.upper("B")], reps={})
    eta = Tensor("eta", [adj.lower("A"), adj.lower("B")],
                 symmetric_pairs=[(0, 1)], reps={})
    expr = TensorProduct(TensorProduct(W1, W2), eta)
    out = absorb_einstein_metric(expr)
    assert "eta" not in str(out)


# ─── M9.6-C: end-to-end W·W SU(2) invariance ─────────────


def test_c_ww_su2_adjoint_invariance(adj, st):
    """W^A_μ W^B_ν η_{AB} η^{μν} is SU(2) invariant — δ(W·W) → 0
    after metric absorption + antisym swap."""
    g = Group("SU(2)", dim=3, abelian=False)
    g.add_rep("adj", dim=3)
    g.add_rep("singlet", dim=1)
    gen = make_su_n_generator(g, adj, parameter_name="P")

    W1 = Tensor("W", [adj.upper("A"), st.lower("μ")],
                reps={"SU(2)": "adj"})
    W2 = Tensor("W", [adj.upper("B"), st.lower("ν")],
                reps={"SU(2)": "adj"})
    eta_adj = Tensor("eta", [adj.lower("A"), adj.lower("B")],
                     symmetric_pairs=[(0, 1)], reps={})
    eta_st = Tensor("eta", [st.upper("μ"), st.upper("ν")],
                    symmetric_pairs=[(0, 1)], reps={})
    L = TensorProduct(TensorProduct(W1, W2), TensorProduct(eta_adj, eta_st))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── M9.6-D: end-to-end F·F SU(2) + Lorentz invariance ────


def test_d_ff_su2_invariance(adj, st):
    """F^A_{μν} F^B_{ρσ} η_{AB} η^{μρ} η^{νσ} is SU(2)-invariant — δ(F·F)
    under adjoint produces 2 terms, both single-term antisym×sym = 0
    after metric absorption.

    Lorentz invariance on this hand-built form involves 4 terms (F has
    two frame slots × 2 F instances) and needs antisym-sign-aware
    ``collect_scalar_terms`` — that lives in M9.7. The end-to-end LIONS
    B2 labeler test uses the enumerator's post-simplified form, where
    the cancellation does succeed.
    """
    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("singlet", dim=1)
    su2_gen = make_su_n_generator(su2, adj, parameter_name="P")

    F1 = Tensor("F",
                [adj.upper("A"), st.lower("μ"), st.lower("ν")],
                antisymmetric_pairs=[(1, 2)],
                reps={"SU(2)": "adj", "Lorentz": "vector"})
    F2 = Tensor("F",
                [adj.upper("B"), st.lower("ρ"), st.lower("σ")],
                antisymmetric_pairs=[(1, 2)],
                reps={"SU(2)": "adj", "Lorentz": "vector"})
    eta_adj = Tensor("eta", [adj.lower("A"), adj.lower("B")],
                     symmetric_pairs=[(0, 1)], reps={})
    eta_mu = Tensor("eta", [st.upper("μ"), st.upper("ρ")],
                    symmetric_pairs=[(0, 1)], reps={})
    eta_nu = Tensor("eta", [st.upper("ν"), st.upper("σ")],
                    symmetric_pairs=[(0, 1)], reps={})
    L = TensorProduct(
        TensorProduct(F1, F2),
        TensorProduct(eta_adj, TensorProduct(eta_mu, eta_nu)),
    )

    delta = apply_generator(L, su2_gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── M9.6-E: self-trace is left untouched (deferred) ──────


def test_e_self_trace_not_absorbed(st):
    """η_{μν} T^{μν} (self-trace within one tensor) is NOT absorbed by
    this rule — host_0 == host_1 in the algorithm. Deferred milestone.
    """
    T = Tensor("T", [st.upper("μ"), st.upper("ν")], reps={})
    eta = Tensor("eta", [st.lower("μ"), st.lower("ν")],
                 symmetric_pairs=[(0, 1)], reps={})
    expr = TensorProduct(T, eta)
    out = absorb_einstein_metric(expr)
    assert "eta" in str(out), "self-trace must not absorb (deferred case)"


# ─── M9.6-F → D21b: adjoint Killing inner product is invariant ──


def test_f_xy_eta_adjoint_inner_product_invariant(adj):
    """η_{AB} X^A Y^B (X ≠ Y, both adjoint) IS SU(2)-invariant — the Killing
    inner product is Ad-invariant for a totally antisymmetric f:
    δL = f_{Bbc}(X^c Y^B + X^B Y^c) = (antisym in B,c)·(sym in B,c) = 0.

    M9.6 left this as a non-zero sum because the cancellation is *inter-term*
    (between the two Leibniz terms), which is_zero_by_antisym_swap — a per-
    product rule — cannot see. The D21b is_zero_by_antisym_term_cancellation
    pass folds the antisym parity across terms and now confirms δL = 0.
    (Distinct host tensors are fine: it is the structure, not instance
    identity, that matters.)"""
    g = Group("SU(2)", dim=3, abelian=False)
    g.add_rep("adj", dim=3)
    g.add_rep("singlet", dim=1)
    gen = make_su_n_generator(g, adj, parameter_name="P")

    X = Tensor("X", [adj.upper("A")], reps={"SU(2)": "adj"})
    Y = Tensor("Y", [adj.upper("B")], reps={"SU(2)": "adj"})
    eta_adj = Tensor("eta", [adj.lower("A"), adj.lower("B")],
                     symmetric_pairs=[(0, 1)], reps={})
    L = TensorProduct(TensorProduct(X, Y), eta_adj)

    final = simplify(apply_generator(L, gen))
    assert isinstance(final, ZeroTensor)


def test_f_xy_free_indices_not_zeroed(adj):
    """Control: X^A Y^B with *free* adjoint indices (no η contraction) is not a
    scalar and must NOT be zeroed — guards the D21b pass against over-firing."""
    g = Group("SU(2)", dim=3, abelian=False)
    g.add_rep("adj", dim=3)
    g.add_rep("singlet", dim=1)
    gen = make_su_n_generator(g, adj, parameter_name="P")

    X = Tensor("X", [adj.upper("A")], reps={"SU(2)": "adj"})
    Y = Tensor("Y", [adj.upper("B")], reps={"SU(2)": "adj"})
    final = simplify(apply_generator(TensorProduct(X, Y), gen))
    assert not isinstance(final, ZeroTensor)
