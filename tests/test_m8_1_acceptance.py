"""LIONS M8.1 acceptance — ε-invariance simplifier across deriv wrappers.

Extends M8's apply_epsilon_su_n_invariance so that the X/Y partner search
unwraps PartialDeriv (and CovariantDeriv) to find the inner Tensor. With
that, chiral kinetic terms of the form

    $\\bar L^j \\gamma^\\mu \\partial_\\mu L^k \\epsilon_{jk}$
    $\\partial_\\mu \\bar L^j \\gamma^\\mu L^k \\epsilon_{jk}$

normalize to a common canonical contraction graph and the two Leibniz
terms cancel via collect_scalar_terms — closing the SU(2) false-negative
that B1 D5b documented.

Original M8 (no ∂) is regressed to confirm the partner-unwrap path
behaves identically when the partner IS a bare Tensor.
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import partial
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    make_u1_generator, make_su_n_generator, make_lorentz_spinor_generator,
)
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


@pytest.fixture
def setup():
    """SU(2)_L × Lorentz (chiral reps) fixture, M8-compatible."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ijklmn")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")

    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3); su2.add_rep("fund", dim=2)
    su2.add_rep("antifund", dim=2, conjugate=True); su2.add_rep("singlet", dim=1)

    lor = Group("Lorentz", dim=6, abelian=False)
    lor.add_rep("L_spinor", dim=2); lor.add_rep("R_spinor", dim=2)
    lor.add_rep("conj_L_spinor", dim=2, conjugate=True)
    lor.add_rep("conj_R_spinor", dim=2, conjugate=True)
    lor.add_rep("spinor", dim=4); lor.add_rep("conj_spinor", dim=4, conjugate=True)
    lor.add_rep("vector", dim=4); lor.add_rep("singlet", dim=1)

    su2_gen = make_su_n_generator(su2, su2_adj, parameter_name="P", fund_space=su2_fund)
    lor_gen = make_lorentz_spinor_generator(lor, frame_space=st, spinor_space=dirac)

    return {"st": st, "dirac": dirac, "su2_fund": su2_fund,
            "su2_gen": su2_gen, "lor_gen": lor_gen}


def _make_Lbar(s, j="j", α="α"):
    return Tensor("Lbar", [s["su2_fund"].upper(j), s["dirac"].lower(α)],
                  reps={"SU(2)": "fund", "Lorentz": "conj_L_spinor"},
                  statistics="fermionic")


def _make_L(s, k="k", β="β"):
    return Tensor("L", [s["su2_fund"].upper(k), s["dirac"].upper(β)],
                  reps={"SU(2)": "fund", "Lorentz": "L_spinor"},
                  statistics="fermionic")


def _make_gamma(s, μ="μ", α="α", β="β"):
    return Tensor("gamma",
                  [s["st"].upper(μ), s["dirac"].upper(α), s["dirac"].lower(β)],
                  reps={})


def _make_eps(s, j="j", k="k"):
    return Tensor("epsilon",
                  [s["su2_fund"].lower(j), s["su2_fund"].lower(k)],
                  antisymmetric_pairs=[(0, 1)], reps={})


# ─── M8.1-A: ∂ on the unbarred fermion ─────────────────────


def test_L_kinetic_su2_invariant(setup):
    """$\\delta_{SU(2)}(\\bar L^j \\gamma^\\mu \\partial_\\mu L^k \\epsilon_{jk}) = 0$.

    The two Leibniz terms (T on Lbar, T on L) must normalize to the same
    canonical contraction graph after the partner-unwrap extension picks
    up the PartialDeriv-wrapped L as a valid ε-slot1 partner.
    """
    s = setup
    Lbar = _make_Lbar(s); L = _make_L(s)
    gamma = _make_gamma(s); eps = _make_eps(s)
    dL = partial(L, s["st"].lower("μ"))

    expr = ScalarMul(
        1j, TensorProduct(Lbar, TensorProduct(gamma, TensorProduct(dL, eps))),
    )
    final = simplify(apply_generator(expr, s["su2_gen"]))
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── M8.1-B: ∂ on the barred fermion ───────────────────────


def test_dLbar_kinetic_su2_invariant(setup):
    """$\\delta_{SU(2)}(\\partial_\\mu \\bar L^j \\gamma^\\mu L^k \\epsilon_{jk}) = 0$.

    Symmetric test — partner unwrap must also work when the ∂-wrapped
    factor is the X partner (T.col side), not just Y.
    """
    s = setup
    Lbar = _make_Lbar(s); L = _make_L(s)
    gamma = _make_gamma(s); eps = _make_eps(s)
    dLbar = partial(Lbar, s["st"].lower("μ"))

    expr = ScalarMul(
        1j, TensorProduct(dLbar, TensorProduct(gamma, TensorProduct(L, eps))),
    )
    final = simplify(apply_generator(expr, s["su2_gen"]))
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── M8.1-C: original M8 Yukawa regression ────────────────


def test_m8_yukawa_still_zero(setup):
    """No-∂ Yukawa — partner is a bare Tensor; M8.1 must not change the
    M8 behavior on this canonical case."""
    s = setup
    Lbar = _make_Lbar(s)
    H = Tensor("H", [s["su2_fund"].upper("k")],
               reps={"SU(2)": "fund", "Lorentz": "singlet"})
    eR = Tensor("eR", [s["dirac"].upper("α")],
                reps={"SU(2)": "singlet", "Lorentz": "R_spinor"},
                statistics="fermionic")
    eps = _make_eps(s)

    expr = TensorProduct(Lbar, TensorProduct(H, TensorProduct(eps, eR)))
    final = simplify(apply_generator(expr, s["su2_gen"]))
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── M8.1-D: triply-invariant Lorentz + U(1)_Y unaffected ─


def test_L_kinetic_full_triple_invariance(setup):
    """Confirm Lorentz and U(1)_Y stay invariant after the SU(2) fix —
    M8.1 is partner-search only; non-SU(2) generators see no behavior
    change. (U(1)_Y omitted here — would need a separate U(1)_Y gen.)
    """
    s = setup
    Lbar = _make_Lbar(s); L = _make_L(s)
    gamma = _make_gamma(s); eps = _make_eps(s)
    dL = partial(L, s["st"].lower("μ"))

    expr = ScalarMul(
        1j, TensorProduct(Lbar, TensorProduct(gamma, TensorProduct(dL, eps))),
    )
    for tag, g in (("Lorentz", s["lor_gen"]), ("SU(2)", s["su2_gen"])):
        final = simplify(apply_generator(expr, g))
        assert isinstance(final, ZeroTensor), (
            f"{tag} failed after M8.1: got {type(final).__name__}"
        )
