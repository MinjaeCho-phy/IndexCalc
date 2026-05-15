"""LIONS M9.5 acceptance — [Σ^{ab}, P_L] = [Σ^{ab}, P_R] = [Σ^{ab}, γ_5] = 0.

신규 도구: ``apply_sigma_projector_commute`` — Σ 를 chiral projector / γ_5 의
오른쪽으로 push 하는 정규화. Σ ∝ [γ^a, γ^b] 가 짝수 개의 γ → P_{L,R}/γ_5 와
commute.

End-to-end target: $\\bar\\psi P_L \\psi$, $\\bar\\psi P_R \\psi$ Lorentz invariance
— 두 Leibniz 항이 Σ 위치만 다르고 부호 반대 → simplify 가 normal form 으로
정규화 후 collect_scalar_terms 가 cancel → ZeroTensor.

설계 노트: `notes/m9_5_sigma_projector_commute.md` (LIONS).
"""

import pytest

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_lorentz_spinor_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify, apply_sigma_projector_commute


# ─── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def setup():
    """Minkowski + Dirac spinor + Lorentz (spinor / conj_spinor / vector / singlet)."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδερστυφ")

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )
    return st, dirac, lorentz, gen


def make_psi(dirac, idx="α"):
    return Tensor(
        "psi", [dirac.upper(idx)],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )


def make_psibar(dirac, idx="α"):
    return Tensor(
        "psibar", [dirac.lower(idx)],
        reps={"Lorentz": "conj_spinor"}, statistics="fermionic",
    )


def make_P_L(dirac, up="α", down="β"):
    return Tensor("P_L", [dirac.upper(up), dirac.lower(down)], reps={})


def make_P_R(dirac, up="α", down="β"):
    return Tensor("P_R", [dirac.upper(up), dirac.lower(down)], reps={})


def make_gamma_5(dirac, up="α", down="β"):
    return Tensor("gamma_5", [dirac.upper(up), dirac.lower(down)], reps={})


def make_Sigma(st, dirac, a="a", b="b", row="α", col="β"):
    return Tensor(
        "Sigma",
        [Index(a, st, "upper"),
         Index(b, st, "upper"),
         dirac.upper(row),
         dirac.lower(col)],
        antisymmetric_pairs=[(0, 1)],
    )


# ─── 1. Σ · P_L → P_L · Σ (단일 rewrite) ─────────────────────


def test_sigma_PL_commutes_to_right(setup):
    """Σ^{ab,α}{}_β · P_L^β{}_γ → P_L^α{}_ρ · Σ^{ab,ρ}{}_γ.

    apply_sigma_projector_commute 직접 호출 — sign 없음.
    """
    st, dirac, *_ = setup
    Sigma = make_Sigma(st, dirac, "a", "b", "α", "β")
    PL = make_P_L(dirac, "β", "γ")
    expr = TensorProduct(Sigma, PL)

    out = apply_sigma_projector_commute(expr)

    assert isinstance(out, TensorProduct)
    # P_L is now on the left, Σ on the right
    assert isinstance(out.left, Tensor) and out.left.name == "P_L"
    assert isinstance(out.right, Tensor) and out.right.name == "Sigma"

    # Outer indices preserved: P_L row=α (was Σ.row), Σ col=γ (was P.col)
    assert out.left.indices[0].name == "α"
    assert out.right.indices[3].name == "γ"


# ─── 2. Σ · γ_5 → γ_5 · Σ ──────────────────────────────────


def test_sigma_gamma5_commutes_to_right(setup):
    st, dirac, *_ = setup
    Sigma = make_Sigma(st, dirac, "a", "b", "α", "β")
    g5 = make_gamma_5(dirac, "β", "γ")
    expr = TensorProduct(Sigma, g5)

    out = apply_sigma_projector_commute(expr)
    assert isinstance(out, TensorProduct)
    assert isinstance(out.left, Tensor) and out.left.name == "gamma_5"
    assert isinstance(out.right, Tensor) and out.right.name == "Sigma"


# ─── 3. End-to-end Lorentz: $\bar\psi P_L \psi$ ──────────────


def test_psibar_PL_psi_lorentz_invariant(setup):
    """$\\delta_{\\rm Lorentz}(\\bar\\psi P_L \\psi) = 0$.

    δψ̄ = +(i/2) ψ̄ Σ → +(i/2) ψ̄ Σ P_L ψ → +(i/2) ψ̄ P_L Σ ψ (commute).
    δψ  = -(i/2) Σ ψ → -(i/2) ψ̄ P_L Σ ψ.
    합 = 0 (cancel via collect_scalar_terms).
    """
    st, dirac, _lorentz, gen = setup
    psibar = make_psibar(dirac, "α")
    PL = make_P_L(dirac, "α", "β")
    psi = make_psi(dirac, "β")

    L = TensorProduct(psibar, TensorProduct(PL, psi))
    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── 4. End-to-end Lorentz: $\bar\psi P_R \psi$ ──────────────


def test_psibar_PR_psi_lorentz_invariant(setup):
    st, dirac, _lorentz, gen = setup
    psibar = make_psibar(dirac, "α")
    PR = make_P_R(dirac, "α", "β")
    psi = make_psi(dirac, "β")

    L = TensorProduct(psibar, TensorProduct(PR, psi))
    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"


# ─── 5. Sanity: $\bar\psi \psi$ Lorentz invariance regression ─


def test_dirac_mass_term_still_lorentz_invariant(setup):
    """M4 mass term regression — Σ-projector commute simplifier 추가 후에도
    plain $\\bar\\psi \\psi$ Lorentz invariance 가 유지되는지."""
    st, dirac, _lorentz, gen = setup
    psibar = make_psibar(dirac, "α")
    psi = make_psi(dirac, "α")

    L = TensorProduct(psibar, psi)
    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {type(final).__name__}: {final!r}"
