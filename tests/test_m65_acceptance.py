"""LIONS M6.5 acceptance — free Dirac kinetic term Lorentz invariance via Clifford simplifier.

검증 대상:

1. **Clifford rewriter** ``apply_clifford_sigma_gamma`` (단독 단위 검증).
   IR-level Clifford-vector consistency identity (M4 generator convention):

   $$\\Sigma^{ab,\\alpha}{}_\\beta\\,\\gamma^{\\mu,\\beta}{}_\\gamma
   \\;\\to\\;
   \\gamma^{\\mu,\\alpha}{}_\\beta\\,\\Sigma^{ab,\\beta}{}_\\gamma
   + (-2i)\\,M^{ab,\\mu}{}_\\rho\\,\\gamma^{\\rho,\\alpha}{}_\\gamma$$

   ``Sigma`` 부호 ($-i/2$, $+i/2$) + vector $M$ 부호 ($\\delta V_\\mu = -V_\\nu M^{ab,\\nu}{}_\\mu$)
   에서 유도된 coefficient $-2i$. 이 identity 가 ``simplify`` 파이프라인에 들어가야
   $\\delta(i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi)=0$ 가 IR 만으로 검증된다.

2. **End-to-end:** $\\delta_{\\rm Lorentz}(i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi) = 0$.

   Leibniz 세 항:
       (a) $\\delta\\bar\\psi$ : $-\\tfrac12 \\bar\\psi\\Sigma\\gamma\\partial\\psi$
       (b) $\\delta\\psi$    : $+\\tfrac12 \\bar\\psi\\gamma\\Sigma\\partial\\psi$
       (c) $\\delta\\partial$: $-i\\bar\\psi\\gamma^\\mu M^{ab,\\nu}{}_\\mu \\partial_\\nu\\psi$

   (a) 에 Clifford 적용 → $-\\tfrac12 \\bar\\psi\\gamma\\Sigma\\partial\\psi$ + $i\\bar\\psi M\\gamma\\partial\\psi$.
   (a-swap) + (b) cancel. (a-correction) + (c) cancel
   ($M\\gamma$ vs $\\gamma M$ factor commutativity).
"""

import pytest

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul, TensorSum
from indexcalc.core.deriv import PartialDeriv, partial
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_lorentz_spinor_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify, apply_clifford_sigma_gamma


# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def setup():
    """Minkowski + Dirac + Lorentz (spinor + conj_spinor + vector + singlet)."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )
    return st, dirac, lorentz, gen


def make_psi(dirac, idx_name="α"):
    return Tensor(
        "psi", [dirac.upper(idx_name)],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )


def make_psibar(dirac, idx_name="α"):
    return Tensor(
        "psibar", [dirac.lower(idx_name)],
        reps={"Lorentz": "conj_spinor"}, statistics="fermionic",
    )


def make_gamma(st, dirac, mu="μ", row="α", col="β"):
    """γ^μ — invariant tensor (no rep tag); frame + spinor (row up, col down)."""
    return Tensor(
        "gamma",
        [st.upper(mu), dirac.upper(row), dirac.lower(col)],
    )


# ─── M6.5-A: Clifford rewriter standalone behavior ───────────


def test_clifford_sigma_gamma_produces_two_terms(setup):
    """``apply_clifford_sigma_gamma`` 가 Σ-γ 패턴에서 두 term sum 을 만든다.

    Input pattern: $\\bar\\psi \\cdot \\Sigma^{ab,\\beta}{}_\\alpha \\cdot \\gamma^{\\mu,\\alpha}{}_\\gamma \\cdot \\psi$.
    Expected: TensorSum with
        Term 1: ``Sigma`` 와 ``gamma`` factor 둘 다 살아남되 spinor contraction 순서
                 바뀜 (γ-Σ).
        Term 2: ScalarMul(-2j, ...) 에 ``Sigma`` 없음 + ``M_vec`` 출현 + ``gamma``는
                 vector index 가 dummy 로 바뀌어 M_vec 와 contract.
    """
    st, dirac, *_ = setup
    psibar = make_psibar(dirac, "β")
    psi = make_psi(dirac, "γ")
    Sigma = Tensor(
        "Sigma",
        [
            Index("a", st, "upper"),
            Index("b", st, "upper"),
            dirac.upper("β"),   # row → ψ̄
            dirac.lower("α"),   # col → γ.row
        ],
        antisymmetric_pairs=[(0, 1)],
    )
    gamma = make_gamma(st, dirac, mu="μ", row="α", col="γ")
    expr = TensorProduct(psibar, TensorProduct(Sigma, TensorProduct(gamma, psi)))

    out = apply_clifford_sigma_gamma(expr)
    assert isinstance(out, TensorSum), f"expected TensorSum, got {type(out).__name__}"

    def collect_names(e, acc):
        if isinstance(e, Tensor):
            acc.append(e.name)
        elif isinstance(e, TensorProduct):
            collect_names(e.left, acc); collect_names(e.right, acc)
        elif isinstance(e, ScalarMul):
            collect_names(e.expr, acc)
        return acc

    # Term 1: still has Sigma + gamma (swap), no -2j scalar.
    names_t1 = collect_names(out.left, [])
    assert "Sigma" in names_t1
    assert "gamma" in names_t1
    assert "M_vec" not in names_t1

    # Term 2: ScalarMul(-2j, ...) with M_vec + gamma, no Sigma.
    t2 = out.right
    assert isinstance(t2, ScalarMul)
    assert t2.scalar == -2j
    names_t2 = collect_names(t2.expr, [])
    assert "M_vec" in names_t2
    assert "gamma" in names_t2
    assert "Sigma" not in names_t2


def test_clifford_no_pattern_returns_unchanged(setup):
    """Σ 없거나 γ 없으면 무변환."""
    _st, dirac, *_ = setup
    psi = make_psi(dirac, "α")
    psibar = make_psibar(dirac, "α")
    expr = TensorProduct(psibar, psi)
    out = apply_clifford_sigma_gamma(expr)
    assert out is expr


# ─── M6.5-B: End-to-end Dirac kinetic Lorentz invariance ────


def test_dirac_kinetic_lorentz_invariant(setup):
    """$\\delta_{\\rm Lorentz}\\bigl(i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi\\bigr) = 0$.

    Clifford rewrite (Σγ → γΣ -2i M·γ) + factor commutativity (M·γ vs γ·M)
    + dummy renaming + collect_scalar_terms → ZeroTensor.
    """
    st, dirac, _lorentz, gen = setup
    psibar = make_psibar(dirac, "α")
    psi = make_psi(dirac, "γ")
    gamma = make_gamma(st, dirac, mu="μ", row="α", col="γ")
    dpsi = partial(psi, st.lower("μ"))  # ∂_μ ψ (μ contracts with γ vector)

    # L = i ψ̄ γ^μ ∂_μ ψ
    L = ScalarMul(1j, TensorProduct(psibar, TensorProduct(gamma, dpsi)))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_full_dirac_lagrangian_lorentz_invariant(setup):
    """$\\delta_{\\rm Lorentz}\\bigl(i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi - m\\bar\\psi\\psi\\bigr) = 0$.

    Kinetic part: M6.5 Clifford 경로. Mass part: M4 path (Σ row/col cancel).
    """
    st, dirac, _lorentz, gen = setup

    # 운동 부분
    psibar_k = make_psibar(dirac, "α")
    psi_k = make_psi(dirac, "γ")
    gamma = make_gamma(st, dirac, mu="μ", row="α", col="γ")
    dpsi = partial(psi_k, st.lower("μ"))
    L_kin = ScalarMul(1j, TensorProduct(psibar_k, TensorProduct(gamma, dpsi)))

    # 질량 부분 — 다른 인덱스 이름 사용
    psibar_m = make_psibar(dirac, "δ")
    psi_m = make_psi(dirac, "δ")
    L_mass = ScalarMul(-1.0, TensorProduct(psibar_m, psi_m))

    L = TensorSum(L_kin, L_mass)
    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"
