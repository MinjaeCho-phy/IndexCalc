"""LIONS M1 acceptance test — free complex scalar with U(1) symmetry.

End-to-end 시연:
  1. U(1) group 등록 (charge ±1 reps)
  2. complex scalar φ, φ* 를 reps tag와 함께 Tensor로 선언
  3. kinetic term L_kin = (∂_μ φ)(∂^μ φ^*) 빌드
  4. apply_generator로 δL 계산
  5. δL이 두 Leibniz 항을 가지며 각각 scalar coefficient ±i를 갖는지 확인
     (full cancellation은 simplifier가 들어오는 M2의 일.)

이 테스트는 M1의 모든 신규 모듈을 한 파이프라인에 묶어 검증한다.
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.deriv import PartialDeriv, partial
from indexcalc.core.group import Group
from indexcalc.core.invariant_tensors import (
    InvariantTensorRegistry,
    standard_lorentz_invariants,
)
from indexcalc.core.generator import make_u1_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.variation import ZeroTensor


# ─── Setup ─────────────────────────────────────────────────


@pytest.fixture
def setup():
    """U(1) + complex scalar 시스템."""
    # Index space
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")

    # Group + reps
    u1 = Group("U(1)", dim=1, abelian=True)
    u1.add_rep("+1", dim=1, charge=1.0)
    u1.add_rep("-1", dim=1, charge=-1.0)

    # Generator
    gen = make_u1_generator(u1)

    # Invariant tensors (η used implicitly by IndexSpace; eta tag here
    # just demonstrates registry usage)
    inv_reg = InvariantTensorRegistry()
    for inv in standard_lorentz_invariants():
        inv_reg.declare(inv.name, inv.group_name, inv.index_pattern, inv.symmetry)

    return st, u1, gen, inv_reg


# ─── Pipeline pieces ─────────────────────────────────────────


def test_field_with_rep_tag(setup):
    """φ는 reps 태그를 갖는 Tensor."""
    st, *_ = setup
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    phistar = Tensor("phistar", [], reps={"U(1)": "-1"})
    assert phi.reps == {"U(1)": "+1"}
    assert phistar.reps == {"U(1)": "-1"}
    assert phi.statistics == "bosonic"


def test_apply_to_charged_scalar(setup):
    """δφ = i·(+1)·φ, δφ* = i·(-1)·φ*."""
    st, _u1, gen, _ = setup
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    phistar = Tensor("phistar", [], reps={"U(1)": "-1"})

    dphi = apply_generator(phi, gen)
    dphistar = apply_generator(phistar, gen)

    assert isinstance(dphi, ScalarMul) and dphi.scalar == 1j
    assert isinstance(dphistar, ScalarMul) and dphistar.scalar == -1j


def test_apply_to_partial_derivative_commutes(setup):
    """δ(∂_μ φ) = ∂_μ(δφ) — global symmetry이므로 ∂와 δ는 commute."""
    st, _u1, gen, _ = setup
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    mu = st.lower("μ")
    dmu_phi = partial(phi, mu)

    result = apply_generator(dmu_phi, gen)
    assert isinstance(result, PartialDeriv)
    inner = result.expr
    assert isinstance(inner, ScalarMul)
    assert inner.scalar == 1j


# ─── Full kinetic term ──────────────────────────────────────


def test_kinetic_term_leibniz_expansion(setup):
    """L_kin = (∂_μ φ)·(∂^μ φ*); δL_kin은 두 Leibniz 항.

    (∂_μ(δφ))·(∂^μ φ*) + (∂_μ φ)·(∂^μ(δφ*))
    = i (∂_μ φ)·(∂^μ φ*)  +  (-i) (∂_μ φ)·(∂^μ φ*)

    두 항의 scalar 계수는 각각 +i, -i. (full cancellation 검증은 M2/E9)
    """
    st, _u1, gen, _ = setup
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    phistar = Tensor("phistar", [], reps={"U(1)": "-1"})
    mu_lo = st.lower("μ")
    mu_up = st.upper("μ")

    dmu_phi = partial(phi, mu_lo)        # ∂_μ φ
    dmu_phistar = partial(phistar, mu_up)  # ∂^μ φ*  (생성자에서 flip)
    L_kin = TensorProduct(dmu_phi, dmu_phistar)

    delta_L = apply_generator(L_kin, gen)

    # Leibniz: TensorSum of two TensorProducts
    assert isinstance(delta_L, TensorSum)
    left, right = delta_L.left, delta_L.right
    assert isinstance(left, TensorProduct)
    assert isinstance(right, TensorProduct)

    # 좌항: (∂_μ(δφ)) · (∂^μ φ*)
    #   = ∂_μ(i·φ) · (∂^μ φ*)  ← inner ScalarMul
    inner_d_dphi = left.left  # PartialDeriv
    assert isinstance(inner_d_dphi, PartialDeriv)
    sc = inner_d_dphi.expr
    assert isinstance(sc, ScalarMul) and sc.scalar == 1j

    # 우항: (∂_μ φ) · (∂^μ(δφ*))
    inner_d_dphistar = right.right  # PartialDeriv on δφ*
    assert isinstance(inner_d_dphistar, PartialDeriv)
    sc = inner_d_dphistar.expr
    assert isinstance(sc, ScalarMul) and sc.scalar == -1j


def test_singlet_field_unchanged(setup):
    """U(1) singlet인 field — reps에 U(1) tag 없음 — 의 변환은 0."""
    st, _u1, gen, _ = setup
    psi = Tensor("psi", [], reps={})  # singlet w.r.t. U(1)
    mu = st.lower("μ")
    L = TensorProduct(partial(psi, mu), Tensor("c", []))  # ∂_μ ψ · c
    # c는 ZeroTensor가 아니지만 reps={}이므로 generator 작용은 둘 다 0.
    # 결과: 0 + 0 = ZeroTensor (또는 simplify 후 ZeroTensor).
    delta_L = apply_generator(L, gen)
    assert isinstance(delta_L, ZeroTensor)


# ─── Smoke: invariant tensor registry usage ────────────────


def test_lorentz_eta_registered(setup):
    """spacetime metric η가 registry에 invariant tensor로 등록되어 있다."""
    _, _, _, inv_reg = setup
    assert inv_reg.is_invariant("eta", "Lorentz")
    eta = inv_reg.get("eta", "Lorentz")
    assert eta.symmetry == "symmetric"
    assert len(eta.index_pattern) == 2
