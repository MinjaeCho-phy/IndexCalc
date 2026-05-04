"""LIONS M3 acceptance — Dirac Lagrangian global U(1) invariance.

검증 대상: $L_D = i\\bar\\psi \\gamma^\\mu \\partial_\\mu \\psi - m \\bar\\psi \\psi$.
Global U(1): $\\delta\\psi = +iq\\psi,\\ \\delta\\bar\\psi = -iq\\bar\\psi$. 각 항이
bilinear in $(\\psi, \\bar\\psi)$이므로 phase가 cancel → $\\delta L_D = 0$.

검증 파이프라인: ``apply_generator → simplify`` (pull_scalars + collect_scalar_terms).

**제외 (M4 예정):**
- Lorentz invariance 증명: Σ-γ commutator + Clifford algebra 적용 simplifier 필요.
- Local U(1) (gauge): $A_\\mu \\to A_\\mu + \\frac{1}{e}\\partial_\\mu\\alpha$, 매개변수
  $\\alpha(x)$의 spacetime 의존성과 cancellation.
- 본격적 fermion 부호 추적 (Yukawa, Fierz reordering).
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import partial
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_u1_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


# ─── Setup ──────────────────────────────────────────────────


@pytest.fixture
def setup():
    """Spacetime + spinor + U(1) + Dirac fields."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")  # no metric (자체 metric 없음)

    u1 = Group("U(1)", dim=1, abelian=True)
    u1.add_rep("+1", dim=1, charge=1.0)
    u1.add_rep("-1", dim=1, charge=-1.0)
    gen = make_u1_generator(u1)

    return st, dirac, u1, gen


def make_psi(dirac, idx_name="α"):
    """ψ^α — Dirac field with U(1) charge +1, fermionic."""
    return Tensor(
        "psi",
        [dirac.upper(idx_name)],
        reps={"U(1)": "+1"},
        statistics="fermionic",
    )


def make_psibar(dirac, idx_name="α"):
    """ψ̄_α — Dirac conjugate with U(1) charge -1, fermionic."""
    return Tensor(
        "psibar",
        [dirac.lower(idx_name)],
        reps={"U(1)": "-1"},
        statistics="fermionic",
    )


def make_gamma(st, dirac, mu_name="μ", upper_name="α", lower_name="β"):
    """γ^μ — Lorentz frame index μ + spinor (upper, lower) — no rep tag (invariant)."""
    return Tensor(
        "gamma",
        [
            st.upper(mu_name),
            dirac.upper(upper_name),
            dirac.lower(lower_name),
        ],
    )


# ─── Mass term ──────────────────────────────────────────────


def test_dirac_mass_u1_invariant(setup):
    """δ(-m ψ̄ ψ) = 0 under global U(1).

    Leibniz: $-m[(\\delta\\bar\\psi)\\psi + \\bar\\psi(\\delta\\psi)]$
    $= -m[(-iq)\\bar\\psi\\psi + iq \\bar\\psi\\psi] = 0$.
    """
    st, dirac, u1, gen = setup
    psi = make_psi(dirac, "α")
    psibar = make_psibar(dirac, "α")
    mass = ScalarMul(-1.0, TensorProduct(psibar, psi))  # -m factored as -1 here

    delta = apply_generator(mass, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── Kinetic term ───────────────────────────────────────────


def test_dirac_kinetic_u1_invariant(setup):
    """δ(i ψ̄ γ^μ ∂_μ ψ) = 0 under global U(1).

    Leibniz: $(\\delta\\bar\\psi) γ^μ ∂_μ\\psi + \\bar\\psi γ^μ ∂_μ(\\delta\\psi)$
    $= (-iq)\\bar\\psi γ^μ ∂_μ\\psi + iq \\bar\\psi γ^μ ∂_μ\\psi = 0$
    (γ^μ는 invariant, ∂_μ는 변환과 commute, scalar factor는 ∂ 통과).
    """
    st, dirac, u1, gen = setup
    psi = make_psi(dirac, "β")
    psibar = make_psibar(dirac, "α")
    gamma = make_gamma(st, dirac, "μ", "α", "β")
    # ∂_μ ψ  (μ contracts with γ's μ upper)
    dpsi = partial(psi, st.lower("μ"))

    # i * ψ̄ * γ^μ * ∂_μ ψ
    body = TensorProduct(psibar, TensorProduct(gamma, dpsi))
    L_kin = ScalarMul(1j, body)

    delta = apply_generator(L_kin, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── Full Dirac Lagrangian ──────────────────────────────────


def test_full_dirac_lagrangian_u1_invariant(setup):
    """δ(i ψ̄ γ^μ ∂_μ ψ - m ψ̄ ψ) = 0 under global U(1)."""
    st, dirac, u1, gen = setup

    # 운동 부분
    psi_k = make_psi(dirac, "β")
    psibar_k = make_psibar(dirac, "α")
    gamma = make_gamma(st, dirac, "μ", "α", "β")
    dpsi = partial(psi_k, st.lower("μ"))
    L_kin = ScalarMul(1j, TensorProduct(psibar_k, TensorProduct(gamma, dpsi)))

    # 질량 부분 (별도 인덱스 이름 사용 — 두 항이 합쳐질 때 free 인덱스 0 매칭만 필요)
    psi_m = make_psi(dirac, "γ")
    psibar_m = make_psibar(dirac, "γ")
    L_mass = ScalarMul(-1.0, TensorProduct(psibar_m, psi_m))

    from indexcalc.core.tensor import TensorSum
    L_D = TensorSum(L_kin, L_mass)

    delta = apply_generator(L_D, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── Sanity: 비-bilinear는 invariant 아님 ───────────────────


def test_non_invariant_term_not_zero(setup):
    """ψ ψ (둘 다 charge +1) — 항은 invariant 아님.

    $\\delta(\\psi\\psi) = (iq)\\psi\\psi + \\psi(iq)\\psi = 2iq\\psi\\psi$. 0 아님.
    """
    st, dirac, u1, gen = setup
    psi1 = make_psi(dirac, "α")
    psi2 = make_psi(dirac, "α")  # 같은 charge
    # (실제로 ψ ψ는 spinor 인덱스가 같으면 contract 시도되지만 same position이라 invalid;
    # 그러나 여기선 의도적으로 free index로 둠 — TensorProduct는 contract 시도)
    # 안전하게 다른 spinor 인덱스 사용
    psi_a = make_psi(dirac, "α")
    psi_b = make_psi(dirac, "β")

    expr = TensorProduct(psi_a, psi_b)
    delta = apply_generator(expr, gen)
    final = simplify(delta)
    assert not isinstance(final, ZeroTensor)
