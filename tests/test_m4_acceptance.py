"""LIONS M4 acceptance — Dirac mass term Lorentz invariance.

검증 대상: $\\delta_{\\rm Lorentz}(\\bar\\psi_\\alpha \\psi^\\alpha) = 0$.

작용:
    $\\delta\\psi^\\alpha = -\\tfrac{i}{2} \\Sigma^{ab}{}^\\alpha{}_\\beta \\psi^\\beta$
    $\\delta\\bar\\psi_\\alpha = +\\tfrac{i}{2} \\bar\\psi_\\beta \\Sigma^{ab}{}^\\beta{}_\\alpha$

Leibniz 두 항의 body는 dummy 인덱스만 다르고 구조 동일 →
``canonical_form_modulo_dummies``로 같은 group, ±0.5j cancel → ZeroTensor.

**제외 (M4.5/M5 예정):**
- 자유 Dirac 운동항 $i\\bar\\psi\\gamma^\\mu\\partial_\\mu\\psi$의 Lorentz invariance.
  $[\\Sigma^{ab}, \\gamma^c] = i(\\eta^{bc}\\gamma^a - \\eta^{ac}\\gamma^b)$ Clifford
  identity 적용 simplifier가 필요.
- Lorentz vector generator on $\\partial_\\mu, A_\\mu, V^\\mu$ 등 — vector rep 작용
  추가 필요.
- Yukawa, $\\epsilon_{ij}$ contraction, chirality $P_{L,R}$.
- Local gauge ($\\partial_\\mu\\alpha$).
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_lorentz_spinor_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


@pytest.fixture
def setup():
    """Minkowski + Dirac + Lorentz Group with spinor reps."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")  # no metric

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("singlet", dim=1)

    gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )
    return st, dirac, lorentz, gen


def make_psi(dirac, idx_name="α"):
    return Tensor(
        "psi",
        [dirac.upper(idx_name)],
        reps={"Lorentz": "spinor"},
        statistics="fermionic",
    )


def make_psibar(dirac, idx_name="α"):
    return Tensor(
        "psibar",
        [dirac.lower(idx_name)],
        reps={"Lorentz": "conj_spinor"},
        statistics="fermionic",
    )


# ─── M4-B: 본 acceptance ─────────────────────────────────────


def test_dirac_mass_lorentz_invariant(setup):
    """$\\delta_{Lorentz}(\\bar\\psi\\psi) = 0$.

    Leibniz의 두 항 ($+\\tfrac{i}{2}$ 와 $-\\tfrac{i}{2}$ 계수)이 dummy renaming
    후 같은 body로 인식되어 cancel.
    """
    _st, dirac, _lorentz, gen = setup
    psi = make_psi(dirac, "α")
    psibar = make_psibar(dirac, "α")
    mass = TensorProduct(psibar, psi)  # ψ̄ ψ scalar

    delta = apply_generator(mass, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_dirac_mass_term_with_coefficient(setup):
    """$\\delta(-m \\bar\\psi\\psi) = -m \\cdot 0 = 0$ — 외부 ScalarMul도 통과."""
    _st, dirac, _lorentz, gen = setup
    psi = make_psi(dirac, "α")
    psibar = make_psibar(dirac, "α")
    mass = ScalarMul(-1.0, TensorProduct(psibar, psi))

    delta = apply_generator(mass, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor)


# ─── 작용 구조 sanity ───────────────────────────────────────


def test_action_on_psi_upper(setup):
    """δψ^α 의 구조: ``ScalarMul(-0.5j, TP(Σ, ψ_renamed))``."""
    _st, dirac, _lorentz, gen = setup
    psi = make_psi(dirac, "α")
    result = apply_generator(psi, gen)

    assert isinstance(result, ScalarMul)
    assert result.scalar == -0.5j
    inner = result.expr
    assert isinstance(inner, TensorProduct)
    Sigma, psi_r = inner.left, inner.right
    assert Sigma.name == "Sigma"
    # Σ에 antisym (0,1) 페어
    assert (0, 1) in Sigma.antisymmetric_pairs
    # ψ_renamed의 spinor 인덱스 = Σ의 마지막 (lower) — contract
    assert psi_r.indices[0].name == Sigma.indices[3].name
    assert psi_r.indices[0].position == "upper"


def test_action_on_psibar_lower(setup):
    """δψ̄_α 의 구조: ``ScalarMul(+0.5j, TP(ψ̄_renamed, Σ))``."""
    _st, dirac, _lorentz, gen = setup
    psibar = make_psibar(dirac, "α")
    result = apply_generator(psibar, gen)

    assert isinstance(result, ScalarMul)
    assert result.scalar == 0.5j
    inner = result.expr
    assert isinstance(inner, TensorProduct)
    psibar_r, Sigma = inner.left, inner.right
    assert Sigma.name == "Sigma"
    # ψ̄_renamed의 spinor lo = Σ의 slot 2 (upper, dummy)와 contract
    assert psibar_r.indices[0].name == Sigma.indices[2].name


# ─── Negative — singlet field은 변환 안 함 ──────────────────


def test_singlet_field_zero(setup):
    _st, _dirac, _lorentz, gen = setup
    s = Tensor("s", [], reps={"Lorentz": "singlet"})
    result = apply_generator(s, gen)
    assert isinstance(result, ZeroTensor)


# ─── M4-D: Lorentz vector + $V^\\mu V_\\mu$ invariance ───────


@pytest.fixture
def setup_with_vector():
    """Lorentz Group with spinor + vector reps."""
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


def test_vector_squared_lorentz_invariant(setup_with_vector):
    """$\\delta_{Lorentz}(V^\\mu V_\\mu) = 0$ via mass-term-style cancellation."""
    st, _dirac, _lorentz, gen = setup_with_vector
    Vu = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    Vd = Tensor("V", [st.lower("μ")], reps={"Lorentz": "vector"})
    L = TensorProduct(Vu, Vd)  # V^μ V_μ

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_lorentz_action_on_vector_upper(setup_with_vector):
    """δV^μ 의 구조: ``TensorProduct(M, V_renamed)``."""
    st, *_, gen = setup_with_vector
    V = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    result = apply_generator(V, gen)

    assert isinstance(result, TensorProduct)
    M = result.left
    assert M.name == "M_vec"
    # M에 antisym (a, b) 페어 (slot 0, 1)
    assert (0, 1) in M.antisymmetric_pairs
    # M의 row matches input ('μ', upper), col is dummy (lower)
    assert M.indices[2].name == "μ"
    assert M.indices[2].position == "upper"
    assert M.indices[3].position == "lower"


def test_lorentz_action_on_vector_lower(setup_with_vector):
    """δV_μ 의 구조: ``ScalarMul(-1, TP(V_renamed, M))``."""
    st, *_, gen = setup_with_vector
    V = Tensor("V", [st.lower("μ")], reps={"Lorentz": "vector"})
    result = apply_generator(V, gen)

    assert isinstance(result, ScalarMul)
    assert result.scalar == -1.0
    inner = result.expr
    assert isinstance(inner, TensorProduct)
    V_r, M = inner.left, inner.right
    assert M.name == "M_vec"
    # input (μ lower) → M의 col matches lower
    assert M.indices[3].name == "μ"
    assert M.indices[3].position == "lower"


# ─── 통합: Dirac mass + V·V 동시 검증 ────────────────────────


def test_lagrangian_with_vector_and_spinor(setup_with_vector):
    """$\\bar\\psi\\psi + V^\\mu V_\\mu$ — 두 항 모두 Lorentz invariant이므로 합도 invariant.

    Free index 구조 일치 확인 (둘 다 0 free)을 위해 ScalarMul로 별도 wrap.
    """
    from indexcalc.core.tensor import TensorSum

    st, dirac, _lorentz, gen = setup_with_vector
    psi = make_psi(dirac, "α")
    psibar = make_psibar(dirac, "α")
    Vu = Tensor("V", [st.upper("ν")], reps={"Lorentz": "vector"})
    Vd = Tensor("V", [st.lower("ν")], reps={"Lorentz": "vector"})

    L = TensorSum(
        TensorProduct(psibar, psi),       # ψ̄ ψ
        TensorProduct(Vu, Vd),            # V^μ V_μ
    )

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor)
