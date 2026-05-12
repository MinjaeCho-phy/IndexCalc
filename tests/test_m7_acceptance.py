"""LIONS M7 acceptance — rank-≥2 Lorentz tensor rep via multi-slot Leibniz.

``lorentz_vector_action``의 N-frame-slot 일반화:

    $\\delta T^{\\mu\\nu} = M^{ab,\\mu}{}_\\rho T^{\\rho\\nu} + M^{ab,\\nu}{}_\\rho T^{\\mu\\rho}$
    $\\delta T_{\\mu\\nu} = -T_{\\rho\\nu} M^{ab,\\rho}{}_\\mu - T_{\\mu\\rho} M^{ab,\\rho}{}_\\nu$
    (혼합 위치도 mixed Leibniz)

비-frame 인덱스 (예: $W^A_{\\mu\\nu}$의 SU(2) adj $A$) 는 그대로 전달.

검증:
- M7-A: 단일 슬롯 (rank-1) 회귀 — 기존 M4 패턴, 출력 구조 그대로.
- M7-B: rank-2 generic $T^{\\mu\\nu}T_{\\mu\\nu}$ Lorentz invariance.
- M7-C: rank-2 antisym $W^A_{\\mu\\nu}W_A^{\\mu\\nu}$ Lorentz invariance (4-term cancel
        via canonical_form_modulo_dummies, antisym W slot은 simplifier가 자동 처리).
- M7-D: $V^\\mu \\partial_\\mu T^{\\rho\\sigma}$ — multi-slot vector × deriv_index hook
        혼합 invariance.
"""

import pytest

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.deriv import PartialDeriv, partial
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_lorentz_spinor_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify


@pytest.fixture
def setup():
    """Minkowski + Lorentz Group with vector + singlet reps (spinor 미사용)."""
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


# ─── M7-A: rank-1 회귀 ─────────────────────────────────────


def test_rank1_vector_dot_vector_invariant(setup):
    """$V^\\mu V_\\mu$ Lorentz invariance — 단일-슬롯 path 그대로 동작."""
    st, _dirac, _lorentz, gen = setup
    Vu = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    Vl = Tensor("V", [st.lower("μ")], reps={"Lorentz": "vector"})
    L = TensorProduct(Vu, Vl)

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── M7-B: rank-2 generic ──────────────────────────────────


def test_rank2_tensor_contracted_invariant(setup):
    """$T^{\\mu\\nu} T_{\\mu\\nu}$ Lorentz invariance — generic rank-2 (no symmetry).

    각 T는 frame slot 2개 → δT 가 2개 항 (Leibniz). 두 T를 곱하면 4개 cross-term
    전부 pairwise cancel.
    """
    st, _dirac, _lorentz, gen = setup
    Tu = Tensor(
        "T", [st.upper("μ"), st.upper("ν")],
        reps={"Lorentz": "vector"},
    )
    Tl = Tensor(
        "T", [st.lower("μ"), st.lower("ν")],
        reps={"Lorentz": "vector"},
    )
    L = TensorProduct(Tu, Tl)

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_rank2_tensor_trace_swap_invariant(setup):
    """$T^{\\mu\\nu} T_{\\nu\\mu}$ Lorentz invariance — slot 순서 다른 contraction.

    위 패턴과 다른 contraction graph지만 cancellation은 동일하게 발생.
    """
    st, _dirac, _lorentz, gen = setup
    Tu = Tensor(
        "T", [st.upper("μ"), st.upper("ν")],
        reps={"Lorentz": "vector"},
    )
    Tl = Tensor(
        "T", [st.lower("ν"), st.lower("μ")],
        reps={"Lorentz": "vector"},
    )
    L = TensorProduct(Tu, Tl)

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── M7-C: rank-2 antisym (Yang-Mills field strength) ──────


def test_w_field_strength_lorentz_invariant(setup):
    """$W^A_{\\mu\\nu} W_A^{\\mu\\nu}$ Lorentz invariance — antisym (μν) rank-2.

    Adj index $A$는 비-frame이라 그대로 contract. M_vec 회전은 frame 인덱스
    (μ, ν)에 대해서만 — antisym slot이 ``canonical_form_modulo_dummies`` 와
    ``is_zero_by_antisym_swap`` 에서 처리.
    """
    st, _dirac, _lorentz, gen = setup
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")

    W_low = Tensor(
        "W",
        [su2_adj.upper("A"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"Lorentz": "vector"},  # frame 인덱스만 회전
    )
    W_up = Tensor(
        "W",
        [su2_adj.lower("A"), st.upper("μ"), st.upper("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"Lorentz": "vector"},
    )
    L = ScalarMul(-0.25, TensorProduct(W_low, W_up))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


# ─── M7-D: multi-slot × deriv_index hook 혼합 ─────────────


def test_v_partial_rank2_invariant(setup):
    """$V^\\mu \\partial_\\mu T^{\\rho}{}_{\\rho}$ Lorentz invariance — vector dot scalar.

    $T^\\rho{}_\\rho$ 는 frame 인덱스가 contracted 된 사실상 스칼라이지만 generator는
    각 frame slot 마다 회전을 만든다 — 두 항이 같은 M_vec 구조로 dummy renaming
    후 cancel. 외부 $V^\\mu \\partial_\\mu$ 부분은 M6의 vector × deriv_index 경로.
    """
    st, _dirac, _lorentz, gen = setup
    Vu = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    T = Tensor(
        "T", [st.upper("ρ"), st.lower("ρ")],
        reps={"Lorentz": "vector"},
    )
    L = TensorProduct(Vu, PartialDeriv(T, st.lower("μ")))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"
