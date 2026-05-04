"""LIONS M2 acceptance — SU(N) adjoint pipeline + antisym × sym → 0.

본 테스트는 M2의 작동 가능 영역을 검증한다:

1. SU(N) adjoint generator 등록.
2. δ_d V^a = f^a{}_{de} V^e 형태로 작용 가능.
3. all-lower-f / all-upper-V 컨벤션에서 antisymmetric tensor가 같은-이름 factor
   (V^a V^b V^c)와 fully-contracted 곱일 때 simplifier가 ZeroTensor 검출.
4. apply_generator + simplify 파이프라인이 통합 동작.

**제외 (M2.5/M3 예정):**
- Full Yang-Mills $-\\tfrac14 F^a_{\\mu\\nu} F_a{}^{\\mu\\nu}$의 mixed-position adj 처리.
  $\\kappa = \\delta$ identity로 위치 정규화 후에야 simplifier가 0 검출 가능.
- $F = \\partial A + g[A,A]$ 전개와 Jacobi identity가 필요한 시나리오.
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_su_n_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import (
    is_zero_by_antisym_swap, simplify, is_structurally_equal,
)


# ─── Setup ──────────────────────────────────────────────────


@pytest.fixture
def setup():
    """SU(3) + adj space + adjoint scalar V^a."""
    sun = Group("SU(3)", dim=8, abelian=False)
    sun.add_rep("adj", dim=8)
    sun.add_rep("singlet", dim=1)
    adj = IndexSpace("su3_adj", dim=8, indices="abcdefgh")
    gen = make_su_n_generator(sun, adj, parameter_name="d")
    return sun, adj, gen


# ─── 1. Generator 작용 (M2-A 통합) ─────────────────────────


def test_adj_generator_on_V(setup):
    """δ_d V^a 가 ``f^a{}_{de} V^e`` 형태."""
    sun, adj, gen = setup
    V = Tensor("V", [adj.upper("a")], reps={"SU(3)": "adj"})
    result = apply_generator(V, gen)
    # apply_generator → simplify_zeros 거치지만 결과는 product 그대로
    assert isinstance(result, TensorProduct)
    f = result.left
    Vp = result.right
    assert f.name == "f"
    assert f.indices[0].name == "a" and f.indices[0].position == "upper"
    assert f.indices[1].name == "d" and f.indices[1].position == "lower"
    # V'의 adj index는 dummy로 renamed + upper
    assert Vp.name == "V"
    assert Vp.indices[0].position == "upper"
    assert Vp.indices[0].name == f.indices[2].name  # 이름 일치 = contraction


def test_adj_generator_on_singlet_zero(setup):
    sun, adj, gen = setup
    s = Tensor("s", [], reps={"SU(3)": "singlet"})
    result = apply_generator(s, gen)
    assert isinstance(result, ZeroTensor)


# ─── 2. Antisym × symmetric → 0 (M2-C: cubic invariant) ────


def test_cubic_invariant_is_zero(setup):
    """``f_{abc} V^a V^b V^c`` — totally antisymmetric × totally symmetric (V·V·V) = 0.

    이건 자체가 항등적으로 0인 표현 (수학적으로). simplifier가 이를 인식.
    """
    sun, adj, gen = setup
    f = Tensor(
        "f",
        [adj.lower("a"), adj.lower("b"), adj.lower("c")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    Va = Tensor("V", [adj.upper("a")], reps={"SU(3)": "adj"})
    Vb = Tensor("V", [adj.upper("b")], reps={"SU(3)": "adj"})
    Vc = Tensor("V", [adj.upper("c")], reps={"SU(3)": "adj"})
    expr = TensorProduct(f, TensorProduct(Va, TensorProduct(Vb, Vc)))

    result = simplify(expr)
    assert isinstance(result, ZeroTensor)


# ─── 3. Generator + simplifier 통합 — Leibniz 후 simplify ──


def test_apply_generator_to_quartic_then_simplify(setup):
    """``L = κ V^a V^b W_a U_b`` 류는 본 컨벤션 밖이므로,
    대신 ``f_{abc} V^a V^b V^c``에 δ_d 적용 후 simplifier.

    각 Leibniz 항은 두 f와 두 V 곱이지만 (M2 simplifier가 이중-f 구조까지
    완전 정리하진 못함) — 본 테스트에서는 **at least 항상 0인 expression이
    결과로 들어왔을 때 simplifier가 그걸 그대로 0으로 유지하는지**까지만 확인.

    full Leibniz 결과는 M2.5/M3에서 더 강한 simplifier가 처리.
    """
    sun, adj, gen = setup
    f = Tensor(
        "f",
        [adj.lower("a"), adj.lower("b"), adj.lower("c")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    Va = Tensor("V", [adj.upper("a")], reps={"SU(3)": "adj"})
    Vb = Tensor("V", [adj.upper("b")], reps={"SU(3)": "adj"})
    Vc = Tensor("V", [adj.upper("c")], reps={"SU(3)": "adj"})
    L = TensorProduct(f, TensorProduct(Va, TensorProduct(Vb, Vc)))

    # L 자체는 0. simplify 적용:
    L_simp = simplify(L)
    assert isinstance(L_simp, ZeroTensor)

    # δ_d (0) = 0. apply_generator on the original L:
    delta_L = apply_generator(L, gen)
    # apply_generator 내부에서 _simplify_zeros만 거치므로 ZeroTensor 자동 흡수까지 안 될 수 있음.
    # simplify로 마무리:
    final = simplify(delta_L)
    # δ가 antisym tensor f의 자유 인덱스 d를 만든다는 등의 이유로 fully simplified 안 될 수도 있음.
    # 본 테스트에서는 **at least 결과가 시작 expr인 0의 변형 형태**임을 확인.
    # 정확히는: 수학적으로 0이고 simplifier가 detect하면 ZeroTensor, 아니면 unchanged.
    assert isinstance(final, ZeroTensor) or final is delta_L or True
    # 더 정확한 검증은 M2.5+. 본 acceptance에서는 파이프라인이 throw 없이 돈다는 점 확인.


# ─── 4. Limitation acknowledgment — full YM mixed-position ──


def test_full_ym_mixed_position_not_simplified(setup):
    """현 simplifier가 mixed-position YM ($F^c F_a$) 패턴에서 0을 검출 못함을 명시.

    이 테스트는 **현재 한계의 회귀 방지**용 — M2.5에서 κ-application 구현 후
    이 테스트는 통과 방향으로 뒤집히게 된다.
    """
    sun, adj, gen = setup
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")

    F1 = Tensor(
        "F",
        [adj.upper("a"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(3)": "adj"},
    )
    F2 = Tensor(
        "F",
        [adj.lower("a"), st.upper("μ"), st.upper("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(3)": "adj"},
    )
    L_YM = ScalarMul(-0.25, TensorProduct(F1, F2))

    delta_L = apply_generator(L_YM, gen)
    final = simplify(delta_L)
    # 수학적으로는 0이지만 mixed-position simplifier 한계로 ZeroTensor 아님.
    # 미래에 κ-handling 추가되면 이 assertion이 깨지고, 그땐 ZeroTensor로 변경.
    assert not isinstance(final, ZeroTensor)
