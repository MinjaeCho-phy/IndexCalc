"""LIONS M2.5 acceptance — full Yang-Mills $-\\tfrac14 F^a_{\\mu\\nu} F_a{}^{\\mu\\nu}$ invariance.

M2에서 deferred됐던 mixed-position adj 처리. 핵심 변경: ``IndexSpace.metric``이
non-empty이면 simplifier의 canonical form이 그 공간의 position을 ``"*"``로
collapse한다. 이는 raise/lower가 component identity인 공간 ($\\kappa = \\delta$ for
compact adj; $\\eta$ for Lorentz frame)에선 위치 구분이 canonical 비교에 영향을
주지 말아야 한다는 물리적 사실을 반영.

이 변경 하나로 $F^c F_a$와 $F^a F_c$ 같은 mixed-position 표현이 strict
multiset 비교에서 같아져, antisym × sym = 0 swap-prove-zero가 발화한다.
"""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import make_su_n_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify, is_structurally_equal


# ─── Setup with metric on adj space ────────────────────────


@pytest.fixture
def setup():
    sun = Group("SU(3)", dim=8, abelian=False)
    sun.add_rep("adj", dim=8)
    sun.add_rep("singlet", dim=1)
    # 핵심: adj space에 metric 설정 → canonical form에서 position-collapse
    adj = IndexSpace("su3_adj", dim=8, indices="abcdefgh", metric="κ")
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    gen = make_su_n_generator(sun, adj, parameter_name="d")
    return sun, adj, st, gen


# ─── Position-collapse 동작 확인 ──────────────────────────────


def test_metric_space_position_collapses(setup):
    """metric이 있는 공간은 canonical form에서 upper와 lower가 같은 token으로 처리."""
    sun, adj, st, gen = setup
    F_up = Tensor("F", [adj.upper("c")])
    F_lo = Tensor("F", [adj.lower("c")])
    # adj에 metric이 있으므로 위치 차이가 canonical에서 collapse → 같음
    assert is_structurally_equal(F_up, F_lo)


def test_no_metric_space_position_preserved():
    """metric이 없는 공간에선 position 보존."""
    no_metric = IndexSpace("abstract", dim=3, indices="abc")  # metric=""
    A_up = Tensor("A", [no_metric.upper("a")])
    A_lo = Tensor("A", [no_metric.lower("a")])
    assert not is_structurally_equal(A_up, A_lo)


# ─── Full YM invariance ────────────────────────────────────


def test_full_ym_lagrangian_is_invariant(setup):
    """$L_{YM} = -\\tfrac14 F^a_{\\mu\\nu} F_a{}^{\\mu\\nu}$의 SU(N) gauge invariance.

    완료 시 ``apply_generator(L) → simplify → ZeroTensor``.
    """
    sun, adj, st, gen = setup

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

    assert isinstance(final, ZeroTensor), (
        f"Expected ZeroTensor (YM is gauge-invariant), got {type(final).__name__}: {final!r}"
    )
    # free index = d (parameter)
    assert len(final.free_indices) == 1
    assert final.free_indices[0].name == "d"


def test_individual_leibniz_term_is_zero(setup):
    """Leibniz의 한 항만 따로 — $f^a{}_{dc} F^c F_a$ — 도 0이어야."""
    sun, adj, st, gen = setup

    f = Tensor(
        "f",
        [adj.upper("a"), adj.lower("d"), adj.lower("c_1")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    F_renamed = Tensor(
        "F",
        [adj.upper("c_1"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(3)": "adj"},
    )
    F2 = Tensor(
        "F",
        [adj.lower("a"), st.upper("μ"), st.upper("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(3)": "adj"},
    )
    expr = TensorProduct(f, TensorProduct(F_renamed, F2))

    result = simplify(expr)
    assert isinstance(result, ZeroTensor)


# ─── Negative test: 이름이 다른 factor면 여전히 0 아님 ────────


def test_distinct_factor_names_not_zero_even_with_metric(setup):
    """adj에 metric이 있어도, X·Y 류 (서로 다른 factor 이름)는 0 검출 안 됨.

    수학적으로 옳다 — antisym × X·Y는 일반적으로 0이 아니다.
    """
    sun, adj, st, gen = setup
    f = Tensor(
        "f",
        [adj.upper("a"), adj.lower("d"), adj.lower("c_1")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    X = Tensor(
        "X",
        [adj.upper("c_1"), st.lower("μ")],
        reps={"SU(3)": "adj"},
    )
    Y = Tensor(
        "Y",
        [adj.lower("a"), st.upper("μ")],
        reps={"SU(3)": "adj"},
    )
    expr = TensorProduct(f, TensorProduct(X, Y))
    result = simplify(expr)
    assert not isinstance(result, ZeroTensor)
