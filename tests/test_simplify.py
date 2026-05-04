"""Simplify 모듈 테스트 (LIONS M2 / E9 기초)."""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import (
    rename_index, collect_factors,
    canonical_form, is_structurally_equal,
    is_zero_by_antisym_swap, simplify,
)


# ─── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def adj():
    return IndexSpace("adj", dim=8, indices="abcdefgh")


@pytest.fixture
def st():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")


# ─── rename_index ───────────────────────────────────────────


class TestRenameIndex:
    def test_rename_in_tensor(self, adj):
        T = Tensor("T", [adj.upper("a"), adj.lower("b")])
        new = rename_index(T, {"a": "x"})
        assert new.indices[0].name == "x"
        assert new.indices[0].position == "upper"
        assert new.indices[1].name == "b"

    def test_rename_preserves_metadata(self, adj, st):
        T = Tensor(
            "T",
            [adj.upper("a"), st.lower("μ"), st.lower("ν")],
            antisymmetric_pairs=[(1, 2)],
            reps={"SU(3)": "adj"},
            statistics="bosonic",
        )
        new = rename_index(T, {"a": "z"})
        assert new.antisymmetric_pairs == ((1, 2),)
        assert new.reps == {"SU(3)": "adj"}

    def test_rename_in_product(self, adj):
        A = Tensor("A", [adj.upper("a")])
        B = Tensor("B", [adj.lower("a")])
        P = TensorProduct(A, B)
        new = rename_index(P, {"a": "k"})
        assert new.left.indices[0].name == "k"
        assert new.right.indices[0].name == "k"

    def test_rename_swap_two_names(self, adj):
        T = Tensor("T", [adj.upper("a"), adj.lower("b")])
        # a ↔ b 동시 swap을 한 번의 mapping으로
        new = rename_index(T, {"a": "b", "b": "a"})
        assert new.indices[0].name == "b"
        assert new.indices[1].name == "a"


# ─── collect_factors ────────────────────────────────────────


def test_collect_factors_flat(adj):
    A = Tensor("A", [adj.upper("a")])
    B = Tensor("B", [adj.lower("a")])
    C = Tensor("C", [])
    p = TensorProduct(A, TensorProduct(B, C))
    facs = collect_factors(p)
    assert len(facs) == 3
    assert facs[0].name == "A"
    assert facs[2].name == "C"


def test_collect_factors_single_returns_list_of_one(adj):
    A = Tensor("A", [adj.upper("a")])
    assert collect_factors(A) == [A]


# ─── canonical_form / structural equality ──────────────────


class TestCanonicalEquality:
    def test_bosonic_commute_equal(self, adj, st):
        A = Tensor("A", [adj.upper("a")])
        B = Tensor("B", [adj.lower("a")])
        e1 = TensorProduct(A, B)
        e2 = TensorProduct(B, A)
        assert is_structurally_equal(e1, e2)

    def test_different_factors_not_equal(self, adj):
        A = Tensor("A", [adj.upper("a")])
        B = Tensor("B", [adj.lower("a")])
        C = Tensor("C", [adj.lower("a")])
        assert not is_structurally_equal(
            TensorProduct(A, B), TensorProduct(A, C)
        )

    def test_same_factor_repeated_with_swapped_dummy_names(self, adj):
        """V^a V^b (all upper, same factor) vs V^b V^a — 같음 (bosonic commute)."""
        Va = Tensor("V", [adj.upper("a")])
        Vb = Tensor("V", [adj.upper("b")])
        assert is_structurally_equal(
            TensorProduct(Va, Vb), TensorProduct(Vb, Va)
        )

    def test_mixed_position_not_equal_strict(self, adj):
        """F^c F_a vs F^a F_c — strict semantics에선 다름.

        수학적으로는 $\\kappa = \\delta$ identity로 같지만, 그 정규화는 별도 모듈
        (M2.5/M3 예정). 현재 strict canonical form은 multiset만 비교.
        """
        e1 = TensorProduct(
            Tensor("F", [adj.upper("c")]), Tensor("F", [adj.lower("a")]),
        )
        e2 = TensorProduct(
            Tensor("F", [adj.upper("a")]), Tensor("F", [adj.lower("c")]),
        )
        assert not is_structurally_equal(e1, e2)


# ─── is_zero_by_antisym_swap ────────────────────────────────


class TestZeroByAntisymSwap:
    def test_all_lower_f_with_three_upper_V(self, adj):
        """$f_{abc} V^a V^b V^c$ — totally antisymmetric × totally symmetric (V·V·V) = 0.

        all-lower convention에서 antisym tensor와 같은-이름 factor 다중 곱은
        strict multiset 비교만으로도 swap-invariant가 명확해 0 검출.
        """
        f = Tensor(
            "f",
            [adj.lower("a"), adj.lower("b"), adj.lower("c")],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )
        Va = Tensor("V", [adj.upper("a")])
        Vb = Tensor("V", [adj.upper("b")])
        Vc = Tensor("V", [adj.upper("c")])
        expr = TensorProduct(f, TensorProduct(Va, TensorProduct(Vb, Vc)))

        result = is_zero_by_antisym_swap(expr)
        assert isinstance(result, ZeroTensor)
        # free index 없음 (b의 매개변수가 아님; 위 식은 모든 인덱스 contracted)
        assert len(result.free_indices) == 0

    def test_distinct_factor_names_not_zero(self, adj):
        """X·Y·Z 등 이름이 다른 factor면 swap 후 multiset이 달라 → 0 검출 안 됨."""
        f = Tensor(
            "f",
            [adj.lower("a"), adj.lower("b"), adj.lower("c")],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )
        Xa = Tensor("X", [adj.upper("a")])
        Yb = Tensor("Y", [adj.upper("b")])
        Zc = Tensor("Z", [adj.upper("c")])
        expr = TensorProduct(f, TensorProduct(Xa, TensorProduct(Yb, Zc)))

        result = is_zero_by_antisym_swap(expr)
        # X^a, Y^b 등 factor가 모두 다른 이름 → swap (a↔b) 시 X^b, Y^a로 이름 cross-binding이
        # 바뀌므로 multiset 다름 → 0 검출 안 됨 (수학적으로 일반적으로 0 아님)
        assert not isinstance(result, ZeroTensor)

    def test_mixed_position_not_detected(self, adj):
        """Mixed-position에서는 strict multiset 비교가 0을 놓칠 수 있다.

        $F^c F_a$ 같은 구조는 $\\kappa$-application 후에야 같아진다 (M2.5).
        본 테스트는 ``현재 simplifier의 한계``를 명시적으로 문서화.
        """
        f = Tensor(
            "f",
            [adj.upper("a"), adj.lower("b"), adj.lower("c")],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )
        Fc = Tensor("F", [adj.upper("c")])
        Fa = Tensor("F", [adj.lower("a")])
        expr = TensorProduct(f, TensorProduct(Fc, Fa))

        result = is_zero_by_antisym_swap(expr)
        # 수학적으로는 0 (κ=δ identity 후 antisym × sym = 0)이지만,
        # strict multiset 비교는 위치 차이를 구분 → 검출 안 됨.
        assert not isinstance(result, ZeroTensor)

    def test_no_antisym_factor_returns_unchanged(self, adj):
        A = Tensor("A", [adj.upper("a")])
        B = Tensor("B", [adj.lower("a")])
        expr = TensorProduct(A, B)
        result = is_zero_by_antisym_swap(expr)
        assert result is expr  # 변경 없음

    def test_non_dummy_indices_not_swapped(self, adj):
        """antisym pair 이름 중 하나가 free index면 swap 후보 아님."""
        f = Tensor(
            "f",
            [adj.lower("a"), adj.lower("b"), adj.lower("c")],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )
        expr = TensorProduct(f, Tensor("dummy", []))
        result = is_zero_by_antisym_swap(expr)
        assert not isinstance(result, ZeroTensor)


# ─── simplify (top-level) ───────────────────────────────────


class TestSimplifyTopLevel:
    def test_simplify_inside_sum(self, adj):
        """TensorSum 안의 product에 zero detection 적용 (all-lower-f, all-upper-V)."""
        f = Tensor(
            "f",
            [adj.lower("a"), adj.lower("b"), adj.lower("c")],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )
        Va = Tensor("V", [adj.upper("a")])
        Vb = Tensor("V", [adj.upper("b")])
        Vc = Tensor("V", [adj.upper("c")])
        zero_term = TensorProduct(
            f, TensorProduct(Va, TensorProduct(Vb, Vc)),
        )

        nonzero = Tensor("Y", [])  # 0개 free index — TensorSum과 호환 (둘 다 0 free)
        expr = TensorSum(zero_term, nonzero)

        result = simplify(expr)
        # zero_term이 ZeroTensor로 → ZeroTensor + Y → Y
        assert isinstance(result, Tensor)
        assert result.name == "Y"

    def test_simplify_idempotent(self, adj):
        Y = Tensor("Y", [adj.lower("b")])
        result1 = simplify(Y)
        result2 = simplify(result1)
        assert result2 is result1 or result2 == result1
