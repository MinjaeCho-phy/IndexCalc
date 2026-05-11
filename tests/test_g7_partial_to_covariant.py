"""G7: ∂_μ T → ∇̄_μ T - Σ Γ̄·T forward 변환 회귀 테스트.

규칙:
    - upper slot: ``∂_μ T^ρ = ∇̄_μ T^ρ - Γ^ρ_{μα} T^α``
    - lower slot: ``∂_μ T_ρ = ∇̄_μ T_ρ + Γ^α_{μρ} T_α``
    - Tensor 속성(antisymmetric_pairs 등) 보존
    - Sum/Product/ScalarMul 안에서 재귀
    - only_for로 특정 leaf만 변환
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry,
    LeviCivitaConnection, CovariantDeriv,
    PartialDeriv, partial_to_covariant,
    TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.index import Index
from indexcalc.core.simplify import simplify, is_structurally_equal


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλα", metric="g")


@pytest.fixture
def conn_setup(st):
    g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")], symmetric_pairs=[(0, 1)])
    g_up = Tensor("g", [st.upper("μ"), st.upper("ν")], symmetric_pairs=[(0, 1)])
    mreg = MetricRegistry()
    mreg.register(g_lo, g_up, st)
    conn = LeviCivitaConnection(g_lo, g_up, st)
    return mreg, conn


# ─── Sign / structure ───────────────────────────────────────


class TestSignsAndStructure:
    def test_upper_slot_correction_negative(self, st, conn_setup):
        """∂_μ T^ρ → ∇̄_μ T^ρ + ScalarMul(-1, Γ^ρ_μα T^α)."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn)
        # 결과는 TensorSum(CovariantDeriv, ScalarMul(-1, Γ·T))
        assert isinstance(out, TensorSum)
        assert isinstance(out.left, CovariantDeriv)
        assert isinstance(out.right, ScalarMul)
        assert out.right.scalar == -1
        # 보정 항은 Γ × T_replaced
        prod = out.right.expr
        assert isinstance(prod, TensorProduct)
        assert prod.left.name == "Γ"

    def test_lower_slot_correction_positive(self, st, conn_setup):
        """∂_μ T_ρ → ∇̄_μ T_ρ + Γ^α_μρ T_α (no ScalarMul -1)."""
        _, conn = conn_setup
        T = Tensor("T", [st.lower("ρ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn)
        assert isinstance(out, TensorSum)
        assert isinstance(out.left, CovariantDeriv)
        # right는 Γ·T (TensorProduct, no ScalarMul -1 wrapper)
        assert isinstance(out.right, TensorProduct)

    def test_mixed_slots_one_each(self, st, conn_setup):
        """∂_μ T^ρ_σ → ∇̄ + (-)Γ^ρ_μα T^α_σ + Γ^β_μσ T^ρ_β. 3개 항."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ"), st.lower("σ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn)
        # nested TensorSum: ((cov + (-)Γ·T) + Γ·T)
        # flatten
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(out)
        assert len(terms) == 3
        # 첫 항은 CovariantDeriv
        assert isinstance(terms[0], CovariantDeriv)

    def test_free_indices_match(self, st, conn_setup):
        """변환 전후 free_indices 일치."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ"), st.lower("σ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn)
        before = sorted(i.name for i in expr.free_indices)
        after = sorted(i.name for i in out.free_indices)
        assert before == after


# ─── Attribute preservation ─────────────────────────────────


class TestAttributePreservation:
    def test_antisymmetric_pairs_preserved(self, st, conn_setup):
        """antisym 속성이 보정 항의 T_replaced에 보존."""
        _, conn = conn_setup
        T = Tensor(
            "F", [st.lower("μ"), st.lower("ν")],
            antisymmetric_pairs=[(0, 1)],
        )
        expr = PartialDeriv(T, st.lower("ρ"))
        out = partial_to_covariant(expr, conn)
        # 보정 항 안의 T_replaced를 찾아 antisym 확인
        from indexcalc.core.simplify import _flatten_sum
        for term in _flatten_sum(out):
            # 보정 항: TensorProduct(Γ, T_replaced) 형태
            if isinstance(term, TensorProduct):
                replaced = term.right
                assert isinstance(replaced, Tensor)
                assert replaced.antisymmetric_pairs == ((0, 1),)


# ─── Recursive walk ─────────────────────────────────────────


class TestRecursive:
    def test_inside_product(self, st, conn_setup):
        """∂_μ T · S → (∇̄ + Γ corrections) · S."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        S = Tensor("S", [st.lower("ρ")])
        expr = TensorProduct(PartialDeriv(T, st.lower("μ")), S)
        out = partial_to_covariant(expr, conn)
        # 변환된 표현식은 product of (sum of ∇̄ + corrections) × S
        assert isinstance(out, TensorProduct)
        assert isinstance(out.left, TensorSum)

    def test_inside_sum(self, st, conn_setup):
        """∂_μ T + ∂_μ S → (∇̄T+Γ) + (∇̄S+Γ)."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        S = Tensor("S", [st.upper("ρ")])
        expr = TensorSum(
            PartialDeriv(T, st.lower("μ")),
            PartialDeriv(S, st.lower("μ")),
        )
        out = partial_to_covariant(expr, conn)
        # 두 leg 모두 변환됨
        assert isinstance(out, TensorSum)
        assert isinstance(out.left, TensorSum)  # ∇̄T + correction
        assert isinstance(out.right, TensorSum)  # ∇̄S + correction


# ─── only_for filter ───────────────────────────────────────


class TestOnlyForFilter:
    def test_only_for_targets_specific_name(self, st, conn_setup):
        """only_for={'g'}이면 ∂_μ T는 변환 안 됨."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn, only_for={"g"})
        # 변환 미적용 → 그대로
        assert out is expr

    def test_only_for_includes_target(self, st, conn_setup):
        """only_for={'T'}이면 ∂_μ T는 변환됨."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        expr = PartialDeriv(T, st.lower("μ"))
        out = partial_to_covariant(expr, conn, only_for={"T"})
        assert isinstance(out, TensorSum)


# ─── Cancellation through simplify ─────────────────────────


class TestSimplifyCancellation:
    def test_partial_minus_self_cancels(self, st, conn_setup):
        """∂_μ T - ∂_μ T → 0. partial_to_covariant 후 simplify로 cancel."""
        _, conn = conn_setup
        T = Tensor("T", [st.upper("ρ")])
        expr = TensorSum(
            PartialDeriv(T, st.lower("μ")),
            ScalarMul(-1, PartialDeriv(T, st.lower("μ"))),
        )
        out = partial_to_covariant(expr, conn)
        result = simplify(out)
        # ZeroTensor 또는 None
        from indexcalc.core.variation import ZeroTensor
        assert isinstance(result, ZeroTensor)
