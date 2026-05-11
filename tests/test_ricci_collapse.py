"""δR_σν → textbook ∇̄ form collapse 회귀 — IndexCalc 전체 파이프라인 검증.

핵심 입력:
    raw δR_σν = ∂_ρ δΓ^ρ_νσ - ∂_ν δΓ^ρ_ρσ
              + δΓ^ρ_ρλ Γ^λ_νσ + Γ^ρ_ρλ δΓ^λ_νσ
              - δΓ^ρ_νλ Γ^λ_ρσ - Γ^ρ_νλ δΓ^λ_ρσ
핵심 출력:
    δR_σν = ∇_ρ δΓ^ρ_νσ - ∇_ν δΓ^ρ_ρσ
(textbook covariant form)

검증:
    - Christoffel sym_pairs=[(1,2)] 적용 (chr.make_tensor).
    - δΓ도 sym_pairs 보존 (delta_of preserving).
    - covariant_collapse가 mreg + simplify로 self-cancelling 정리.
    - distribute_products로 ScalarMul(-1, Sum) 평탄화.
    - 6항 → 2항 (∇ 형태)으로 collapse.
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry, LeviCivitaConnection,
    Variation, VariationRegistry, expand_variation,
    PartialDeriv, CovariantDeriv,
    TensorProduct, covariant_collapse,
)
from indexcalc.core.index import Index
from indexcalc.core.simplify import _flatten_sum


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλ", metric="g")


@pytest.fixture
def setup(st):
    g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")], symmetric_pairs=[(0, 1)])
    g_up = Tensor("g", [st.upper("μ"), st.upper("ν")], symmetric_pairs=[(0, 1)])
    mreg = MetricRegistry()
    mreg.register(g_lo, g_up, st)
    chr = LeviCivitaConnection(g_lo, g_up, st)
    vreg = VariationRegistry()
    vreg.declare_varying("g")
    vreg.declare_varying("Γ")
    return st, mreg, chr, vreg


def _build_delta_R(setup):
    """δR_σν 구성: Ricci 정의 (μ↔ρ trace의 Riemann) + Variation."""
    st, mreg, chr, vreg = setup

    def G(u, l1, l2):
        return chr.make_tensor(u, l1, l2)

    t1 = PartialDeriv(G("ρ", "ν", "σ"), Index("ρ", st, "lower"))
    t2 = PartialDeriv(G("ρ", "ρ", "σ"), Index("ν", st, "lower"))
    q1 = TensorProduct(G("ρ", "ρ", "λ"), G("λ", "ν", "σ"))
    q2 = TensorProduct(G("ρ", "ν", "λ"), G("λ", "ρ", "σ"))
    ricci_def = (t1 - t2) + (q1 - q2)
    return expand_variation(Variation(ricci_def), vreg, mreg), mreg, chr


class TestRicciTextbookCollapse:
    def test_collapses_to_two_nabla_terms(self, setup):
        delta_R, mreg, chr = _build_delta_R(setup)
        # raw은 6 항
        # (단, ScalarMul(-1, TensorSum) wrapping으로 _flatten_sum이 4개를 볼 수도; covariant_collapse 내부에서 distribute로 평탄화함)

        collapsed = covariant_collapse(
            delta_R, chr, only_for={"δΓ"}, mreg=mreg,
        )
        # 결과는 2 항 (TensorSum of 두 CovariantDeriv)
        terms = _flatten_sum(collapsed)
        assert len(terms) == 2

    def test_both_terms_are_covariant_deriv(self, setup):
        delta_R, mreg, chr = _build_delta_R(setup)
        collapsed = covariant_collapse(
            delta_R, chr, only_for={"δΓ"}, mreg=mreg,
        )
        from indexcalc.core.tensor import ScalarMul as _SM
        for term in _flatten_sum(collapsed):
            # 각 항은 CovariantDeriv 또는 ScalarMul(±, CovariantDeriv)
            cur = term
            if isinstance(cur, _SM):
                cur = cur.expr
            assert isinstance(cur, CovariantDeriv)
            # inner는 δΓ leaf
            assert cur.expr.name == "δΓ"

    def test_free_indices_preserved(self, setup):
        delta_R, mreg, chr = _build_delta_R(setup)
        collapsed = covariant_collapse(
            delta_R, chr, only_for={"δΓ"}, mreg=mreg,
        )
        before = sorted(i.name for i in delta_R.free_indices)
        after = sorted(i.name for i in collapsed.free_indices)
        assert before == after
        assert after == ["ν", "σ"]
