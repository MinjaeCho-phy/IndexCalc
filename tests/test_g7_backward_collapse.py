"""G7 backward: covariant_collapse 회귀.

규칙:
    - ∂T + Γ corrections (∇̄T expand 형태) → ∇̄T로 collapse.
    - upper slot의 +Γ correction, lower slot의 -Γ correction 부호 일치.
    - dummy 이름 다른 보정 매칭.
    - 매칭 실패 (보정 일부 누락 / 다른 부호) → 변경 없음.
    - only_for로 leaf 필터.
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry, LeviCivitaConnection,
    PartialDeriv, CovariantDeriv, expand_covariant,
    partial_to_covariant, covariant_collapse,
    TensorSum, TensorProduct, ScalarMul,
)


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλα", metric="g")


@pytest.fixture
def chr(st):
    g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")], symmetric_pairs=[(0, 1)])
    g_up = Tensor("g", [st.upper("μ"), st.upper("ν")], symmetric_pairs=[(0, 1)])
    return LeviCivitaConnection(g_lo, g_up, st)


# ─── Round-trip from CovariantDeriv expand ───────────────


class TestRoundTrip:
    def test_upper_only(self, st, chr):
        """∇̄_μ T^ρ → expand → collapse → ∇̄_μ T^ρ."""
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        back = covariant_collapse(raw, chr)
        assert isinstance(back, CovariantDeriv)
        assert back.expr.name == "T"

    def test_lower_only(self, st, chr):
        """∇̄_μ T_ρ → expand → collapse → ∇̄_μ T_ρ."""
        T = Tensor("T", [st.lower("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        back = covariant_collapse(raw, chr)
        assert isinstance(back, CovariantDeriv)

    def test_mixed_slots(self, st, chr):
        """∇̄_μ T^ρ_σ → expand (3 terms) → collapse → ∇̄_μ T^ρ_σ."""
        T = Tensor("T", [st.upper("ρ"), st.lower("σ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        # raw should have 3 terms: ∂T + Γ·T - Γ·T
        from indexcalc.core.simplify import _flatten_sum
        assert len(_flatten_sum(raw)) == 3
        back = covariant_collapse(raw, chr)
        assert isinstance(back, CovariantDeriv)


# ─── Partial cancellation cases ─────────────────────────


class TestPartialMatch:
    def test_missing_correction_no_collapse(self, st, chr):
        """Γ correction이 하나만 있고 다른 거 없으면 collapse 안 함."""
        T = Tensor("T", [st.upper("ρ"), st.lower("σ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        # raw에서 마지막 항 제거 (보정 부분 깨기)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(raw)
        # 첫 두 항만 (∂T + 일부 Γ)
        broken = terms[0] + terms[1]
        back = covariant_collapse(broken, chr)
        # collapse 실패 → 그대로 (TensorSum)
        assert isinstance(back, TensorSum)

    def test_no_partial_no_collapse(self, st, chr):
        """∂가 없는 sum이면 변경 없음."""
        T = Tensor("T", [st.upper("ρ")])
        S = Tensor("S", [st.upper("ρ")])
        expr = T + S  # 두 Tensor의 합
        back = covariant_collapse(expr, chr)
        assert back is expr

    def test_extra_summand_preserved(self, st, chr):
        """∂T + corrections + (관계없는 X) → ∇̄T + X."""
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)  # 2 terms (∂T + Γ·T)
        X = Tensor("X", [st.lower("μ"), st.upper("ρ")])
        expr = raw + X
        back = covariant_collapse(expr, chr)
        # 결과는 ∇̄T + X (TensorSum)
        assert isinstance(back, TensorSum)
        from indexcalc.core.simplify import _flatten_sum
        terms = _flatten_sum(back)
        assert len(terms) == 2
        # 한쪽이 CovariantDeriv여야 함
        kinds = sorted(type(t).__name__ for t in terms)
        assert "CovariantDeriv" in kinds


# ─── only_for ─────────────────────────────────────────────


class TestOnlyFor:
    def test_only_for_excludes(self, st, chr):
        """only_for={'X'}이면 ∂T는 collapse 안 됨."""
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        back = covariant_collapse(raw, chr, only_for={"X"})
        # T 이름이 only_for에 없음 → 변경 없음
        assert back is raw

    def test_only_for_includes(self, st, chr):
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        back = covariant_collapse(raw, chr, only_for={"T"})
        assert isinstance(back, CovariantDeriv)


# ─── Dummy name agnostic matching ────────────────────────


class TestDummyAgnostic:
    def test_user_term_with_different_dummy(self, st, chr):
        """Same content with different dummy name in correction → still matches.

        Note: Dummy 이름은 raw 생성 시 자동으로 mu_1, mu_2 등이 붙음. 사용자가
        다른 이름(α, β)으로 직접 작성했어도 canonical_form_modulo_dummies가
        동일성 인식. Γ는 ``chr.make_tensor``로 만들어 sym_pairs=[(1,2)] 보장.
        """
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        # raw = ∂_μ T^ρ + Γ^ρ_{μ μ_1} T^{μ_1}
        # 사용자 버전 (수동): ∂_μ T^ρ + Γ^ρ_{μ α} T^{α} (chr.make_tensor로 sym 부여)
        partial = PartialDeriv(T, st.lower("μ"))
        gamma = chr.make_tensor("ρ", "μ", "α")
        T_alpha = Tensor("T", [st.upper("α")])
        user_correction = TensorProduct(gamma, T_alpha)
        user_form = partial + user_correction
        back = covariant_collapse(user_form, chr)
        assert isinstance(back, CovariantDeriv)
