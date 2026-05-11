"""Christoffel symmetry-aware: LeviCivita Γ가 sym_pairs=[(1,2)] 보유, δΓ도
Palatini correction 시 동일 sym 보존, simplify가 antisym×Γ→0 같은 cancel 활용.
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, MetricRegistry,
    LeviCivitaConnection, CovariantDeriv, expand_covariant,
    Variation, VariationRegistry, expand_variation, ZeroTensor,
    TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.simplify import simplify, is_zero_by_antisym_swap


@pytest.fixture
def st():
    return IndexSpace("st", dim=4, indices="μνρσλα", metric="g")


@pytest.fixture
def chr(st):
    g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")], symmetric_pairs=[(0, 1)])
    g_up = Tensor("g", [st.upper("μ"), st.upper("ν")], symmetric_pairs=[(0, 1)])
    return LeviCivitaConnection(g_lo, g_up, st)


# ─── Christoffel sym 자체 ─────────────────────────────────


class TestChristoffelSym:
    def test_make_tensor_has_sym_pairs(self, chr):
        gamma = chr.make_tensor("a", "b", "c")
        assert gamma.symmetric_pairs == ((1, 2),)
        assert gamma.name == "Γ"

    def test_christoffel_alias(self, chr):
        g1 = chr.make_tensor("a", "b", "c")
        g2 = chr.christoffel("a", "b", "c")
        assert g1.symmetric_pairs == g2.symmetric_pairs
        assert g1.name == g2.name


# ─── expand_covariant이 sym Γ를 만들어내는지 ─────────────


class TestExpandCovariantSym:
    def test_correction_gamma_has_sym(self, st, chr):
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        raw = expand_covariant(cov)
        # raw = ∂T + Γ·T_replaced. Γ tensor를 찾아 sym 확인
        from indexcalc.core.simplify import _flatten_sum, collect_factors
        for term in _flatten_sum(raw):
            for f in collect_factors(term):
                if isinstance(f, Tensor) and f.name == "Γ":
                    assert f.symmetric_pairs == ((1, 2),)


# ─── δΓ via Palatini도 sym 보존 ──────────────────────────


class TestPalatiniSymPreserved:
    def test_delta_gamma_has_sym(self, st, chr):
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        vreg.declare_varying_connection("Γ")
        T = Tensor("T", [st.upper("ρ")])
        cov = CovariantDeriv(T, st.lower("μ"), chr)
        # δ(∇_μ T^ρ) → ∇_μ δT^ρ + δΓ^ρ_{μ α} T^α
        delta = expand_variation(Variation(cov), vreg)
        from indexcalc.core.simplify import _flatten_sum, collect_factors
        found = False
        for term in _flatten_sum(delta):
            for f in collect_factors(term):
                if isinstance(f, Tensor) and f.name == "δΓ":
                    found = True
                    assert f.symmetric_pairs == ((1, 2),)
        assert found, "δΓ tensor not found in expansion"


# ─── antisym × Γ slot → 0 (sym 활용) ─────────────────────


class TestAntisymVsChristoffel:
    def test_antisym_factor_with_gamma_sym_slots(self, st, chr):
        """A_{μν} (antisym μ↔ν) × Γ^ρ_{μν} (sym μ↔ν) 동일 dummy → 0.

        is_zero_by_antisym_swap이 sym slot canonicalize로 검출해야.
        """
        # A^{μν} antisymmetric
        A = Tensor("A", [st.upper("μ"), st.upper("ν")], antisymmetric_pairs=[(0, 1)])
        # Γ^ρ_{μν} (sym μν) — chr.make_tensor로 만들면 sym 자동
        gamma = chr.make_tensor("ρ", "μ", "ν")
        expr = TensorProduct(A, gamma)
        result = is_zero_by_antisym_swap(expr)
        assert isinstance(result, ZeroTensor)


# ─── Backward compat: 모든 기존 테스트 통과 ───────────


class TestBackwardCompat:
    def test_simplify_no_change_on_simple_gamma_t(self, st, chr):
        """Γ·T (특별한 cancel 패턴 없음) → simplify 변경 없음."""
        gamma = chr.make_tensor("ρ", "μ", "ν")
        T = Tensor("T", [st.lower("ν"), st.upper("μ")])
        # ν와 μ가 contract됨 (T의 ν lower가 Γ의 ν lower와 contract — 부적합)
        # 그냥 contract되는 식으로 만들자
        T2 = Tensor("T", [st.upper("ν"), st.upper("μ")])
        # Γ^ρ_μν T^μν contract → free=[ρ]
        expr = TensorProduct(gamma, T2)
        # T2가 sym/antisym 없으면 cancel 없음
        result = simplify(expr)
        # 결과는 zero가 아님 (T2가 일반 tensor)
        assert not isinstance(result, ZeroTensor)
