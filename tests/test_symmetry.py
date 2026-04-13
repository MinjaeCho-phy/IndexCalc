"""Antisymmetric tensor canonicalization tests."""

import pytest
from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.symmetry import canonicalize_antisym
from indexcalc.parse.display import to_latex


@pytest.fixture
def st():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")


class TestAntisymTensor:
    def test_pairs_stored(self, st):
        B = Tensor("B", [st.lower("μ"), st.lower("ν")], antisymmetric_pairs=[(0, 1)])
        assert B.antisymmetric_pairs == ((0, 1),)

    def test_no_pairs_default(self, st):
        T = Tensor("T", [st.lower("μ"), st.lower("ν")])
        assert T.antisymmetric_pairs == ()


class TestCanonicalize:
    def test_canonical_order_unchanged(self, st):
        """B_μν는 이미 canonical이라 그대로."""
        B = Tensor("B", [st.lower("μ"), st.lower("ν")], antisymmetric_pairs=[(0, 1)])
        result = canonicalize_antisym(B)
        assert isinstance(result, Tensor)
        assert result.indices[0].name == "μ"
        assert result.indices[1].name == "ν"

    def test_swap_produces_minus(self, st):
        """B_νμ → -B_μν"""
        B = Tensor("B", [st.lower("ν"), st.lower("μ")], antisymmetric_pairs=[(0, 1)])
        result = canonicalize_antisym(B)
        assert isinstance(result, ScalarMul)
        assert result.scalar == -1
        assert result.expr.indices[0].name == "μ"
        assert result.expr.indices[1].name == "ν"

    def test_non_antisym_not_touched(self, st):
        """antisymmetric_pairs가 없으면 건드리지 않음."""
        T = Tensor("T", [st.lower("ν"), st.lower("μ")])
        result = canonicalize_antisym(T)
        assert result is T

    def test_product_of_two_swaps_plus(self, st):
        """두 번 -1 → +1."""
        B1 = Tensor("B", [st.lower("ν"), st.lower("μ")], antisymmetric_pairs=[(0, 1)])
        B2 = Tensor("C", [st.lower("σ"), st.lower("λ")], antisymmetric_pairs=[(0, 1)])
        prod = TensorProduct(B1, B2)
        result = canonicalize_antisym(prod)
        assert isinstance(result, TensorProduct)
        assert result.left.indices[0].name == "μ"
        assert result.right.indices[0].name == "λ"

    def test_in_sum(self, st):
        B1 = Tensor("B", [st.lower("ν"), st.lower("μ")], antisymmetric_pairs=[(0, 1)])
        A = Tensor("A", [st.lower("μ"), st.lower("ν")])
        expr = TensorSum(A, B1)
        result = canonicalize_antisym(expr)
        latex = to_latex(result)
        assert "- B" in latex
        assert "A_{\\mu \\nu}" in latex

    def test_dft_style_bfield(self, st):
        """DFT B-field: B_μν antisym, positional swap tracks sign."""
        B_bad = Tensor(
            "B", [st.upper("σ"), st.lower("μ")],
            antisymmetric_pairs=[(0, 1)],
        )
        # σ (upper) vs μ (lower): 이름은 σ>μ 이지만 position upper가 우선
        # canonical key: (name, 0 if upper else 1). σ는 ("σ", 0), μ는 ("μ", 1).
        # σ < μ in canonical key (이름순에서 μ<σ지만 튜플 첫 요소로 μ가 먼저)
        # 즉 ("μ", 1) vs ("σ", 0): "μ" < "σ" 이므로 μ가 먼저여야 함.
        # 원래 순서: [σ_up, μ_low] → key: [("σ",0), ("μ",1)]. ("σ",0) > ("μ",1).
        # 스왑 필요 → 부호 -1.
        result = canonicalize_antisym(B_bad)
        assert isinstance(result, ScalarMul)
        assert result.scalar == -1
