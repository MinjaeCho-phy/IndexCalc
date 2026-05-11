"""G5: 텐서 속성(traceless / transverse / symmetric_pairs)을 활용한
simplification 규칙 회귀 테스트.

규칙:
    - traceless × metric → 0 :  γ^{ij} h^{TT}_{ij} = 0
    - transverse × ∂ (direct) → 0 :  ∂^i BV_i = 0
    - transverse × ∂ via metric → 0 :  γ^{ij} ∂_i BV_j = 0
    - antisym × symmetric_pairs slot → 0 :  A_{[ab]} S^{ab}_{(...)} = 0
"""

import pytest

from indexcalc import (
    IndexSpace, Tensor, TensorProduct, MetricRegistry,
    PartialDeriv,
)
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import (
    is_zero_by_traceless_metric,
    is_zero_by_transverse_deriv,
    is_zero_by_antisym_swap,
    simplify,
)


# ─── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def sp():
    return IndexSpace("sp", dim=3, indices="ijklmn", metric="γ")


@pytest.fixture
def mreg(sp):
    i, j = sp.lower("i"), sp.lower("j")
    iU, jU = sp.upper("i"), sp.upper("j")
    g_lo = Tensor("γ", [i, j], symmetric_pairs=[(0, 1)])
    g_up = Tensor("γ", [iU, jU], symmetric_pairs=[(0, 1)])
    m = MetricRegistry()
    m.register(g_lo, g_up, sp)
    return m


# ─── G5a: traceless × metric → 0 ────────────────────────────


class TestTracelessMetric:
    def test_inverse_metric_traces_traceless_tensor(self, sp, mreg):
        """γ^{ij} h^{TT}_{ij} → 0."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        hTT = Tensor(
            "hTT", [i, j],
            symmetric_pairs=[(0, 1)],
            traceless=[(0, 1)],
        )
        γU = Tensor("γ", [iU, jU], symmetric_pairs=[(0, 1)])
        expr = TensorProduct(γU, hTT)
        result = is_zero_by_traceless_metric(expr, mreg)
        assert isinstance(result, ZeroTensor)
        assert result.free_indices == []

    def test_metric_lower_traces_upper_traceless(self, sp, mreg):
        """γ_{ij} T^{ij} where T^{ij} is traceless → 0."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        T = Tensor(
            "T", [iU, jU],
            symmetric_pairs=[(0, 1)],
            traceless=[(0, 1)],
        )
        γL = Tensor("γ", [i, j], symmetric_pairs=[(0, 1)])
        expr = TensorProduct(γL, T)
        result = is_zero_by_traceless_metric(expr, mreg)
        assert isinstance(result, ZeroTensor)

    def test_partial_contraction_not_zero(self, sp, mreg):
        """h^{TT}_{ij} alone (free) is not zero — only metric trace vanishes."""
        i, j = sp.lower("i"), sp.lower("j")
        hTT = Tensor("hTT", [i, j], traceless=[(0, 1)])
        # 동반자가 없으면 TensorProduct 아님 → 그대로
        result = is_zero_by_traceless_metric(hTT, mreg)
        assert result is hTT

    def test_no_traceless_property_no_reduction(self, sp, mreg):
        """traceless 속성 없는 텐서는 metric trace 가능."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        T = Tensor("T", [i, j])  # 평범한 텐서
        γU = Tensor("γ", [iU, jU])
        expr = TensorProduct(γU, T)
        result = is_zero_by_traceless_metric(expr, mreg)
        # T가 traceless 아니므로 reduction 없음
        assert not isinstance(result, ZeroTensor)

    def test_unrelated_metric_does_not_trigger(self, sp, mreg):
        """다른 IndexSpace의 metric은 trigger 안 됨."""
        st = IndexSpace("st", dim=4, indices="μνρσ", metric="g")
        i, j = sp.lower("i"), sp.lower("j")
        μU, νU = st.upper("μ"), st.upper("ν")
        hTT = Tensor("hTT", [i, j], traceless=[(0, 1)])
        gU_st = Tensor("g", [μU, νU])
        expr = TensorProduct(gU_st, hTT)
        result = is_zero_by_traceless_metric(expr, mreg)
        assert not isinstance(result, ZeroTensor)


# ─── G5b: transverse × deriv → 0 ────────────────────────────


class TestTransverseDeriv:
    def test_direct_partial_into_transverse_vector(self, sp):
        """∂^i BV_i pattern: transverse=[0] 인 BV의 인덱스가 ∂의 raised index와 contract.

        실제 표현은 PartialDeriv(BV_with_upper_i, lower_i):
            BV는 i가 upper position(transverse 인정 = upper면 ∂_i가 contract)
            wait: convention — direct case 작동 조건은 T.transverse slot이 upper
            이고, deriv index name과 같다.
        """
        # BV^i (upper index, transverse slot at 0)
        iU = sp.upper("i")
        i = sp.lower("i")
        BV_up = Tensor("BV", [iU], transverse=[0])
        expr = PartialDeriv(BV_up, i)  # ∂_i BV^i — direct contraction
        result = is_zero_by_transverse_deriv(expr)
        assert isinstance(result, ZeroTensor)

    def test_via_metric(self, sp, mreg):
        """γ^{ij} ∂_i BV_j → 0 (raise i to contract with BV_j transverse).

        Raised index 'i' contracts with BV's lower transverse 'j' through metric γ^{ij}.
        Algorithm matches: deriv_index name 'i' ↔ metric ↔ transverse name 'j'.
        """
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        BV = Tensor("BV", [j], transverse=[0])
        γU = Tensor("γ", [iU, jU], symmetric_pairs=[(0, 1)])
        deriv = PartialDeriv(BV, i)  # ∂_i BV_j
        expr = TensorProduct(γU, deriv)
        result = is_zero_by_transverse_deriv(expr, mreg)
        assert isinstance(result, ZeroTensor)

    def test_no_transverse_no_zero(self, sp, mreg):
        """transverse 속성 없는 vector는 metric으로 raise해서 contract해도 0 아님."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        V = Tensor("V", [j])  # 평범한 vector
        γU = Tensor("γ", [iU, jU])
        deriv = PartialDeriv(V, i)
        expr = TensorProduct(γU, deriv)
        result = is_zero_by_transverse_deriv(expr, mreg)
        assert not isinstance(result, ZeroTensor)

    def test_free_index_does_not_trigger(self, sp):
        """transverse slot 인덱스 이름이 deriv index와 같지만, 둘 다 free면 0 아님."""
        # ∂_i BV^i 만들면 사실 contracted; 별도 구성 — single PartialDeriv 직접 검사 OK
        # 여기선 단일 BV에 ∂_i 적용 후 다른 인덱스 이름으로 — pattern 자체가 없는 case
        iU = sp.upper("i")
        kL = sp.lower("k")
        BV_up = Tensor("BV", [iU], transverse=[0])
        expr = PartialDeriv(BV_up, kL)  # ∂_k BV^i — different names, no contraction
        result = is_zero_by_transverse_deriv(expr)
        assert not isinstance(result, ZeroTensor)


# ─── G5c: antisym × symmetric_pairs slot → 0 ───────────────


class TestAntisymVsSymmetricPair:
    def test_antisym_pair_meets_symmetric_pair(self, sp):
        """A_{[ab]} S^{ab} where S has symmetric_pairs=[(0,1)] → 0.

        rest = S after dummy swap a↔b: S 인덱스가 sym 슬롯 안에서 자리바꿈됨.
        symmetric_pairs slot 정렬을 _factor_key_no_swap에 추가했으므로 key
        invariant → ZeroTensor 검출.
        """
        adj = IndexSpace("adj", dim=8, indices="abcdefgh")
        a, b = adj.lower("a"), adj.lower("b")
        aU, bU = adj.upper("a"), adj.upper("b")
        A = Tensor("A", [a, b], antisymmetric_pairs=[(0, 1)])
        S = Tensor("S", [aU, bU], symmetric_pairs=[(0, 1)])
        expr = TensorProduct(A, S)
        result = is_zero_by_antisym_swap(expr)
        assert isinstance(result, ZeroTensor)

    def test_antisym_pair_meets_nonsymmetric_factor(self, sp):
        """A_{[ab]} X^a Y^b — X·Y는 sym 속성 없음, swap 후 이름 cross 불일치 → 0 아님."""
        adj = IndexSpace("adj", dim=8, indices="abcdefgh")
        a, b = adj.lower("a"), adj.lower("b")
        aU, bU = adj.upper("a"), adj.upper("b")
        A = Tensor("A", [a, b], antisymmetric_pairs=[(0, 1)])
        X = Tensor("X", [aU])
        Y = Tensor("Y", [bU])
        expr = TensorProduct(A, TensorProduct(X, Y))
        result = is_zero_by_antisym_swap(expr)
        assert not isinstance(result, ZeroTensor)


# ─── top-level simplify integration ────────────────────────


class TestSimplifyIntegration:
    def test_simplify_with_mreg_collapses_traceless_trace(self, sp, mreg):
        """top-level simplify(expr, mreg)도 traceless × metric → 0."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        hTT = Tensor("hTT", [i, j], traceless=[(0, 1)])
        γU = Tensor("γ", [iU, jU])
        expr = TensorProduct(γU, hTT)
        result = simplify(expr, mreg)
        assert isinstance(result, ZeroTensor)

    def test_simplify_without_mreg_preserves_expr(self, sp):
        """mreg 없으면 traceless rule이 비활성 → 그대로."""
        i, j = sp.lower("i"), sp.lower("j")
        iU, jU = sp.upper("i"), sp.upper("j")
        hTT = Tensor("hTT", [i, j], traceless=[(0, 1)])
        γU = Tensor("γ", [iU, jU])
        expr = TensorProduct(γU, hTT)
        result = simplify(expr)
        assert not isinstance(result, ZeroTensor)
