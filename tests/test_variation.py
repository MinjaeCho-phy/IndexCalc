"""Variation operator (δ) 테스트 — P1~P4."""

import pytest
from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv, Connection
from indexcalc.core.variation import (
    Variation, ZeroTensor, VariationRegistry, expand_variation, _simplify_zeros,
)
from indexcalc.parse.latex import IndexRegistry, parse
from indexcalc.parse.display import to_latex


# ─── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def spaces():
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    lr = IndexSpace("lorentz", dim=4, indices="pqrs", metric="η")
    return st, lr


@pytest.fixture
def registry(spaces):
    st, lr = spaces
    reg = IndexRegistry()
    reg.register(st)
    reg.register(lr)
    return reg


# ─── P1: Variation node ─────────────────────────────────────

class TestVariationNode:
    def test_free_indices(self, spaces):
        st, _ = spaces
        T = Tensor("T", [st.upper("μ"), st.lower("ν")])
        v = Variation(T)
        assert len(v.free_indices) == 2
        assert v.free_indices[0].name == "μ"

    def test_repr(self, spaces):
        st, _ = spaces
        T = Tensor("T", [st.upper("μ")])
        v = Variation(T)
        assert "δ" in repr(v)


# ─── P1: ZeroTensor ─────────────────────────────────────────

class TestZeroTensor:
    def test_free_indices(self, spaces):
        st, _ = spaces
        z = ZeroTensor([st.lower("μ"), st.lower("ν")])
        assert len(z.free_indices) == 2

    def test_repr(self):
        z = ZeroTensor([])
        assert repr(z) == "0"


# ─── P1: VariationRegistry ──────────────────────────────────

class TestVariationRegistry:
    def test_varying_default(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("μ"), st.lower("ν")])
        result = vreg.delta_of(T)
        assert isinstance(result, Tensor)
        assert result.name == "δT"
        assert len(result.indices) == 2

    def test_background(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_background("η")
        eta = Tensor("η", [st.lower("μ"), st.lower("ν")])
        result = vreg.delta_of(eta)
        assert isinstance(result, ZeroTensor)
        assert len(result.free_indices) == 2

    def test_undeclared_raises(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        T = Tensor("T", [st.upper("μ")])
        with pytest.raises(ValueError, match="not declared"):
            vreg.delta_of(T)


# ─── P1: expand_variation — Leibniz ─────────────────────────

class TestExpandVariation:
    def test_single_tensor(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("μ")])
        expr = Variation(T)
        result = expand_variation(expr, vreg)
        assert isinstance(result, Tensor)
        assert result.name == "δT"

    def test_sum(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("A")
        vreg.declare_varying("B")
        A = Tensor("A", [st.upper("μ")])
        B = Tensor("B", [st.upper("μ")])
        expr = Variation(TensorSum(A, B))
        result = expand_variation(expr, vreg)
        assert isinstance(result, TensorSum)
        assert to_latex(result) == r"δA^{\mu} + δB^{\mu}"

    def test_product_leibniz(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("A")
        vreg.declare_varying("B")
        A = Tensor("A", [st.upper("μ")])
        B = Tensor("B", [st.lower("ν")])
        expr = Variation(TensorProduct(A, B))
        result = expand_variation(expr, vreg)
        # δ(A*B) = δA*B + A*δB
        assert isinstance(result, TensorSum)

    def test_scalar_mul(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("μ")])
        expr = Variation(ScalarMul(3, T))
        result = expand_variation(expr, vreg)
        assert isinstance(result, ScalarMul)
        assert result.scalar == 3

    def test_background_eliminated(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("A")
        vreg.declare_background("B")
        A = Tensor("A", [st.upper("μ")])
        B = Tensor("B", [st.lower("μ")])
        # δ(A * B) = δA*B + A*δB = δA*B + A*0 = δA*B
        expr = Variation(TensorProduct(A, B))
        result = expand_variation(expr, vreg)
        # A*0 should be simplified away, leaving only δA*B
        assert isinstance(result, TensorProduct)
        assert to_latex(result.left) == r"δA^{\mu}"

    def test_sum_both_background(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_background("A")
        vreg.declare_background("B")
        A = Tensor("A", [st.upper("μ")])
        B = Tensor("B", [st.upper("μ")])
        expr = Variation(TensorSum(A, B))
        result = expand_variation(expr, vreg)
        assert isinstance(result, ZeroTensor)


# ─── P2: _simplify_zeros ────────────────────────────────────

class TestSimplifyZeros:
    def test_sum_left_zero(self, spaces):
        st, _ = spaces
        z = ZeroTensor([st.upper("μ")])
        T = Tensor("T", [st.upper("μ")])
        result = _simplify_zeros(TensorSum(z, T))
        assert isinstance(result, Tensor)
        assert result.name == "T"

    def test_sum_right_zero(self, spaces):
        st, _ = spaces
        T = Tensor("T", [st.upper("μ")])
        z = ZeroTensor([st.upper("μ")])
        result = _simplify_zeros(TensorSum(T, z))
        assert isinstance(result, Tensor)
        assert result.name == "T"

    def test_product_zero(self, spaces):
        st, _ = spaces
        T = Tensor("T", [st.upper("μ")])
        z = ZeroTensor([st.lower("μ")])
        prod = TensorProduct(T, z)
        result = _simplify_zeros(prod)
        assert isinstance(result, ZeroTensor)

    def test_scalar_zero(self, spaces):
        st, _ = spaces
        z = ZeroTensor([st.upper("μ")])
        result = _simplify_zeros(ScalarMul(5, z))
        assert isinstance(result, ZeroTensor)


# ─── P3: ∂/∇ 교환 ──────────────────────────────────────────

class TestDerivExchange:
    def test_partial_exchange(self, spaces):
        """δ(∂_μ T^ν) = ∂_μ(δT^ν)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("ν")])
        expr = Variation(PartialDeriv(T, st.lower("μ")))
        result = expand_variation(expr, vreg)
        assert isinstance(result, PartialDeriv)
        assert isinstance(result.expr, Tensor)
        assert result.expr.name == "δT"

    def test_covariant_exchange(self, spaces):
        """δ(∇_μ T^ν) = ∇_μ(δT^ν) (P1: background connection)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        conn = Connection("Γ", st)
        T = Tensor("T", [st.upper("ν")])
        expr = Variation(CovariantDeriv(T, st.lower("μ"), conn))
        result = expand_variation(expr, vreg)
        assert isinstance(result, CovariantDeriv)
        assert isinstance(result.expr, Tensor)
        assert result.expr.name == "δT"


# ─── P4: Parser + Display ───────────────────────────────────

class TestParserDisplay:
    def test_parse_var_braces(self, spaces, registry):
        """\\Var{T^{μ}} → Variation(Tensor("T", [^μ]))"""
        expr = parse(r"\Var{T^{\mu}}", registry)
        assert isinstance(expr, Variation)
        assert isinstance(expr.expr, Tensor)
        assert expr.expr.name == "T"

    def test_parse_var_compound(self, spaces, registry):
        """\\Var{A^{μ} + B^{μ}} → Variation(TensorSum(...))"""
        expr = parse(r"\Var{A^{\mu} + B^{\mu}}", registry)
        assert isinstance(expr, Variation)
        assert isinstance(expr.expr, TensorSum)

    def test_display_variation(self, spaces):
        st, _ = spaces
        T = Tensor("T", [st.upper("μ")])
        v = Variation(T)
        assert to_latex(v) == r"\delta(T^{\mu})"

    def test_display_zero_tensor(self):
        z = ZeroTensor([])
        assert to_latex(z) == "0"

    def test_roundtrip_parse_display(self, spaces, registry):
        """parse → to_latex roundtrip for un-expanded Variation"""
        expr = parse(r"\Var{T^{\mu}}", registry)
        latex = to_latex(expr)
        assert latex == r"\delta(T^{\mu})"


# ─── Acceptance criterion ───────────────────────────────────

class TestAcceptanceCriterion:
    def test_dft_first_equation(self):
        """사진 1번 식: δ(e η + B e) = δe·η + δB·e + B·δe"""
        st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
        lr = IndexSpace("lorentz", dim=4, indices="pqrs", metric="η")
        reg = IndexRegistry()
        reg.register(st)
        reg.register(lr)

        latex_in = r"\Var{ e_{\mu}{}^{q} \eta_{p q} + B_{\mu \sigma} e^{\sigma}{}_{p} }"
        expr = parse(latex_in, reg)
        assert isinstance(expr, Variation)

        vreg = VariationRegistry()
        vreg.declare_varying("e")
        vreg.declare_varying("B")
        vreg.declare_background("η")

        result = expand_variation(expr, vreg)
        latex_out = to_latex(result)

        expected = (
            r"δe_{\mu}{}^{q} \eta_{p q} + "
            r"δB_{\mu \sigma} e^{\sigma}{}_{p} + "
            r"B_{\mu \sigma} δe^{\sigma}{}_{p}"
        )
        assert latex_out == expected


# ─── Trace 케이스 ───────────────────────────────────────────

class TestTraceVariation:
    def test_delta_of_trace_tensor(self, spaces):
        """δ(Tr T) = Tr(δT)"""
        from indexcalc.core.contract import Trace
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("μ"), st.lower("μ")])
        tr = Trace(T, "μ")
        result = expand_variation(Variation(tr), vreg)
        assert isinstance(result, Trace)
        assert result.tensor.name == "δT"

    def test_delta_of_trace_background(self, spaces):
        """δ(Tr η) = 0"""
        from indexcalc.core.contract import Trace
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_background("η")
        eta = Tensor("η", [st.upper("μ"), st.lower("μ")])
        tr = Trace(eta, "μ")
        result = expand_variation(Variation(tr), vreg)
        assert isinstance(result, ZeroTensor)


# ─── P5: 2차 변분 ───────────────────────────────────────────

class TestSecondOrderVariation:
    def test_delta_squared_single(self, spaces):
        """δ²(T) = δ(δT) = δδT (auto-registered nested)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        T = Tensor("T", [st.upper("μ")])
        expr = Variation(Variation(T))
        result = expand_variation(expr, vreg)
        assert isinstance(result, Tensor)
        assert result.name == "δδT"

    def test_delta_squared_background(self, spaces):
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_background("η")
        eta = Tensor("η", [st.lower("μ")])
        expr = Variation(Variation(eta))
        result = expand_variation(expr, vreg)
        assert isinstance(result, ZeroTensor)

    def test_delta_squared_product(self, spaces):
        """δ²(A·B) = δ²A·B + 2·δA·δB + A·δ²B (구조적 확인)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("A")
        vreg.declare_varying("B")
        A = Tensor("A", [st.upper("μ")])
        B = Tensor("B", [st.lower("ν")])
        expr = Variation(Variation(TensorProduct(A, B)))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        # 네 항: δδA·B, δA·δB, δA·δB, A·δδB
        assert latex.count("δδA") == 1
        assert latex.count("δδB") == 1
        assert latex.count("δA^{\\mu} δB") == 2

    def test_nested_var_parser(self, spaces, registry):
        r"""\\Var{\\Var{T^μ}} 파싱"""
        expr = parse(r"\Var{\Var{T^{\mu}}}", registry)
        assert isinstance(expr, Variation)
        assert isinstance(expr.expr, Variation)


# ─── P6: Palatini δΓ ────────────────────────────────────────

class TestPalatini:
    def test_background_connection_no_correction(self, spaces):
        """Γ가 varying이 아니면 δ(∇T) = ∇(δT) (기존 동작 유지)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        conn = Connection("Γ", st)
        T = Tensor("T", [st.upper("ν")])
        expr = Variation(CovariantDeriv(T, st.lower("μ"), conn))
        result = expand_variation(expr, vreg)
        assert isinstance(result, CovariantDeriv)
        assert result.expr.name == "δT"

    def test_varying_connection_upper_index(self, spaces):
        """δ(∇_μ T^ν) = ∇_μ(δT^ν) + δΓ^ν_{μρ} T^ρ (Γ varying)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        vreg.declare_varying_connection("Γ")
        conn = Connection("Γ", st)
        T = Tensor("T", [st.upper("ν")])
        expr = Variation(CovariantDeriv(T, st.lower("μ"), conn))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        assert "δT" in latex
        assert "δΓ" in latex

    def test_varying_connection_lower_index_sign(self, spaces):
        """δ(∇_μ T_ν) = ∇_μ(δT_ν) - δΓ^ρ_{μν} T_ρ (minus 부호)"""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_varying("T")
        vreg.declare_varying_connection("Γ")
        conn = Connection("Γ", st)
        T = Tensor("T", [st.lower("ν")])
        expr = Variation(CovariantDeriv(T, st.lower("μ"), conn))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        assert " - " in latex  # 마이너스 부호 존재
        assert "δΓ" in latex

    def test_background_tensor_with_varying_connection(self, spaces):
        """δT=0 이어도 δΓ·T 항은 살아남아야 한다."""
        st, _ = spaces
        vreg = VariationRegistry()
        vreg.declare_background("T")
        vreg.declare_varying_connection("Γ")
        conn = Connection("Γ", st)
        T = Tensor("T", [st.upper("ν")])
        expr = Variation(CovariantDeriv(T, st.lower("μ"), conn))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        assert "δΓ" in latex
        # ∇(δT) = ∇(0) = 0이 소거되고 δΓ 항만 남음
        assert "\\nabla" not in latex
