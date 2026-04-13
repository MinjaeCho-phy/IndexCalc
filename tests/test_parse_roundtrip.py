r"""
LaTeX 파서 강화 테스트: 파싱, 디스플레이, roundtrip 검증.

테스트 범주:
  1. 기존 파서 기능 (Unicode 인덱스)
  2. LaTeX 명령 인덱스 파싱 (\mu, \nu 등)
  3. \partial 파싱
  4. \nabla 파싱
  5. \Gamma 등 LaTeX 텐서 이름 파싱
  6. {} 빈 그룹 처리
  7. to_latex → parse roundtrip
"""

import pytest
from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.deriv import (
    PartialDeriv, CovariantDeriv, Connection, partial, covariant,
)
from indexcalc.parse.latex import IndexRegistry, parse
from indexcalc.parse.display import to_latex


# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def spacetime():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")


@pytest.fixture
def lorentz():
    return IndexSpace("lorentz", dim=4, indices="abcde", metric="η")


@pytest.fixture
def reg(spacetime):
    r = IndexRegistry()
    r.register(spacetime)
    return r


@pytest.fixture
def reg2(spacetime, lorentz):
    r = IndexRegistry()
    r.register(spacetime)
    r.register(lorentz)
    return r


# ─── 1. 기존 파서 기능 (기존 동작 보존) ──────────────────────

class TestBasicParsing:
    def test_single_tensor(self, reg, spacetime):
        expr = parse("T^{μ}_{ν}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "T"
        assert len(expr.indices) == 2
        assert expr.indices[0] == Index("μ", spacetime, "upper")
        assert expr.indices[1] == Index("ν", spacetime, "lower")

    def test_tensor_product(self, reg):
        expr = parse("T^{μ}_{ν} S^{ν}_{λ}", reg)
        assert isinstance(expr, TensorProduct)

    def test_tensor_sum(self, reg):
        expr = parse("T^{μ}_{ν} + S^{μ}_{ν}", reg)
        assert isinstance(expr, TensorSum)

    def test_scalar_mul(self, reg):
        expr = parse("2 T^{μ}_{ν}", reg)
        assert isinstance(expr, ScalarMul)
        assert expr.scalar == 2

    def test_frac(self, reg):
        expr = parse(r"\frac{1}{2} T^{μ}_{ν}", reg)
        assert isinstance(expr, ScalarMul)
        assert abs(expr.scalar - 0.5) < 1e-10

    def test_negative(self, reg):
        expr = parse("-T^{μ}_{ν}", reg)
        assert isinstance(expr, ScalarMul)
        assert expr.scalar == -1

    def test_parentheses(self, reg):
        expr = parse("(T^{μ}_{ν} + S^{μ}_{ν}) V^{ν}", reg)
        assert isinstance(expr, TensorProduct)

    def test_shorthand_index(self, reg, spacetime):
        """중괄호 없는 단일 문자 인덱스."""
        expr = parse("T^μ_ν", reg)
        assert isinstance(expr, Tensor)
        assert expr.indices[0] == Index("μ", spacetime, "upper")

    def test_multi_lower_indices(self, reg, spacetime):
        """여러 lower 인덱스."""
        expr = parse("g_{μν}", reg)
        assert isinstance(expr, Tensor)
        assert len(expr.indices) == 2
        assert all(i.position == "lower" for i in expr.indices)


# ─── 2. LaTeX 명령 인덱스 파싱 ─────────────────────────────

class TestLatexCommandIndices:
    def test_latex_upper_index(self, reg, spacetime):
        r"""T^{\mu} → T^μ."""
        expr = parse(r"T^{\mu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.indices[0] == Index("μ", spacetime, "upper")

    def test_latex_lower_index(self, reg, spacetime):
        r"""T_{\nu} → T_ν."""
        expr = parse(r"T_{\nu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.indices[0] == Index("ν", spacetime, "lower")

    def test_latex_mixed_indices(self, reg, spacetime):
        r"""T^{\mu}{}_{\nu} — {} 구분자 포함."""
        expr = parse(r"T^{\mu}{}_{\nu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.indices[0] == Index("μ", spacetime, "upper")
        assert expr.indices[1] == Index("ν", spacetime, "lower")

    def test_latex_multi_indices_in_group(self, reg, spacetime):
        r"""g_{\mu \nu} → g_{μν}."""
        expr = parse(r"g_{\mu \nu}", reg)
        assert isinstance(expr, Tensor)
        assert len(expr.indices) == 2
        assert expr.indices[0] == Index("μ", spacetime, "lower")
        assert expr.indices[1] == Index("ν", spacetime, "lower")

    def test_latex_multi_indices_no_space(self, reg, spacetime):
        r"""g_{\mu\nu} → g_{μν} (공백 없이)."""
        expr = parse(r"g_{\mu\nu}", reg)
        assert isinstance(expr, Tensor)
        assert len(expr.indices) == 2

    def test_latex_product_with_contraction(self, reg):
        r"""T^{\mu}{}_{\nu} S^{\nu}{}_{\lambda}."""
        expr = parse(r"T^{\mu}{}_{\nu} S^{\nu}{}_{\lambda}", reg)
        assert isinstance(expr, TensorProduct)
        assert len(expr.contracted_pairs) == 1


# ─── 3. \partial 파싱 ──────────────────────────────────────

class TestPartialParsing:
    def test_partial_unicode(self, reg, spacetime):
        r"""\partial_{μ} V^{ν}."""
        expr = parse(r"\partial_{μ} V^{ν}", reg)
        assert isinstance(expr, PartialDeriv)
        assert expr.deriv_index == Index("μ", spacetime, "lower")
        assert isinstance(expr.expr, Tensor)
        assert expr.expr.name == "V"

    def test_partial_latex_index(self, reg, spacetime):
        r"""\partial_{\mu} V^{\nu}."""
        expr = parse(r"\partial_{\mu} V^{\nu}", reg)
        assert isinstance(expr, PartialDeriv)
        assert expr.deriv_index == Index("μ", spacetime, "lower")

    def test_partial_on_parenthesized(self, reg):
        r"""\partial_{μ} (T^{μ}_{ν} + S^{μ}_{ν})."""
        expr = parse(r"\partial_{μ} (T^{μ}_{ν} + S^{μ}_{ν})", reg)
        assert isinstance(expr, PartialDeriv)
        assert isinstance(expr.expr, TensorSum)

    def test_nested_partial(self, reg, spacetime):
        r"""\partial_{μ} \partial_{ν} V^{λ}."""
        expr = parse(r"\partial_{μ} \partial_{ν} V^{λ}", reg)
        assert isinstance(expr, PartialDeriv)
        assert expr.deriv_index == Index("μ", spacetime, "lower")
        inner = expr.expr
        assert isinstance(inner, PartialDeriv)
        assert inner.deriv_index == Index("ν", spacetime, "lower")

    def test_partial_in_product(self, reg):
        r"""A^{μ} \partial_{ν} B^{ν} — 암묵적 곱."""
        expr = parse(r"A^{μ} \partial_{ν} B^{ν}", reg)
        assert isinstance(expr, TensorProduct)

    def test_partial_in_sum(self, reg):
        r"""\partial_{μ} A^{ν} + \partial_{μ} B^{ν}."""
        expr = parse(r"\partial_{μ} A^{ν} + \partial_{μ} B^{ν}", reg)
        assert isinstance(expr, TensorSum)

    def test_partial_missing_index_raises(self, reg):
        with pytest.raises(SyntaxError):
            parse(r"\partial V^{μ}", reg)


# ─── 4. \nabla 파싱 ────────────────────────────────────────

class TestNablaParsing:
    def test_nabla_basic(self, reg, spacetime):
        r"""\nabla_{μ} V^{ν}."""
        expr = parse(r"\nabla_{μ} V^{ν}", reg)
        assert isinstance(expr, CovariantDeriv)
        assert expr.deriv_index == Index("μ", spacetime, "lower")
        assert isinstance(expr.expr, Tensor)

    def test_nabla_latex_index(self, reg, spacetime):
        r"""\nabla_{\mu} V^{\nu}."""
        expr = parse(r"\nabla_{\mu} V^{\nu}", reg)
        assert isinstance(expr, CovariantDeriv)
        assert expr.deriv_index == Index("μ", spacetime, "lower")

    def test_nabla_with_connections(self, reg, spacetime):
        conn = Connection("Γ", spacetime)
        conns = {spacetime.name: conn}
        expr = parse(r"\nabla_{μ} V^{ν}", reg, connections=conns)
        assert isinstance(expr, CovariantDeriv)
        assert expr.connections == conns

    def test_nabla_on_parenthesized(self, reg):
        expr = parse(r"\nabla_{μ} (A^{ν} + B^{ν})", reg)
        assert isinstance(expr, CovariantDeriv)
        assert isinstance(expr.expr, TensorSum)

    def test_nabla_missing_index_raises(self, reg):
        with pytest.raises(SyntaxError):
            parse(r"\nabla V^{μ}", reg)


# ─── 5. LaTeX 텐서 이름 파싱 ───────────────────────────────

class TestLatexTensorNames:
    def test_gamma_tensor(self, reg, spacetime):
        r"""\Gamma^{\mu}{}_{\nu \lambda}."""
        expr = parse(r"\Gamma^{\mu}{}_{\nu \lambda}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "Γ"
        assert len(expr.indices) == 3

    def test_eta_tensor(self, reg2, lorentz):
        r"""\eta_{ab}."""
        expr = parse(r"\eta_{ab}", reg2)
        assert isinstance(expr, Tensor)
        assert expr.name == "η"
        assert len(expr.indices) == 2


# ─── 6. {} 빈 그룹 처리 ───────────────────────────────────

class TestEmptyGroup:
    def test_empty_group_skipped(self, reg, spacetime):
        """T^{μ}{}_{ν} — {} 가 무시되어야 함."""
        expr = parse(r"T^{μ}{}_{\nu}", reg)
        assert isinstance(expr, Tensor)
        assert len(expr.indices) == 2
        assert expr.indices[0].position == "upper"
        assert expr.indices[1].position == "lower"


# ─── 7. to_latex → parse roundtrip ─────────────────────────

class TestRoundtrip:
    """to_latex(expr) → parse(latex) 가 원래 구조와 동일한지 검증."""

    def _assert_roundtrip(self, expr, reg, connections=None):
        """expr → to_latex → parse → to_latex 결과가 동일한지 확인."""
        latex1 = to_latex(expr)
        reparsed = parse(latex1, reg, connections=connections)
        latex2 = to_latex(reparsed)
        assert latex1 == latex2, f"Roundtrip failed:\n  {latex1!r}\n  {latex2!r}"

    def test_single_tensor(self, reg, spacetime):
        T = Tensor("T", [spacetime.upper("μ"), spacetime.lower("ν")])
        self._assert_roundtrip(T, reg)

    def test_metric(self, reg, spacetime):
        g = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
        self._assert_roundtrip(g, reg)

    def test_vector(self, reg, spacetime):
        V = Tensor("V", [spacetime.upper("μ")])
        self._assert_roundtrip(V, reg)

    def test_tensor_product(self, reg, spacetime):
        T = Tensor("T", [spacetime.upper("μ"), spacetime.lower("ν")])
        S = Tensor("S", [spacetime.upper("ν"), spacetime.lower("λ")])
        self._assert_roundtrip(T * S, reg)

    def test_tensor_sum(self, reg, spacetime):
        A = Tensor("A", [spacetime.upper("μ")])
        B = Tensor("B", [spacetime.upper("μ")])
        self._assert_roundtrip(A + B, reg)

    def test_tensor_diff(self, reg, spacetime):
        A = Tensor("A", [spacetime.upper("μ")])
        B = Tensor("B", [spacetime.upper("μ")])
        self._assert_roundtrip(A - B, reg)

    def test_scalar_mul(self, reg, spacetime):
        T = Tensor("T", [spacetime.upper("μ"), spacetime.lower("ν")])
        self._assert_roundtrip(ScalarMul(2, T), reg)

    def test_frac_scalar(self, reg, spacetime):
        T = Tensor("T", [spacetime.upper("μ"), spacetime.lower("ν")])
        self._assert_roundtrip(ScalarMul(0.5, T), reg)

    def test_negative(self, reg, spacetime):
        T = Tensor("T", [spacetime.upper("μ")])
        self._assert_roundtrip(-T, reg)

    def test_partial_deriv(self, reg, spacetime):
        V = Tensor("V", [spacetime.upper("ν")])
        dV = PartialDeriv(V, spacetime.lower("μ"))
        self._assert_roundtrip(dV, reg)

    def test_covariant_deriv(self, reg, spacetime):
        V = Tensor("V", [spacetime.upper("ν")])
        conn = Connection("Γ", spacetime)
        conns = {spacetime.name: conn}
        nabla_V = CovariantDeriv(V, spacetime.lower("μ"), conns)
        self._assert_roundtrip(nabla_V, reg, connections=conns)

    def test_nested_partial(self, reg, spacetime):
        V = Tensor("V", [spacetime.upper("λ")])
        d2V = PartialDeriv(PartialDeriv(V, spacetime.lower("ν")), spacetime.lower("μ"))
        self._assert_roundtrip(d2V, reg)

    def test_partial_on_product(self, reg, spacetime):
        A = Tensor("A", [spacetime.upper("ν")])
        B = Tensor("B", [spacetime.lower("ν")])
        expr = PartialDeriv(A * B, spacetime.lower("μ"))
        self._assert_roundtrip(expr, reg)

    def test_complex_expression(self, reg, spacetime):
        """R^μ_ν - 1/2 g^μ_ν R 형태."""
        R_ud = Tensor("R", [spacetime.upper("μ"), spacetime.lower("ν")])
        g_ud = Tensor("g", [spacetime.upper("μ"), spacetime.lower("ν")])
        R_scalar = Tensor("R", [])
        expr = R_ud - ScalarMul(0.5, g_ud * R_scalar)
        self._assert_roundtrip(expr, reg)

    def test_connection_tensor(self, reg, spacetime):
        """Γ^μ_{νλ} roundtrip."""
        gamma = Tensor("Γ", [
            spacetime.upper("μ"),
            spacetime.lower("ν"),
            spacetime.lower("λ"),
        ])
        self._assert_roundtrip(gamma, reg)

    def test_two_space_tensor(self, reg2, spacetime, lorentz):
        """e^a_μ (vielbein) — 두 공간 인덱스."""
        e = Tensor("e", [lorentz.upper("a"), spacetime.lower("μ")])
        self._assert_roundtrip(e, reg2)


# ─── 8. Decorator parsing (\bar, \hat, \tilde) ─────────────

class TestDecoratorParsing:
    """Fix 1: \\bar{X}, \\hat{X} on tensor names and indices."""

    def test_bar_on_tensor_name(self, reg, spacetime):
        r"""\bar{e}^{\mu} → tensor name "ē"."""
        expr = parse(r"\bar{e}^{\mu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "e\u0304"
        assert expr.indices[0] == Index("μ", spacetime, "upper")

    def test_hat_on_tensor_name(self, reg, spacetime):
        r"""\hat{B}_{\mu\nu} → tensor name "B̂"."""
        expr = parse(r"\hat{B}_{\mu\nu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "B\u0302"
        assert len(expr.indices) == 2

    def test_tilde_on_tensor_name(self, reg, spacetime):
        expr = parse(r"\tilde{h}^{\mu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "h\u0303"

    def test_bar_roundtrip(self, reg, spacetime):
        r"""\bar{e} → to_latex → parse yields the same structure."""
        expr = parse(r"\bar{e}^{\mu}{}_{\nu}", reg)
        latex = to_latex(expr)
        reparsed = parse(latex, reg)
        assert to_latex(reparsed) == latex
        assert reparsed.name == expr.name

    def test_nested_decorators(self, reg):
        r"""\bar{\hat{e}} applies decorators inside-out."""
        expr = parse(r"\bar{\hat{e}}^{\mu}", reg)
        assert isinstance(expr, Tensor)
        # inner \hat → U+0302, outer \bar → U+0304
        assert expr.name == "e\u0302\u0304"

    def test_decorator_on_greek_base(self, reg):
        r"""\bar{\mu} as a tensor name: base is μ, decorated."""
        expr = parse(r"\bar{\mu}^{\nu}", reg)
        assert isinstance(expr, Tensor)
        assert expr.name == "μ\u0304"

    def test_decorated_index_alphabet(self):
        r"""IndexSpace with decorated index chars ("p̄q̄") registers cleanly."""
        lorbar = IndexSpace(
            "lorentz_bar", dim=4, indices="p\u0304q\u0304r\u0304s\u0304", metric="ηbar"
        )
        r = IndexRegistry()
        r.register(lorbar)
        # Each grapheme (p̄, q̄, ...) is a separate index character.
        assert r.resolve("p\u0304") is lorbar
        assert r.resolve("q\u0304") is lorbar

    def test_barred_index_via_latex(self):
        r"""\bar{e}^{\bar{p}}_{\mu} with a registered barred-Lorentz space."""
        st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
        lorbar = IndexSpace(
            "lorentz_bar", dim=4, indices="p\u0304q\u0304r\u0304s\u0304", metric="ηbar"
        )
        r = IndexRegistry()
        r.register(st)
        r.register(lorbar)
        expr = parse(r"\bar{e}^{\bar{p}}_{\mu}", r)
        assert isinstance(expr, Tensor)
        assert expr.name == "e\u0304"
        assert expr.indices[0].name == "p\u0304"
        assert expr.indices[0].space is lorbar
        assert expr.indices[1].name == "μ"
        assert expr.indices[1].space is st


# ─── 9. Einstein validation with deriv indices ─────────────

class TestDerivIndexValidation:
    """Fix 2: validate_einstein must see PartialDeriv / CovariantDeriv indices."""

    def test_nested_same_deriv_is_invalid(self, reg, spacetime):
        r"""∂_μ ∂_μ E — same lower index twice, must fail validation."""
        from indexcalc import validate_einstein
        E = Tensor("E", [])
        mu_lo = spacetime.lower("μ")
        bad = PartialDeriv(PartialDeriv(E, mu_lo), mu_lo)
        info = validate_einstein(bad)
        assert info["valid"] is False
        assert any("μ" in err for err in info["errors"])

    def test_divergence_is_valid(self, reg, spacetime):
        r"""∂_μ V^μ — legitimate divergence, deriv index contracts with V."""
        from indexcalc import validate_einstein
        V = Tensor("V", [spacetime.upper("μ")])
        divV = PartialDeriv(V, spacetime.lower("μ"))
        info = validate_einstein(divV)
        assert info["valid"] is True
        assert len(info["contracted"]) == 1
        assert info["contracted"][0][0].name == "μ"
        assert info["free"] == []

    def test_laplacian_via_metric_trace(self, reg, spacetime):
        r"""g^{μν} ∂_μ ∂_ν φ — two distinct deriv indices, metric-contracted."""
        from indexcalc import validate_einstein
        phi = Tensor("φ", [])
        g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])
        inner = PartialDeriv(PartialDeriv(phi, spacetime.lower("ν")), spacetime.lower("μ"))
        lap = g_inv * inner
        info = validate_einstein(lap)
        assert info["valid"] is True
        assert len(info["contracted"]) == 2
