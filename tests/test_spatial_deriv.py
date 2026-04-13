"""SpatialCovariantDeriv (D_i) 테스트 — B1.2, B1.3."""

import pytest
from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum
from indexcalc.core.deriv import (
    CovariantDeriv, Connection, LeviCivitaConnection, PartialDeriv,
    expand_covariant,
)
from indexcalc.core.spatial_deriv import (
    SpatialCovariantDeriv, spatial_covariant, expand_spatial_covariant,
)
from indexcalc.core.variation import (
    Variation, VariationRegistry, expand_variation, ZeroTensor,
)
from indexcalc.parse.latex import IndexRegistry, parse
from indexcalc.parse.display import to_latex


@pytest.fixture
def sp_space():
    return IndexSpace("spatial", dim=3, indices="ijklmn", metric="Ω")


@pytest.fixture
def spatial_conn(sp_space):
    Om = Tensor("Ω", [sp_space.lower("i"), sp_space.lower("j")])
    Oinv = Tensor("Ω", [sp_space.upper("i"), sp_space.upper("j")])
    return LeviCivitaConnection(Om, Oinv, sp_space)


@pytest.fixture
def reg(sp_space):
    r = IndexRegistry()
    r.register(sp_space)
    return r


# ─── B1.2: SpatialCovariantDeriv 기본 ────────────────────────

class TestSpatialCovariantDeriv:
    def test_subclass_of_covariant(self, sp_space, spatial_conn):
        V = Tensor("V", [sp_space.upper("i")])
        D = SpatialCovariantDeriv(V, sp_space.lower("j"), spatial_conn)
        assert isinstance(D, CovariantDeriv)
        assert isinstance(D, SpatialCovariantDeriv)

    def test_free_indices(self, sp_space, spatial_conn):
        V = Tensor("V", [sp_space.upper("i")])
        D = SpatialCovariantDeriv(V, sp_space.lower("j"), spatial_conn)
        names = [idx.name for idx in D.free_indices]
        assert names == ["j", "i"]

    def test_helper_upper_flips(self, sp_space, spatial_conn):
        """spatial_covariant()가 upper index를 lower로 flip."""
        V = Tensor("V", [sp_space.upper("i")])
        D = spatial_covariant(V, sp_space.upper("j"), spatial_conn)
        assert D.deriv_index.position == "lower"

    def test_display_uses_D(self, sp_space, spatial_conn):
        V = Tensor("V", [sp_space.upper("i")])
        D = SpatialCovariantDeriv(V, sp_space.lower("j"), spatial_conn)
        latex = to_latex(D)
        assert latex.startswith("D_{")
        assert "\\nabla" not in latex

    def test_expansion_produces_partial_plus_gamma(self, sp_space, spatial_conn):
        """D_j V^i = ∂_j V^i + Γ^i_{j,dummy} V^dummy"""
        V = Tensor("V", [sp_space.upper("i")])
        D = SpatialCovariantDeriv(V, sp_space.lower("j"), spatial_conn)
        expanded = expand_spatial_covariant(D)
        latex = to_latex(expanded)
        assert "\\partial_{j}" in latex
        assert "\\Gamma" in latex


# ─── B1.2: Parser + display roundtrip ───────────────────────

class TestParser:
    def test_parse_D_scalar(self, reg, sp_space):
        """D_{i} φ 파싱"""
        expr = parse(r"D_{i} \phi", reg)
        assert isinstance(expr, SpatialCovariantDeriv)
        assert expr.deriv_index.name == "i"

    def test_parse_D_vector(self, reg, sp_space):
        """D_{i} V^{j} 파싱"""
        expr = parse(r"D_{i} V^{j}", reg)
        assert isinstance(expr, SpatialCovariantDeriv)
        assert isinstance(expr.expr, Tensor)
        assert expr.expr.name == "V"

    def test_nested_D_D(self, reg):
        """D_{i} D_{j} φ (2차 미분)"""
        expr = parse(r"D_{i} D_{j} \phi", reg)
        assert isinstance(expr, SpatialCovariantDeriv)
        assert isinstance(expr.expr, SpatialCovariantDeriv)

    def test_roundtrip(self, reg):
        expr = parse(r"D_{i} V^{j}", reg)
        assert to_latex(expr) == r"D_{i} V^{j}"

    def test_D_not_tensor_when_followed_by_lower(self, reg):
        """D_{i} 뒤에 atom이 오면 연산자로 해석 (tensor 이름 D 사용 불가)."""
        expr = parse(r"D_{i} \phi", reg)
        assert not isinstance(expr, Tensor)
        assert isinstance(expr, SpatialCovariantDeriv)


# ─── B1.2: Leibniz via type-preserving expansion ─────────────

class TestLeibnizPreservesType:
    def test_product_rule(self, sp_space, spatial_conn):
        """D_i(φ·ψ) → (D_iφ)·ψ + φ·(D_iψ). 중간 노드들도 SpatialCovariantDeriv."""
        phi = Tensor("φ", [])
        psi = Tensor("ψ", [])
        D = SpatialCovariantDeriv(
            TensorProduct(phi, psi), sp_space.lower("i"), spatial_conn,
        )
        # 수동으로 Leibniz 확인 — _distribute_nabla_once 검증
        from indexcalc.core.variation import _distribute_nabla_once
        distributed = _distribute_nabla_once(D)
        # 전개 결과의 CovariantDeriv들이 모두 SpatialCovariantDeriv인지
        from indexcalc.core.tensor import TensorSum as _S, TensorProduct as _P
        assert isinstance(distributed, _S)
        left = distributed.left
        right = distributed.right
        assert isinstance(left, _P) and isinstance(left.left, SpatialCovariantDeriv)
        assert isinstance(right, _P) and isinstance(right.right, SpatialCovariantDeriv)


# ─── B1.3: Variation + D_i ──────────────────────────────────

class TestVariationWithD:
    def test_delta_of_D_background_connection(self, sp_space, spatial_conn):
        """γ가 background: δ(D_i φ) = D_i(δφ)"""
        vreg = VariationRegistry()
        vreg.declare_varying("φ")
        phi = Tensor("φ", [])
        expr = Variation(SpatialCovariantDeriv(phi, sp_space.lower("i"), spatial_conn))
        result = expand_variation(expr, vreg)
        assert isinstance(result, SpatialCovariantDeriv)
        assert result.expr.name == "δφ"

    def test_delta_of_D_varying_connection(self, sp_space, spatial_conn):
        """γ가 varying: δ(D_i V^j) = D_i(δV^j) + δΓ^j_{ik} V^k"""
        vreg = VariationRegistry()
        vreg.declare_varying("V")
        vreg.declare_varying_connection("Γ")
        V = Tensor("V", [sp_space.upper("j")])
        expr = Variation(SpatialCovariantDeriv(V, sp_space.lower("i"), spatial_conn))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        assert "D_{i}" in latex
        assert "δV" in latex
        assert "δΓ" in latex

    def test_delta_of_D_product_leibniz(self, sp_space, spatial_conn):
        """δ(D_i (φ·ψ)) — 복합 내부 Leibniz 분배 후 δ 적용"""
        vreg = VariationRegistry()
        vreg.declare_varying("φ")
        vreg.declare_varying("ψ")
        phi = Tensor("φ", [])
        psi = Tensor("ψ", [])
        expr = Variation(SpatialCovariantDeriv(
            TensorProduct(phi, psi), sp_space.lower("i"), spatial_conn,
        ))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        # D_i(δφ)·ψ, φ·D_i(δψ), D_i(φ)·δψ, δφ·D_i(ψ) 조합이 나타나야 함
        assert "δφ" in latex
        assert "δψ" in latex
        assert "D_{i}" in latex

    def test_background_tensor_varying_connection(self, sp_space, spatial_conn):
        """δT=0 여도 δγ·T 항만 남아야 한다."""
        vreg = VariationRegistry()
        vreg.declare_background("V")
        vreg.declare_varying_connection("Γ")
        V = Tensor("V", [sp_space.upper("j")])
        expr = Variation(SpatialCovariantDeriv(V, sp_space.lower("i"), spatial_conn))
        result = expand_variation(expr, vreg)
        latex = to_latex(result)
        assert "δΓ" in latex
        # D_i(δV)=D_i(0)=0 이므로 covariant 본체 없어야 함
        assert "D_{i} δV" not in latex
