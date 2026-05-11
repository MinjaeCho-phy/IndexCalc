"""Frontend 2/3 회귀:
    - parse_line_element: ds² → SymPy matrix.
    - parse_curvature_components: SymbolicCurvatureResult.latex() → dict.

Round-trip: SymbolicCurvatureResult.latex(t) → parse_curvature_components → 원본 array와 일치.
"""

import pytest
import sympy as sp

from indexcalc import (
    Metric,
    parse_line_element, parse_curvature_components,
)


# ─── Frontend 3: parse_line_element ───────────────────────


class TestLineElement:
    def test_flat_2d(self):
        g = parse_line_element("dx**2 + dy**2", ["x", "y"])
        assert g == sp.Matrix([[1, 0], [0, 1]])

    def test_flat_3d(self):
        g = parse_line_element("dx**2 + dy**2 + dz**2", ["x", "y", "z"])
        assert g == sp.eye(3)

    def test_polar_2d(self):
        g = parse_line_element("dr**2 + r**2*dtheta**2", ["r", "theta"])
        r = sp.Symbol("r", real=True)
        assert g == sp.Matrix([[1, 0], [0, r**2]])

    def test_minkowski(self):
        g = parse_line_element(
            "-dt**2 + dx**2 + dy**2 + dz**2", ["t", "x", "y", "z"],
        )
        expected = sp.diag(-1, 1, 1, 1)
        assert g == expected

    def test_flrw(self):
        g = parse_line_element(
            "-dt**2 + a**2*(dx**2 + dy**2 + dz**2)", ["t", "x", "y", "z"],
        )
        # 'a'는 coords 밖 → 파서가 생성한 Symbol vs 외부 Symbol 비교는
        # attribute 차이로 == 가 False일 수 있어, simplify-difference로 검증.
        a = sp.Symbol("a")
        expected = sp.diag(-1, a**2, a**2, a**2)
        diff = g - expected
        assert sp.simplify(diff) == sp.zeros(4, 4)

    def test_schwarzschild(self):
        g = parse_line_element(
            "-(1 - 2/r)*dt**2 + dr**2/(1 - 2/r) + r**2*dtheta**2 + r**2*sin(theta)**2*dphi**2",
            ["t", "r", "theta", "phi"],
        )
        r, theta = sp.symbols("r theta", real=True)
        # g_tt = -(1-2/r), g_rr = 1/(1-2/r), g_θθ = r^2, g_φφ = r^2 sin²θ
        assert sp.simplify(g[0, 0] - (-(1 - 2/r))) == 0
        assert sp.simplify(g[1, 1] - 1/(1 - 2/r)) == 0
        assert sp.simplify(g[2, 2] - r**2) == 0
        assert sp.simplify(g[3, 3] - r**2 * sp.sin(theta)**2) == 0

    def test_cross_term_factor_two(self):
        """2 dt dx → g_{tx} = 1 (textbook 컨벤션 검증)."""
        g = parse_line_element("dt**2 + 2*dt*dx + dx**2", ["t", "x"])
        assert g == sp.Matrix([[1, 1], [1, 1]])

    def test_caret_exponent(self):
        """``^`` → ``**`` 자동 변환."""
        g = parse_line_element("dx^2 + dy^2", ["x", "y"])
        assert g == sp.eye(2)

    def test_unrecognized_differential_raises(self):
        with pytest.raises(ValueError, match="unrecognized"):
            # dz는 coords에 없음
            parse_line_element("dx**2 + dz**2", ["x", "y"])

    def test_empty_coords_raises(self):
        with pytest.raises(ValueError, match="coords"):
            parse_line_element("dx**2", [])


# ─── Frontend 2: parse_curvature_components round-trip ───


class TestCurvatureRoundTrip:
    def _polar_result(self):
        r, theta = sp.symbols("r theta", real=True)
        metric = sp.Matrix([[1, 0], [0, r**2]])
        return Metric(metric, ["r", "theta"]).symbolic([r, theta])

    def test_christoffel_round_trip(self):
        result = self._polar_result()
        latex = result.latex("christoffel")
        parsed = parse_curvature_components(latex, ["r", "theta"], tensor="christoffel")
        # parser는 새 Symbol("r", real=True) 생성 — outer test의 r과 attribute는
        # 같지만 객체 동일성은 다를 수 있어 simplify-difference로 비교.
        r_sym = sp.Symbol("r", real=True)
        assert sp.simplify(parsed[(0, 1, 1)] + r_sym) == 0
        # Γ^θ_{rθ} = 1/r — slot (1, 0, 1)
        assert sp.simplify(parsed[(1, 0, 1)] - 1/r_sym) == 0

    def test_ricci_scalar_round_trip(self):
        result = self._polar_result()
        latex = result.latex("ricci_scalar")
        parsed = parse_curvature_components(latex, ["r", "theta"], tensor="ricci_scalar")
        assert sp.simplify(parsed) == 0

    def test_metric_round_trip(self):
        result = self._polar_result()
        latex = result.latex("metric")
        parsed = parse_curvature_components(latex, ["r", "theta"], tensor="metric")
        r_sym = sp.Symbol("r", real=True)
        assert parsed[(0, 0)] == 1
        assert sp.simplify(parsed[(1, 1)] - r_sym**2) == 0

    def test_ricci_tensor_zero(self):
        result = self._polar_result()
        latex = result.latex("ricci")
        parsed = parse_curvature_components(latex, ["r", "theta"], tensor="ricci")
        # 2D flat polar → Ricci tensor = 0
        # latex output가 "(all components zero)"이면 parsed는 빈 dict
        assert parsed == {}

    def test_unknown_tensor_raises(self):
        with pytest.raises(ValueError, match="unknown tensor"):
            parse_curvature_components("R = 0", ["r"], tensor="not_a_tensor")
