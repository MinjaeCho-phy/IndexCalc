"""
Phase 6f Symbolic Demo: Symbolic Curvature Computation.

SymPy Matrix로 metric을 정의하고, Christoffel/Riemann/Ricci를
symbolic 수식으로 계산한다.

테스트 시나리오:
  1. FLRW 우주론 — R = 12H²
  2. 2-Sphere — R = 2/r₀²
  3. Schwarzschild — R = 0, K = 48M²/r⁶
  4. show() 출력 확인
  5. Coordinates API + SymPy Metric 연동
"""

import sys
sys.path.insert(0, "/home/minjae/Minjae/IndexCalc")

import sympy as sp
from indexcalc import Metric

print("=" * 60)
print("Phase 6f Symbolic: Curvature from SymPy Metric")
print("=" * 60)


# ─── Test 1: FLRW 우주론 ─────────────────────────────────────

print("\n[Test 1] FLRW: ds² = -dt² + e^{2Ht}(dx² + dy² + dz²)")
print("-" * 50)

t, x, y, z = sp.symbols('t x y z')
H = sp.Symbol('H', positive=True)

g_flrw = sp.diag(-1, sp.exp(2*H*t), sp.exp(2*H*t), sp.exp(2*H*t))
metric = Metric(g_flrw, "cartesian", signature=(3, 1))
print(f"  {metric}")

sym = metric.symbolic([t, x, y, z])

R_expected = 12 * H**2
print(f"  R = {sym.R}")
print(f"  expected: {R_expected}")
assert sp.simplify(sym.R - R_expected) == 0, f"R mismatch: {sym.R}"

print("\n  Christoffel (비영):")
sym.show("christoffel")

print("\n  Ricci tensor:")
sym.show("ricci")

print(f"\n  summary: {sym.summary()}")
print("  ✓ R = 12H² 확인")


# ─── Test 2: 2-Sphere ────────────────────────────────────────

print(f"\n[Test 2] 2-sphere: ds² = r₀²(dθ² + sin²θ dφ²)")
print("-" * 50)

theta, phi = sp.symbols('theta phi')
r0 = sp.Symbol('r_0', positive=True)

g_s2 = sp.diag(r0**2, r0**2 * sp.sin(theta)**2)
metric_s2 = Metric(g_s2, "spherical", dim=2)

sym_s2 = metric_s2.symbolic([theta, phi])

R_s2_expected = 2 / r0**2
print(f"  R = {sym_s2.R}")
print(f"  expected: {R_s2_expected}")
assert sp.simplify(sym_s2.R - R_s2_expected) == 0

print("\n  Christoffel:")
sym_s2.show("christoffel")

# Γ^θ_{φφ} = -sinθ cosθ
gamma_expected = -sp.sin(theta) * sp.cos(theta)
assert sp.simplify(sym_s2.Γ[0, 1, 1] - gamma_expected) == 0
print(f"\n  Γ^θ_{{φφ}} = {sym_s2.Γ[0, 1, 1]}  (해석: {gamma_expected})")

print("  ✓ R = 2/r₀², Christoffel 일치")


# ─── Test 3: Schwarzschild ────────────────────────────────────

print(f"\n[Test 3] Schwarzschild: vacuum R = 0")
print("-" * 50)

r = sp.Symbol('r', positive=True)
M = sp.Symbol('M', positive=True)
rs = 2 * M

f = 1 - rs / r
g_sch = sp.diag(-f, 1/f, r**2, r**2 * sp.sin(theta)**2)
metric_sch = Metric(g_sch, "spherical", signature=(3, 1))

sym_sch = metric_sch.symbolic([t, r, theta, phi])

print(f"  R = {sym_sch.R}")
assert sp.simplify(sym_sch.R) == 0, f"R ≠ 0: {sym_sch.R}"

# Ricci tensor = 0 (vacuum)
for i in range(4):
    for j in range(4):
        assert sp.simplify(sym_sch.Ric[i, j]) == 0, \
            f"R_{{{i}{j}}} ≠ 0: {sym_sch.Ric[i, j]}"
print("  R_{μν} = 0 (모든 성분)")

# Kretschner scalar
K_expected = 48 * M**2 / r**6
K_diff = sp.simplify(sym_sch.K - K_expected)
print(f"  K = {sp.simplify(sym_sch.K)}")
print(f"  expected: {K_expected}")
assert K_diff == 0, f"K mismatch: diff = {K_diff}"

print("\n  Christoffel (비영):")
sym_sch.show("christoffel")

# 값 대입 검증
K_numeric = float(sym_sch.K.subs({M: 1, r: 5}))
K_analytic = 48 / 5**6
print(f"\n  K(M=1, r=5) = {K_numeric:.10f}  (해석: {K_analytic:.10f})")
assert abs(K_numeric - K_analytic) < 1e-12

print("  ✓ R = 0, K = 48M²/r⁶, subs() 검증 완료")


# ─── Test 4: show() 다양한 텐서 ──────────────────────────────

print(f"\n[Test 4] show() — FLRW Riemann 텐서")
print("-" * 50)
sym.show("riemann")


# ─── Test 5: de Sitter Einstein 방정식 ────────────────────────

print(f"\n[Test 5] de Sitter: G_{{μν}} + Λg_{{μν}} = 0")
print("-" * 50)

Lambda = sp.Symbol('Lambda', positive=True)
f_ds = 1 - Lambda * r**2 / 3
g_ds = sp.diag(-f_ds, 1/f_ds, r**2, r**2 * sp.sin(theta)**2)
metric_ds = Metric(g_ds, "spherical", signature=(3, 1))

sym_ds = metric_ds.symbolic([t, r, theta, phi])

print(f"  R = {sp.simplify(sym_ds.R)}")
R_ds_expected = 4 * Lambda
assert sp.simplify(sym_ds.R - R_ds_expected) == 0

# G_{μν} + Λ g_{μν} = 0
for i in range(4):
    for j in range(4):
        residual = sp.simplify(sym_ds.G[i, j] + Lambda * g_ds[i, j])
        assert residual == 0, f"G_{{{i}{j}}} + Λg_{{{i}{j}}} = {residual}"

print("  G_{μν} + Λg_{μν} = 0 ✓ (모든 성분)")


# ─── Summary ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 6f Symbolic: 모든 테스트 통과!")
print("=" * 60)
print(f"  • FLRW: R = 12H²")
print(f"  • 2-Sphere: R = 2/r₀², Γ^θ_{{φφ}} = -sinθcosθ")
print(f"  • Schwarzschild: R = 0, K = 48M²/r⁶")
print(f"  • de Sitter: G_{{μν}} = -Λg_{{μν}}")
print(f"  • subs()로 수치 대입 검증 완료")
