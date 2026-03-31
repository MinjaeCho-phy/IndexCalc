"""
Coordinate Transform Demo.

CoordinateTransform으로 좌표 변환 정의, Jacobian 자동 계산,
metric 변환, 일반 텐서 변환을 테스트한다.

테스트 시나리오:
  1. Spherical → Cartesian (3D flat metric) — symbolic
  2. Jacobian 해석적 검증
  3. Metric 변환: ds²=dr²+r²dΩ² → ds²=dx²+dy²+dz²
  4. Vector 변환: e_r → (sinθcosφ, sinθsinφ, cosθ)
  5. (0,2) tensor 변환
  6. Numeric 변환 (JAX)
"""

import sys
sys.path.insert(0, "/home/minjae/Minjae/IndexCalc")

import sympy as sp
from indexcalc import CoordinateTransform, Metric, Coordinates

print("=" * 60)
print("Coordinate Transform Demo")
print("=" * 60)


# ─── 공통 심볼 ────────────────────────────────────────────────

r, θ, φ = sp.symbols('r θ φ', positive=True)
x, y, z = sp.symbols('x y z')


# ─── Test 1: Spherical → Cartesian 변환 정의 ─────────────────

print("\n[Test 1] Spherical → Cartesian 변환 정의")
print("-" * 50)

T = CoordinateTransform(
    source=["r", "θ", "φ"],
    target=["x", "y", "z"],
    forward=[
        r * sp.sin(θ) * sp.cos(φ),
        r * sp.sin(θ) * sp.sin(φ),
        r * sp.cos(θ),
    ],
    source_symbols=[r, θ, φ],
)

print(f"  {T}")
print(f"  source: {T.source}")
print(f"  target: {T.target}")
print(f"  dim: {T.dim}")


# ─── Test 2: Jacobian 검증 ───────────────────────────────────

print(f"\n[Test 2] Jacobian ∂(x,y,z)/∂(r,θ,φ)")
print("-" * 50)

J = T.jacobian()
print(f"  J =")
sp.pprint(J)

# 해석적 검증: J[0,0] = ∂x/∂r = sinθcosφ
assert sp.simplify(J[0, 0] - sp.sin(θ) * sp.cos(φ)) == 0
# J[2,0] = ∂z/∂r = cosθ
assert sp.simplify(J[2, 0] - sp.cos(θ)) == 0
# J[0,1] = ∂x/∂θ = r·cosθ·cosφ
assert sp.simplify(J[0, 1] - r * sp.cos(θ) * sp.cos(φ)) == 0

print("  ✓ Jacobian 해석적 결과 일치")

# Jacobian 역행렬
J_inv = T.jacobian_inv()
print(f"\n  J^{{-1}} =")
sp.pprint(sp.simplify(J_inv))

# J · J^{-1} = I
identity_check = sp.simplify(J * J_inv)
assert identity_check == sp.eye(3), f"J·J^{{-1}} ≠ I: {identity_check}"
print("  ✓ J · J⁻¹ = I 검증")


# ─── Test 3: Metric 변환 ─────────────────────────────────────

print(f"\n[Test 3] Flat metric: spherical → cartesian")
print("-" * 50)

# Spherical flat metric: ds² = dr² + r²dθ² + r²sin²θ dφ²
g_sph = sp.diag(1, r**2, r**2 * sp.sin(θ)**2)
metric_sph = Metric(g_sph, ["r", "θ", "φ"])

print(f"  g_sph =")
sp.pprint(g_sph)

# 변환
metric_cart = T.transform_metric(metric_sph)

print(f"\n  g_cart =")
g_cart_matrix = sp.simplify(metric_cart._sympy_metric)
sp.pprint(g_cart_matrix)

# 기대: 단위행렬 (flat space in Cartesian)
assert g_cart_matrix == sp.eye(3), f"g_cart ≠ I: {g_cart_matrix}"
print("  ✓ g_cart = I (평탄 공간 단위행렬)")


# ─── Test 4: Vector 변환 ─────────────────────────────────────

print(f"\n[Test 4] Vector 변환: e_r → Cartesian")
print("-" * 50)

# e_r = (1, 0, 0) in spherical
e_r = sp.Matrix([1, 0, 0])
e_r_cart = T.transform_components(e_r, rank=(1, 0))

print(f"  e_r (spherical) = {e_r.T}")
print(f"  e_r (cartesian) = {sp.simplify(e_r_cart).T}")

# 기대: (sinθcosφ, sinθsinφ, cosθ)
expected = sp.Matrix([sp.sin(θ)*sp.cos(φ), sp.sin(θ)*sp.sin(φ), sp.cos(θ)])
assert sp.simplify(e_r_cart - expected) == sp.zeros(3, 1)
print("  ✓ e_r → (sinθcosφ, sinθsinφ, cosθ)")

# e_θ = (0, 1, 0) in spherical
e_theta = sp.Matrix([0, 1, 0])
e_theta_cart = T.transform_components(e_theta, rank=(1, 0))
print(f"  e_θ (cartesian) = {sp.simplify(e_theta_cart).T}")


# ─── Test 5: Covector 변환 ───────────────────────────────────

print(f"\n[Test 5] Covector (1-form) 변환")
print("-" * 50)

# dr (covector) = (1, 0, 0) in spherical
dr = sp.Matrix([1, 0, 0])
dr_cart = T.transform_components(dr, rank=(0, 1))
print(f"  dr (cartesian) = {sp.simplify(dr_cart).T}")

# 기대: dr = sinθcosφ dx + sinθsinφ dy + cosθ dz
# 즉 (sinθcosφ, sinθsinφ, cosθ) — J_inv^T · dr
expected_dr = sp.Matrix([sp.sin(θ)*sp.cos(φ), sp.sin(θ)*sp.sin(φ), sp.cos(θ)])
assert sp.simplify(dr_cart - expected_dr) == sp.zeros(3, 1)
print("  ✓ dr → (sinθcosφ, sinθsinφ, cosθ)")


# ─── Test 6: (0,2) tensor 변환 ───────────────────────────────

print(f"\n[Test 6] (0,2) tensor 변환 — metric itself")
print("-" * 50)

# metric은 (0,2) tensor → transform_components와 transform_metric 결과 일치
g_cart_via_comp = T.transform_components(g_sph, rank=(0, 2))
g_cart_via_comp = sp.simplify(sp.Matrix(g_cart_via_comp))
print(f"  transform_components(g, (0,2)) =")
sp.pprint(g_cart_via_comp)

assert g_cart_via_comp == sp.eye(3)
print("  ✓ transform_metric과 transform_components 결과 일치")


# ─── Test 7: LaTeX 출력 ──────────────────────────────────────

print(f"\n[Test 7] LaTeX 출력")
print("-" * 50)
print(f"  {T.latex()}")


# ─── Test 8: 2D Polar → Cartesian (numeric) ──────────────────

print(f"\n[Test 8] 2D Polar → Cartesian (numeric, JAX)")
print("-" * 50)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def polar_to_cart(x):
    r, theta = x[0], x[1]
    return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

def cart_to_polar(x):
    return jnp.array([jnp.sqrt(x[0]**2 + x[1]**2), jnp.arctan2(x[1], x[0])])

T_num = CoordinateTransform(
    source=["r", "θ"],
    target=["x", "y"],
    forward=polar_to_cart,
    inverse=cart_to_polar,
)
print(f"  {T_num}")

# Numeric Jacobian at r=2, θ=π/4
coords_polar = jnp.array([2.0, jnp.pi / 4])
J_num = T_num.jacobian(at=coords_polar)
print(f"  J at (r=2, θ=π/4):")
print(f"    {J_num}")

# 기대: J = [[cosθ, -rsinθ], [sinθ, rcosθ]]
import numpy as np
theta_val = np.pi / 4
J_expected = np.array([
    [np.cos(theta_val), -2 * np.sin(theta_val)],
    [np.sin(theta_val),  2 * np.cos(theta_val)],
])
assert np.allclose(J_num, J_expected, atol=1e-10)
print("  ✓ Numeric Jacobian 일치")

# Metric 변환: polar → cartesian
def polar_metric(x):
    r = x[0]
    return jnp.diag(jnp.array([1.0, r**2]))

metric_polar = Metric(polar_metric, ["r", "θ"])
metric_cart_num = T_num.transform_metric(metric_polar)

# 평가: (x,y) = (1, 1) → (r,θ) = (√2, π/4)
result = metric_cart_num.at(jnp.array([1.0, 1.0]))
print(f"  g_cart at (1,1):")
print(f"    {result.metric}")
assert jnp.allclose(result.metric, jnp.eye(2), atol=1e-10)
print("  ✓ 평탄 metric → 단위행렬")

# Curvature invariance: R=0 in both
print(f"  R (polar)    = {float(Metric(polar_metric, ['r','θ']).at(coords_polar).R):.2e}")
print(f"  R (cartesian) = {float(result.R):.2e}")
assert abs(float(result.R)) < 1e-8
print("  ✓ R = 0 (곡률 불변)")


# ─── Summary ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Coordinate Transform: 모든 테스트 통과!")
print("=" * 60)
print(f"  • Spherical → Cartesian: Jacobian 검증")
print(f"  • Metric 변환: ds²=dr²+r²dΩ² → ds²=dx²+dy²+dz²")
print(f"  • Vector 변환: e_r → (sinθcosφ, sinθsinφ, cosθ)")
print(f"  • Covector 변환: dr 검증")
print(f"  • (0,2) tensor 변환 = metric 변환 일치")
print(f"  • Numeric (JAX): polar → cartesian, R=0 불변")
