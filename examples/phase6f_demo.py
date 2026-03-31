"""
Phase 6f Demo: Curvature Tensor Auto-Derivation.

Metric 클래스로 metric 함수 + 좌표계를 정의하고,
JAX autodiff로 Christoffel → Riemann → Ricci → Einstein 텐서를 자동 계산한다.

테스트 시나리오:
  1. Minkowski 시공간 — 모든 곡률 = 0
  2. 2-sphere (S²) — R = 2/r², Christoffel 해석적 검증
  3. Schwarzschild 시공간 — R = 0 (진공), Kretschner = 48M²/r⁶
  4. de Sitter 시공간 — R = 4Λ, G_{μν} = -Λg_{μν}
  5. FLRW 우주론 — R = 12H² (exponential expansion)
  6. Metric.show() — 좌표 이름으로 비영 성분 출력
"""

import sys
sys.path.insert(0, "/home/minjae/Minjae/IndexCalc")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from indexcalc import Metric, Coordinates

print("=" * 60)
print("Phase 6f: Curvature Tensor Auto-Derivation")
print("=" * 60)


# ─── Test 1: Minkowski 시공간 ────────────────────────────────

print("\n[Test 1] Minkowski spacetime η_{μν} = diag(-1, 1, 1, 1)")
print("-" * 50)

def minkowski(x):
    return jnp.diag(jnp.array([-1., 1., 1., 1.]))

g_mink = Metric(minkowski, "cartesian", signature=(3, 1))
print(f"  좌표: {g_mink.coords}")
print(f"  signature: {g_mink.signature}")

result = g_mink.at(jnp.array([0., 1., 2., 3.]))

print(f"  |Γ| max    = {float(jnp.max(jnp.abs(result.Γ))):.2e}")
print(f"  |R^ρ_σμν| = {float(jnp.max(jnp.abs(result.riemann))):.2e}")
print(f"  R          = {float(result.R):.2e}")
print(f"  |G_{'{μν}'}| = {float(jnp.max(jnp.abs(result.G))):.2e}")
print(f"  K          = {float(result.K):.2e}")

assert jnp.allclose(result.Γ, 0., atol=1e-14), "Γ ≠ 0!"
assert jnp.allclose(result.riemann, 0., atol=1e-14), "Riemann ≠ 0!"
assert abs(float(result.R)) < 1e-14, "R ≠ 0!"
print("  ✓ 모든 곡률 텐서 = 0 (평탄 시공간)")


# ─── Test 2: 2-Sphere (S²) ──────────────────────────────────

print(f"\n[Test 2] 2-sphere: ds² = r²(dθ² + sin²θ dφ²)")
print("-" * 50)

r0 = 3.0

def sphere_metric(x):
    theta = x[0]
    return r0**2 * jnp.array([
        [1., 0.],
        [0., jnp.sin(theta)**2],
    ])

g_s2 = Metric(sphere_metric, "spherical", dim=2)
print(f"  좌표: {g_s2.coords}")

theta0, phi0 = jnp.pi / 3, 0.5
result_s2 = g_s2.at(jnp.array([theta0, phi0]))

# 해석적 Christoffel
gamma_analytic = np.zeros((2, 2, 2))
gamma_analytic[0, 1, 1] = -np.sin(theta0) * np.cos(theta0)  # Γ^θ_{φφ}
gamma_analytic[1, 0, 1] = np.cos(theta0) / np.sin(theta0)    # Γ^φ_{θφ}
gamma_analytic[1, 1, 0] = np.cos(theta0) / np.sin(theta0)    # Γ^φ_{φθ}

print(f"  θ = π/3, φ = 0.5, r = {r0}")
print(f"  Γ^θ_{{φφ}} = {float(result_s2.Γ[0, 1, 1]):.6f}"
      f"  (해석: {gamma_analytic[0, 1, 1]:.6f})")
print(f"  Γ^φ_{{θφ}} = {float(result_s2.Γ[1, 0, 1]):.6f}"
      f"  (해석: {gamma_analytic[1, 0, 1]:.6f})")

assert jnp.allclose(result_s2.Γ, gamma_analytic, atol=1e-10), "Christoffel mismatch!"

R_analytic = 2.0 / r0**2
print(f"  R = {float(result_s2.R):.8f}  (해석: {R_analytic:.8f})")
assert abs(float(result_s2.R) - R_analytic) < 1e-10

K_gauss = float(result_s2.R) / 2
print(f"  Gaussian curvature K = {K_gauss:.8f}  (해석: {1/r0**2:.8f})")

print("\n  비영 Christoffel 성분:")
result_s2.show("christoffel")
print("  ✓ Christoffel, Ricci scalar 해석적 결과와 일치")


# ─── Test 3: Schwarzschild 시공간 ────────────────────────────

print(f"\n[Test 3] Schwarzschild: ds² = -(1-rₛ/r)dt² + dr²/(1-rₛ/r) + r²dΩ²")
print("-" * 50)

M = 1.0
rs = 2 * M

def schwarzschild(x):
    r, theta = x[1], x[2]
    f = 1 - rs / r
    return jnp.diag(jnp.array([-f, 1 / f, r**2, r**2 * jnp.sin(theta)**2]))

g_sch = Metric(schwarzschild, "spherical", signature=(3, 1))
print(f"  좌표: {g_sch.coords}")

r_eval = 5.0
coords_sch = jnp.array([0., r_eval, jnp.pi / 2, 0.])
result_sch = g_sch.at(coords_sch)

print(f"  r = {r_eval}, θ = π/2")
print(f"  |R_{'{μν}'}| max = {float(jnp.max(jnp.abs(result_sch.Ric))):.2e}")
print(f"  R = {float(result_sch.R):.2e}")
print(f"  |G_{'{μν}'}| max = {float(jnp.max(jnp.abs(result_sch.G))):.2e}")

assert jnp.allclose(result_sch.Ric, 0., atol=1e-8), "Ricci ≠ 0 for vacuum!"
assert abs(float(result_sch.R)) < 1e-8, "R ≠ 0 for vacuum!"

# Kretschner: K = 48M²/r⁶
K_analytic = 48 * M**2 / r_eval**6
print(f"  Kretschner K = {float(result_sch.K):.10f}")
print(f"         해석해: {K_analytic:.10f}")
assert abs(float(result_sch.K) - K_analytic) / K_analytic < 1e-6

# Γ^t_{tr} 확인
f_val = 1 - rs / r_eval
gamma_ttr_analytic = rs / (2 * r_eval**2 * f_val)
print(f"  Γ^t_{{tr}} = {float(result_sch.Γ[0, 0, 1]):.8f}"
      f"  (해석: {gamma_ttr_analytic:.8f})")
assert abs(float(result_sch.Γ[0, 0, 1]) - gamma_ttr_analytic) < 1e-8

print("\n  비영 Christoffel 성분:")
result_sch.show("christoffel")
print("  ✓ 진공해 R = 0, Kretschner 일치")


# ─── Test 4: de Sitter 시공간 ────────────────────────────────

print(f"\n[Test 4] de Sitter: ds² = -(1-Λr²/3)dt² + dr²/(1-Λr²/3) + r²dΩ²")
print("-" * 50)

Lambda = 0.3

def de_sitter(x):
    r, theta = x[1], x[2]
    f = 1 - Lambda * r**2 / 3
    return jnp.diag(jnp.array([-f, 1 / f, r**2, r**2 * jnp.sin(theta)**2]))

g_ds = Metric(de_sitter, "spherical", signature=(3, 1))
result_ds = g_ds.at(jnp.array([0., 1.0, jnp.pi / 2, 0.]))

R_ds_analytic = 4 * Lambda
print(f"  R = {float(result_ds.R):.8f}  (해석: {R_ds_analytic:.8f})")
assert abs(float(result_ds.R) - R_ds_analytic) < 1e-6

# G_{μν} + Λ g_{μν} ≈ 0
residual = result_ds.G + Lambda * result_ds.g
print(f"  |G_{'{μν}'} + Λg_{'{μν}'}| max = {float(jnp.max(jnp.abs(residual))):.2e}")
assert jnp.allclose(residual, 0., atol=1e-6)

print("  ✓ R = 4Λ, Einstein 방정식 G_{μν} = -Λg_{μν} 검증")


# ─── Test 5: FLRW 우주론 ─────────────────────────────────────

print(f"\n[Test 5] FLRW (flat): ds² = -dt² + a(t)²(dx² + dy² + dz²)")
print("-" * 50)

H = 0.5

def flrw_flat(x):
    t = x[0]
    a2 = jnp.exp(2 * H * t)
    return jnp.diag(jnp.array([-1., a2, a2, a2]))

g_flrw = Metric(flrw_flat, "cartesian", signature=(3, 1))
t_eval = 1.0
result_flrw = g_flrw.at(jnp.array([t_eval, 0., 0., 0.]))

R_flrw_analytic = 12 * H**2
print(f"  H = {H}, t = {t_eval}")
print(f"  R = {float(result_flrw.R):.8f}  (해석: {R_flrw_analytic:.8f})")
assert abs(float(result_flrw.R) - R_flrw_analytic) < 1e-6

a2_val = np.exp(2 * H * t_eval)
gamma_txx = float(result_flrw.Γ[0, 1, 1])
gamma_txx_analytic = a2_val * H
print(f"  Γ^t_{{xx}} = {gamma_txx:.8f}  (해석: {gamma_txx_analytic:.8f})")
assert abs(gamma_txx - gamma_txx_analytic) < 1e-8

gamma_xtx = float(result_flrw.Γ[1, 0, 1])
print(f"  Γ^x_{{tx}} = {gamma_xtx:.8f}  (해석: {H:.8f})")
assert abs(gamma_xtx - H) < 1e-8

print("  ✓ FLRW R = 12H², Christoffel 일치")


# ─── Test 6: show() + Coordinates API ────────────────────────

print(f"\n[Test 6] Metric API / Coordinates 확인")
print("-" * 50)

# Custom 좌표
g_custom = Metric(minkowski, ["t", "x", "y", "z"], signature=(3, 1))
print(f"  Custom: {g_custom}")

# Preset
g_preset = Metric(minkowski, "cartesian", signature=(3, 1))
print(f"  Preset: {g_preset}")

# Riemannian (signature 생략)
g_riem = Metric(sphere_metric, ["θ", "φ"])
print(f"  Riemannian: {g_riem}")
print(f"  Riemannian signature: {g_riem.signature}")
assert g_riem.signature == (1, 1), "Riemannian signature should be all +"

# Preset dim only
coords_sph3 = Coordinates.preset("spherical", dim=3)
print(f"  Spherical dim=3: {coords_sph3}")
assert coords_sph3.names == ("r", "θ", "φ")

# Preset with signature
coords_sph4 = Coordinates.preset("spherical", signature=(3, 1))
print(f"  Spherical sig=(3,1): {coords_sph4}")
assert coords_sph4.names == ("t", "r", "θ", "φ")

# Explicit signature
g_explicit = Metric(minkowski, ["t", "x", "y", "z"], signature=(-1, 1, 1, 1))
print(f"  Explicit sig: {g_explicit}")

print("  ✓ Metric/Coordinates API 정상 동작")


# ─── Test 7: show() 출력 ─────────────────────────────────────

print(f"\n[Test 7] show() — Schwarzschild 비영 Riemann 성분")
print("-" * 50)
result_sch.show("riemann")

print(f"\n  summary: {result_sch.summary()}")


# ─── Summary ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Phase 6f: 모든 테스트 통과!")
print("=" * 60)
print(f"  • Minkowski: 평탄 시공간 (R = 0)")
print(f"  • 2-Sphere:  Gaussian curvature K = 1/r²")
print(f"  • Schwarzschild: 진공 R = 0, Kretschner = 48M²/r⁶")
print(f"  • de Sitter: R = 4Λ, G_{{μν}} = -Λg_{{μν}}")
print(f"  • FLRW: R = 12H² (exponential expansion)")
print(f"  • Metric API: custom/preset/show() 정상 동작")
