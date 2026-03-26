"""
Phase 6e 데모: Functional Components + JAX Autodiff.

텐서 component를 좌표 함수로 제공하면, ∂_μ V^ν를 jax.jacfwd로 자동 계산.
"""

import jax
import jax.numpy as jnp
from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, to_latex, evaluate,
    partial, expand_partial,
    Connection, LeviCivitaConnection,
    covariant, expand_covariant,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")

# ============================================================
# 1. 기본: V(x) 함수 → 배열로 평가
# ============================================================
print("=== 1. Functional component: V(x) ===\n")

def V_func(x):
    """V^μ = (x^0 squared, sin(x^1), 0, 0)"""
    return jnp.array([x[0]**2, jnp.sin(x[1]), 0., 0.])

V_t = Tensor("V", [spacetime.upper("μ")])
x0 = jnp.array([2., jnp.pi/6, 0., 0.])

result1 = evaluate(V_t, {"V": V_func}, backend="jax", coords=x0)
print(f"  x = {x0}")
print(f"  V(x) = {result1}")
print(f"  기대: [4, 0.5, 0, 0]")
print()

# ============================================================
# 2. Autodiff: ∂_μ V^ν — "∂V" 키 없이 자동 계산
# ============================================================
print("=== 2. Autodiff: ∂_μ V^ν ===\n")

dV = partial(V_t, spacetime.lower("ν"))
print(f"  {to_latex(dV)}")

result2 = evaluate(dV, {"V": V_func}, backend="jax", coords=x0)
print(f"  ∂_ν V^μ 자동미분 결과:")
print(f"  {result2}")

# 수동 검증: ∂_0 V^0 = 2*x^0 = 4, ∂_1 V^1 = cos(x^1) = cos(π/6) ≈ 0.866
print(f"\n  기대: ∂_0 V^0 = {2*x0[0]:.3f}, ∂_1 V^1 = {jnp.cos(x0[1]):.3f}")
print(f"  결과: ∂_0 V^0 = {result2[0,0]:.3f}, ∂_1 V^1 = {result2[1,1]:.3f}")
print()

# ============================================================
# 3. Metric 함수 + index lowering
# ============================================================
print("=== 3. Functional metric: η_{ab} V^b(x) ===\n")

def eta_func(x):
    """상수 Minkowski metric — 좌표에 무관."""
    return jnp.diag(jnp.array([-1., 1., 1., 1.]))

eta_T = Tensor("η", [spacetime.lower("μ"), spacetime.lower("ν")])
V_up  = Tensor("V", [spacetime.upper("ν")])
expr3 = eta_T * V_up

result3 = evaluate(expr3, {"η": eta_func, "V": V_func}, backend="jax", coords=x0)
print(f"  η_{{μν}} V^ν(x) = {result3}")
print(f"  기대: [-4, 0.5, 0, 0]")
print()

# ============================================================
# 4. 전개된 공변미분 + autodiff
# ============================================================
print("=== 4. ∇_μ V^ν = ∂_μ V^ν + Γ^ν_{μρ} V^ρ (autodiff) ===\n")

g     = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])
christoffel = LeviCivitaConnection(g, g_inv, spacetime)

V_cov = Tensor("V", [spacetime.upper("ν")])
mu = spacetime.lower("μ")
nabla_V = covariant(V_cov, mu, christoffel)
expanded = expand_covariant(nabla_V)

print(f"  {to_latex(expanded)}")

# 평탄 시공간: Γ=0
Gamma_zero = jnp.zeros((4, 4, 4))

result4 = evaluate(expanded, {"V": V_func, "Γ": Gamma_zero},
                   backend="jax", coords=x0)
print(f"\n  Flat spacetime (Γ=0):")
print(f"  ∇_μ V^ν = \n{result4}")
print(f"  (∂_μ V^ν와 같아야 함)")
print()

# ============================================================
# 5. 2차 미분: ∂_μ ∂_ν f (scalar 함수)
# ============================================================
print("=== 5. 2차 미분: ∂_μ ∂_ν f ===\n")

def f_func(x):
    """f = x^0 * x^1 + (x^1)^3"""
    return x[0] * x[1] + x[1]**3

f_T = Tensor("f", [])  # scalar — 인덱스 없음
df = partial(f_T, spacetime.lower("μ"))
ddf = partial(df, spacetime.lower("ν"))

result5 = evaluate(ddf, {"f": f_func}, backend="jax", coords=x0)
print(f"  f(x) = x^0 * x^1 + (x^1)^3")
print(f"  x = {x0}")
print(f"  ∂_μ ∂_ν f = \n{result5}")

# 수동 검증:
# ∂_0 f = x^1, ∂_1 f = x^0 + 3(x^1)^2
# ∂_0 ∂_0 f = 0, ∂_0 ∂_1 f = 1
# ∂_1 ∂_0 f = 1, ∂_1 ∂_1 f = 6*x^1
x1 = float(x0[1])
print(f"\n  기대: ∂_0∂_0 = 0, ∂_0∂_1 = 1, ∂_1∂_0 = 1, ∂_1∂_1 = {6*x1:.3f}")
print()

# ============================================================
# 6. 혼합: 배열 + 함수
# ============================================================
print("=== 6. 혼합: 배열(Γ) + 함수(V) ===\n")

# 일부는 배열, 일부는 함수
Gamma_nonzero = jnp.zeros((4, 4, 4)).at[0, 1, 0].set(0.5)

result6 = evaluate(expanded, {"V": V_func, "Γ": Gamma_nonzero},
                   backend="jax", coords=x0)
print(f"  Γ^0_{{10}} = 0.5, 나머지 0")
print(f"  V(x) = {V_func(x0)}")
print(f"  ∇_μ V^ν = \n{result6}")
print(f"\n  (1,0) 성분 = {result6[1,0]:.3f}")
print(f"  기대: ∂_1 V^0 + Γ^0_{{10}} V^0 = 0 + 0.5*4 = {0.5*float(x0[0]**2):.3f}")

print("\n=== Phase 6e 완료! ===")
