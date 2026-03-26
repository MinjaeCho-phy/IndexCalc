"""
Phase 6d 데모: Component Evaluation — TensorExpr → numeric array via einsum.
"""

import numpy as np
from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse, to_latex, evaluate,
    trace, Trace, expand_covariant, covariant, LeviCivitaConnection,
    partial, expand_partial,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

# Minkowski metric
eta = np.diag([-1., 1., 1., 1.])

# ============================================================
# 1. 기본: η_{μν} V^ν → V_μ (index lowering)
# ============================================================
print("=== 1. Index lowering: η_{μν} V^ν ===\n")

V = np.array([2., 1., 0., -1.])
eta_T = Tensor("η", [lorentz.lower("a"), lorentz.lower("b")])
V_T   = Tensor("V", [lorentz.upper("b")])

expr = eta_T * V_T  # η_{ab} V^b → contraction on b
result = evaluate(expr, {"η": eta, "V": V})
print(f"  V^a = {V}")
print(f"  η_{{ab}} V^b = {result}")
print(f"  (기대: [-2, 1, 0, -1])")
print()

# ============================================================
# 2. 행렬 곱: T^a_b S^b_c → (TS)^a_c
# ============================================================
print("=== 2. 행렬 곱: T^a_b S^b_c ===\n")

T_arr = np.array([[1, 2], [3, 4]], dtype=float)
S_arr = np.array([[5, 6], [7, 8]], dtype=float)

small = IndexSpace("small", dim=2, indices="ij", metric="δ")
T_t = Tensor("T", [small.upper("i"), small.lower("j")])
S_t = Tensor("S", [small.upper("j"), small.lower("k")])
# 주의: j는 T에서 lower, S에서 upper → contraction
# 하지만 여기서는 직접 만든 index가 아니라...
# 수정: small space에서 j가 lower(T) and upper(S) → contracts

# 올바른 구성: T^i_j * S^j_k
prod = T_t * S_t
result2 = evaluate(prod, {"T": T_arr, "S": S_arr})
print(f"  T = \n{T_arr}")
print(f"  S = \n{S_arr}")
print(f"  T^i_j S^j_k = \n{result2}")
print(f"  np.matmul 확인 = \n{T_arr @ S_arr}")
print()

# ============================================================
# 3. Trace: T^a_a → scalar
# ============================================================
print("=== 3. Trace: T^a_a ===\n")

T_trace = Tensor("T", [small.upper("i"), small.lower("i")])
tr = trace(T_trace, "i")
result3 = evaluate(tr, {"T": T_arr})
print(f"  T = \n{T_arr}")
print(f"  Tr(T) = {result3}")
print(f"  np.trace 확인 = {np.trace(T_arr)}")
print()

# ============================================================
# 4. ScalarMul & TensorSum
# ============================================================
print("=== 4. ScalarMul & TensorSum ===\n")

A_t = Tensor("A", [small.upper("i"), small.lower("j")])
B_t = Tensor("B", [small.upper("i"), small.lower("j")])

A_arr = np.eye(2)
B_arr = np.ones((2, 2))

# 2*A + B
expr4 = 2 * A_t + B_t
result4 = evaluate(expr4, {"A": A_arr, "B": B_arr})
print(f"  2*A + B = \n{result4}")
print(f"  확인 = \n{2*A_arr + B_arr}")
print()

# ============================================================
# 5. PartialDeriv: ∂_μ V^ν
# ============================================================
print("=== 5. PartialDeriv ===\n")

V_tensor = Tensor("V", [spacetime.upper("ν")])
dV = partial(V_tensor, spacetime.lower("μ"))

# ∂_μ V^ν 의 component: 4×4 배열
# axis 0 = μ (deriv), axis 1 = ν (tensor index)
dV_arr = np.random.randn(4, 4).round(2)

result5 = evaluate(dV, {"∂V": dV_arr})
print(f"  ∂_μ V^ν = (shape {result5.shape})")
print(f"  {result5}")
print()

# ============================================================
# 6. 전개된 공변미분: ∇_μ V^ν = ∂_μ V^ν + Γ^ν_{μρ} V^ρ
# ============================================================
print("=== 6. Expanded Covariant Derivative ===\n")

g     = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])

christoffel = LeviCivitaConnection(g, g_inv, spacetime)

V_up = Tensor("V", [spacetime.upper("ν")])
mu = spacetime.lower("μ")
nabla_V = covariant(V_up, mu, christoffel)
expanded = expand_covariant(nabla_V)

print(f"  ∇_μ V^ν = {to_latex(expanded)}")
print()

# Flat spacetime: Γ=0 → ∇V = ∂V
V_arr = np.array([1., 0., 0., 0.])
dV_flat = np.zeros((4, 4))
dV_flat[1, 0] = 0.5  # ∂_1 V^0 = 0.5
Gamma_zero = np.zeros((4, 4, 4))

result6 = evaluate(expanded, {"∂V": dV_flat, "Γ": Gamma_zero, "V": V_arr})
print(f"  Flat spacetime (Γ=0):")
print(f"  ∇_μ V^ν = \n{result6}")
print(f"  (∂_μ V^ν와 같아야 함)")
print()

# ============================================================
# 7. (name, positions) 키로 metric 구분
# ============================================================
print("=== 7. Metric 구분: (name, pos) 키 ===\n")

# g_{μν} → ("g", "dd"), g^{μν} → ("g", "uu")
g_lower = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_upper = Tensor("g", [spacetime.upper("ν"), spacetime.upper("λ")])

# Flat metric example
metric = np.diag([-1., 1., 1., 1.])
inv_metric = np.diag([-1., 1., 1., 1.])  # Minkowski is self-inverse

# g_{μν} g^{νλ} = δ^λ_μ (identity)
prod7 = g_lower * g_upper
# 주의: ν는 g_lower에서 lower, g_upper에서 upper → contraction
result7 = evaluate(prod7, {("g", "dd"): metric, ("g", "uu"): inv_metric})
print(f"  g_{{μν}} g^{{νλ}} = \n{result7}")
print(f"  (δ^λ_μ = identity여야 함)")
print()

print("=== Phase 6d 완료! ===")
