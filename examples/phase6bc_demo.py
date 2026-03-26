"""
Phase 6b/6c 데모: Connection, CovariantDeriv, 자동 전개.
"""

from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse, to_latex,
    Connection, LeviCivitaConnection,
    CovariantDeriv, covariant, expand_covariant,
    partial, expand_partial,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

g     = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])

# ============================================================
# 1. Connection 정의
# ============================================================
print("=== Connection ===\n")

christoffel = LeviCivitaConnection(g, g_inv, spacetime)
print(f"  {christoffel}")

# Γ^μ_νλ 텐서 생성
gamma = christoffel.make_tensor("μ", "ν", "λ")
print(f"  Γ tensor: {gamma}")
print(f"  LaTeX:    {to_latex(gamma)}")
print()

# Spin connection
spin_conn = Connection("ω", lorentz, deriv_space=spacetime)
print(f"  {spin_conn}")
omega = spin_conn.make_tensor("a", "μ", "b")
print(f"  ω tensor: {omega}")
print(f"  LaTeX:    {to_latex(omega)}")
print()

# ============================================================
# 2. Christoffel symbol 정의식
# ============================================================
print("=== Christoffel 정의 ===\n")

defn = christoffel.definition()
print(f"  Γ^μ_νλ = {to_latex(defn)}")
print()

# ============================================================
# 3. CovariantDeriv — 기본
# ============================================================
print("=== CovariantDeriv 기본 ===\n")

V = Tensor("V", [spacetime.upper("ν")])
mu = spacetime.lower("μ")

nabla_V = covariant(V, mu, christoffel)
print(f"  ∇_μ V^ν = {nabla_V}")
print(f"  LaTeX:    {to_latex(nabla_V)}")
print(f"  free:     {nabla_V.free_indices}")
print()

# ============================================================
# 4. expand_covariant — ∇ → ∂ + Γ
# ============================================================
print("=== expand_covariant ===\n")

# ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μρ V^ρ
expanded = expand_covariant(nabla_V)
print(f"  ∇_μ V^ν 전개:")
print(f"  {to_latex(expanded)}")
print()

# ∇_μ V_ν = ∂_μ V_ν - Γ^ρ_μν V_ρ
V_lower = Tensor("V", [spacetime.lower("ν")])
nabla_Vl = covariant(V_lower, mu, christoffel)
expanded_l = expand_covariant(nabla_Vl)
print(f"  ∇_μ V_ν 전개:")
print(f"  {to_latex(expanded_l)}")
print()

# ============================================================
# 5. (1,1)-tensor: ∇_μ T^ν_λ
# ============================================================
print("=== (1,1)-tensor ===\n")

T = Tensor("T", [spacetime.upper("ν"), spacetime.lower("λ")])
nabla_T = covariant(T, mu, christoffel)
expanded_T = expand_covariant(nabla_T)
print(f"  ∇_μ T^ν_λ 전개:")
print(f"  {to_latex(expanded_T)}")
print()

# ============================================================
# 6. (0,2)-tensor (metric): ∇_μ g_νλ
# ============================================================
print("=== metric ∇_μ g_νλ ===\n")

g_nl = Tensor("g", [spacetime.lower("ν"), spacetime.lower("λ")])
nabla_g = covariant(g_nl, mu, christoffel)
expanded_g = expand_covariant(nabla_g)
print(f"  ∇_μ g_νλ 전개:")
print(f"  {to_latex(expanded_g)}")
print("  (metric compatibility: 이 값이 0이어야 함)")
print()

# ============================================================
# 7. 다중 connection: T^{ν a} (spacetime + lorentz)
# ============================================================
print("=== 다중 connection ===\n")

T_mixed = Tensor("T", [spacetime.upper("ν"), lorentz.upper("a")])
connections = {
    spacetime.name: christoffel,
    lorentz.name: spin_conn,
}
nabla_Tm = covariant(T_mixed, mu, connections)
expanded_Tm = expand_covariant(nabla_Tm)
print(f"  ∇_μ T^ν^a 전개 (Γ for spacetime, ω for lorentz):")
print(f"  {to_latex(expanded_Tm)}")
print()

# ============================================================
# 8. Connection이 없는 space는 무시
# ============================================================
print("=== connection 없는 space ===\n")

# lorentz connection 없이 spacetime만
nabla_Tm2 = covariant(T_mixed, mu, christoffel)
expanded_Tm2 = expand_covariant(nabla_Tm2)
print(f"  ∇_μ T^ν^a (spacetime connection만):")
print(f"  {to_latex(expanded_Tm2)}")
print("  → lorentz index a에는 Γ 항 없음")
