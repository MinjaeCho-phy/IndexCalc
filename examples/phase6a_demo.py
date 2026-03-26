"""
Phase 6a 데모: PartialDeriv, Leibniz rule 전개.
"""

from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse, to_latex,
    partial, expand_partial, PartialDeriv,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

# ============================================================
# 1. 기본 편미분
# ============================================================
print("=== 기본 편미분 ===\n")

V = Tensor("V", [spacetime.upper("ν")])
mu = spacetime.lower("μ")

dV = partial(V, mu)
print(f"  ∂_μ V^ν = {dV}")
print(f"  LaTeX:    {to_latex(dV)}")
print(f"  free:     {dV.free_indices}")
print(f"  rank:     {dV.rank}")
print()

# metric의 편미분
g = Tensor("g", [spacetime.lower("ν"), spacetime.lower("λ")])
dg = partial(g, mu)
print(f"  ∂_μ g_νλ = {dg}")
print(f"  LaTeX:     {to_latex(dg)}")
print(f"  free:      {dg.free_indices}")
print()

# ============================================================
# 2. upper index로 줘도 자동으로 lower로 변환
# ============================================================
print("=== upper → lower 자동 변환 ===\n")

mu_up = spacetime.upper("μ")
dV2 = partial(V, mu_up)
print(f"  partial(V, ^μ) → {dV2}")
print(f"  deriv_index position: {dV2.deriv_index.position}")
print()

# ============================================================
# 3. 2차 미분
# ============================================================
print("=== 2차 미분 ===\n")

nu_idx = spacetime.lower("ν")
d2V = partial(partial(V, mu), nu_idx)
print(f"  ∂_ν ∂_μ V^... = {d2V}")
print(f"  LaTeX: {to_latex(d2V)}")
print(f"  free:  {d2V.free_indices}")
print()

# ============================================================
# 4. Leibniz rule: ∂_μ (A * B) → (∂_μ A) B + A (∂_μ B)
# ============================================================
print("=== Leibniz rule ===\n")

A = Tensor("A", [spacetime.upper("ν")])
B = Tensor("B", [spacetime.lower("λ")])
AB = A * B

dAB = partial(AB, mu)
print(f"  ∂_μ (A^ν B_λ) = {dAB}")
print(f"  LaTeX: {to_latex(dAB)}")
print()

expanded = expand_partial(dAB)
print(f"  전개:  {expanded}")
print(f"  LaTeX: {to_latex(expanded)}")
print()

# ============================================================
# 5. Leibniz rule: 합에 대한 미분
# ============================================================
print("=== 합의 미분 ===\n")

C = Tensor("C", [spacetime.upper("ν")])
ApC = A + C
dApC = partial(ApC, mu)
print(f"  ∂_μ (A^ν + C^ν) = {dApC}")
print(f"  LaTeX: {to_latex(dApC)}")

expanded2 = expand_partial(dApC)
print(f"  전개:  LaTeX: {to_latex(expanded2)}")
print()

# ============================================================
# 6. 스칼라곱의 미분
# ============================================================
print("=== 스칼라곱의 미분 ===\n")

sA = 3 * A
dsA = partial(sA, mu)
expanded3 = expand_partial(dsA)
print(f"  ∂_μ (3 A^ν)")
print(f"  전개: {to_latex(expanded3)}")
print()

# ============================================================
# 7. 복합: ∂_μ (g_{νλ} V^λ)
# ============================================================
print("=== 복합 표현식 ===\n")

g2 = Tensor("g", [spacetime.lower("ν"), spacetime.lower("λ")])
Vl = Tensor("V", [spacetime.upper("λ")])
gV = g2 * Vl

dgV = partial(gV, mu)
print(f"  ∂_μ (g_νλ V^λ)")
print(f"  LaTeX: {to_latex(dgV)}")

expanded4 = expand_partial(dgV)
print(f"  전개:  {to_latex(expanded4)}")
print()

# ============================================================
# 8. _repr_latex_ 확인
# ============================================================
print("=== _repr_latex_ ===\n")
print(f"  {dV._repr_latex_()}")
print(f"  {expanded4._repr_latex_()}")
