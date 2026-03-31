"""
Cosmological Perturbation Theory: Linearized Ricci Tensor.

Background: ḡ_{μν} = diag(-N², a²Ω_{ij})  with Ω_{ij} = δ_{ij}/f², f = 1+kr²/4
Perturbation (scalar sector, Newtonian gauge B=E=0):
  δg_{00} = -2N²A,  δg_{ij} = 2a²CΩ_{ij}

Uses SymPy direct computation with Palatini identity for linearized Ricci.
"""

import sys
sys.path.insert(0, "/home/minjae/Minjae/IndexCalc")

import sympy as sp

print("=" * 70)
print("Cosmological Perturbation Theory: δR_{μν}")
print("=" * 70)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# k=0, general N(t), Newtonian gauge (B=E=0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[Scalar sector] k=0, Newtonian gauge, general N(t)")
print("-" * 60)

t, x, y, z = sp.symbols('t x y z')
eps = sp.Symbol('eps')
spatial = [x, y, z]
coords = [t, x, y, z]
dim = 4

# Background
a = sp.Function('a', positive=True)(t)
N = sp.Function('N', positive=True)(t)

# Perturbations
A = sp.Function('A')(t, x, y, z)
C = sp.Function('C')(t, x, y, z)


def linearize(expr):
    """ε의 1차까지 전개."""
    return expr.series(eps, 0, 2).removeO()


def compute_christoffel(g, g_inv, coords, dim):
    """Christoffel symbols, linearized in ε."""
    Gamma = [[[None]*dim for _ in range(dim)] for _ in range(dim)]
    for s in range(dim):
        for m in range(dim):
            for n in range(m, dim):
                val = sp.S(0)
                for rho in range(dim):
                    if g_inv[s][rho] == 0:
                        continue
                    val += g_inv[s][rho] * (
                        sp.diff(g[rho][m], coords[n])
                        + sp.diff(g[rho][n], coords[m])
                        - sp.diff(g[m][n], coords[rho])
                    )
                val = linearize(sp.expand(sp.Rational(1, 2) * val))
                Gamma[s][m][n] = val
                Gamma[s][n][m] = val
    return Gamma


def split_bg_pert(Gamma, dim, eps):
    """Background (ε⁰)와 perturbation (ε¹) 분리."""
    bg = [[[G.coeff(eps, 0) for G in row] for row in mat] for mat in Gamma]
    dG = [[[G.coeff(eps, 1) for G in row] for row in mat] for mat in Gamma]
    return bg, dG


def ricci_linearized(mu, nu, bg, dG, coords, dim):
    """Palatini identity: δR_{μν} = ∇̄_ρ δΓ^ρ_{μν} - ∇̄_ν δΓ^ρ_{μρ}."""
    val = sp.S(0)
    for rho in range(dim):
        val += sp.diff(dG[rho][mu][nu], coords[rho])
        val -= sp.diff(dG[rho][mu][rho], coords[nu])
        for sigma in range(dim):
            val += bg[rho][rho][sigma] * dG[sigma][mu][nu]
            val += dG[rho][rho][sigma] * bg[sigma][mu][nu]
            val -= bg[rho][mu][sigma] * dG[sigma][nu][rho]
            val -= dG[rho][mu][sigma] * bg[sigma][nu][rho]
    return sp.expand(val)


# ── Metric 구성 ──
g = [[sp.S(0)]*dim for _ in range(dim)]
g[0][0] = -N**2 * (1 + 2*eps*A)
for i in range(1, 4):
    g[i][i] = a**2 * (1 + 2*eps*C)

g_inv = [[sp.S(0)]*dim for _ in range(dim)]
g_inv[0][0] = -1/N**2 * (1 - 2*eps*A)
for i in range(1, 4):
    g_inv[i][i] = 1/a**2 * (1 - 2*eps*C)


# ── 계산 ──
print("  Christoffel symbols 계산 중...")
Gamma = compute_christoffel(g, g_inv, coords, dim)
Gamma_bg, dGamma = split_bg_pert(Gamma, dim, eps)

print("  δR_{μν} 계산 중 (Palatini identity)...")
idx = ['t', 'x', 'y', 'z']

results = {}
for mu in range(dim):
    for nu in range(mu, dim):
        val = ricci_linearized(mu, nu, Gamma_bg, dGamma, coords, dim)
        results[(mu, nu)] = val


# ── 표기법 정리 ──
# ∂_t → dot,  ∂_i∂_j → 공간 미분
Adot = sp.Derivative(A, t)
Cdot = sp.Derivative(C, t)


def nabla2(f):
    """Flat spatial Laplacian."""
    return sum(sp.diff(f, xi, 2) for xi in spatial)


# ── 출력 ──
print("\n  ══════════════════════════════════════════════════════════")
print("  ds² = -N(t)²(1+2A)dt² + a(t)²(1+2C)(dx²+dy²+dz²)")
print("  Notation: H = ȧ/a, dot = ∂/∂t, ∇² = spatial Laplacian")
print("  ══════════════════════════════════════════════════════════")

# δR_{00}
print(f"\n  δR_{{00}} =")
val00 = results[(0, 0)]
print(f"    {val00}")

# δR_{0i}: check they're all proportional to ∂_i
print(f"\n  δR_{{0i}} (= ∂_i × [...]):")
val01 = results[(0, 1)]
# Factor out ∂/∂x from the x-component
print(f"    δR_{{0x}} = {val01}")

# δR_{ij}: split trace and traceless
print(f"\n  δR_{{ij}} (spatial):")
val11 = results[(1, 1)]  # δR_{xx}
val12 = results.get((1, 2), sp.S(0))  # δR_{xy}
val22 = results[(2, 2)]  # δR_{yy}

print(f"    δR_{{xx}} = {val11}")
print(f"    δR_{{xy}} = {val12}")

# 확인: δR_{yy} vs δR_{xx} (isotropic 구조)
diff_11_22 = sp.expand(val11.subs(x, sp.Symbol('X')).subs(y, x).subs(sp.Symbol('X'), y) - val22)
if sp.simplify(diff_11_22) == 0:
    print(f"    δR_{{yy}} = δR_{{xx}}|_{{x↔y}} ✓ (isotropy)")

# ── δR_{ij}를 trace + traceless로 분해 ──
print(f"\n  ── Trace-traceless 분해 ──")
# δR_{ij} = [trace part]·δ_{ij} + ∂_i∂_j[traceless part]
# trace: Σ_i δR_{ii} / 3
trace_part = sp.expand(sp.Rational(1, 3) * (
    results[(1,1)] + results[(2,2)] + results[(3,3)]
))
# traceless: δR_{xy} = ∂_x∂_y × (something)
if val12 != 0:
    # δR_{xy} / (∂_x∂_y) should give the traceless coefficient
    # extract it symbolically
    print(f"    δR_{{ij}} = [trace]·δ_{{ij}} + ∂_i∂_j[traceless]")
    print(f"    trace part (1/3 Σ δR_{{ii}}) = {trace_part}")
    print(f"    traceless: δR_{{xy}} = {val12}")


# ── Ricci Scalar ──
print(f"\n  ══════════════════════════════════════════════════════════")
print(f"  δR (Ricci scalar)")
print(f"  ══════════════════════════════════════════════════════════")

# δR = ḡ^{μν}δR_{μν} + δg^{μν}R̄_{μν}

# Background Ricci
R_bg = {}
for mu in range(dim):
    for nu in range(mu, dim):
        val = sp.S(0)
        for rho in range(dim):
            val += sp.diff(Gamma_bg[rho][mu][nu], coords[rho])
            val -= sp.diff(Gamma_bg[rho][mu][rho], coords[nu])
            for sigma in range(dim):
                val += Gamma_bg[rho][rho][sigma] * Gamma_bg[sigma][mu][nu]
                val -= Gamma_bg[rho][mu][sigma] * Gamma_bg[sigma][nu][rho]
        R_bg[(mu, nu)] = sp.expand(val)

print(f"\n  Background Ricci:")
print(f"    R̄_{{00}} = {sp.simplify(R_bg[(0,0)])}")
print(f"    R̄_{{ii}} = {sp.simplify(R_bg[(1,1)])}")

# δR = ḡ^{μν}δR_{μν} + δg^{μν}R̄_{μν}
delta_R = sp.S(0)
delta_R += (-1/N**2) * results[(0, 0)]
for i in range(1, 4):
    delta_R += (1/a**2) * results[(i, i)]
delta_R += (2*A/N**2) * R_bg[(0, 0)]
for i in range(1, 4):
    delta_R += (-2*C/a**2) * R_bg[(i, i)]

delta_R = sp.expand(delta_R)
print(f"\n  δR = {delta_R}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# N=1 특수화 + 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*70}")
print(f"  N=1 (cosmic time) 특수화")
print(f"{'='*70}")

N1 = {N: 1, sp.Derivative(N, t): 0, sp.Derivative(N, t, 2): 0}

for key in [(0,0), (0,1), (1,1), (1,2)]:
    val = results[key].subs(N1)
    val = sp.simplify(val)
    label = f"δR_{{{idx[key[0]]}{idx[key[1]]}}}"
    print(f"\n  {label} = {val}")

delta_R_N1 = delta_R.subs(N1)
delta_R_N1 = sp.simplify(delta_R_N1)
print(f"\n  δR = {delta_R_N1}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검증: Metric.symbolic()와 교차 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*70}")
print(f"  Verification: Metric.symbolic() cross-check (static a, N=1)")
print(f"{'='*70}")

from indexcalc import Metric

a0 = sp.Symbol('a_0', positive=True)
Phi = sp.Function('Phi')(t, x, y, z)

# Static background (ȧ=0) + perturbation
g_test = sp.diag(
    -(1 + 2*Phi),
    a0**2 * (1 + 2*Phi),  # C = Phi (anisotropic stress = 0)
    a0**2 * (1 + 2*Phi),
    a0**2 * (1 + 2*Phi),
)

print(f"  g = diag(-(1+2Φ), a₀²(1+2Φ), ...) [static, A=C=Φ]")

m = Metric(g_test, ["t", "x", "y", "z"], signature=(3, 1))
sym = m.symbolic([t, x, y, z])

# For static a, the background R=0
# δR should be proportional to ∇²Φ
R_full = sp.expand(sym.R)
print(f"  R (full, exact) = {R_full}")

# Compare: our perturbation formula with a=a0 (const), N=1, A=C=Phi
our_R = delta_R.subs(N1).subs({
    a: a0, sp.Derivative(a, t): 0, sp.Derivative(a, t, 2): 0,
    A: Phi, C: Phi,
})
our_R = sp.expand(our_R)
print(f"  δR (perturbation) = {our_R}")

# Linearize the exact result (already linear if Φ is small)
# The exact result includes higher-order terms in Φ
# Extract linear coefficient: substitute Φ → ε*Φ, expand, take O(ε)
Phi_sub = sp.Function('Phi')(t, x, y, z)
R_lin = R_full  # already linear for small Φ if background R=0
print(f"\n  Match: {sp.simplify(R_full - our_R) == 0}")
if sp.simplify(R_full - our_R) != 0:
    print(f"  Difference: {sp.simplify(R_full - our_R)}")
    print(f"  (Higher-order terms expected if Metric.symbolic() gives exact result)")


print(f"\n{'='*70}")
print("Done.")
print(f"{'='*70}")
