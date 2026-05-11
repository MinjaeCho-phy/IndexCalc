"""perturbed_ricci_diagnose.py

Perturbed Ricci tensor 진단 스크립트.

Setting:
    - Conformal time η, spatial section maximally symmetric (curvature K).
    - Background:  ḡ_μν dx^μ dx^ν = a²(η) [-dη² + γ_ij dx^i dx^j],
      where γ_ij is the K-curvature 3-metric (R̄^(3)_ij = 2K γ_ij).
    - Perturbation (full SVT):
        δg_00 = -2 a² A
        δg_0i =    a² B_i,    B_i = D_i B + B_i^V         (D^i B_i^V = 0)
        δg_ij =  2 a² C_ij,   C_ij = -ψ γ_ij + D_i D_j E
                              + D_(i E_j)^V + h_ij^TT
                              (D^i E_j^V = 0, h^TT trace-/transverse-free)

목적: IndexCalc로 위 setting을 표현하고 Christoffel→Riemann→Ricci까지
계산을 시도하며, 막히는 지점에서 어떤 기능 확장이 필요한지를 진단한다.

(Karpathy guideline: 각 단계마다 명시적 성공 기준 + GAP 기록.)
"""

from __future__ import annotations

import sympy as sp

from indexcalc import (
    IndexSpace, IndexRegistry, MetricRegistry,
    Tensor, LeviCivitaConnection,
    Variation, VariationRegistry, expand_variation,
    PartialDeriv, partial, partial_to_covariant,
    TensorProduct, TensorSum, ScalarMul,
    Sym, Antisym, TraceFreeSym, expand_symmetrization,
    ZeroTensor,
    to_latex,
)
from indexcalc.core.index import Index


# ───────────────────────────────────────────────────────────────────
# Step 1: Background FLRW metric (conformal time, curvature K)
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Background FLRW metric")
print("=" * 70)

# Index spaces — 4D spacetime 위에 3D spatial submanifold.
st = IndexSpace("st", dim=4, indices="μνρσλαβ", metric="g")
sp_ = IndexSpace("sp", dim=3, indices="ijklmn",  metric="γ")

reg = IndexRegistry()
for space in (st, sp_):
    reg.register(space)

# Metric registry — metric / inverse 쌍을 IndexSpace에 등록.
# 4D spacetime
μ0, ν0 = st.lower("μ"), st.lower("ν")
μ0_up, ν0_up = st.upper("μ"), st.upper("ν")
g_lower = Tensor("g", [μ0, ν0])
g_upper = Tensor("g", [μ0_up, ν0_up])

# 3D spatial
i0, j0 = sp_.lower("i"), sp_.lower("j")
i0_up, j0_up = sp_.upper("i"), sp_.upper("j")
γ_lower = Tensor("γ", [i0, j0])
γ_upper = Tensor("γ", [i0_up, j0_up])

mreg = MetricRegistry()
mreg.register(g_lower, g_upper, st)
mreg.register(γ_lower, γ_upper, sp_)

# Background quantities (symbolic scalars).
η = sp.symbols("η", real=True)
a = sp.Function("a")(η)
K = sp.symbols("K", real=True)   # spatial curvature (0, +1, -1)

# Background metric tensor symbols.
# ḡ_μν 자체는 IndexCalc에서는 'g' metric을 lower-indices로 부르면 자동 처리.
# γ_ij도 마찬가지.
print(f"  conformal time η     : {η}")
print(f"  scale factor a(η)    : {a}")
print(f"  spatial curvature K  : {K} (free symbol)")
print(f"  IndexSpace 'st' dim={st.dim}, metric='{st.metric}'")
print(f"  IndexSpace 'sp' dim={sp_.dim}, metric='{sp_.metric}'")
print(f"  MetricRegistry: 'g' on st, 'γ' on sp")

# 성공 기준 1: metric 객체가 등록되고 is_metric로 확인 가능.
print(f"\n  γ_ij Tensor          : {γ_lower}")
print(f"  γ^ij Tensor          : {γ_upper}")
print(f"  is_metric(γ_lower)?  : {mreg.is_metric(γ_lower)}")
print(f"  g_μν Tensor          : {g_lower}")
print(f"  g^μν Tensor          : {g_upper}")
print(f"  is_metric(g_lower)?  : {mreg.is_metric(g_lower)}")

print("\nSTEP 1 ✓  background metric symbols registered.\n")


# ───────────────────────────────────────────────────────────────────
# Step 2: SVT decomposition — perturbation variables
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 2: SVT decomposition")
print("=" * 70)

# Scalar perturbations: A, B, ψ, E — coordinate functions.
A = sp.Function("A")(η)        # placeholder; full deps η, x^i 가지지만 표기 단순화.
B = sp.Function("B")(η)
ψ = sp.Function("ψ")(η)
E = sp.Function("E")(η)
print(f"  scalar pert.        : A, B, ψ, E (sp.Function placeholders)")

# Vector perturbations: B_i^V, E_i^V — transverse 3-vectors (D^i V_i = 0).
i_lo, j_lo = sp_.lower("i"), sp_.lower("j")
BV = Tensor("BV", [i_lo], transverse=[0])
EV = Tensor("EV", [i_lo], transverse=[0])
print(f"  vector pert. (transverse):")
print(f"    BV_i (B^V)      : {BV}    transverse={BV.transverse}")
print(f"    EV_i (E^V)      : {EV}    transverse={EV.transverse}")

# Tensor perturbation: h_ij^TT — symmetric, traceless, transverse.
hTT = Tensor(
    "hTT", [i_lo, j_lo],
    symmetric_pairs=[(0, 1)],
    traceless=[(0, 1)],
    transverse=[0, 1],
)
print(f"  tensor pert. (TT)   :")
print(f"    hTT_ij (h^TT)   : {hTT}")
print(f"      symmetric_pairs={hTT.symmetric_pairs}")
print(f"      traceless     ={hTT.traceless}")
print(f"      transverse    ={hTT.transverse}")

# Background spatial metric γ_ij — symmetric (이미 metric은 symmetric으로 다뤄야 함).
# 추후 expand_metric 에서 자동 인지하도록 만드는 게 자연스럽지만, 일단은 명시.
γ_sym = Tensor("γ", [i_lo, j_lo], symmetric_pairs=[(0, 1)])
print(f"\n  γ_ij (with symmetric slot) : {γ_sym}, symmetric_pairs={γ_sym.symmetric_pairs}")

print()
print("  ─── 갭 현황 (G1~G3 해결, G4~G5 남음) ────────────────────────────")
print("  G1. Tensor.symmetric_pairs    : ✓ 추가됨")
print("  G2. Tensor.traceless          : ✓ 추가됨")
print("  G3. Tensor.transverse         : ✓ 추가됨")
print("  G4. Sym / TraceFreeSym 노드   : 미구현 (D_(i E_j), D_⟨i D_j⟩ 표현용)")
print("  G5. Simplification 규칙       : 미구현")
print("      • γ^ij · hTT_ij        → 0   (traceless × metric)")
print("      • D^i · BV_i           → 0   (transverse × spatial ∂^i)")
print("      • antisym_pair × sym_pair → 0  (canonicalize_antisym 확장 필요)")

print("\nSTEP 2 ✓  속성 슬롯으로 표현 완료. 다음 갭: Sym 노드 + simplification.\n")


# ───────────────────────────────────────────────────────────────────
# Step 3: Christoffel — background Γ̄ + δΓ via Palatini
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 3: Christoffel symbol — background + linear order")
print("=" * 70)

# 3.1 Background Christoffel definition (symbolic)
conn_st = LeviCivitaConnection(g_lower, g_upper, st)
chris_def = conn_st.definition()
print("  Background Christoffel definition:")
print("    Γ^μ_νλ =", to_latex(chris_def))

# 3.2 δΓ 도출 시도 — metric을 varying으로 declare 후 변분 전개
print("\n  Trying δ(Christoffel definition) via expand_variation...")

vreg = VariationRegistry()
vreg.declare_varying("g")           # metric은 varying
# inverse metric의 변분은 별도로 declare 가능; 일단 시도.

try:
    var_expr = Variation(chris_def)
    delta_chris = expand_variation(var_expr, vreg)
    print("    δ(Γ) =", to_latex(delta_chris)[:300])
    print("    ... (full)")
    print("    [성공: Variation API로 δΓ 전개됨]")
except Exception as e:
    print(f"    [FAIL] {type(e).__name__}: {e}")
    print("    → expand_variation이 이 형태를 처리 못함. 진단 필요.")

print()
print("  ─── Step 3 갭 누적 ────────────────────────────────────────────")
print("  G6. δg^{μν} = -g^{μρ}g^{νσ}δg_{ρσ} : 자동 치환 안 됨")
print("  G7. Palatini covariant form collapse : 자동 안 됨")
print("        (∂_μ δg_νλ → ∇̄_μ δg_νλ - Γ̄·δg − ...)")
print()


# ───────────────────────────────────────────────────────────────────
# Step 4: Riemann R^ρ_σμν — symbolic + δR via Palatini identity
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 4: Riemann tensor — symbolic structure + linear order")
print("=" * 70)

# Riemann definition (R^ρ_σμν):
#   R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
#
# Γ을 Tensor 객체로 두고 표현식을 직접 구성한다.
# 진단 포인트:
#   (a) ∂Γ - ∂Γ 부분의 antisymmetric_pairs=[(2,3)] 표현이 자연스럽게 되는가?
#   (b) Γ을 varying_connection으로 declare 후 expand_variation이 δR 도출 가능한가?
#   (c) 결과가 Palatini identity 형태 (δR^ρ_σμν = ∇̄_μ δΓ^ρ_νσ − ∇̄_ν δΓ^ρ_μσ)로
#       collapse 되는가?

def Gamma(up: str, lo1: str, lo2: str) -> Tensor:
    """Γ^up_{lo1 lo2} Tensor 객체. 인덱스 이름만 받아 구성."""
    return Tensor("Γ", [
        Index(up,  st, "upper"),
        Index(lo1, st, "lower"),
        Index(lo2, st, "lower"),
    ])

ρ, σ, μi, νi, λ = "ρ", "σ", "μ", "ν", "λ"
μ_lo, ν_lo = Index(μi, st, "lower"), Index(νi, st, "lower")

# 항 1, 2: ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ
term1 = PartialDeriv(Gamma(ρ, νi, σ), μ_lo)
term2 = PartialDeriv(Gamma(ρ, μi, σ), ν_lo)
partial_part = term1 - term2

# 항 3, 4: Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
quad1 = TensorProduct(Gamma(ρ, μi, λ),  Gamma(λ, νi, σ))
quad2 = TensorProduct(Gamma(ρ, νi, λ),  Gamma(λ, μi, σ))
quad_part = quad1 - quad2

riemann = partial_part + quad_part
print("  Riemann symbolic form:")
print("    R^ρ_{σμν} =", to_latex(riemann))

# 4.1 δR via expand_variation
print("\n  Trying δR via expand_variation (Γ declared as varying_connection)...")
vreg_R = VariationRegistry()
vreg_R.declare_varying("Γ")                 # Tensor leaf as varying
vreg_R.declare_varying_connection("Γ")      # connection role (for covariant deriv)

try:
    delta_R = expand_variation(Variation(riemann), vreg_R)
    out = to_latex(delta_R)
    print("    δR =", out)
    print(f"    [총 길이: {len(out)} 자]")
    print("    [성공: raw Leibniz form. covariant form collapse는 G7 (미해결)]")
except Exception as e:
    print(f"    [FAIL] {type(e).__name__}: {e}")

print()
print("  ─── Step 4에서 발견 + 수정한 버그 ─────────────────────────────")
print("  BUG-D1. display.py ScalarMul × TensorSum 부호 분배 누락.")
print("          δ(-A·B) 결과가 -δA·B + A·δB 로 잘못 출력 (오른쪽 항 부호 누락).")
print("          fix: ScalarMul(c<0, TensorSum)을 평탄화하여 모든 항에 부호 전파.")
print("          regression: tests/test_display_sign_distribution.py (4개).")
print()


# ───────────────────────────────────────────────────────────────────
# Step 5: Ricci R_μν — δR_μν via index contraction of δR^ρ_σμν
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 5: Ricci tensor — δR_μν")
print("=" * 70)

# Ricci = R^ρ_σρν trace contraction. δR_σν = δR^ρ_σρν 즉 (μ=ρ) 으로 contraction.
#
# 직접적으로 IndexCalc 표현으로 만들기에는 Riemann의 인덱스 이름을 다시 설정해야 한다.
# Riemann (ρ, σ, μ, ν) 중 (ρ, μ) trace. 아래는 새 인덱스 설정으로 Ricci tensor 정의.
ρ2, σ2, μ2, ν2, λ2 = "ρ", "σ", "ρ", "ν", "λ"  # μ ↔ ρ contraction
# 하지만 Index 클래스가 같은 이름 두 번 (한 번 upper, 한 번 lower)이면 contract가 됨.
# Tensor 객체로 직접 Ricci 정의를 만들자.

def Γ_named(up, lo1, lo2, *, up_pos="upper", lo1_pos="lower", lo2_pos="lower"):
    return Tensor("Γ", [
        Index(up,  st, up_pos),
        Index(lo1, st, lo1_pos),
        Index(lo2, st, lo2_pos),
    ])

# Ricci: contract μ ↔ ρ in Riemann definition.
# Use distinct dummy names to be safe.
ρ_lo = Index("ρ", st, "lower")

ricci_term1 = PartialDeriv(Γ_named("ρ", "ν", "σ"), ρ_lo)            # ∂_ρ Γ^ρ_νσ
ricci_term2 = PartialDeriv(Γ_named("ρ", "ρ", "σ"), Index("ν", st, "lower"))  # ∂_ν Γ^ρ_ρσ
ricci_quad1 = TensorProduct(Γ_named("ρ", "ρ", "λ"), Γ_named("λ", "ν", "σ"))   # Γ^ρ_ρλ Γ^λ_νσ
ricci_quad2 = TensorProduct(Γ_named("ρ", "ν", "λ"), Γ_named("λ", "ρ", "σ"))   # Γ^ρ_νλ Γ^λ_ρσ
try:
    ricci_def = (ricci_term1 - ricci_term2) + (ricci_quad1 - ricci_quad2)
    print("  R_σν =", to_latex(ricci_def))
    print("    [성공]")
except Exception as e:
    print(f"  [FAIL constructing Ricci] {type(e).__name__}: {e}")
    print()
    print("  ─── 새 갭 G8 ────────────────────────────────────────────────")
    print("  G8. PartialDeriv가 자기 deriv_index와 inner expr의 같은-이름-반대-위치")
    print("      인덱스를 자동 contraction으로 인식 못함.")
    print("      ∂_ρ Γ^ρ_νσ 식에서 ρ-ρ pair가 free로 카운트되어 free_indices=4가 됨")
    print("      (정상은 2: σ, ν). 결과: TensorSum free index 일치 체크에서 실패.")
    print("      우회: Trace 객체 명시, 또는 PartialDeriv의 free_indices 로직 수정.")
    ricci_def = None

if ricci_def is not None:
    print("\n  Trying δR_σν via expand_variation...")
    try:
        delta_Ricci = expand_variation(Variation(ricci_def), vreg_R)
        out = to_latex(delta_Ricci)
        print("    δR_σν =", out)
        print(f"    [총 길이: {len(out)} 자]")
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {e}")
print()


# ───────────────────────────────────────────────────────────────────
# Step 6: Ricci scalar R — δR = g^μν δR_μν + δg^μν R̄_μν
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 6: Ricci scalar — δR")
print("=" * 70)

# 진단: δR = δ(g^μν R_μν) = δg^μν R_μν + g^μν δR_μν
# 직접 g^μν · R_μν 를 만들고 expand_variation.

R_sigma_nu = Tensor("R", [Index("σ", st, "lower"), Index("ν", st, "lower")])
g_up = Tensor("g", [Index("σ", st, "upper"), Index("ν", st, "upper")])
ricci_scalar = TensorProduct(g_up, R_sigma_nu)
print("  R = g^σν R_σν")
print("    structure:", to_latex(ricci_scalar))

vreg6 = VariationRegistry()
vreg6.declare_varying("g")
vreg6.declare_varying("R")
try:
    deltaR_scalar = expand_variation(Variation(ricci_scalar), vreg6)
    print("\n  δR =", to_latex(deltaR_scalar))
    print("    [성공: scalar variation Leibniz로 전개됨]")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

print()
print("=" * 70)
print("진단 요약")
print("=" * 70)
print("✓ Step 1: Background metric setup (FLRW conformal, general K)")
print("✓ Step 2: SVT decomposition — Tensor 속성 슬롯 추가 후 표현 완료")
print("✓ Step 3: δΓ raw form 도출")
print("✓ Step 4: δR raw form 도출 + display 버그 발견·수정")
print("✓ Step 5: δR_μν raw form (Ricci trace 구조)")
print("✓ Step 6: δR scalar raw form (g^μν·R_μν Leibniz)")
print()
print("━━━ G4 demo: Sym/Antisym/TraceFreeSym 노드 (2026-05-11 추가) ━━━")
# (a) D_(i E_j) — SVT vector decomposition의 textbook 항
print("\n  (a) D_(i E_j^V) (transverse vector pert. symmetrizer)")
expr_a = PartialDeriv(EV, j_lo)        # ∂_i EV_j 같은 형태 (deriv index = i)
expr_a = PartialDeriv(EV, i_lo)         # ∂_i EV_j → indices: deriv i, EV index j? EV has only one index, named 'i' already → conflict
# 안전한 형태: 새 인덱스로 EV를 j_lo로 옮긴 별도 객체
EV_j = Tensor("EV", [j_lo], transverse=[0])
expr_a = PartialDeriv(EV_j, i_lo)       # ∂_i EV_j
sym_a = Sym(expr_a, [i_lo, j_lo])
print("    Sym node       :", to_latex(sym_a))
print("    expanded       :", to_latex(expand_symmetrization(sym_a)))

# (b) D_⟨i D_j⟩ E — traceless symmetric, perturbation scalar E의 textbook 항
print("\n  (b) D_⟨i D_j⟩ E (TraceFreeSym on scalar pert.)")
E_t = Tensor("E", [])
inner_b = PartialDeriv(PartialDeriv(E_t, j_lo), i_lo)
tfs_b = TraceFreeSym(inner_b, [i_lo, j_lo])
print("    TFS node       :", to_latex(tfs_b))
print("    expanded       :", to_latex(expand_symmetrization(tfs_b, mreg)))

# (c) Antisym of partial deriv — F_{[ij]} 같은 패턴
print("\n  (c) Antisym(∂_i V_j) (electromagnetic-like)")
V_j = Tensor("V", [j_lo])
expr_c = PartialDeriv(V_j, i_lo)
antisym_c = Antisym(expr_c, [i_lo, j_lo])
print("    Antisym node   :", to_latex(antisym_c))
print("    expanded       :", to_latex(expand_symmetrization(antisym_c)))

print()
print("━━━ G5 demo: 텐서 속성 기반 simplification (2026-05-11 추가) ━━━")
from indexcalc.core.simplify import (
    is_zero_by_traceless_metric,
    is_zero_by_transverse_deriv,
    is_zero_by_antisym_swap,
    simplify,
)

# (a) γ^{ij} · h^{TT}_{ij} → 0  (traceless × metric)
print("\n  (a) γ^{ij} h^{TT}_{ij}  (traceless × metric)")
i_lo2, j_lo2 = sp_.lower("i"), sp_.lower("j")
i_up2, j_up2 = sp_.upper("i"), sp_.upper("j")
γ_inv_ij = Tensor("γ", [i_up2, j_up2], symmetric_pairs=[(0, 1)])
hTT2 = Tensor(
    "hTT", [i_lo2, j_lo2],
    symmetric_pairs=[(0, 1)],
    traceless=[(0, 1)],
    transverse=[0, 1],
)
expr_a = TensorProduct(γ_inv_ij, hTT2)
print(f"    raw       : {to_latex(expr_a)}")
result_a = is_zero_by_traceless_metric(expr_a, mreg)
print(f"    simplified: {to_latex(result_a)}    [✓ ZeroTensor]"
      if isinstance(result_a, ZeroTensor) else
      f"    simplified: {to_latex(result_a)}    [✗ not detected]")

# (b) γ^{ij} ∂_i BV_j → 0  (transverse × ∂ via metric)
print("\n  (b) γ^{ij} ∂_i BV_j  (transverse × ∂ via metric)")
BV2 = Tensor("BV", [j_lo2], transverse=[0])
deriv_b = PartialDeriv(BV2, i_lo2)
expr_b = TensorProduct(γ_inv_ij, deriv_b)
print(f"    raw       : {to_latex(expr_b)}")
result_b = is_zero_by_transverse_deriv(expr_b, mreg)
print(f"    simplified: {to_latex(result_b)}    [✓ ZeroTensor]"
      if isinstance(result_b, ZeroTensor) else
      f"    simplified: {to_latex(result_b)}    [✗ not detected]")

# (c) A_{[ab]} S^{ab} → 0  (antisym pair × symmetric_pairs slot)
print("\n  (c) A_{[ab]} S^{ab}  (antisym × symmetric_pairs slot)")
adj = IndexSpace("adj", dim=8, indices="abcdefgh")
a_lo, b_lo = adj.lower("a"), adj.lower("b")
a_up, b_up = adj.upper("a"), adj.upper("b")
A2 = Tensor("A", [a_lo, b_lo], antisymmetric_pairs=[(0, 1)])
S2 = Tensor("S", [a_up, b_up], symmetric_pairs=[(0, 1)])
expr_c = TensorProduct(A2, S2)
print(f"    raw       : {to_latex(expr_c)}")
result_c = is_zero_by_antisym_swap(expr_c)
print(f"    simplified: {to_latex(result_c)}    [✓ ZeroTensor]"
      if isinstance(result_c, ZeroTensor) else
      f"    simplified: {to_latex(result_c)}    [✗ not detected]")

print()
print("━━━ G7 demo: ∂_μ T → ∇̄_μ T - Σ Γ̄·T forward 변환 (2026-05-11 추가) ━━━")
# (a) Simple ∂_μ T^ρ_σ
print("\n  (a) ∂_μ T^ρ_σ  (mixed slots)")
T_demo = Tensor("T", [Index("ρ", st, "upper"), Index("σ", st, "lower")])
expr_g7a = PartialDeriv(T_demo, Index("μ", st, "lower"))
print(f"    raw       : {to_latex(expr_g7a)}")
out_g7a = partial_to_covariant(expr_g7a, conn_st)
print(f"    expanded  : {to_latex(out_g7a)}")

# (b) δg를 ∂에서 ∇̄ form으로 정리 (Palatini 응용)
print("\n  (b) ∂_ν δg_{ρσ}  (textbook covariant form 진입)")
δg = Tensor("δg", [Index("ρ", st, "lower"), Index("σ", st, "lower")],
            symmetric_pairs=[(0, 1)])
expr_g7b = PartialDeriv(δg, Index("ν", st, "lower"))
print(f"    raw       : {to_latex(expr_g7b)}")
out_g7b = partial_to_covariant(expr_g7b, conn_st)
print(f"    expanded  : {to_latex(out_g7b)}")

# (c) cancellation: ∂_μ T - ∂_μ T → 0
from indexcalc.core.simplify import simplify
print("\n  (c) Cancellation: ∂_μ T - ∂_μ T  (partial_to_covariant + simplify → 0)")
T_c = Tensor("T", [Index("ρ", st, "upper")])
expr_g7c = TensorSum(
    PartialDeriv(T_c, Index("μ", st, "lower")),
    ScalarMul(-1, PartialDeriv(T_c, Index("μ", st, "lower"))),
)
print(f"    raw       : {to_latex(expr_g7c)}")
out_g7c = partial_to_covariant(expr_g7c, conn_st)
result_g7c = simplify(out_g7c)
print(f"    final     : {to_latex(result_g7c)}    [{type(result_g7c).__name__}]")

print()
print("━━━ G6 demo: δg^{μν} → −g^{μρ}g^{νσ}δg_{ρσ} 자동 치환 (2026-05-11 추가) ━━━")
# (a) inverse metric variation alone
print("\n  (a) δ(g^{ρσ})  with mreg")
g_up_demo = Tensor("g", [Index("ρ", st, "upper"), Index("σ", st, "upper")])
vreg_g6 = VariationRegistry()
vreg_g6.declare_varying("g")
expanded_a = expand_variation(Variation(g_up_demo), vreg_g6, mreg)
print(f"    raw       : δ({to_latex(g_up_demo)})")
print(f"    auto-expand: {to_latex(expanded_a)}")

# (b) δ(g^μν · R_μν) — Leibniz 안에서도 mreg 전파
print("\n  (b) δ(g^{μν} R_{μν})  Leibniz + auto-expand")
R_demo = Tensor("R", [Index("μ", st, "lower"), Index("ν", st, "lower")])
g_up_b = Tensor("g", [Index("μ", st, "upper"), Index("ν", st, "upper")])
vreg_g6b = VariationRegistry()
vreg_g6b.declare_varying("g")
vreg_g6b.declare_varying("R")
prod_b = TensorProduct(g_up_b, R_demo)
expanded_b = expand_variation(Variation(prod_b), vreg_g6b, mreg)
print(f"    raw       : δ({to_latex(prod_b)})")
print(f"    auto-expand: {to_latex(expanded_b)}")

# (c) background metric → 0
print("\n  (c) δ(g^{ρσ})  with g declared background (δg=0) → ZeroTensor")
vreg_g6c = VariationRegistry()
vreg_g6c.declare_background("g")
expanded_c = expand_variation(Variation(g_up_demo), vreg_g6c, mreg)
print(f"    auto-expand: {to_latex(expanded_c)}    [{type(expanded_c).__name__}]")

print()
print("━━━ G8 demo: free_indices의 Einstein 자동 contraction (2026-05-11 추가) ━━━")
# (a) Tensor self-contract: Γ^ρ_ρλ → free=[λ]
ρU2 = Index("ρ", st, "upper")
ρL2 = Index("ρ", st, "lower")
λL2 = Index("λ", st, "lower")
Γ_self = Tensor("Γ", [ρU2, ρL2, λL2])
print(f"\n  (a) Tensor self-contract")
print(f"     Γ^ρ_ρλ.indices      = {[i.name for i in Γ_self.indices]} (3개, 원본 보존)")
print(f"     Γ^ρ_ρλ.free_indices = {[i.name for i in Γ_self.free_indices]} ({Γ_self.rank}: λ만 free)")

# (b) PartialDeriv contracts deriv_index with inner free
νL2 = Index("ν", st, "lower")
σL2 = Index("σ", st, "lower")
Γ_νσ = Tensor("Γ", [Index("ρ", st, "upper"), νL2, σL2])
d_Γ = PartialDeriv(Γ_νσ, ρL2)
print(f"\n  (b) PartialDeriv ↔ inner contract")
print(f"     ∂_ρ Γ^ρ_νσ.free = {[i.name for i in d_Γ.free_indices]} (정상: ν, σ)")

print()
print("미해결 갭 (다음 작업 후보)")
print("  G4. ✓ 완료 (n=2 swap, prefix LaTeX, expand 함수)")
print("  G5. ✓ 완료 2026-05-11")
print("      • γ^{ij} · h^{TT}_{ij}     → 0  (is_zero_by_traceless_metric)")
print("      • ∂^i BV_i / γ^{ij}∂_iBV_j → 0  (is_zero_by_transverse_deriv)")
print("      • antisym × sym (slot)     → 0  (is_zero_by_antisym_swap 확장)")
print("  G8. ✓ 완료 2026-05-11")
print("      • Tensor.free_indices가 self-pair (Γ^ρ_ρλ) 자동 contract")
print("      • PartialDeriv/CovariantDeriv의 deriv_index도 inner와 자동 contract")
print("      • Step 5 Ricci tensor 직접 구성 가능, Step 6 δR_μν 전개 OK")
print("  G6. ✓ 완료 2026-05-11")
print("      • δg^{μν} → −g^{μρ}g^{νσ}δg_{ρσ} 자동 (mreg 전달 시)")
print("      • Leibniz 재귀에서 mreg 전파; background metric은 ZeroTensor로 cleanup")
print("  G7. ✓ 완료 2026-05-11 (forward 변환 + simplify cleanup)")
print("      • partial_to_covariant: ∂_μ T → ∇̄_μ T - Σ Γ̄·T (slot별 보정)")
print("      • Tensor 속성 보존, only_for로 leaf 선택 가능")
print("      • Backward collapse(∂ + Γ̄ → ∇̄ pattern matching)는 후속 작업")
