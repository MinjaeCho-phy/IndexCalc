"""
Phase 4 데모: Metric raise/lower, absorb, expand.
"""

from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse,
    MetricRegistry, raise_index, lower_index, absorb_metric, expand_metric,
    summarize,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

# Metric 텐서 정의 & 등록
g     = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])
eta     = Tensor("η", [lorentz.lower("a"), lorentz.lower("b")])
eta_inv = Tensor("η", [lorentz.upper("a"), lorentz.upper("b")])

metrics = MetricRegistry()
metrics.register(g, g_inv, spacetime)
metrics.register(eta, eta_inv, lorentz)

# ============================================================
# 1. raise_index — lower → upper (inverse metric 삽입)
# ============================================================
print("=== raise_index ===\n")

V_lower = parse("V_{μ}", reg)
raised = raise_index(V_lower, "μ", metrics)
print(f"  V_μ → {raised}")
print(f"  free: {raised.free_indices}")
print()

# (1,1)-tensor에서 lower index 올리기
T = parse("T^{μ}_{ν}", reg)
T_raised = raise_index(T, "ν", metrics)
print(f"  T^μ_ν → {T_raised}")
print(f"  free: {T_raised.free_indices}")
print()

# ============================================================
# 2. lower_index — upper → lower (metric 삽입)
# ============================================================
print("=== lower_index ===\n")

V_upper = parse("V^{μ}", reg)
lowered = lower_index(V_upper, "μ", metrics)
print(f"  V^μ → {lowered}")
print(f"  free: {lowered.free_indices}")
print()

# ============================================================
# 3. absorb_metric — metric을 텐서에 흡수
# ============================================================
print("=== absorb_metric ===\n")

# g_{μν} V^{ν} → V_{μ}
expr1 = parse("g_{μν} V^{ν}", reg)
abs1 = absorb_metric(expr1, metrics)
print(f"  g_μν V^ν  →  {abs1}")
print(f"  free: {abs1.free_indices}")
print()

# g^{μν} T_{νλ} → T^{μ}_{λ}
expr2 = parse("g^{μν} T_{νλ}", reg)
abs2 = absorb_metric(expr2, metrics)
print(f"  g^μν T_νλ  →  {abs2}")
print(f"  free: {abs2.free_indices}")
print()

# η_{ab} V^{b} → V_{a}  (lorentz space)
expr3 = parse("η_{ab} V^{b}", reg)
abs3 = absorb_metric(expr3, metrics)
print(f"  η_ab V^b  →  {abs3}")
print(f"  free: {abs3.free_indices}")
print()

# ============================================================
# 4. absorb_metric — 연쇄 contraction
# ============================================================
print("=== absorb_metric (연쇄) ===\n")

# g^{μν} g_{νλ} → δ^μ_λ 같은 효과
expr4 = parse("g^{μν} g_{νλ}", reg)
abs4 = absorb_metric(expr4, metrics)
print(f"  g^μν g_νλ  →  {abs4}")
print(f"  free: {abs4.free_indices}")
print()

# ============================================================
# 5. expand_metric — absorb의 역연산
# ============================================================
print("=== expand_metric ===\n")

V_low = parse("V_{μ}", reg)
expanded = expand_metric(V_low, "μ", metrics)
print(f"  V_μ → {expanded}")
print(f"  free: {expanded.free_indices}")
print()

V_up = parse("V^{μ}", reg)
expanded2 = expand_metric(V_up, "μ", metrics)
print(f"  V^μ → {expanded2}")
print(f"  free: {expanded2.free_indices}")
print()

# ============================================================
# 6. 왕복 테스트: raise → absorb → 원래로?
# ============================================================
print("=== 왕복 테스트 ===\n")

original = parse("V_{μ}", reg)
print(f"  원본:    {original}   free={original.free_indices}")

step1 = raise_index(original, "μ", metrics)
print(f"  raise:   {step1}   free={step1.free_indices}")

step2 = absorb_metric(step1, metrics)
print(f"  absorb:  {step2}   free={step2.free_indices}")
print()

# ============================================================
# 7. 에러 케이스
# ============================================================
print("=== 에러 케이스 ===\n")

try:
    raise_index(parse("V^{μ}", reg), "μ", metrics)
except ValueError as e:
    print(f"  이미 upper: {e}")

try:
    lower_index(parse("V_{μ}", reg), "μ", metrics)
except ValueError as e:
    print(f"  이미 lower: {e}")

try:
    raise_index(parse("V_{μ}", reg), "ν", metrics)
except ValueError as e:
    print(f"  없는 index: {e}")
