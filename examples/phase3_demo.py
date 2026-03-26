"""
Phase 3 데모: Contraction 분석, Trace, Einstein convention 검증.
"""

from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse,
    validate_einstein, collect_tensors, collect_all_indices,
    trace, summarize,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

# ============================================================
# 1. validate_einstein — 인덱스 구조 분석
# ============================================================
print("=== Einstein Convention 검증 ===\n")

# 정상: T^μ_ν S^ν_λ  (ν contracted, μ λ free)
expr = parse("T^{μ}_{ν} S^{ν}_{λ}", reg)
info = validate_einstein(expr)
print(f"  {expr}")
print(f"  valid: {info['valid']}")
print(f"  free: {info['free']}")
print(f"  contracted: {info['contracted']}")
print()

# 정상: 3개 텐서 연쇄 contraction
expr2 = parse("T^{μ}_{ν} g^{νλ} S_{λρ}", reg)
info2 = validate_einstein(expr2)
print(f"  {expr2}")
print(f"  valid: {info2['valid']}")
print(f"  free: {[str(i) for i in info2['free']]}")
print(f"  contracted: {[(a.name, a.space.name) for a, b in info2['contracted']]}")
print()

# ============================================================
# 2. Einstein convention 위반 감지
# ============================================================
print("=== Convention 위반 감지 ===\n")

# 위반: 같은 위치에서 반복 (둘 다 upper)
bad = Tensor("A", [spacetime.upper("μ"), spacetime.lower("ν")]) * \
      Tensor("B", [spacetime.upper("μ"), spacetime.lower("λ")])
info_bad = validate_einstein(bad)
print(f"  {bad}")
print(f"  valid: {info_bad['valid']}")
for err in info_bad["errors"]:
    print(f"    ⚠ {err}")
print()

# ============================================================
# 3. collect_tensors — 표현식에서 텐서 추출
# ============================================================
print("=== 텐서 수집 ===\n")

expr3 = parse("T^{μ}_{ν} g^{νλ} S_{λρ}", reg)
tensors = collect_tensors(expr3)
print(f"  Expression: {expr3}")
print(f"  Tensors: {tensors}")
print(f"  All indices: {collect_all_indices(expr3)}")
print()

# ============================================================
# 4. Trace — 같은 텐서 내 contraction
# ============================================================
print("=== Trace ===\n")

# T^μ_μ → scalar (trace of mixed tensor)
T_mixed = Tensor("T", [spacetime.upper("μ"), spacetime.lower("μ")])
tr = trace(T_mixed, "μ")
print(f"  T = {T_mixed}")
print(f"  Tr(T) = {tr}")
print(f"  rank: {tr.rank}  (scalar)")
print()

# R^μ_ν^ν_λ → trace over ν, leaves μ and λ
R = Tensor("R", [
    spacetime.upper("μ"),
    spacetime.lower("ν"),
    spacetime.upper("ν"),
    spacetime.lower("λ"),
])
tr_R = trace(R, "ν")
print(f"  R = {R}")
print(f"  Tr_ν(R) = {tr_R}")
print(f"  free: {tr_R.free_indices}")
print(f"  rank: {tr_R.rank}")
print()

# Trace 에러: 같은 위치 2개
try:
    bad_tensor = Tensor("X", [spacetime.upper("μ"), spacetime.upper("μ")])
    trace(bad_tensor, "μ")
except ValueError as e:
    print(f"  예상된 에러: {e}")
print()

# ============================================================
# 5. summarize — 한눈에 보기
# ============================================================
print("=== 표현식 요약 ===\n")

# Vielbein 곱
expr4 = parse("e^{a}_{μ} e^{b}^{μ}", reg)
print(summarize(expr4))
print()

# 복합 표현식
expr5 = parse("T^{μ}_{ν} g^{νλ} S_{λρ}", reg)
print(summarize(expr5))
print()

# Metric lowering
expr6 = parse("g_{μν} V^{ν}", reg)
print(summarize(expr6))
