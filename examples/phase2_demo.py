"""
Phase 2 데모: LaTeX 파서 사용법.

LaTeX 문자열을 TensorExpr로 파싱하는 과정을 보여준다.
"""

from indexcalc import IndexSpace
from indexcalc.parse.latex import IndexRegistry, parse

# ============================================================
# 1. IndexSpace 정의 & 레지스트리 등록
# ============================================================

spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

print("=== 레지스트리 ===")
print(f"  'μ' → {reg.resolve('μ')}")
print(f"  'a' → {reg.resolve('a')}")
print()

# ============================================================
# 2. 단일 텐서 파싱
# ============================================================
print("=== 단일 텐서 ===")

# 기본: 중괄호 표기
t1 = parse("T^{μ}_{ν}", reg)
print(f"  T^{{μ}}_{{ν}}  →  {t1}   rank={t1.rank}")

# 중괄호 안에 여러 인덱스 (e.g., metric)
t2 = parse("g_{μν}", reg)
print(f"  g_{{μν}}       →  {t2}   rank={t2.rank}")

# 단일 문자면 중괄호 생략 가능
t3 = parse("V^μ", reg)
print(f"  V^μ            →  {t3}    rank={t3.rank}")

# Riemann-like
t4 = parse("R^{μ}_{νλρ}", reg)
print(f"  R^{{μ}}_{{νλρ}}  →  {t4}   rank={t4.rank}")
print()

# ============================================================
# 3. 텐서곱 & 자동 contraction
# ============================================================
print("=== 텐서곱 (자동 contraction) ===")

# T^μ_ν S^ν_λ  →  ν 축약
expr1 = parse("T^{μ}_{ν} S^{ν}_{λ}", reg)
print(f"  T^{{μ}}_{{ν}} S^{{ν}}_{{λ}}  →  {expr1}")
print(f"    free: {expr1.free_indices}   rank={expr1.rank}")

# Vielbein 곱: e^a_μ e^b^μ  → μ 축약, lorentz 인덱스만 남음
expr2 = parse("e^{a}_{μ} e^{b}^{μ}", reg)
print(f"  e^{{a}}_{{μ}} e^{{b}}^{{μ}}  →  {expr2}")
print(f"    free: {expr2.free_indices}")
print()

# ============================================================
# 4. 스칼라곱
# ============================================================
print("=== 스칼라곱 ===")

s1 = parse("2 T^{μ}_{ν}", reg)
print(f"  2 T^{{μ}}_{{ν}}           →  {s1}")

s2 = parse("-T^{μ}_{ν}", reg)
print(f"  -T^{{μ}}_{{ν}}            →  {s2}")

s3 = parse(r"\frac{1}{2} T^{μ}_{ν}", reg)
print(f"  \\frac{{1}}{{2}} T^{{μ}}_{{ν}}  →  {s3}")
print()

# ============================================================
# 5. 합과 차
# ============================================================
print("=== 합/차 ===")

sum1 = parse("A^{μ}_{ν} + B^{μ}_{ν}", reg)
print(f"  A + B  →  {sum1}")

diff1 = parse("A^{μ}_{ν} - B^{μ}_{ν}", reg)
print(f"  A - B  →  {diff1}")
print()

# ============================================================
# 6. 괄호
# ============================================================
print("=== 괄호 ===")

p1 = parse("(A^{μ}_{ν} + B^{μ}_{ν}) V^{ν}", reg)
print(f"  (A + B) V^ν  →  {p1}")
print(f"    free: {p1.free_indices}")
print()

# ============================================================
# 7. 복합 표현식
# ============================================================
print("=== 복합 표현식 ===")

# 물리학에서 흔한 형태: metric contraction
expr3 = parse("g_{μν} V^{ν}", reg)
print(f"  g_{{μν}} V^{{ν}}  →  {expr3}")
print(f"    free: {expr3.free_indices}  (= V_μ, covector)")

# 세 텐서의 곱
expr4 = parse("T^{μ}_{ν} g^{νλ} S_{λρ}", reg)
print(f"  T^μ_ν g^{{νλ}} S_{{λρ}}  →  {expr4}")
print(f"    free: {expr4.free_indices}")
