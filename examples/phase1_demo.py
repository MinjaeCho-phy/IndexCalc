"""
Phase 1 데모: IndexSpace, Index, TensorExpr 기본 사용법.

이 스크립트는 IndexCalc의 핵심 자료구조가 어떻게 동작하는지 보여준다.
"""

from indexcalc import IndexSpace, Tensor

# ============================================================
# 1. IndexSpace 정의
# ============================================================
# 각 index가 "어떤 공간에 속하는지"를 먼저 정의한다.
# indices 문자열은 convention 표시용이고, metric은 나중에 raise/lower에 쓰인다.

spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

print("=== Index Spaces ===")
print(f"  {spacetime}")
print(f"  {lorentz}")
print()

# ============================================================
# 2. Index 생성
# ============================================================
# IndexSpace.upper("μ") → contravariant index ^μ
# IndexSpace.lower("μ") → covariant index _μ

mu  = spacetime.upper("μ")
nu  = spacetime.lower("ν")
lam = spacetime.lower("λ")
a   = lorentz.upper("a")
b   = lorentz.lower("b")

print("=== Indices ===")
print(f"  mu  = {mu}   (space: {mu.space.name}, position: {mu.position})")
print(f"  nu  = {nu}   (space: {nu.space.name}, position: {nu.position})")
print(f"  mu.flip() = {mu.flip()}")
print(f"  mu contracts with _μ? {mu.contracts_with(spacetime.lower('μ'))}")
print(f"  mu contracts with ^μ? {mu.contracts_with(spacetime.upper('μ'))}")
print(f"  mu contracts with _a? {mu.contracts_with(lorentz.lower('μ'))}")
print()

# ============================================================
# 3. Tensor 생성
# ============================================================
# Tensor는 이름 + 인덱스 목록으로 정의한다.
# repr이 물리학 표기법을 따른다: T^μ_ν

T = Tensor("T", [mu, nu])                         # T^μ_ν  (1,1)-tensor
g = Tensor("g", [spacetime.lower("μ"), nu])        # g_μν   metric
R = Tensor("R", [mu, nu, lam, spacetime.lower("ρ")])  # R^μ_ν_λ_ρ  Riemann-like

print("=== Tensors ===")
print(f"  T = {T}        rank = {T.rank}")
print(f"  g = {g}      rank = {g.rank}")
print(f"  R = {R}  rank = {R.rank}")
print()

# ============================================================
# 4. TensorProduct & 자동 Contraction
# ============================================================
# * 연산자로 텐서곱을 만든다.
# 같은 이름 + 같은 공간 + 반대 위치(upper↔lower)인 인덱스 쌍이
# 자동으로 contraction(Einstein summation)으로 인식된다.

# T^μ_ν * S^ν_λ  →  ν가 contracted, free indices = [^μ, _λ]
S = Tensor("S", [spacetime.upper("ν"), lam])  # S^ν_λ
product = T * S

print("=== TensorProduct (자동 contraction) ===")
print(f"  T = {T}")
print(f"  S = {S}")
print(f"  T * S = {product}")
print(f"  Free indices: {product.free_indices}")
print(f"  Contracted pairs: {product.contracted_pairs}")
print(f"  Result rank: {product.rank}")
print()

# contraction이 없는 경우: T^μ_ν * V^λ  →  free indices 3개
V = Tensor("V", [spacetime.upper("λ")])
no_contract = T * V
print(f"  T * V = {no_contract}")
print(f"  Free indices: {no_contract.free_indices}  (no contraction)")
print()

# ============================================================
# 5. 다른 공간의 인덱스는 축약되지 않는다
# ============================================================
# spacetime의 _a와 lorentz의 ^a는 이름이 같아도 공간이 다르므로 축약 안 됨

X = Tensor("X", [spacetime.lower("a")])  # spacetime의 _a
Y = Tensor("Y", [lorentz.upper("a")])    # lorentz의 ^a
mixed = X * Y

print("=== 다른 공간은 축약 안 됨 ===")
print(f"  X = {X}  (space: {X.indices[0].space.name})")
print(f"  Y = {Y}  (space: {Y.indices[0].space.name})")
print(f"  X * Y = {mixed}")
print(f"  Contracted: {mixed.contracted_pairs}  (빈 리스트 — 공간이 다르므로)")
print()

# ============================================================
# 6. Vielbein 예시: 두 공간을 잇는 텐서
# ============================================================
# e^a_μ: lorentz upper a, spacetime lower μ

e = Tensor("e", [a, spacetime.lower("μ")])
print("=== Vielbein ===")
print(f"  e = {e}  (lorentz ^a, spacetime _μ)")

# e^a_μ * e^b_?  — 여기서 μ를 공유하려면 e의 두 번째 인덱스가 upper여야
e2 = Tensor("e", [lorentz.upper("b"), spacetime.upper("μ")])
ee = e * e2
print(f"  e^a_μ * e^b^μ = {ee}")
print(f"  Contracted: {[p[0].name for p in ee.contracted_pairs]}")
print(f"  Free: {ee.free_indices}  → lorentz 인덱스만 남음")
print()

# ============================================================
# 7. 산술 연산: 합, 차, 스칼라곱
# ============================================================

A = Tensor("A", [mu, nu])
B = Tensor("B", [mu, nu])

print("=== 산술 연산 ===")
print(f"  A + B = {A + B}")
print(f"  A - B = {A - B}")
print(f"  3 * A = {3 * A}")
print(f"  -A    = {-A}")
print()

# Free index 개수가 다르면 에러
try:
    _ = A + V  # (1,1) + (1,0) → 에러
except ValueError as err:
    print(f"  A + V → ValueError: {err}")
print()

# ============================================================
# 8. 표현식 체이닝
# ============================================================
# 복합 표현식도 자연스럽게 구성된다.

expr = 2 * T * S + (-1) * Tensor("U", [mu, lam])
print("=== 복합 표현식 ===")
print(f"  2*T*S - U = {expr}")
print(f"  rank = {expr.rank}")
