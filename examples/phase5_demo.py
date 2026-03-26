"""
Phase 5 데모: LaTeX 출력.
"""

from indexcalc import (
    IndexSpace, Tensor, IndexRegistry, parse, to_latex,
    MetricRegistry, raise_index, lower_index, absorb_metric,
    trace,
)

# ─── Setup ───────────────────────────────────────────────────
spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
lorentz   = IndexSpace("lorentz",   dim=4, indices="abcde", metric="η")

reg = IndexRegistry()
reg.register(spacetime)
reg.register(lorentz)

g     = Tensor("g", [spacetime.lower("μ"), spacetime.lower("ν")])
g_inv = Tensor("g", [spacetime.upper("μ"), spacetime.upper("ν")])
eta     = Tensor("η", [lorentz.lower("a"), lorentz.lower("b")])
eta_inv = Tensor("η", [lorentz.upper("a"), lorentz.upper("b")])

metrics = MetricRegistry()
metrics.register(g, g_inv, spacetime)
metrics.register(eta, eta_inv, lorentz)

# ============================================================
# 1. 단일 텐서
# ============================================================
print("=== 단일 텐서 ===\n")

cases = [
    "T^{μ}_{ν}",
    "g_{μν}",
    "V^μ",
    "R^{μ}_{νλρ}",
    "η_{ab}",
]
for tex in cases:
    expr = parse(tex, reg)
    print(f"  {tex:25s} →  {to_latex(expr)}")
print()

# ============================================================
# 2. 텐서곱 & contraction
# ============================================================
print("=== 텐서곱 ===\n")

products = [
    "T^{μ}_{ν} S^{ν}_{λ}",
    "g_{μν} V^{ν}",
    "e^{a}_{μ} e^{b}^{μ}",
    "T^{μ}_{ν} g^{νλ} S_{λρ}",
]
for tex in products:
    expr = parse(tex, reg)
    print(f"  {tex:35s} →  {to_latex(expr)}")
print()

# ============================================================
# 3. 스칼라곱 & 분수
# ============================================================
print("=== 스칼라곱 ===\n")

scalars = [
    "2 T^{μ}_{ν}",
    "-T^{μ}_{ν}",
    r"\frac{1}{2} T^{μ}_{ν}",
    r"\frac{3}{4} g_{μν}",
]
for tex in scalars:
    expr = parse(tex, reg)
    print(f"  {tex:35s} →  {to_latex(expr)}")
print()

# ============================================================
# 4. 합과 차
# ============================================================
print("=== 합/차 ===\n")

sums = [
    "A^{μ}_{ν} + B^{μ}_{ν}",
    "A^{μ}_{ν} - B^{μ}_{ν}",
    r"\frac{1}{2} A^{μ}_{ν} - \frac{1}{3} B^{μ}_{ν}",
]
for tex in sums:
    expr = parse(tex, reg)
    print(f"  {tex:50s} →  {to_latex(expr)}")
print()

# ============================================================
# 5. 괄호
# ============================================================
print("=== 괄호 ===\n")

expr = parse("(A^{μ}_{ν} + B^{μ}_{ν}) V^{ν}", reg)
print(f"  (A + B) V^ν  →  {to_latex(expr)}")
print()

# ============================================================
# 6. Metric raise → LaTeX
# ============================================================
print("=== raise/lower → LaTeX ===\n")

V_low = parse("V_{μ}", reg)
raised = raise_index(V_low, "μ", metrics)
print(f"  raise V_μ   →  {to_latex(raised)}")

V_up = parse("V^{μ}", reg)
lowered = lower_index(V_up, "μ", metrics)
print(f"  lower V^μ   →  {to_latex(lowered)}")

# absorb 후
absorbed = absorb_metric(parse("g_{μν} V^{ν}", reg), metrics)
print(f"  absorb g_μν V^ν  →  {to_latex(absorbed)}")
print()

# ============================================================
# 7. Trace
# ============================================================
print("=== Trace ===\n")

T_trace = Tensor("T", [spacetime.upper("μ"), spacetime.lower("μ")])
tr = trace(T_trace, "μ")
print(f"  Tr(T^μ_μ)  →  {to_latex(tr)}")

R = Tensor("R", [
    spacetime.upper("μ"),
    lorentz.upper("a"),
    spacetime.lower("μ"),
    lorentz.lower("b"),
])
tr_R = trace(R, "μ")
print(f"  Tr_μ(R^μa_μb)  →  {to_latex(tr_R)}")
print()

# ============================================================
# 8. _repr_latex_ (Jupyter에서 자동 렌더링될 문자열)
# ============================================================
print("=== _repr_latex_ ===\n")

expr = parse("T^{μ}_{ν} g^{νλ} S_{λρ}", reg)
print(f"  {expr._repr_latex_()}")

expr2 = parse(r"\frac{1}{2} R^{μ}_{νλρ}", reg)
print(f"  {expr2._repr_latex_()}")
