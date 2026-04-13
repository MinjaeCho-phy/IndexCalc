"""
LaTeX 출력 포매터: TensorExpr → LaTeX 문자열 변환.

물리학 표준 convention에 따라:
  - 연속 같은 위치 인덱스를 하나의 {}로 그룹핑
  - 그리스 문자 / 특수 기호를 LaTeX 명령으로 변환
  - 분수를 \\frac으로 표시
  - Trace는 기본적으로 contracted index 형태로 출력
"""

from __future__ import annotations
from fractions import Fraction
from indexcalc.core.index import Index
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.contract import Trace


# ─── 문자 → LaTeX 매핑 ──────────────────────────────────────

# Unicode combining marks → LaTeX decorator command.
# Mirrors _DECORATOR_COMBINING in parse/latex.py; both directions must agree.
_COMBINING_TO_DECORATOR = {
    "\u0304": "bar",
    "\u0302": "hat",
    "\u0303": "tilde",
    "\u0307": "dot",
    "\u0308": "ddot",
}


def _split_grapheme(name: str) -> tuple[str, list[str]]:
    """Split name into (base, [combining_marks]) using first grapheme only.

    "ē" (e + U+0304) → ("e", ["\\u0304"]).
    "e"              → ("e", []).
    "μ̄"              → ("μ", ["\\u0304"]).
    "ab"             → ("ab", [])  — no combining marks, multi-char kept whole.
    """
    if not name:
        return name, []
    marks = []
    # Find the first combining mark; everything before it is the base.
    base_end = 1
    while base_end < len(name) and not (
        0x0300 <= ord(name[base_end]) <= 0x036F
    ):
        base_end += 1
    base = name[:base_end]
    i = base_end
    while i < len(name) and 0x0300 <= ord(name[i]) <= 0x036F:
        marks.append(name[i])
        i += 1
    # Anything after the combining marks is returned as trailing base (rare)
    if i < len(name):
        base = base + "?" + name[i:]  # should not happen in practice
    return base, marks


def _wrap_decorators(latex_base: str, marks: list[str]) -> str:
    """Wrap a LaTeX base string with decorator commands from combining marks.

    marks=[U+0304, U+0302]  ⇒  \\hat{\\bar{<base>}}  (outermost last).
    Unknown combining marks pass through unchanged.
    """
    for mark in marks:
        decorator = _COMBINING_TO_DECORATOR.get(mark)
        if decorator:
            latex_base = f"\\{decorator}{{{latex_base}}}"
        else:
            latex_base = latex_base + mark
    return latex_base


_GREEK_MAP = {
    # lowercase
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # uppercase
    "Α": r"\Alpha", "Β": r"\Beta", "Γ": r"\Gamma", "Δ": r"\Delta",
    "Ε": r"\Epsilon", "Ζ": r"\Zeta", "Η": r"\Eta", "Θ": r"\Theta",
    "Ι": r"\Iota", "Κ": r"\Kappa", "Λ": r"\Lambda", "Μ": r"\Mu",
    "Ν": r"\Nu", "Ξ": r"\Xi", "Π": r"\Pi", "Ρ": r"\Rho",
    "Σ": r"\Sigma", "Τ": r"\Tau", "Υ": r"\Upsilon", "Φ": r"\Phi",
    "Χ": r"\Chi", "Ψ": r"\Psi", "Ω": r"\Omega",
}


def _latex_char(name: str) -> str:
    """인덱스 이름이나 텐서 이름의 단일 문자를 LaTeX로 변환한다.

    - 그리스 문자:    μ → \\mu
    - 장식된 문자:    p̄ → \\bar{p},  B̂ → \\hat{B}
    - dummy index:   μ_1 → \\mu_{1},  p̄_1 → \\bar{p}_{1}
    - 로마자:         그대로
    """
    # dummy index: base_number 형식
    if "_" in name:
        base, suffix = name.split("_", 1)
        base_core, marks = _split_grapheme(base)
        base_latex = _GREEK_MAP.get(base_core, base_core)
        base_latex = _wrap_decorators(base_latex, marks)
        return f"{base_latex}_{{{suffix}}}"

    base_core, marks = _split_grapheme(name)
    base_latex = _GREEK_MAP.get(base_core, base_core)
    return _wrap_decorators(base_latex, marks)


def _latex_tensor_name(name: str) -> str:
    """텐서 이름을 LaTeX로 변환한다.

    - 단일 그리스 문자:   η → \\eta
    - 장식된 문자:        ē → \\bar{e},  B̂ → \\hat{B}
    - 여러 글자 이름:      그대로 (이미 LaTeX 호환)
    - \\로 시작하면 그대로 (이미 LaTeX 명령)
    """
    if name.startswith("\\"):
        return name

    # Single grapheme (1 base + optional combining marks)
    base_core, marks = _split_grapheme(name)
    if base_core + "".join(marks) == name:
        # pure single-grapheme name
        base_latex = _GREEK_MAP.get(base_core, base_core)
        return _wrap_decorators(base_latex, marks)

    # Multi-character name: preserve as-is unless it's a known Greek word
    if name in _GREEK_MAP:
        return _GREEK_MAP[name]
    return name


# ─── 인덱스 그룹핑 ───────────────────────────────────────────

def _format_indices(indices: tuple[Index, ...] | list[Index]) -> str:
    """인덱스 리스트를 LaTeX 표기로 변환한다.

    연속된 같은 위치(upper/lower)의 인덱스를 하나의 {}로 그룹핑한다.

    Examples:
        [_μ, _ν]         → _{\\mu\\nu}
        [^μ, _ν]         → ^{\\mu}{}_{\\nu}
        [^μ, _ν, _λ, _ρ] → ^{\\mu}{}_{\\nu\\lambda\\rho}
    """
    if not indices:
        return ""

    # 연속 같은 위치끼리 그룹핑
    groups: list[tuple[str, list[str]]] = []
    for idx in indices:
        latex_name = _latex_char(idx.name)
        if groups and groups[-1][0] == idx.position:
            groups[-1][1].append(latex_name)
        else:
            groups.append((idx.position, [latex_name]))

    parts = []
    for i, (position, names) in enumerate(groups):
        prefix = "^" if position == "upper" else "_"
        content = " ".join(names)

        # 위치가 바뀔 때 {} 구분자 삽입 (첫 그룹 제외)
        if i > 0 and groups[i - 1][0] != position:
            parts.append("{}")

        parts.append(f"{prefix}{{{content}}}")

    return "".join(parts)


# ─── 스칼라 포매팅 ────────────────────────────────────────────

def _format_scalar(value) -> str:
    """스칼라 값을 LaTeX로 변환한다.

    깔끔한 분수이면 \\frac, 아니면 숫자 그대로.
    """
    if isinstance(value, int):
        return str(value)

    # float → 깔끔한 분수 시도
    try:
        frac = Fraction(value).limit_denominator(1000)
        if abs(float(frac) - value) < 1e-10:
            if frac.denominator == 1:
                return str(frac.numerator)
            sign = "-" if frac.numerator < 0 else ""
            num = abs(frac.numerator)
            den = frac.denominator
            return f"{sign}\\frac{{{num}}}{{{den}}}"
    except (ValueError, OverflowError):
        pass

    return f"{value}"


# ─── TensorExpr → LaTeX ──────────────────────────────────────

def to_latex(expr: TensorExpr) -> str:
    """TensorExpr을 LaTeX 문자열로 변환한다.

    Parameters
    ----------
    expr : TensorExpr
        변환할 표현식.

    Returns
    -------
    str
        LaTeX 문자열 ($ 없이).
    """
    if isinstance(expr, Tensor):
        name = _latex_tensor_name(expr.name)
        indices = _format_indices(expr.indices)
        return f"{name}{indices}"

    if isinstance(expr, TensorProduct):
        left = to_latex(expr.left)
        right = to_latex(expr.right)

        # 왼쪽이 합이면 괄호 필요
        if isinstance(expr.left, TensorSum):
            left = f"({left})"
        if isinstance(expr.right, TensorSum):
            right = f"({right})"

        return f"{left} {right}"

    if isinstance(expr, TensorSum):
        left = to_latex(expr.left)
        right_expr = expr.right

        # 오른쪽이 -1 * X 형태면 "- X"로 표시
        if isinstance(right_expr, ScalarMul) and right_expr.scalar == -1:
            right = to_latex(right_expr.expr)
            return f"{left} - {right}"
        if isinstance(right_expr, ScalarMul) and right_expr.scalar < 0:
            # -n * X → "- n X"
            pos_scalar = -right_expr.scalar
            inner = to_latex(right_expr.expr)
            coeff = _format_scalar(pos_scalar)
            if coeff == "1":
                return f"{left} - {inner}"
            return f"{left} - {coeff} {inner}"

        right = to_latex(right_expr)
        return f"{left} + {right}"

    if isinstance(expr, ScalarMul):
        coeff = _format_scalar(expr.scalar)
        inner = to_latex(expr.expr)

        if expr.scalar == 1:
            return inner
        if expr.scalar == -1:
            return f"-{inner}"

        # _ScalarOne 체크
        from indexcalc.parse.latex import _ScalarOne
        if isinstance(expr.expr, _ScalarOne):
            return coeff

        return f"{coeff} {inner}"

    if isinstance(expr, Trace):
        # 기본: contracted index 형태로 출력 (Einstein convention)
        return to_latex(expr.tensor)

    # ZeroTensor: 0
    from indexcalc.core.variation import Variation, ZeroTensor
    if isinstance(expr, ZeroTensor):
        return "0"

    # Variation: δ(expr)
    if isinstance(expr, Variation):
        inner = to_latex(expr.expr)
        return f"\\delta({inner})"

    # PartialDeriv: ∂_μ T^ν_λ
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
    if isinstance(expr, PartialDeriv):
        idx_latex = _latex_char(expr.deriv_index.name)
        inner = to_latex(expr.expr)
        if isinstance(expr.expr, (TensorProduct, TensorSum)):
            inner = f"({inner})"
        return f"\\partial_{{{idx_latex}}} {inner}"

    # SpatialCovariantDeriv: D_i T (CovariantDeriv의 subclass이므로 먼저 검사)
    from indexcalc.core.spatial_deriv import SpatialCovariantDeriv
    if isinstance(expr, SpatialCovariantDeriv):
        idx_latex = _latex_char(expr.deriv_index.name)
        inner = to_latex(expr.expr)
        if isinstance(expr.expr, (TensorProduct, TensorSum)):
            inner = f"({inner})"
        return f"D_{{{idx_latex}}} {inner}"

    # CovariantDeriv: ∇_μ T^ν_λ
    if isinstance(expr, CovariantDeriv):
        idx_latex = _latex_char(expr.deriv_index.name)
        inner = to_latex(expr.expr)
        if isinstance(expr.expr, (TensorProduct, TensorSum)):
            inner = f"({inner})"
        return f"\\nabla_{{{idx_latex}}} {inner}"

    # 폴백
    return str(expr)
