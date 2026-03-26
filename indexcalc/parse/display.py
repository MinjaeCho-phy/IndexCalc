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

    - 그리스 문자: μ → \\mu
    - dummy index: μ_1 → \\mu_1
    - 로마자: 그대로
    """
    # dummy index: base_number 형식
    if "_" in name:
        base, suffix = name.split("_", 1)
        base_latex = _GREEK_MAP.get(base, base)
        return f"{base_latex}_{{{suffix}}}"

    return _GREEK_MAP.get(name, name)


def _latex_tensor_name(name: str) -> str:
    """텐서 이름을 LaTeX로 변환한다.

    - 단일 그리스 문자: η → \\eta
    - 여러 글자 이름: 그대로 (이미 LaTeX 호환)
    - \\로 시작하면 그대로 (이미 LaTeX 명령)
    """
    if name.startswith("\\"):
        return name
    if len(name) == 1:
        return _GREEK_MAP.get(name, name)
    # 여러 글자인데 첫 글자가 그리스면 변환
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

    # PartialDeriv: ∂_μ T^ν_λ
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
    if isinstance(expr, PartialDeriv):
        idx_latex = _latex_char(expr.deriv_index.name)
        inner = to_latex(expr.expr)
        if isinstance(expr.expr, (TensorProduct, TensorSum)):
            inner = f"({inner})"
        return f"\\partial_{{{idx_latex}}} {inner}"

    # CovariantDeriv: ∇_μ T^ν_λ
    if isinstance(expr, CovariantDeriv):
        idx_latex = _latex_char(expr.deriv_index.name)
        inner = to_latex(expr.expr)
        if isinstance(expr.expr, (TensorProduct, TensorSum)):
            inner = f"({inner})"
        return f"\\nabla_{{{idx_latex}}} {inner}"

    # 폴백
    return str(expr)
