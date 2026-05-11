"""SymbolicCurvatureResult.latex(...) 출력을 component dict로 재파싱.

``\\Gamma^{x}_{yz} &= ...`` 같은 LaTeX equation block을 해석해
``{(x, y, z): sp.Expr}`` 형태로 변환한다. round-trip 검증·외부 인용·재계산
편의용.

지원 텐서:
    - ``christoffel`` (Γ^σ_{μν})  → key (σ, μ, ν) 의 정수 인덱스
    - ``riemann``    (R^ρ_{σμν})  → key (ρ, σ, μ, ν)
    - ``ricci``      (R_{μν})     → key (μ, ν)
    - ``einstein``   (G_{μν})     → key (μ, ν)
    - ``metric``     (g_{μν})     → key (μ, ν)
    - ``ricci_scalar`` / ``kretschner`` → 단일 sp.Expr 반환

좌표 인덱스 → 정수 매핑은 ``coord_names`` 인자로 결정.
"""

from __future__ import annotations

import re

import sympy as sp


# 텐서별 prefix 패턴
_TENSOR_PATTERNS = {
    "christoffel": (r"\\Gamma\^\{([^}]*)\}_\{([^}]*)\}", 3),  # σ, μν → 3 indices
    "riemann":     (r"R\^\{([^}]*)\}_\{([^}]*)\}", 4),         # ρ, σμν → 4 indices
    "ricci":       (r"R_\{([^}]*)\}", 2),                      # μν → 2 indices
    "einstein":    (r"G_\{([^}]*)\}", 2),                      # μν → 2 indices
    "metric":      (r"g_\{([^}]*)\}", 2),                      # μν → 2 indices
}


def _split_index_string(s: str, expected: int, names_lookup: dict[str, int]) -> tuple[int, ...]:
    """Index string (e.g., ``"r theta"`` 또는 ``"rθ"``)을 좌표명 시퀀스로 분리.

    먼저 공백 split 시도; 그 실패하면 longest-match로 좌표명 추출.
    """
    s = s.strip()
    if " " in s:
        parts = [p for p in s.split() if p]
    else:
        # longest-match
        parts = []
        i = 0
        while i < len(s):
            matched = None
            # 길이 내림차순으로 매칭
            for name in sorted(names_lookup.keys(), key=len, reverse=True):
                if s.startswith(name, i):
                    matched = name
                    break
            if matched is None:
                raise ValueError(
                    f"failed to tokenize index string {s!r} with names "
                    f"{list(names_lookup.keys())}"
                )
            parts.append(matched)
            i += len(matched)
    if len(parts) != expected:
        raise ValueError(
            f"expected {expected} indices, got {len(parts)} from {s!r}"
        )
    return tuple(names_lookup[p] for p in parts)


def _strip_align_block(s: str) -> str:
    """``\\begin{aligned} ... \\end{aligned}`` 또는 align 환경의 본문 추출."""
    m = re.search(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", s, re.DOTALL)
    if m:
        return m.group(1)
    return s


def _parse_sympy_rhs(rhs: str, namespace: dict) -> sp.Expr:
    """LaTeX RHS 일부분(e.g., ``\\frac{2 M}{r^{2}}``)을 SymPy로 해석.

    sympy.parsing.latex이 있으면 그것을 우선; 없거나 실패하면 단순화된 변환
    (``\\frac{a}{b}`` → ``(a)/(b)``, ``^`` → ``**``, ``\\theta`` → ``theta`` 등)
    후 sympify.
    """
    # 1) sympy.parsing.latex (antlr 필요)
    try:
        from sympy.parsing.latex import parse_latex
        try:
            return parse_latex(rhs)
        except Exception:
            pass
    except ImportError:
        pass

    # 2) fallback: minimal LaTeX → python 변환
    s = rhs.strip()
    # \\frac{a}{b} → (a)/(b)  (단일 레벨; nested는 반복 적용)
    while True:
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    # 일반적인 LaTeX 변수 → python 식별자
    replacements = {
        r"\\theta": "theta", r"\\varphi": "phi", r"\\phi": "phi",
        r"\\rho": "rho", r"\\sigma": "sigma", r"\\mu": "mu",
        r"\\nu": "nu", r"\\lambda": "lam", r"\\alpha": "alpha",
        r"\\beta": "beta", r"\\gamma": "gamma",
        r"\\sin": "sin", r"\\cos": "cos", r"\\tan": "tan",
        r"\\exp": "exp", r"\\log": "log", r"\\sqrt": "sqrt",
        r"\\left": "", r"\\right": "",
        r"\\,": " ", r"\\!": "",
    }
    for k, v in replacements.items():
        s = re.sub(k, v, s)
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    try:
        return sp.sympify(s, locals=namespace)
    except (sp.SympifyError, SyntaxError) as e:
        raise ValueError(f"failed to parse RHS {rhs!r}: {e}") from e


def parse_curvature_components(
    latex_str: str,
    coord_names: list[str],
    *,
    tensor: str = "christoffel",
) -> dict[tuple[int, ...], sp.Expr] | sp.Expr:
    """SymbolicCurvatureResult.latex(tensor) 출력을 컴포넌트 dict로 변환.

    Parameters
    ----------
    latex_str : str
        ``SymbolicCurvatureResult.latex(tensor=...)`` 의 반환값.
    coord_names : list[str]
        좌표 이름 (LaTeX 표기 기준 — 예: ``"theta"`` not ``"\\theta"``).
        ``(좌표명) → 정수 인덱스`` 매핑에 사용.
    tensor : str
        대상 텐서 종류 — ``christoffel``, ``riemann``, ``ricci``, ``einstein``,
        ``metric``, ``ricci_scalar``, ``kretschner`` 중 하나.

    Returns
    -------
    dict 또는 sp.Expr
        component tensor의 경우 ``{(slot indices tuple): sp.Expr}``.
        scalar 텐서 (ricci_scalar/kretschner)의 경우 단일 ``sp.Expr``.
    """
    if tensor in ("ricci_scalar", "kretschner"):
        # "R = ..." 또는 "K = ..." 형태
        m = re.match(r"\s*[RK]\s*=\s*(.+)", latex_str.strip(), re.DOTALL)
        if m is None:
            raise ValueError(f"unable to parse scalar form: {latex_str[:80]}")
        ns = {n: sp.Symbol(n, real=True) for n in coord_names}
        return _parse_sympy_rhs(m.group(1), ns)

    if tensor not in _TENSOR_PATTERNS:
        raise ValueError(
            f"unknown tensor {tensor!r}; supported: "
            f"{list(_TENSOR_PATTERNS) + ['ricci_scalar', 'kretschner']}"
        )

    pattern, n_indices = _TENSOR_PATTERNS[tensor]
    body = _strip_align_block(latex_str)

    # 라인 단위로 split (\\\\ 또는 줄바꿈)
    lines = re.split(r"\\\\|\n", body)

    # 이름 → int 매핑. LaTeX backslash 변형 (\\theta, \\varphi 등)도 같은 인덱스로
    # 매핑해서 SymbolicCurvatureResult.latex() round-trip이 동작하도록 한다.
    _LATEX_VARIANTS = {
        "theta": [r"\theta"], "phi": [r"\varphi", r"\phi"],
        "varphi": [r"\varphi"], "rho": [r"\rho"], "psi": [r"\psi"],
        "alpha": [r"\alpha"], "beta": [r"\beta"], "gamma": [r"\gamma"],
        "lambda": [r"\lambda"], "mu": [r"\mu"], "nu": [r"\nu"],
        "sigma": [r"\sigma"], "tau": [r"\tau"], "omega": [r"\omega"],
    }
    name_to_int: dict[str, int] = {}
    for i, n in enumerate(coord_names):
        name_to_int[n] = i
        for variant in _LATEX_VARIANTS.get(n, []):
            name_to_int[variant] = i
    namespace = {n: sp.Symbol(n, real=True) for n in coord_names}

    out: dict[tuple[int, ...], sp.Expr] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.search(rf"{pattern}\s*&?=\s*(.+)", line)
        if m is None:
            continue
        # m.groups(): 텐서별로 1~2개의 index group + 마지막 RHS
        # christoffel: 2 groups (σ, μν), 마지막은 RHS
        # ricci: 1 group (μν), 마지막은 RHS
        groups = m.groups()
        index_strs = groups[:-1]
        rhs = groups[-1]

        # 모든 index strings 합쳐서 n_indices개로 토큰화
        # christoffel: groups[0] = σ, groups[1] = "μν"; concat 후 split
        # ricci: groups[0] = "μν"; split
        # riemann: groups[0] = "ρ", groups[1] = "σμν"
        all_index_str = " ".join(index_strs)
        try:
            slot_tuple = _split_index_string(all_index_str, n_indices, name_to_int)
        except ValueError:
            # 첫 group + 두 번째 group 별도 분리
            try:
                upper = _split_index_string(index_strs[0], 1, name_to_int) if len(index_strs) >= 1 else ()
                lower = _split_index_string(index_strs[1], n_indices - 1, name_to_int) if len(index_strs) >= 2 else ()
                slot_tuple = upper + lower
            except ValueError as e:
                raise ValueError(f"failed to parse index in line {line!r}: {e}") from e

        rhs = rhs.strip().rstrip(",")
        if rhs.endswith(r"\\"):
            rhs = rhs[:-2].strip()
        out[slot_tuple] = _parse_sympy_rhs(rhs, namespace)

    return out
