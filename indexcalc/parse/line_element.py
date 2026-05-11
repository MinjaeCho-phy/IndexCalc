"""ds² 라인 요소 파싱 → SymPy metric matrix.

LaTeX 또는 python 스타일 ds² 표현식을 받아 g_{μν} matrix 반환.

사용 예
-------
>>> g = parse_line_element("-dt**2 + dr**2 + r**2*(dtheta**2 + sin(theta)**2*dphi**2)",
...                        coords=["t", "r", "theta", "phi"])
>>> g.shape
(4, 4)

기본 규칙:
    - 좌표 이름 ``c`` 마다 differential 토큰 ``dc``를 인식 (단어 경계 검사).
    - ``dc**2`` 또는 ``dc*dc`` (대각): coefficient → g_{cc}.
    - ``dc1*dc2`` (비대각, c1≠c2): coefficient → 2 g_{c1c2} (textbook 컨벤션,
      대칭화); 결과 matrix에 g_{c1c2} = g_{c2c1} = coeff/2.
    - ``^`` 자동으로 ``**``로 변환 (LaTeX 호환).
    - SymPy 함수 (sin, cos, exp, sqrt, log) 자동 namespace에 포함.

LaTeX 백슬래시 (e.g., ``\\theta``)는 사용자가 미리 일반 이름 (``theta``)로
preprocessing 후 사용. 좌표 이름 자체에 backslash 포함은 미지원.
"""

from __future__ import annotations

import re

import sympy as sp


def _normalize_input(s: str) -> str:
    """LaTeX-style → python-style: ``^`` → ``**``."""
    return s.replace("^", "**")


def parse_line_element(
    line_str: str,
    coords: list[str],
    *,
    extra_symbols: dict[str, sp.Symbol] | None = None,
) -> sp.Matrix:
    """ds² 표현식을 g_{μν} SymPy matrix로 변환.

    Parameters
    ----------
    line_str : str
        ds²의 RHS만 (e.g., ``-dt**2 + dr**2 + r**2*dtheta**2``).
        ``ds^2 =`` 같은 prefix는 포함하지 말 것.
    coords : list[str]
        좌표 이름 순서. 결과 matrix의 행/열 순서를 정함.
    extra_symbols : dict[str, Symbol], optional
        파싱 namespace에 추가할 SymPy symbol/function. 기본은 좌표 + 표준
        수학 함수 (sin, cos, ...).

    Returns
    -------
    sp.Matrix
        Symmetric ``dim × dim`` metric matrix.

    Raises
    ------
    ValueError
        파싱 실패 또는 좌표 이름 충돌 시.
    """
    n = len(coords)
    if n == 0:
        raise ValueError("coords list cannot be empty")
    if len(set(coords)) != n:
        raise ValueError(f"coordinate names must be unique: {coords}")

    s = _normalize_input(line_str)

    # 사전 검사: 등록되지 않은 d<coord> 토큰이 있는지 (예: coords가 [x,y]인데
    # dz가 입력에 있으면 사용자 실수). d 다음에 alpha-id 형태의 모든 토큰 수집.
    all_d_tokens = set(re.findall(
        r"(?<![A-Za-z_0-9])d([A-Za-z_][A-Za-z_0-9]*)", s,
    ))
    unknown = sorted(t for t in all_d_tokens if t not in coords)
    if unknown:
        raise ValueError(
            f"unrecognized differential token(s) {['d' + t for t in unknown]!r}; "
            f"provided coords: {coords}"
        )

    # Differential symbols: d<coord>는 sp.Symbol("_d_<coord>")로 매핑
    d_syms = {c: sp.Symbol(f"_d_{c}", real=True) for c in coords}
    v_syms = {c: sp.Symbol(c, real=True) for c in coords}

    # 좌표 이름 token "dc" → "_d_c". 단, "d" + 좌표명만 정확히 매칭하기 위해 word boundary 사용.
    # 길이 내림차순으로 처리해 prefix 충돌 (예: "theta" vs "th") 회피.
    for c in sorted(coords, key=len, reverse=True):
        # \b 로 시작/끝 경계, escape coord name
        s = re.sub(rf"(?<![A-Za-z_0-9])d{re.escape(c)}(?![A-Za-z_0-9])",
                   d_syms[c].name, s)

    # Namespace
    namespace: dict[str, object] = {
        **{d.name: d for d in d_syms.values()},
        **{v.name: v for v in v_syms.values()},
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Rational": sp.Rational, "pi": sp.pi,
    }
    if extra_symbols:
        namespace.update(extra_symbols)

    try:
        expr = sp.sympify(s, locals=namespace)
    except (sp.SympifyError, SyntaxError) as e:
        raise ValueError(f"failed to parse line element: {e}") from e

    expr = sp.expand(expr)

    # Matrix 채우기
    g = sp.zeros(n, n)
    for i, ci in enumerate(coords):
        di = d_syms[ci]
        # 대각: coefficient of di**2
        coeff_diag = expr.coeff(di, 2)
        g[i, i] = sp.simplify(coeff_diag)

        for j in range(i + 1, n):
            cj = coords[j]
            dj = d_syms[cj]
            # 비대각: SymPy는 di*dj == dj*di (commutative)이므로 한 번만 추출.
            # ds² 컨벤션: di*dj 계수 = 2 g_{ij} → g_{ij} = coeff/2.
            coeff_ij = expr.coeff(di * dj)
            g_ij = sp.simplify(coeff_ij) / 2
            g[i, j] = g_ij
            g[j, i] = g_ij

    # 안전 검사: 모든 differential 항을 다 인식했는지 — 남은 d_ symbol이 expr에 있으면 경고
    used = set()
    for c in coords:
        used.add(d_syms[c])
    free = expr.free_symbols
    leftover = [sym for sym in free if sym.name.startswith("_d_") and sym not in used]
    if leftover:
        raise ValueError(
            f"unrecognized differential symbols in line element: {leftover}. "
            f"Provided coords: {coords}"
        )

    return g
