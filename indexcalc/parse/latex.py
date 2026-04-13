"""
LaTeX 파서: LaTeX 문자열을 TensorExpr로 변환한다.

지원하는 문법:
  - 단일 텐서:    "T^{μ}_{ν}",  "g_{μν}",  "V^{μ}"
  - 축약 표기:    "T^μ_ν"  (단일 문자면 중괄호 생략 가능)
  - 텐서곱:       "T^{μ}_{ν} S^{ν}_{λ}"  (공백으로 구분)
  - 스칼라곱:     "2 T^{μ}_{ν}",  "-T^{μ}_{ν}"
  - 합/차:        "T^{μ}_{ν} + S^{μ}_{ν}",  "A^{μ} - B^{μ}"
  - 괄호:         "(T^{μ}_{ν} + S^{μ}_{ν}) V^{ν}"
  - 분수:         "\\frac{1}{2} T^{μ}_{ν}"
  - 편미분:       "\\partial_{μ} V^{ν}",  "\\partial_{\\mu} V^{\\nu}"
  - 공변미분:     "\\nabla_{μ} V^{ν}",  "\\nabla_{\\mu} V^{\\nu}"
  - LaTeX 그리스: "T^{\\mu}_{\\nu}" (Unicode와 LaTeX 명령 모두 지원)

인덱스-공간 매핑은 IndexRegistry를 통해 관리한다.
파서는 등록된 인덱스 문자를 보고 어떤 공간에 속하는지 자동으로 판단한다.
"""

from __future__ import annotations
import re
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul, TensorExpr
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv, Connection
from indexcalc.core.spatial_deriv import SpatialCovariantDeriv


class IndexRegistry:
    """인덱스 문자 → IndexSpace 매핑을 관리한다.

    IndexSpace를 등록하면, 그 space의 indices 문자열에 포함된 각 문자가
    해당 공간에 매핑된다. 파서가 인덱스 문자를 만나면 이 레지스트리를 참조하여
    어느 공간에 속하는지 결정한다.

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> lr = IndexSpace("lorentz", dim=4, indices="abcde", metric="η")
    >>> reg = IndexRegistry()
    >>> reg.register(st)
    >>> reg.register(lr)
    >>> reg.resolve("μ")
    IndexSpace('spacetime', dim=4)
    """

    def __init__(self):
        self._map: dict[str, IndexSpace] = {}

    def register(self, space: IndexSpace) -> None:
        """IndexSpace를 등록한다. space.indices의 각 grapheme를 매핑에 추가.

        A 'grapheme' is a base character plus any following Unicode combining
        marks — so a space declared with ``indices="p̄q̄"`` (e + U+0304 each)
        registers two distinct index characters rather than four codepoints.
        """
        for ch in _iter_graphemes(space.indices):
            if ch in self._map and self._map[ch] != space:
                raise ValueError(
                    f"Index character '{ch}' is already registered to "
                    f"'{self._map[ch].name}', cannot register to '{space.name}'"
                )
            self._map[ch] = space

    def resolve(self, name: str) -> IndexSpace:
        """인덱스 문자를 IndexSpace로 해석한다.

        dummy index(μ_1 등)는 base 문자(μ)로 공간을 찾는다.
        """
        if name in self._map:
            return self._map[name]

        # dummy index: base_suffix 형식
        if "_" in name:
            base = name.split("_", 1)[0]
            if base in self._map:
                return self._map[base]

        raise KeyError(
            f"Unknown index '{name}'. "
            f"Register an IndexSpace containing this character first."
        )

    def make_index(self, name: str, position: str) -> Index:
        """인덱스 문자와 위치로 Index 객체를 생성한다."""
        space = self.resolve(name)
        return Index(name, space, position)


# ─── LaTeX → Unicode 역매핑 ──────────────────────────────────

_LATEX_TO_UNICODE = {
    # lowercase
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    # uppercase
    r"\Alpha": "Α", r"\Beta": "Β", r"\Gamma": "Γ", r"\Delta": "Δ",
    r"\Epsilon": "Ε", r"\Zeta": "Ζ", r"\Eta": "Η", r"\Theta": "Θ",
    r"\Iota": "Ι", r"\Kappa": "Κ", r"\Lambda": "Λ", r"\Mu": "Μ",
    r"\Nu": "Ν", r"\Xi": "Ξ", r"\Pi": "Π", r"\Rho": "Ρ",
    r"\Sigma": "Σ", r"\Tau": "Τ", r"\Upsilon": "Υ", r"\Phi": "Φ",
    r"\Chi": "Χ", r"\Psi": "Ψ", r"\Omega": "Ω",
}

_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")


# ─── Decorator commands (\bar, \hat, \tilde, \dot, \ddot) ────
#
# \bar{X} is represented by appending a Unicode combining mark to the
# base character, producing a multi-codepoint name like "ē" = "e" + U+0304.
# Tensor and index names can thereby carry decorators transparently; the
# parser preprocesses the LaTeX source before tokenization, and display.py
# reverses the transformation for roundtrip output.
_DECORATOR_COMBINING = {
    "bar":   "\u0304",  # combining macron
    "hat":   "\u0302",  # combining circumflex
    "tilde": "\u0303",  # combining tilde
    "dot":   "\u0307",  # combining dot above
    "ddot":  "\u0308",  # combining diaeresis
}

# Matches \bar{<inner>} where <inner> contains no braces (handled iteratively
# so nested decorators like \bar{\hat{X}} are applied inside-out).
_DECORATOR_RE = re.compile(
    r"\\(" + "|".join(_DECORATOR_COMBINING.keys()) + r")\{([^{}]*)\}"
)


def _apply_decorators(text: str) -> str:
    """Rewrite \\bar{X}, \\hat{X}, ... as base + Unicode combining mark.

    Applied iteratively so nested decorators are resolved from the innermost
    outward: \\bar{\\hat{e}} → ê → ê̄.
    Inner LaTeX commands (\\mu etc.) are resolved inline so the combining mark
    attaches to the already-unicode base character.
    """

    def repl(match: re.Match) -> str:
        kind = match.group(1)
        inner = match.group(2).strip()
        combining = _DECORATOR_COMBINING[kind]

        if not inner:
            return ""

        # Resolve a leading LaTeX command (\mu → μ) so the combining mark
        # attaches to a single base character. Anything else passes through.
        if inner.startswith("\\"):
            cmd_match = _LATEX_CMD_RE.match(inner)
            if cmd_match:
                cmd = cmd_match.group()
                rest = inner[cmd_match.end():]
                base = _LATEX_TO_UNICODE.get(cmd, cmd)
                return base + combining + rest

        # Plain character (possibly Unicode Greek already): attach to last char.
        return inner[:-1] + inner[-1] + combining

    while True:
        new_text = _DECORATOR_RE.sub(repl, text)
        if new_text == text:
            return text
        text = new_text


def _is_combining(ch: str) -> bool:
    """True if ch is a Unicode combining diacritical mark (U+0300–U+036F)."""
    return len(ch) == 1 and 0x0300 <= ord(ch) <= 0x036F


def _iter_graphemes(text: str):
    """Yield each base character bundled with its trailing combining marks.

    "p̄q" (= 'p' + U+0304 + 'q') yields "p̄", then "q".
    """
    i = 0
    while i < len(text):
        ch = text[i]
        i += 1
        while i < len(text) and _is_combining(text[i]):
            ch += text[i]
            i += 1
        yield ch


def _parse_index_content(content: str) -> list[str]:
    """인덱스 그룹 내용을 Unicode 인덱스 이름 리스트로 파싱한다.

    'μν' → ["μ", "ν"],  '\\mu \\nu' → ["μ", "ν"],
    '\\mu_{1}' → ["μ_1"],  'ab' → ["a", "b"]

    Combining marks (from \\bar{X} → X + U+0304) are attached to the
    preceding base character: "p̄q" → ["p̄", "q"].
    """
    result = []
    pos = 0
    content = content.strip()

    while pos < len(content):
        if content[pos].isspace():
            pos += 1
            continue

        # LaTeX 명령: \mu, \nu 등
        if content[pos] == "\\":
            m = _LATEX_CMD_RE.match(content, pos)
            if m:
                cmd = m.group()
                pos = m.end()
                base = _LATEX_TO_UNICODE.get(cmd, cmd)

                # Absorb any trailing combining marks (from \bar{\mu} → μ̄)
                while pos < len(content) and _is_combining(content[pos]):
                    base += content[pos]
                    pos += 1

                # 첨자 접미사: _{1} 또는 _1
                if pos < len(content) and content[pos] == "_":
                    if pos + 1 < len(content) and content[pos + 1] == "{":
                        end = content.index("}", pos + 2)
                        suffix = content[pos + 2 : end]
                        pos = end + 1
                        result.append(f"{base}_{suffix}")
                    elif pos + 1 < len(content):
                        result.append(f"{base}_{content[pos + 1]}")
                        pos += 2
                    else:
                        result.append(base)
                else:
                    result.append(base)
                continue

        # 일반 문자 (Unicode 그리스 포함)
        ch = content[pos]
        pos += 1

        # Absorb any trailing combining marks (multi-codepoint grapheme)
        while pos < len(content) and _is_combining(content[pos]):
            ch += content[pos]
            pos += 1

        # 첨자 접미사
        if pos < len(content) and content[pos] == "_":
            if pos + 1 < len(content) and content[pos + 1] == "{":
                end = content.index("}", pos + 2)
                suffix = content[pos + 2 : end]
                pos = end + 1
                result.append(f"{ch}_{suffix}")
            elif pos + 1 < len(content):
                result.append(f"{ch}_{content[pos + 1]}")
                pos += 2
            else:
                result.append(ch)
            continue

        result.append(ch)

    return result


# ─── Tokenizer ───────────────────────────────────────────────

# 토큰 타입 — 순서 중요: 먼저 매치되는 것이 우선
_TOKEN_TYPES = [
    ("FRAC",        r"\\frac\{[^}]*\}\{[^}]*\}"),
    ("PARTIAL",     r"\\partial"),
    ("NABLA",       r"\\nabla"),
    ("VAR",         r"\\Var"),
    # 인덱스 그룹: 한 단계 중첩 {}를 허용 (dummy index _{1} 지원)
    ("UPPER",       r"\^\{(?:[^{}]|\{[^}]*\})*\}|\^[A-Za-z\u0370-\u03FF]"),
    ("LOWER",       r"_\{(?:[^{}]|\{[^}]*\})*\}|_[A-Za-z\u0370-\u03FF]"),
    ("EMPTY_GROUP", r"\{\}"),
    ("PLUS",        r"\+"),
    ("MINUS",       r"-"),
    ("LPAREN",      r"\("),
    ("RPAREN",      r"\)"),
    ("NUMBER",      r"\d+(?:\.\d+)?"),
    ("NAME",        r"[A-Za-z\u0370-\u03FF\\][A-Za-z0-9\u0370-\u03FF\u0300-\u036F]*"),
    ("SPACE",       r"\s+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_TYPES))

_FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")


def _tokenize(text: str) -> list[tuple[str, str]]:
    """LaTeX 문자열을 (kind, value) 토큰 리스트로 변환한다."""
    tokens = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise SyntaxError(
                f"Unexpected character at position {pos}: '{text[pos:pos+5]}...'"
            )
        kind = m.lastgroup
        value = m.group()
        pos = m.end()
        if kind not in ("SPACE", "EMPTY_GROUP"):
            tokens.append((kind, value))
    return tokens


# ─── Parser ──────────────────────────────────────────────────

class _Parser:
    """재귀 하강 파서. 토큰 스트림을 TensorExpr로 변환한다.

    문법 (비형식적):
      expr     → term (('+' | '-') term)*
      term     → factor (factor)*          ← 공백/인접 = 암묵적 곱
      factor   → ('-')? atom
      atom     → '(' expr ')' | scalar | tensor | partial | nabla
      tensor   → NAME ('^' indices)? ('_' indices)?
      partial  → '\\partial' LOWER atom
      nabla    → '\\nabla' LOWER atom
      indices  → '{' chars '}' | single_char
    """

    def __init__(
        self,
        tokens: list,
        registry: IndexRegistry,
        connections: dict[str, Connection] | None = None,
    ):
        self.tokens = tokens
        self.pos = 0
        self.registry = registry
        self.connections = connections or {}

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(
                f"Expected {expected_type}, got {tok[0]} ('{tok[1]}')"
            )
        self.pos += 1
        return tok

    def parse(self) -> TensorExpr:
        result = self.parse_expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(
                f"Unexpected token at end: '{self.tokens[self.pos][1]}'"
            )
        return result

    def parse_expr(self) -> TensorExpr:
        """expr → term (('+' | '-') term)*"""
        left = self.parse_term()

        while self.peek() and self.peek()[0] in ("PLUS", "MINUS"):
            op = self.consume()[0]
            right = self.parse_term()
            if op == "MINUS":
                right = ScalarMul(-1, right)
            left = TensorSum(left, right)

        return left

    def parse_term(self) -> TensorExpr:
        """term → factor (factor)*   (암묵적 곱: 공백으로 구분된 인접 factor)"""
        left = self.parse_factor()

        while self.peek() and self.peek()[0] in (
            "NAME", "NUMBER", "LPAREN", "FRAC", "PARTIAL", "NABLA", "VAR",
        ):
            right = self.parse_factor()
            left = TensorProduct(left, right)

        return left

    def parse_factor(self) -> TensorExpr:
        """factor → '-' atom | atom | frac atom"""
        # 선행 마이너스
        if self.peek() and self.peek()[0] == "MINUS":
            self.consume()
            atom = self.parse_atom()
            return ScalarMul(-1, atom)

        return self.parse_atom()

    def parse_atom(self) -> TensorExpr:
        """atom → '(' expr ')' | frac | number | tensor"""
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")

        # 괄호
        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_expr()
            self.consume("RPAREN")
            return expr

        # \frac{a}{b}
        if tok[0] == "FRAC":
            _, value = self.consume("FRAC")
            m = _FRAC_RE.match(value)
            num = float(m.group(1))
            den = float(m.group(2))
            scalar = num / den
            # frac 뒤에 텐서가 오면 곱으로 처리 (parse_term이 처리)
            return ScalarMul(scalar, _ScalarOne())

        # 숫자
        if tok[0] == "NUMBER":
            _, value = self.consume("NUMBER")
            scalar = float(value) if "." in value else int(value)
            return ScalarMul(scalar, _ScalarOne())

        # 편미분: \partial_{μ} expr
        if tok[0] == "PARTIAL":
            return self._parse_partial()

        # 공변미분: \nabla_{μ} expr
        if tok[0] == "NABLA":
            return self._parse_nabla()

        # 변분: \Var{expr} or \Var atom
        if tok[0] == "VAR":
            return self._parse_variation()

        # 공간 공변미분: D_{i} atom  (대문자 D + LOWER 다음이 오면 연산자로 해석)
        # 주의: 텐서 이름으로 "D"를 쓰고 싶으면 \mathcal{D} 등으로 회피.
        if (
            tok[0] == "NAME"
            and tok[1] == "D"
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "LOWER"
        ):
            return self._parse_spatial_covariant()

        # 텐서
        if tok[0] == "NAME":
            return self._parse_tensor()

        raise SyntaxError(f"Unexpected token: {tok[0]} ('{tok[1]}')")

    def _parse_tensor(self) -> Tensor:
        """tensor → NAME (UPPER | LOWER)*"""
        _, name = self.consume("NAME")

        # LaTeX 명령을 Unicode로 변환: \Gamma → Γ, \eta → η
        if name.startswith("\\"):
            name = _LATEX_TO_UNICODE.get(name, name)

        indices = []
        while self.peek() and self.peek()[0] in ("UPPER", "LOWER"):
            kind, value = self.consume()
            position = "upper" if kind == "UPPER" else "lower"
            names = self._extract_index_names(value, kind)

            for n in names:
                indices.append(self.registry.make_index(n, position))

        return Tensor(name, indices)

    def _extract_index_names(self, value: str, kind: str) -> list[str]:
        """토큰 값에서 인덱스 이름 리스트를 추출한다.

        Unicode(μν)와 LaTeX 명령(\\mu \\nu) 모두 지원.
        """
        prefix = "^" if kind == "UPPER" else "_"
        rest = value[len(prefix):]

        if rest.startswith("{") and rest.endswith("}"):
            content = rest[1:-1]
            return _parse_index_content(content)

        # 중괄호 없는 단일 문자
        return _parse_index_content(rest)

    def _parse_partial(self) -> PartialDeriv:
        """partial → '\\partial' LOWER atom"""
        self.consume("PARTIAL")

        if not self.peek() or self.peek()[0] != "LOWER":
            raise SyntaxError("Expected lower index after \\partial")

        _, value = self.consume("LOWER")
        names = self._extract_index_names(value, "LOWER")
        if len(names) != 1:
            raise SyntaxError(
                f"\\partial expects exactly one derivative index, got {len(names)}"
            )

        idx = self.registry.make_index(names[0], "lower")
        operand = self.parse_atom()
        return PartialDeriv(operand, idx)

    def _parse_nabla(self) -> CovariantDeriv:
        """nabla → '\\nabla' LOWER atom"""
        self.consume("NABLA")

        if not self.peek() or self.peek()[0] != "LOWER":
            raise SyntaxError("Expected lower index after \\nabla")

        _, value = self.consume("LOWER")
        names = self._extract_index_names(value, "LOWER")
        if len(names) != 1:
            raise SyntaxError(
                f"\\nabla expects exactly one derivative index, got {len(names)}"
            )

        idx = self.registry.make_index(names[0], "lower")
        operand = self.parse_atom()
        return CovariantDeriv(operand, idx, self.connections)

    def _parse_spatial_covariant(self) -> SpatialCovariantDeriv:
        """spatial_covariant → 'D' LOWER atom"""
        self.consume("NAME")  # "D"

        _, value = self.consume("LOWER")
        names = self._extract_index_names(value, "LOWER")
        if len(names) != 1:
            raise SyntaxError(
                f"D expects exactly one derivative index, got {len(names)}"
            )

        idx = self.registry.make_index(names[0], "lower")
        operand = self.parse_atom()
        return SpatialCovariantDeriv(operand, idx, self.connections)

    def _parse_variation(self):
        r"""variation → '\\Var' '(' expr ')' | '\\Var' atom"""
        from indexcalc.core.variation import Variation

        self.consume("VAR")
        if self.peek() and self.peek()[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_expr()
            self.consume("RPAREN")
            return Variation(expr)
        operand = self.parse_atom()
        return Variation(operand)


class _ScalarOne(TensorExpr):
    """스칼라 1을 나타내는 내부 헬퍼. 인덱스가 없는 표현식."""

    @property
    def free_indices(self) -> list[Index]:
        return []

    def __repr__(self) -> str:
        return "1"


# ─── Public API ──────────────────────────────────────────────

def _rewrite_var_braces(text: str) -> str:
    r"""``\Var{...}`` → ``\Var(...)`` 로 변환하여 토크나이저가 처리 가능하게 한다.

    중첩 중괄호를 올바르게 매칭한다.
    """
    prev = None
    while prev != text:
        prev = text
        result: list[str] = []
        i = 0
        prefix = "\\Var{"
        plen = len(prefix)
        while i < len(text):
            if text[i:i + plen] == prefix:
                result.append("\\Var(")
                i += plen
                depth = 1
                while i < len(text) and depth > 0:
                    ch = text[i]
                    if ch == "{":
                        depth += 1
                        result.append(ch)
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            result.append(")")
                        else:
                            result.append(ch)
                    else:
                        result.append(ch)
                    i += 1
            else:
                result.append(text[i])
                i += 1
        text = "".join(result)
    return text


def parse(
    text: str,
    registry: IndexRegistry,
    connections: dict[str, Connection] | None = None,
) -> TensorExpr:
    """LaTeX 문자열을 TensorExpr로 파싱한다.

    Parameters
    ----------
    text : str
        LaTeX 형식의 텐서 표현식.
    registry : IndexRegistry
        인덱스 문자 → IndexSpace 매핑.
    connections : dict[str, Connection] or None
        IndexSpace.name → Connection 매핑. \\nabla 파싱 시 사용.
        None이면 빈 dict로 처리 (roundtrip용).

    Returns
    -------
    TensorExpr
        파싱된 텐서 표현식 트리.

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> reg = IndexRegistry()
    >>> reg.register(st)
    >>> expr = parse("T^{μ}_{ν} S^{ν}_{λ}", reg)
    >>> print(expr)
    (T^μ_ν * S^ν_λ)  [contracted: ν]
    """
    text = _apply_decorators(text)
    text = _rewrite_var_braces(text)
    tokens = _tokenize(text)
    parser = _Parser(tokens, registry, connections)
    result = parser.parse()
    return _simplify_scalars(result)


def _simplify_scalars(expr: TensorExpr) -> TensorExpr:
    """ScalarMul(n, _ScalarOne())을 정리한다.

    예: ScalarMul(2, _ScalarOne()) * Tensor → ScalarMul(2, Tensor)
    """
    if isinstance(expr, TensorProduct):
        left = _simplify_scalars(expr.left)
        right = _simplify_scalars(expr.right)

        # ScalarMul(n, ScalarOne) * X → ScalarMul(n, X)
        if isinstance(left, ScalarMul) and isinstance(left.expr, _ScalarOne):
            return ScalarMul(left.scalar, _simplify_scalars(right))
        # X * ScalarMul(n, ScalarOne) → ScalarMul(n, X)
        if isinstance(right, ScalarMul) and isinstance(right.expr, _ScalarOne):
            return ScalarMul(right.scalar, _simplify_scalars(left))

        return TensorProduct(left, right)

    if isinstance(expr, TensorSum):
        return TensorSum(
            _simplify_scalars(expr.left),
            _simplify_scalars(expr.right),
        )

    if isinstance(expr, ScalarMul):
        inner = _simplify_scalars(expr.expr)
        # ScalarMul(a, ScalarMul(b, X)) → ScalarMul(a*b, X)
        if isinstance(inner, ScalarMul):
            return ScalarMul(expr.scalar * inner.scalar, inner.expr)
        return ScalarMul(expr.scalar, inner)

    return expr
