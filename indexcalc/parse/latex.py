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

인덱스-공간 매핑은 IndexRegistry를 통해 관리한다.
파서는 등록된 인덱스 문자를 보고 어떤 공간에 속하는지 자동으로 판단한다.
"""

from __future__ import annotations
import re
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum, ScalarMul, TensorExpr


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
        """IndexSpace를 등록한다. space.indices의 각 문자를 매핑에 추가."""
        for ch in space.indices:
            if ch in self._map and self._map[ch] != space:
                raise ValueError(
                    f"Index character '{ch}' is already registered to "
                    f"'{self._map[ch].name}', cannot register to '{space.name}'"
                )
            self._map[ch] = space

    def resolve(self, name: str) -> IndexSpace:
        """인덱스 문자를 IndexSpace로 해석한다."""
        if name not in self._map:
            raise KeyError(
                f"Unknown index '{name}'. "
                f"Register an IndexSpace containing this character first."
            )
        return self._map[name]

    def make_index(self, name: str, position: str) -> Index:
        """인덱스 문자와 위치로 Index 객체를 생성한다."""
        space = self.resolve(name)
        return Index(name, space, position)


# ─── Tokenizer ───────────────────────────────────────────────

# 토큰 타입 (캡처 그룹 없이 — 값은 문자열에서 직접 추출)
_TOKEN_TYPES = [
    ("FRAC",    r"\\frac\{[^}]*\}\{[^}]*\}"),
    ("UPPER",   r"\^\{[^}]+\}|\^[A-Za-z\u0370-\u03FF]"),
    ("LOWER",   r"_\{[^}]+\}|_[A-Za-z\u0370-\u03FF]"),
    ("PLUS",    r"\+"),
    ("MINUS",   r"-"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("NUMBER",  r"\d+(?:\.\d+)?"),
    ("NAME",    r"[A-Za-z\u0370-\u03FF\\][A-Za-z0-9\u0370-\u03FF]*"),
    ("SPACE",   r"\s+"),
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
        if kind != "SPACE":
            tokens.append((kind, value))
    return tokens


# ─── Parser ──────────────────────────────────────────────────

class _Parser:
    """재귀 하강 파서. 토큰 스트림을 TensorExpr로 변환한다.

    문법 (비형식적):
      expr     → term (('+' | '-') term)*
      term     → factor (factor)*          ← 공백/인접 = 암묵적 곱
      factor   → ('-')? atom
      atom     → '(' expr ')' | scalar | tensor
      tensor   → NAME ('^' indices)? ('_' indices)?
      indices  → '{' chars '}' | single_char
    """

    def __init__(self, tokens: list, registry: IndexRegistry):
        self.tokens = tokens
        self.pos = 0
        self.registry = registry

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

        while self.peek() and self.peek()[0] in ("NAME", "NUMBER", "LPAREN", "FRAC"):
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

        # 텐서
        if tok[0] == "NAME":
            return self._parse_tensor()

        raise SyntaxError(f"Unexpected token: {tok[0]} ('{tok[1]}')")

    def _parse_tensor(self) -> Tensor:
        """tensor → NAME (UPPER | LOWER)*"""
        _, name = self.consume("NAME")
        indices = []

        while self.peek() and self.peek()[0] in ("UPPER", "LOWER"):
            kind, value = self.consume()
            position = "upper" if kind == "UPPER" else "lower"
            chars = self._extract_index_chars(value, kind)

            for ch in chars:
                if ch.strip():
                    indices.append(self.registry.make_index(ch, position))

        return Tensor(name, indices)

    def _extract_index_chars(self, value: str, kind: str) -> str:
        """토큰 값에서 인덱스 문자들을 추출한다."""
        prefix = "^" if kind == "UPPER" else "_"
        rest = value[len(prefix):]

        if rest.startswith("{") and rest.endswith("}"):
            return rest[1:-1]
        return rest


class _ScalarOne(TensorExpr):
    """스칼라 1을 나타내는 내부 헬퍼. 인덱스가 없는 표현식."""

    @property
    def free_indices(self) -> list[Index]:
        return []

    def __repr__(self) -> str:
        return "1"


# ─── Public API ──────────────────────────────────────────────

def parse(text: str, registry: IndexRegistry) -> TensorExpr:
    """LaTeX 문자열을 TensorExpr로 파싱한다.

    Parameters
    ----------
    text : str
        LaTeX 형식의 텐서 표현식.
    registry : IndexRegistry
        인덱스 문자 → IndexSpace 매핑.

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
    # ScalarMul(n, _ScalarOne()) 을 정리: n * 1 * Tensor → n * Tensor
    tokens = _tokenize(text)
    parser = _Parser(tokens, registry)
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
