"""LIONS IR ↔ prefix-token round-trip.

Token schema: S-expression prefix notation. Each IR node becomes
``( <op> <child1> <child2> ... )``. Atoms are bare strings (white-space
free identifiers, plus typed scalar atoms ``R:0.5`` / ``C:re:im`` / ``I:7``).

Used by the Task 3 decoder in the LIONS pipeline (representation_decision.md
§3). Round-trip with ``tokens_to_expr ∘ expr_to_tokens`` is the guarantee
that the decoder can be validated by re-running ``apply_generator + simplify``
on the parsed IR.

Out of scope here:
- Graph view (D9b — node_features / edge_index / edge_type).
- str→int vocab id mapping (caller / ML framework).
- Numeric scalar embedding.
"""

from __future__ import annotations
from typing import Iterable

from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
from indexcalc.core.variation import ZeroTensor

from indexcalc.lions.dataset import LabeledSample


# Structural tokens that always appear regardless of dataset content.
STRUCTURE_TOKENS = (
    "(", ")",
    "PROD", "SUM", "SCALE", "PARTIAL", "ZERO", "TENSOR", "IDX",
    "IDXLIST", "ANTISYM", "SYM", "TRACELESS", "TRANSVERSE", "REPS",
    "bosonic", "fermionic",
    "upper", "lower",
)


# ─── Scalar atom encoding ────────────────────────────────


def _scalar_to_atom(s) -> str:
    if isinstance(s, complex):
        return f"C:{s.real}:{s.imag}"
    if isinstance(s, int):
        return f"I:{s}"
    if isinstance(s, float):
        return f"R:{s}"
    raise TypeError(f"unsupported scalar type {type(s).__name__}")


def _atom_to_scalar(atom: str):
    if atom.startswith("R:"):
        return float(atom[2:])
    if atom.startswith("I:"):
        return int(atom[2:])
    if atom.startswith("C:"):
        _, re, im = atom.split(":")
        return complex(float(re), float(im))
    raise ValueError(f"not a scalar atom: {atom!r}")


# ─── IR → tokens ─────────────────────────────────────────


def _index_tokens(idx: Index) -> list[str]:
    return ["(", "IDX", idx.name, idx.space.name, idx.position, ")"]


def _pair_tokens(pairs: Iterable[tuple[int, int]]) -> list[str]:
    out: list[str] = []
    for a, b in pairs:
        out.extend(["(", str(a), str(b), ")"])
    return out


def _int_tokens(items: Iterable[int]) -> list[str]:
    return [str(i) for i in items]


def _reps_tokens(reps: dict[str, str]) -> list[str]:
    out: list[str] = []
    for g, r in reps.items():
        out.extend([g, r])
    return out


def expr_to_tokens(expr: TensorExpr) -> list[str]:
    """Serialize ``expr`` to a prefix S-expression token list."""
    if isinstance(expr, Tensor):
        out: list[str] = ["(", "TENSOR", expr.name]
        out.append("("); out.append("IDXLIST")
        for i in expr.indices:
            out.extend(_index_tokens(i))
        out.append(")")
        out.extend(["(", "ANTISYM"])
        out.extend(_pair_tokens(expr.antisymmetric_pairs))
        out.append(")")
        out.extend(["(", "SYM"])
        out.extend(_pair_tokens(expr.symmetric_pairs))
        out.append(")")
        out.extend(["(", "TRACELESS"])
        out.extend(_pair_tokens(expr.traceless))
        out.append(")")
        out.extend(["(", "TRANSVERSE"])
        out.extend(_int_tokens(expr.transverse))
        out.append(")")
        out.extend(["(", "REPS"])
        out.extend(_reps_tokens(expr.reps))
        out.append(")")
        out.append(expr.statistics)
        out.append(")")
        return out
    if isinstance(expr, TensorProduct):
        return ["(", "PROD"] + expr_to_tokens(expr.left) + expr_to_tokens(expr.right) + [")"]
    if isinstance(expr, TensorSum):
        return ["(", "SUM"] + expr_to_tokens(expr.left) + expr_to_tokens(expr.right) + [")"]
    if isinstance(expr, ScalarMul):
        return ["(", "SCALE", _scalar_to_atom(expr.scalar)] + expr_to_tokens(expr.expr) + [")"]
    if isinstance(expr, PartialDeriv):
        return ["(", "PARTIAL"] + expr_to_tokens(expr.expr) + _index_tokens(expr.deriv_index) + [")"]
    if isinstance(expr, ZeroTensor):
        out = ["(", "ZERO", "(", "IDXLIST"]
        for i in expr.free_indices:
            out.extend(_index_tokens(i))
        out.extend([")", ")"])
        return out
    if isinstance(expr, CovariantDeriv):
        raise NotImplementedError(
            "CovariantDeriv tokenization deferred (D8a v1 scope)."
        )
    raise TypeError(f"unsupported expr type {type(expr).__name__}")


# ─── tokens → IR ─────────────────────────────────────────


class _Cursor:
    """Mutable position over a token list — simplifies recursive descent."""
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError("unexpected end of token stream")
        return self.tokens[self.pos]

    def take(self) -> str:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, want: str) -> None:
        got = self.take()
        if got != want:
            raise ValueError(f"expected {want!r}, got {got!r} at pos {self.pos - 1}")

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)


def tokens_to_expr(
    tokens: list[str], spaces: dict[str, IndexSpace],
) -> TensorExpr:
    """Inverse of ``expr_to_tokens``. ``spaces`` resolves space names to
    ``IndexSpace`` instances (use ``serializer.collect_spaces`` to build
    the mapping when round-tripping a known IR)."""
    cur = _Cursor(tokens)
    out = _parse_expr(cur, spaces)
    if not cur.at_end():
        raise ValueError(f"trailing tokens at pos {cur.pos}: {tokens[cur.pos:]}")
    return out


def _parse_expr(cur: _Cursor, spaces: dict[str, IndexSpace]) -> TensorExpr:
    cur.expect("(")
    head = cur.take()
    if head == "PROD":
        L = _parse_expr(cur, spaces)
        R = _parse_expr(cur, spaces)
        cur.expect(")")
        return TensorProduct(L, R)
    if head == "SUM":
        L = _parse_expr(cur, spaces)
        R = _parse_expr(cur, spaces)
        cur.expect(")")
        return TensorSum(L, R)
    if head == "SCALE":
        scalar = _atom_to_scalar(cur.take())
        inner = _parse_expr(cur, spaces)
        cur.expect(")")
        return ScalarMul(scalar, inner)
    if head == "PARTIAL":
        inner = _parse_expr(cur, spaces)
        idx = _parse_index(cur, spaces)
        cur.expect(")")
        return PartialDeriv(inner, idx)
    if head == "ZERO":
        idxs = _parse_idxlist(cur, spaces)
        cur.expect(")")
        return ZeroTensor(idxs)
    if head == "TENSOR":
        name = cur.take()
        indices = _parse_idxlist(cur, spaces)
        antisym = _parse_pair_section(cur, "ANTISYM")
        sym = _parse_pair_section(cur, "SYM")
        traceless = _parse_pair_section(cur, "TRACELESS")
        transverse = _parse_int_section(cur, "TRANSVERSE")
        reps = _parse_reps_section(cur)
        stat = cur.take()
        cur.expect(")")
        return Tensor(
            name, indices,
            antisymmetric_pairs=antisym,
            symmetric_pairs=sym,
            traceless=traceless,
            transverse=transverse,
            reps=reps,
            statistics=stat,
        )
    raise ValueError(f"unknown head token {head!r}")


def _parse_idxlist(cur: _Cursor, spaces: dict[str, IndexSpace]) -> list[Index]:
    cur.expect("("); cur.expect("IDXLIST")
    out: list[Index] = []
    while cur.peek() == "(":
        out.append(_parse_index(cur, spaces))
    cur.expect(")")
    return out


def _parse_index(cur: _Cursor, spaces: dict[str, IndexSpace]) -> Index:
    cur.expect("("); cur.expect("IDX")
    name = cur.take()
    space_name = cur.take()
    position = cur.take()
    cur.expect(")")
    if space_name not in spaces:
        raise KeyError(f"unknown IndexSpace {space_name!r}; provide via spaces dict")
    return Index(name, spaces[space_name], position)


def _parse_pair_section(cur: _Cursor, label: str) -> list[tuple[int, int]]:
    cur.expect("("); cur.expect(label)
    out: list[tuple[int, int]] = []
    while cur.peek() == "(":
        cur.expect("(")
        a = int(cur.take())
        b = int(cur.take())
        cur.expect(")")
        out.append((a, b))
    cur.expect(")")
    return out


def _parse_int_section(cur: _Cursor, label: str) -> list[int]:
    cur.expect("("); cur.expect(label)
    out: list[int] = []
    while cur.peek() != ")":
        out.append(int(cur.take()))
    cur.expect(")")
    return out


def _parse_reps_section(cur: _Cursor) -> dict[str, str]:
    cur.expect("("); cur.expect("REPS")
    out: dict[str, str] = {}
    while cur.peek() != ")":
        k = cur.take()
        v = cur.take()
        out[k] = v
    cur.expect(")")
    return out


# ─── Vocab ───────────────────────────────────────────────


def build_vocab(samples: list[LabeledSample]) -> dict[str, int]:
    """Union all tokens appearing across ``samples`` + structural tokens.

    Returns a deterministic ``str → int`` mapping. Structure tokens get
    the first IDs, then dataset-specific atoms in sorted order so the
    vocab is reproducible for downstream ML embeddings.
    """
    seen: set[str] = set(STRUCTURE_TOKENS)
    for s in samples:
        for t in expr_to_tokens(s.expr):
            seen.add(t)
    # Structure tokens first, then sorted remainder for stability.
    structural = list(STRUCTURE_TOKENS)
    remaining = sorted(seen - set(structural))
    vocab = {t: i for i, t in enumerate(structural + remaining)}
    return vocab
