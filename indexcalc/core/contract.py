"""
Contraction & Einstein summation 분석 모듈.

표현식 트리를 순회하면서:
  - 모든 인덱스를 수집하고
  - contracted pair(dummy index)와 free index를 구분하고
  - Einstein convention 위반을 감지하고
  - Trace(같은 텐서 내 contraction)를 지원한다.
"""

from __future__ import annotations
from collections import Counter
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)


# ─── 표현식 트리 순회 유틸리티 ────────────────────────────────

def collect_tensors(expr: TensorExpr) -> list[Tensor]:
    """표현식 트리에서 모든 Tensor(잎 노드)를 순서대로 수집한다.

    Examples
    --------
    >>> expr = parse("T^{μ}_{ν} S^{ν}_{λ}", reg)
    >>> collect_tensors(expr)
    [T^μ_ν, S^ν_λ]
    """
    from indexcalc.core.variation import Variation, ZeroTensor

    if isinstance(expr, Tensor):
        return [expr]
    if isinstance(expr, TensorProduct):
        return collect_tensors(expr.left) + collect_tensors(expr.right)
    if isinstance(expr, TensorSum):
        # 합의 경우 왼쪽 항의 텐서만 반환 (구조 분석용)
        return collect_tensors(expr.left)
    if isinstance(expr, ScalarMul):
        return collect_tensors(expr.expr)
    if isinstance(expr, Variation):
        return collect_tensors(expr.expr)
    if isinstance(expr, ZeroTensor):
        return []
    return []


def collect_all_indices(expr: TensorExpr) -> list[Index]:
    """표현식의 모든 인덱스를 평탄하게 수집한다.

    Tensor에 달린 인덱스뿐 아니라 ``PartialDeriv``/``CovariantDeriv`` 노드의
    미분 인덱스(``deriv_index``)도 함께 수집한다. 이렇게 해야 ``∂_k ∂_k E``
    처럼 같은 이름의 미분 인덱스가 중첩된 식을 ``validate_einstein``이
    Einstein convention 위반으로 잡아낼 수 있다.
    """
    # Local import to avoid a hard module-load cycle with deriv.py, which
    # imports from tensor.py only. contract.py is loaded before deriv.py in
    # __init__.py, so deferring this import keeps things deterministic.
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
    from indexcalc.core.variation import Variation

    indices: list[Index] = []

    def walk(e: TensorExpr) -> None:
        if isinstance(e, Tensor):
            indices.extend(e.indices)
        elif isinstance(e, (PartialDeriv, CovariantDeriv)):
            indices.append(e.deriv_index)
            walk(e.expr)
        elif isinstance(e, Variation):
            walk(e.expr)
        elif isinstance(e, TensorProduct):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, TensorSum):
            # 합의 경우 왼쪽 항만 구조 분석 대상으로 삼는다 (기존 관례 유지).
            walk(e.left)
        elif isinstance(e, ScalarMul):
            walk(e.expr)
        elif isinstance(e, Trace):
            walk(e.tensor)
        # 기타 미지 노드는 무시 (ZeroTensor 포함)

    walk(expr)
    return indices


# ─── Einstein convention 검증 ─────────────────────────────────

class IndexError(Exception):
    """인덱스 관련 에러."""
    pass


def validate_einstein(expr: TensorExpr) -> dict:
    """표현식이 Einstein summation convention을 만족하는지 검증한다.

    규칙:
      - 같은 이름+공간의 인덱스는 최대 2번만 나타날 수 있다.
      - 2번 나타나면 반드시 하나는 upper, 하나는 lower여야 한다 (contraction).
      - 1번만 나타나면 free index이다.

    Parameters
    ----------
    expr : TensorExpr
        검증할 텐서 표현식.

    Returns
    -------
    dict
        {
            "free": list[Index],       # 1번만 나타나는 인덱스
            "contracted": list[tuple[Index, Index]],  # 축약 쌍
            "valid": bool,
            "errors": list[str],
        }
    """
    all_indices = collect_all_indices(expr)

    # (name, space) 기준으로 그룹핑
    groups: dict[tuple[str, str], list[Index]] = {}
    for idx in all_indices:
        key = (idx.name, idx.space.name)
        groups.setdefault(key, []).append(idx)

    free = []
    contracted = []
    errors = []

    for (name, space_name), idx_list in groups.items():
        count = len(idx_list)

        if count == 1:
            free.append(idx_list[0])

        elif count == 2:
            a, b = idx_list
            if a.position == b.position:
                errors.append(
                    f"Index '{name}' ({space_name}) appears twice "
                    f"with same position '{a.position}'. "
                    f"Contraction requires one upper and one lower."
                )
            else:
                # upper를 먼저 배치
                if a.position == "upper":
                    contracted.append((a, b))
                else:
                    contracted.append((b, a))

        else:
            errors.append(
                f"Index '{name}' ({space_name}) appears {count} times. "
                f"Einstein convention allows at most 2."
            )

    return {
        "free": free,
        "contracted": contracted,
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ─── Trace ────────────────────────────────────────────────────

class Trace(TensorExpr):
    """같은 텐서 내에서 두 인덱스를 축약한다 (trace).

    예: T^μ_μ = Tr(T) — 텐서 T의 μ에 대한 trace.

    Parameters
    ----------
    tensor : Tensor
        Trace를 취할 텐서.
    index_name : str
        축약할 인덱스의 이름. 이 이름을 가진 인덱스가
        정확히 2개 (upper 1, lower 1) 있어야 한다.
    """

    def __init__(self, tensor: Tensor, index_name: str):
        # 해당 이름의 인덱스 찾기
        matching = [
            (i, idx) for i, idx in enumerate(tensor.indices)
            if idx.name == index_name
        ]

        if len(matching) != 2:
            raise ValueError(
                f"Trace requires exactly 2 indices named '{index_name}', "
                f"found {len(matching)} in {tensor}"
            )

        (i1, idx1), (i2, idx2) = matching

        if idx1.position == idx2.position:
            raise ValueError(
                f"Trace requires one upper and one lower '{index_name}', "
                f"but both are {idx1.position} in {tensor}"
            )

        if idx1.space != idx2.space:
            raise ValueError(
                f"Trace indices must be in the same space, "
                f"got {idx1.space.name} and {idx2.space.name}"
            )

        self.tensor = tensor
        self.index_name = index_name
        self.traced_pair = (idx1, idx2)

        # Free indices = 원본에서 traced 쌍 제거
        self._free = [
            idx for j, idx in enumerate(tensor.indices)
            if j != i1 and j != i2
        ]

    @property
    def free_indices(self) -> list[Index]:
        return list(self._free)

    def __repr__(self) -> str:
        if not self._free:
            return f"Tr({self.tensor})"
        return f"Tr_{self.index_name}({self.tensor})"


def trace(tensor: Tensor, index_name: str) -> Trace:
    """텐서의 특정 인덱스에 대한 trace를 계산한다.

    Parameters
    ----------
    tensor : Tensor
        Trace를 취할 텐서.
    index_name : str
        축약할 인덱스의 이름.

    Returns
    -------
    Trace
        Trace 표현식.

    Examples
    --------
    >>> T = Tensor("T", [st.upper("μ"), st.lower("μ")])
    >>> trace(T, "μ")
    Tr(T^μ_μ)
    """
    return Trace(tensor, index_name)


# ─── 표현식 요약 ──────────────────────────────────────────────

def summarize(expr: TensorExpr) -> str:
    """표현식의 인덱스 구조를 사람이 읽기 좋은 형태로 요약한다.

    Parameters
    ----------
    expr : TensorExpr
        요약할 표현식.

    Returns
    -------
    str
        인덱스 구조 요약 문자열.
    """
    info = validate_einstein(expr)

    lines = []
    lines.append(f"Expression: {expr}")
    lines.append(f"Tensors:    {collect_tensors(expr)}")

    if info["free"]:
        free_str = ", ".join(f"{idx}" for idx in info["free"])
        lines.append(f"Free:       {free_str}")
    else:
        lines.append("Free:       (none — scalar)")

    if info["contracted"]:
        pairs_str = ", ".join(
            f"{a.name} ({a.space.name})" for a, b in info["contracted"]
        )
        lines.append(f"Contracted: {pairs_str}")
    else:
        lines.append("Contracted: (none)")

    lines.append(f"Rank:       {expr.rank}")

    if not info["valid"]:
        lines.append("ERRORS:")
        for err in info["errors"]:
            lines.append(f"  ⚠ {err}")

    return "\n".join(lines)
