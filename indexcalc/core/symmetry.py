"""
Antisymmetric/symmetric tensor 속성의 canonicalization.

Tensor에 ``antisymmetric_pairs``가 선언되어 있으면, 해당 slot 쌍의 인덱스가
canonical 순서(이름 오름차순, 단 같은 이름일 땐 position upper 먼저)로
정렬되도록 재배치하고, 교환으로 인한 부호를 ScalarMul로 누적한다.

현재는 slot "쌍"만 지원한다 (예: B_{μν} = -B_{νμ}). DFT B-field, F_{μν}
(electromagnetic), Riemann 부분 대칭 등을 커버하는 최소 기능.
"""

from __future__ import annotations
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)


def _index_key(idx) -> tuple:
    """Canonical 정렬 키. 이름 우선, 같은 이름이면 upper가 먼저."""
    return (idx.name, 0 if idx.position == "upper" else 1)


def _canonicalize_tensor(t: Tensor) -> tuple[Tensor, int]:
    """Tensor의 antisymmetric slot 쌍을 canonical 순서로 정렬.

    Returns
    -------
    (new_tensor, sign) : Tensor, {+1, -1}
    """
    if not t.antisymmetric_pairs:
        return t, 1

    new_indices = list(t.indices)
    sign = 1
    for a, b in t.antisymmetric_pairs:
        if _index_key(new_indices[a]) > _index_key(new_indices[b]):
            new_indices[a], new_indices[b] = new_indices[b], new_indices[a]
            sign = -sign

    if sign == 1 and tuple(new_indices) == t.indices:
        return t, 1

    new_t = Tensor(
        t.name,
        new_indices,
        antisymmetric_pairs=[tuple(p) for p in t.antisymmetric_pairs],
    )
    return new_t, sign


def canonicalize_antisym(expr: TensorExpr) -> TensorExpr:
    """표현식 전체에서 Tensor들의 antisymmetric slot을 canonical 순서로 정렬.

    부호 변화는 ScalarMul로 흡수한다. 표현식 구조(합, 곱, 스칼라곱)는 유지.
    """
    if isinstance(expr, Tensor):
        new_t, sign = _canonicalize_tensor(expr)
        return new_t if sign == 1 else ScalarMul(-1, new_t)

    if isinstance(expr, TensorProduct):
        left = canonicalize_antisym(expr.left)
        right = canonicalize_antisym(expr.right)
        # 양쪽의 ScalarMul(-1, .)을 곱 바깥으로 뽑아낸다
        sign = 1
        if isinstance(left, ScalarMul) and left.scalar == -1:
            left = left.expr
            sign = -sign
        if isinstance(right, ScalarMul) and right.scalar == -1:
            right = right.expr
            sign = -sign
        prod = TensorProduct(left, right)
        return prod if sign == 1 else ScalarMul(-1, prod)

    if isinstance(expr, TensorSum):
        return TensorSum(
            canonicalize_antisym(expr.left),
            canonicalize_antisym(expr.right),
        )

    if isinstance(expr, ScalarMul):
        inner = canonicalize_antisym(expr.expr)
        if isinstance(inner, ScalarMul):
            return ScalarMul(expr.scalar * inner.scalar, inner.expr)
        return ScalarMul(expr.scalar, inner)

    return expr
