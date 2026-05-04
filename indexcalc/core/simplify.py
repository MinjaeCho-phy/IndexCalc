"""
Simplify: 표현식 정규화 및 동등성 검사 기본기.

본 모듈은 LIONS M2의 동기 — *generator 작용 결과가 0이 되는지를 패턴으로 검출*
— 을 위해 필요한 최소 기본기를 제공한다. 일반 CAS의 결정 절차가 아니라,
**정형화된 라그랑지안 기반 검증**에 충분한 한정된 도구 모음.

제공 함수:

- ``rename_index(expr, mapping)`` — 트리 전체에서 인덱스 이름 치환 (위치 보존).
- ``collect_factors(expr)`` — TensorProduct을 factor 리스트로 평탄화.
- ``canonical_form(expr, swap_names=())`` — bosonic commute + (옵션) swap_names의
  canonical renaming까지 반영한 hashable 표현. 동등성 비교에 사용.
- ``is_structurally_equal(e1, e2, swap_names=())`` — canonical_form 비교.
- ``is_zero_by_antisym_swap(expr)`` — TensorProduct 안의 antisymmetric tensor 한
  쌍에 대해 dummy 짝을 swap했을 때 나머지 부분이 invariant이면 → ZeroTensor.
  (antisym × symmetric = 0 패턴 자동 검출.)
"""

from __future__ import annotations
from typing import Sequence

from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
from indexcalc.core.variation import ZeroTensor


# ─── rename_index ───────────────────────────────────────────


def rename_index(expr: TensorExpr, mapping: dict[str, str]) -> TensorExpr:
    """expr 트리 전체에서 인덱스 이름을 ``mapping``에 따라 치환한다.

    위치(upper/lower)와 IndexSpace는 보존; 이름만 변경.
    Tensor의 antisymmetric_pairs / reps / statistics 등 모든 메타도 보존.
    """
    if isinstance(expr, ZeroTensor):
        new_free = [
            Index(mapping.get(idx.name, idx.name), idx.space, idx.position)
            for idx in expr.free_indices
        ]
        return ZeroTensor(new_free)

    if isinstance(expr, Tensor):
        new_indices = [
            Index(mapping.get(idx.name, idx.name), idx.space, idx.position)
            for idx in expr.indices
        ]
        return Tensor(
            expr.name, new_indices,
            antisymmetric_pairs=list(expr.antisymmetric_pairs),
            reps=dict(expr.reps),
            statistics=expr.statistics,
        )

    if isinstance(expr, TensorProduct):
        return TensorProduct(
            rename_index(expr.left, mapping),
            rename_index(expr.right, mapping),
        )
    if isinstance(expr, TensorSum):
        return TensorSum(
            rename_index(expr.left, mapping),
            rename_index(expr.right, mapping),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, rename_index(expr.expr, mapping))

    if isinstance(expr, PartialDeriv):
        new_idx = Index(
            mapping.get(expr.deriv_index.name, expr.deriv_index.name),
            expr.deriv_index.space, expr.deriv_index.position,
        )
        return PartialDeriv(rename_index(expr.expr, mapping), new_idx)

    if isinstance(expr, CovariantDeriv):
        new_idx = Index(
            mapping.get(expr.deriv_index.name, expr.deriv_index.name),
            expr.deriv_index.space, expr.deriv_index.position,
        )
        return type(expr)(
            rename_index(expr.expr, mapping), new_idx, expr.connections,
        )

    raise NotImplementedError(
        f"rename_index not implemented for {type(expr).__name__}"
    )


# ─── collect_factors ────────────────────────────────────────


def collect_factors(expr: TensorExpr) -> list[TensorExpr]:
    """``TensorProduct``를 평탄한 factor 리스트로 변환한다.

    ScalarMul은 factor에 흡수하지 않고 그대로 list에 들어간다 (호출자가 처리).
    그 외 노드는 길이 1 list로 반환.
    """
    if isinstance(expr, TensorProduct):
        return collect_factors(expr.left) + collect_factors(expr.right)
    return [expr]


# ─── canonical_form ─────────────────────────────────────────


def _index_key(idx: Index, swap_names: Sequence[str]) -> tuple:
    """인덱스의 정렬용 key.

    swap_names에 속한 이름은 placeholder로 가린다 (현재 use case에선 unused).
    공간이 metric을 갖고 있으면 (e.g., $\\kappa = \\delta$ for compact adj,
    $\\eta_{\\mu\\nu}$ for Lorentz frame) position을 ``"*"`` 로 collapse —
    raise/lower가 component identity인 공간에선 위치 구분이 canonical 비교에
    영향을 주지 않아야 하므로.
    """
    name_token = "?" if idx.name in swap_names else idx.name
    position = "*" if idx.space.metric else idx.position
    return (idx.space.name, position, name_token)


def _factor_key_no_swap(factor: TensorExpr, swap_names: Sequence[str]) -> tuple:
    """Factor의 sort key — swap_names를 placeholder로 처리한 hashable signature."""
    if isinstance(factor, Tensor):
        idx_keys = tuple(_index_key(idx, swap_names) for idx in factor.indices)
        return (
            "Tensor", factor.name, idx_keys,
            tuple(factor.antisymmetric_pairs),
            factor.statistics,
        )
    if isinstance(factor, ScalarMul):
        return (
            "ScalarMul", str(factor.scalar),
            _factor_key_no_swap(factor.expr, swap_names),
        )
    if isinstance(factor, PartialDeriv):
        return (
            "PartialDeriv",
            _index_key(factor.deriv_index, swap_names),
            _factor_key_no_swap(factor.expr, swap_names),
        )
    if isinstance(factor, CovariantDeriv):
        return (
            "CovariantDeriv",
            _index_key(factor.deriv_index, swap_names),
            _factor_key_no_swap(factor.expr, swap_names),
        )
    if isinstance(factor, ZeroTensor):
        return ("ZeroTensor",)
    return (type(factor).__name__,)


def _collect_factor_index_names(factor: TensorExpr) -> list[str]:
    """Factor 안에 등장하는 모든 인덱스 이름의 리스트 (등장 순서, 중복 포함)."""
    if isinstance(factor, Tensor):
        return [idx.name for idx in factor.indices]
    if isinstance(factor, ScalarMul):
        return _collect_factor_index_names(factor.expr)
    if isinstance(factor, PartialDeriv) or isinstance(factor, CovariantDeriv):
        return [factor.deriv_index.name] + _collect_factor_index_names(factor.expr)
    return []


def canonical_form(
    expr: TensorExpr,
    swap_names: Sequence[str] = (),
) -> tuple:
    """Strict canonical form: bosonic-commute (factor multiset)만 normalize한다.

    Parameters
    ----------
    expr : TensorExpr
        대부분 TensorProduct 또는 단일 factor.
    swap_names : Sequence[str]
        (현 버전에선 사용되지 않음) — API 호환성용 placeholder. v2 ($\\kappa$-handling)
        에 다시 활성화 검토. swap_names를 자유 renaming하는 이전 버전은 X·Y 같은
        ``factor 이름이 다른 경우``에 false positive를 냈음.

    Returns
    -------
    tuple
        factor key의 정렬된 multiset. 두 표현식이 ``bosonic commute`` 외엔
        구조적으로 같아야 같은 값.
    """
    factors = collect_factors(expr)
    keys = sorted(_factor_key_no_swap(f, swap_names=()) for f in factors)
    return tuple(keys)


def is_structurally_equal(
    e1: TensorExpr, e2: TensorExpr, swap_names: Sequence[str] = (),
) -> bool:
    """``canonical_form`` 일치 여부."""
    return canonical_form(e1, swap_names) == canonical_form(e2, swap_names)


def canonical_form_modulo_dummies(expr: TensorExpr) -> tuple:
    """Canonical form with **dummy index renaming** — ``collect_scalar_terms`` 전용.

    Strict ``canonical_form``과의 차이:
        - 정확히 두 번 등장하는 (그리고 expr.free_indices에 없는) 인덱스 이름들은
          dummy로 간주, sorted-factor의 등장 순서대로 ``_d0, _d1, …``로 canonical
          rename 후 비교.
        - 두 표현식이 ``dummy 이름만 다른`` 경우 같은 canonical form을 갖는다.

    Use case: SU(2) doublet 변환에서 두 Leibniz 항의 body가 dummy 인덱스만 다르고
    구조적으로는 같은 식. ``collect_scalar_terms``가 이를 같은 group으로 묶어
    스칼라 합산 → 0 검출.

    NOT used by ``is_zero_by_antisym_swap`` — 거기선 free renaming이 false positive
    (X·Y 류)를 일으키므로 strict ``canonical_form``을 유지.
    """
    factors = collect_factors(expr)

    # 1. dummy 식별 — count == 2 + free_indices에 없음
    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}
    dummy_names = {
        n for n, c in name_counts.items()
        if c == 2 and n not in free_names
    }

    # 2. dummy를 hidden한 채 factors 정렬
    indexed = sorted(
        ((_factor_key_no_swap(f, list(dummy_names)), i)
         for i, f in enumerate(factors)),
    )
    sorted_factors = [factors[i] for _, i in indexed]

    # 3. sorted 순서로 dummy에 canonical 이름 부여
    mapping: dict[str, str] = {}
    counter = 0
    for f in sorted_factors:
        for nm in _collect_factor_index_names(f):
            if nm in dummy_names and nm not in mapping:
                mapping[nm] = f"_d{counter}"
                counter += 1

    # 4. mapping 적용 후 full key로 정렬
    renamed = [rename_index(f, mapping) for f in sorted_factors]
    return tuple(sorted(_factor_key_no_swap(f, ()) for f in renamed))


# ─── antisym × symmetric → 0 ────────────────────────────────


def _index_name_count(factors: list[TensorExpr]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in factors:
        for nm in _collect_factor_index_names(f):
            counts[nm] = counts.get(nm, 0) + 1
    return counts


def _product_of_factors(factors: list[TensorExpr]) -> TensorExpr:
    """Factor 리스트를 다시 좌결합 TensorProduct로 묶는다. 빈 리스트는 ZeroTensor."""
    if not factors:
        return ZeroTensor([])
    result = factors[0]
    for f in factors[1:]:
        result = TensorProduct(result, f)
    return result


def is_zero_by_antisym_swap(expr: TensorExpr) -> TensorExpr:
    """TensorProduct 안의 antisymmetric tensor 한 쌍에 대해 swap-prove-zero 시도.

    알고리즘:
        for each factor T with antisymmetric_pairs:
            for each pair (s_i, s_j) in T.antisymmetric_pairs:
                let n_i, n_j = T.indices[s_i].name, T.indices[s_j].name
                if both n_i, n_j는 expr 전역에서 정확히 두 번 등장하는 dummy:
                    rest = product of factors except T
                    rest_swap = rename(rest, {n_i: n_j, n_j: n_i})
                    if rest_swap == rest (strict canonical multiset):
                        return ZeroTensor(expr.free_indices)
        return expr (변경 없음)

    검출 조건은 **strict multiset 동등성**: rest의 factor multiset이 swap 후 그대로여야 한다.
    이는 mixed-position 컨벤션에선 false negative가 날 수 있고 (예: $F^c F_a$ —
    위치가 달라 multiset이 다름), 이런 경우엔 $\\kappa$-application normalize가
    먼저 필요하다 (M2.5/M3). 같은-position 컨벤션 (all-upper adj 등)에선 안전.

    expr가 TensorProduct가 아니면 그대로 반환.
    """
    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}

    for k, T in enumerate(factors):
        if not isinstance(T, Tensor) or not T.antisymmetric_pairs:
            continue
        for s_i, s_j in T.antisymmetric_pairs:
            n_i = T.indices[s_i].name
            n_j = T.indices[s_j].name
            # 둘 다 dummy (전역 카운트 == 2) 이고 free index가 아니어야 함
            if name_counts.get(n_i, 0) != 2 or name_counts.get(n_j, 0) != 2:
                continue
            if n_i in free_names or n_j in free_names:
                continue
            if n_i == n_j:
                continue

            # rest = factors except T
            rest_factors = [factors[i] for i in range(len(factors)) if i != k]
            rest = _product_of_factors(rest_factors)

            # rest의 dummy 이름 swap
            rest_swap = rename_index(rest, {n_i: n_j, n_j: n_i})

            if is_structurally_equal(rest, rest_swap, swap_names=(n_i, n_j)):
                return ZeroTensor(expr.free_indices)

    return expr


# ─── Distribute TensorProduct over TensorSum (M5) ───────────


def distribute_products(expr: TensorExpr) -> TensorExpr:
    """``TensorProduct``를 ``TensorSum``에 분배해 sum-of-products 형태로 변환.

    .. math::
        T \\otimes (A + B) \\to T \\otimes A + T \\otimes B
        (A + B) \\otimes T \\to A \\otimes T + B \\otimes T
        c \\cdot (A + B) \\to c \\cdot A + c \\cdot B

    이렇게 평탄화되면 ``collect_scalar_terms``가 모든 항을 한 multiset에 모아
    canonical body로 group 가능. 4-field 항이나 ``TS`` 중첩 구조에 필요.
    """
    if isinstance(expr, TensorProduct):
        left = distribute_products(expr.left)
        right = distribute_products(expr.right)
        if isinstance(left, TensorSum):
            return distribute_products(
                TensorSum(
                    TensorProduct(left.left, right),
                    TensorProduct(left.right, right),
                )
            )
        if isinstance(right, TensorSum):
            return distribute_products(
                TensorSum(
                    TensorProduct(left, right.left),
                    TensorProduct(left, right.right),
                )
            )
        if left is not expr.left or right is not expr.right:
            return TensorProduct(left, right)
        return expr

    if isinstance(expr, TensorSum):
        new_l = distribute_products(expr.left)
        new_r = distribute_products(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        inner = distribute_products(expr.expr)
        if isinstance(inner, TensorSum):
            return distribute_products(
                TensorSum(
                    ScalarMul(expr.scalar, inner.left),
                    ScalarMul(expr.scalar, inner.right),
                )
            )
        if inner is not expr.expr:
            return ScalarMul(expr.scalar, inner)
        return expr

    return expr


# ─── Scalar pull-out + collect-like-terms (M3) ──────────────


def pull_scalars(expr: TensorExpr) -> TensorExpr:
    """``ScalarMul``을 가능한 한 ``TensorProduct``/``PartialDeriv`` 밖으로 hoist.

    규칙:
        - ScalarMul(c1, X) * ScalarMul(c2, Y)  →  ScalarMul(c1*c2, X*Y)
        - ScalarMul(c, X) * Y                  →  ScalarMul(c, X*Y)
        - X * ScalarMul(c, Y)                  →  ScalarMul(c, X*Y)
        - ∂_μ(ScalarMul(c, X))                 →  ScalarMul(c, ∂_μ(X))     (linearity)
        - ScalarMul(c1, ScalarMul(c2, X))      →  ScalarMul(c1*c2, X)
        - TensorSum, ScalarMul: 재귀.
    """
    if isinstance(expr, TensorProduct):
        left = pull_scalars(expr.left)
        right = pull_scalars(expr.right)
        if isinstance(left, ScalarMul) and isinstance(right, ScalarMul):
            return ScalarMul(
                left.scalar * right.scalar,
                TensorProduct(left.expr, right.expr),
            )
        if isinstance(left, ScalarMul):
            return ScalarMul(left.scalar, TensorProduct(left.expr, right))
        if isinstance(right, ScalarMul):
            return ScalarMul(right.scalar, TensorProduct(left, right.expr))
        if left is not expr.left or right is not expr.right:
            return TensorProduct(left, right)
        return expr

    if isinstance(expr, TensorSum):
        new_l = pull_scalars(expr.left)
        new_r = pull_scalars(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        inner = pull_scalars(expr.expr)
        if isinstance(inner, ScalarMul):
            return ScalarMul(expr.scalar * inner.scalar, inner.expr)
        if inner is not expr.expr:
            return ScalarMul(expr.scalar, inner)
        return expr

    if isinstance(expr, PartialDeriv):
        inner = pull_scalars(expr.expr)
        if isinstance(inner, ScalarMul):
            return ScalarMul(inner.scalar, PartialDeriv(inner.expr, expr.deriv_index))
        if inner is not expr.expr:
            return PartialDeriv(inner, expr.deriv_index)
        return expr

    if isinstance(expr, CovariantDeriv):
        inner = pull_scalars(expr.expr)
        if isinstance(inner, ScalarMul):
            return ScalarMul(
                inner.scalar,
                type(expr)(inner.expr, expr.deriv_index, expr.connections),
            )
        if inner is not expr.expr:
            return type(expr)(inner, expr.deriv_index, expr.connections)
        return expr

    return expr


def _flatten_sum(expr: TensorExpr) -> list[TensorExpr]:
    """``TensorSum`` 트리를 평탄한 summand 리스트로 변환."""
    if isinstance(expr, TensorSum):
        return _flatten_sum(expr.left) + _flatten_sum(expr.right)
    return [expr]


def _split_scalar(expr: TensorExpr) -> tuple:
    """``ScalarMul(c, X)``  →  ``(c, X)``;  그 외  →  ``(1, expr)``."""
    if isinstance(expr, ScalarMul):
        return expr.scalar, expr.expr
    return 1, expr


def collect_scalar_terms(expr: TensorExpr) -> TensorExpr:
    """``TensorSum`` 안에서 동일 canonical body의 summand를 묶어 scalar 합산.

    - 합이 0인 group은 drop (전체 group들이 모두 cancel되면 ZeroTensor 반환).
    - 합이 1이면 ScalarMul wrapping 없이 body만 반환.
    - 그 외엔 ScalarMul(total, body)로 wrap.
    - ``ScalarMul(c, TensorSum(...))`` 또는 ``TensorProduct(L, TensorSum(...))``
      같은 nested 구조도 안쪽까지 재귀.

    ``pull_scalars``가 먼저 적용되어 있어야 효과적이다 (그래야 nested ScalarMul이
    하나의 (scalar, body) 쌍으로 normalized).
    """
    # ScalarMul wrapping: 안으로 재귀
    if isinstance(expr, ScalarMul):
        inner = collect_scalar_terms(expr.expr)
        if isinstance(inner, ZeroTensor):
            return inner
        if inner is not expr.expr:
            return ScalarMul(expr.scalar, inner)
        return expr

    # TensorProduct: 양쪽으로 재귀 (안쪽 TensorSum이 collected될 수 있음)
    if isinstance(expr, TensorProduct):
        new_l = collect_scalar_terms(expr.left)
        new_r = collect_scalar_terms(expr.right)
        if isinstance(new_l, ZeroTensor) or isinstance(new_r, ZeroTensor):
            return ZeroTensor(expr.free_indices)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    if not isinstance(expr, TensorSum):
        return expr

    summands = _flatten_sum(expr)

    # body의 canonical form (dummy 이름 무관)을 키로 group
    groups: dict[tuple, list] = {}
    has_zero_body = False
    for s in summands:
        scalar, body = _split_scalar(s)
        if isinstance(body, ZeroTensor):
            has_zero_body = True
            continue  # 0 항 무시
        key = canonical_form_modulo_dummies(body)
        groups.setdefault(key, []).append((scalar, body))

    # 변경이 일어날 만한 경우인지 조기 판정 (idempotency 보장):
    # ZeroTensor 항이 있거나, 어떤 group이 다수 항을 갖거나, 합이 0이거나.
    will_change = has_zero_body or any(len(items) > 1 for items in groups.values())
    if not will_change:
        # 모든 group이 단일 항 → 원본을 그대로 반환 (fixed point에서 무한루프 방지)
        return expr

    new_summands: list[TensorExpr] = []
    for key, items in groups.items():
        total = sum(item[0] for item in items)
        if total == 0:
            continue  # 완전 cancel
        body = items[0][1]  # representative
        if total == 1:
            new_summands.append(body)
        else:
            new_summands.append(ScalarMul(total, body))

    if not new_summands:
        return ZeroTensor(expr.free_indices)
    if len(new_summands) == 1:
        return new_summands[0]

    result = new_summands[0]
    for s in new_summands[1:]:
        result = TensorSum(result, s)
    return result


# ─── simplify (top-level) ───────────────────────────────────


def simplify(expr: TensorExpr) -> TensorExpr:
    """모든 정규화 규칙을 fixed-point까지 적용한다.

    적용 규칙 (M2.5 + M3):
        - 재귀: TensorSum / ScalarMul / TensorProduct.
        - is_zero_by_antisym_swap (M2: antisym × sym = 0).
        - pull_scalars (M3: ScalarMul hoist).
        - collect_scalar_terms (M3: TensorSum 같은 body 합산).
        - ZeroTensor 흡수 (variation.py).
    """
    from indexcalc.core.variation import _simplify_zeros

    cur = expr
    for _ in range(20):  # max 20 passes — guards against rule-cycle
        prev = cur
        cur = _simplify_once(cur)
        cur = distribute_products(cur)
        cur = pull_scalars(cur)
        cur = collect_scalar_terms(cur)
        cur = _simplify_zeros(cur)
        if cur is prev:
            break
    return cur


def _simplify_once(expr: TensorExpr) -> TensorExpr:
    """한 번의 simplify pass (재귀 + 한정된 규칙 적용)."""
    if isinstance(expr, TensorProduct):
        new_l = _simplify_once(expr.left)
        new_r = _simplify_once(expr.right)
        prod = TensorProduct(new_l, new_r) if (new_l is not expr.left or new_r is not expr.right) else expr
        # antisym swap 0-detection을 product에 적용
        zero_check = is_zero_by_antisym_swap(prod)
        return zero_check
    if isinstance(expr, TensorSum):
        new_l = _simplify_once(expr.left)
        new_r = _simplify_once(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = _simplify_once(expr.expr)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    return expr
