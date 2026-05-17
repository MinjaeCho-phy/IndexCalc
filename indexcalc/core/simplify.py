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
    Tensor의 antisymmetric_pairs / symmetric_pairs / traceless / transverse /
    reps / statistics 등 모든 메타도 보존.
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
            symmetric_pairs=list(expr.symmetric_pairs),
            traceless=list(expr.traceless),
            transverse=list(expr.transverse),
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
    """Factor의 sort key — swap_names를 placeholder로 처리한 hashable signature.

    Tensor의 ``symmetric_pairs`` slot은 그 두 인덱스 자리의 key를 정렬해 흡수한다.
    이렇게 해야 dummy 이름 swap (n_i↔n_j) 후의 rest가 sym tensor의 key 차원에서
    invariant로 인식되어 ``is_zero_by_antisym_swap``이 antisym × sym = 0 패턴을
    잡을 수 있다.
    """
    if isinstance(factor, Tensor):
        idx_keys = [_index_key(idx, swap_names) for idx in factor.indices]
        for s_a, s_b in factor.symmetric_pairs:
            if idx_keys[s_a] > idx_keys[s_b]:
                idx_keys[s_a], idx_keys[s_b] = idx_keys[s_b], idx_keys[s_a]
        return (
            "Tensor", factor.name, tuple(idx_keys),
            tuple(factor.antisymmetric_pairs),
            tuple(factor.symmetric_pairs),
            tuple(factor.traceless),
            tuple(factor.transverse),
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


# ─── G5: traceless × metric → 0 ─────────────────────────────


def is_zero_by_traceless_metric(expr: TensorExpr, mreg) -> TensorExpr:
    """Tensor의 ``traceless`` slot 쌍이 metric으로 완전히 contract되면 ZeroTensor.

    예: ``γ^{ij} · h^{TT}_{ij}``  (h^TT가 traceless=[(0,1)]) → 0.

    탐지 조건:
        - 어떤 factor T가 ``T.traceless`` 쌍 (s_i, s_j)를 갖는다.
        - T의 두 slot 인덱스 이름 n_i, n_j가 둘 다 dummy (전역 count == 2,
          expr.free_indices에 없음).
        - 다른 factor M이 mreg.is_metric(M)에서 같은 IndexSpace의 metric으로
          확인되고, M의 두 인덱스 이름이 ``{n_i, n_j}``와 일치.

    조건 만족 시 ``ZeroTensor(expr.free_indices)`` 반환, 아니면 expr 그대로.
    """
    if not isinstance(expr, TensorProduct):
        return expr
    if mreg is None:
        return expr

    factors = collect_factors(expr)
    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}

    for k, T in enumerate(factors):
        if not isinstance(T, Tensor) or not T.traceless:
            continue
        for s_i, s_j in T.traceless:
            n_i = T.indices[s_i].name
            n_j = T.indices[s_j].name
            if n_i == n_j:
                continue
            if n_i in free_names or n_j in free_names:
                continue
            if name_counts.get(n_i, 0) != 2 or name_counts.get(n_j, 0) != 2:
                continue
            slot_space = T.indices[s_i].space
            for j, M in enumerate(factors):
                if j == k or not isinstance(M, Tensor):
                    continue
                if len(M.indices) != 2:
                    continue
                space = mreg.is_metric(M)
                if space is None or space != slot_space:
                    continue
                m_names = {M.indices[0].name, M.indices[1].name}
                if m_names == {n_i, n_j}:
                    return ZeroTensor(expr.free_indices)
    return expr


# ─── G5: transverse × deriv → 0 ─────────────────────────────


def _collect_deriv_indices(factors: list[TensorExpr]) -> list[tuple[int, Index]]:
    """factor 리스트에서 (factor_idx, deriv_index) 쌍 — Partial/Covariant deriv 한정."""
    out: list[tuple[int, Index]] = []
    for k, f in enumerate(factors):
        if isinstance(f, (PartialDeriv, CovariantDeriv)):
            out.append((k, f.deriv_index))
    return out


def _innermost_tensor(expr: TensorExpr) -> Tensor | None:
    """PartialDeriv/CovariantDeriv 안쪽의 Tensor leaf를 단일하면 반환.

    여러 factor의 곱이거나 Sum이면 ``None``. transverse-slot 검사용 한정 helper.
    """
    cur = expr
    while isinstance(cur, (PartialDeriv, CovariantDeriv)):
        cur = cur.expr
    if isinstance(cur, Tensor):
        return cur
    return None


def is_zero_by_transverse_deriv(expr: TensorExpr, mreg=None) -> TensorExpr:
    """Tensor의 ``transverse`` slot이 deriv index와 contract하면 ZeroTensor.

    예: ``∂^i BV_i = 0`` (BV.transverse=[0]).

    두 가지 contraction 경로를 인식:
        (A) **Direct** — 어떤 factor가 PartialDeriv/CovariantDeriv이고, 그
            안쪽 Tensor T가 ``T.transverse`` slot s를 가지며, deriv_index name이
            T.indices[s].name과 같다 (positions opposite). 이 경우엔 ``mreg``
            불필요.
        (B) **Via single metric** — TensorProduct 안에서, deriv_index name
            n_d와 어떤 transverse slot의 name n_t가 metric tensor M의 두
            인덱스 이름과 정확히 일치하여 ``γ^{n_d n_t}`` 가 둘을 raising
            contraction으로 묶는다. 이 경우엔 ``mreg`` 필요.

    조건 만족 시 ``ZeroTensor(expr.free_indices)`` 반환.
    """
    # Case (A): 단일 PartialDeriv/CovariantDeriv를 직접 검사 (TensorProduct 아닐 수도)
    if isinstance(expr, (PartialDeriv, CovariantDeriv)):
        T = _innermost_tensor(expr)
        if T is not None and T.transverse:
            # deriv 누적 인덱스 모두 모음
            cur = expr
            deriv_names: list[tuple[str, object]] = []
            while isinstance(cur, (PartialDeriv, CovariantDeriv)):
                deriv_names.append((cur.deriv_index.name, cur.deriv_index.space))
                cur = cur.expr
            for s in T.transverse:
                t_name = T.indices[s].name
                t_space = T.indices[s].space
                t_pos = T.indices[s].position
                for d_name, d_space in deriv_names:
                    # deriv_index는 항상 lower; T의 transverse slot이 upper면 contract
                    if d_name == t_name and d_space == t_space and t_pos == "upper":
                        return ZeroTensor(expr.free_indices)
        return expr

    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    deriv_pairs = _collect_deriv_indices(factors)
    if not deriv_pairs:
        return expr

    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}

    # transverse slot 정보 모음: (name, space, position)
    transverse_slots: list[tuple[str, object, str]] = []
    for f in factors:
        T = _innermost_tensor(f) if isinstance(f, (PartialDeriv, CovariantDeriv)) else f
        if isinstance(T, Tensor):
            for s in T.transverse:
                idx = T.indices[s]
                transverse_slots.append((idx.name, idx.space, idx.position))

    for _, d_idx in deriv_pairs:
        d_name = d_idx.name
        d_space = d_idx.space
        # Case (A) — direct contraction: deriv name == transverse slot name (opposite pos)
        for t_name, t_space, t_pos in transverse_slots:
            if d_space != t_space:
                continue
            if d_name == t_name and t_pos == "upper":
                # 둘 다 dummy 인지 확인
                if d_name in free_names:
                    continue
                if name_counts.get(d_name, 0) != 2:
                    continue
                return ZeroTensor(expr.free_indices)

        # Case (B) — via single metric
        if mreg is None:
            continue
        for M in factors:
            if not isinstance(M, Tensor) or len(M.indices) != 2:
                continue
            space = mreg.is_metric(M)
            if space is None or space != d_space:
                continue
            m_names = {M.indices[0].name, M.indices[1].name}
            if d_name not in m_names:
                continue
            other = next(iter(m_names - {d_name}))
            if d_name == other:
                continue
            for t_name, t_space, t_pos in transverse_slots:
                if t_space != d_space:
                    continue
                if t_name != other:
                    continue
                if d_name in free_names or other in free_names:
                    continue
                if name_counts.get(d_name, 0) != 2 or name_counts.get(other, 0) != 2:
                    continue
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

    if isinstance(expr, PartialDeriv):
        # ∂(A + B) → ∂A + ∂B  (linearity of partial derivative)
        inner = distribute_products(expr.expr)
        if isinstance(inner, TensorSum):
            return distribute_products(
                TensorSum(
                    PartialDeriv(inner.left, expr.deriv_index),
                    PartialDeriv(inner.right, expr.deriv_index),
                )
            )
        if inner is not expr.expr:
            return PartialDeriv(inner, expr.deriv_index)
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


def _is_dynamic_field(expr: TensorExpr) -> bool:
    """``expr``가 dynamic field (= 위치 의존적)을 포함하면 True.

    Heuristic: a leaf ``Tensor`` is dynamic iff its ``reps`` dict is non-empty.
    그룹 invariant tensor (η, δ, f, γ, Σ, M_vec 등)는 ``reps={}``로 둔 IR
    convention을 따르므로 *constant* 로 판정.
    """
    if isinstance(expr, Tensor):
        return bool(expr.reps)
    if isinstance(expr, ZeroTensor):
        return False
    if isinstance(expr, ScalarMul):
        return _is_dynamic_field(expr.expr)
    if isinstance(expr, (TensorProduct, TensorSum)):
        return _is_dynamic_field(expr.left) or _is_dynamic_field(expr.right)
    if isinstance(expr, (PartialDeriv, CovariantDeriv)):
        return _is_dynamic_field(expr.expr)
    return False


def commute_partial_through_constants(expr: TensorExpr) -> TensorExpr:
    """``PartialDeriv``를 *constant* (= dynamic field 아님) factor 밖으로 pull-out.

    .. math::
        \\partial_\\mu (C \\cdot \\psi) \\to C \\cdot \\partial_\\mu \\psi
        \\qquad \\text{if $C$ is constant (e.g. $\\Sigma^{ab}$, $\\gamma^a$, $T^a$)}

    Σ, T, f, γ 같은 group-theoretic 텐서는 IR convention 상 ``reps={}`` 로
    표현되므로 dynamic이 아니다 — spacetime 미분 밖으로 빠진다. 반대로
    ψ, V, A, H 같은 dynamic field는 그대로 ∂ 안에 머문다.

    Notes
    -----
    Index space 기준이 아니라 ``reps`` 기준 — Σ^{ab} 가 frame 인덱스를
    갖더라도 (frame == spacetime인 setup에서) constant 판정.
    """
    if isinstance(expr, PartialDeriv):
        inner = commute_partial_through_constants(expr.expr)
        if isinstance(inner, TensorProduct):
            left = inner.left
            right = inner.right
            left_dyn = _is_dynamic_field(left)
            right_dyn = _is_dynamic_field(right)
            if not left_dyn and right_dyn:
                return TensorProduct(
                    left,
                    commute_partial_through_constants(
                        PartialDeriv(right, expr.deriv_index)
                    ),
                )
            if left_dyn and not right_dyn:
                return TensorProduct(
                    commute_partial_through_constants(
                        PartialDeriv(left, expr.deriv_index)
                    ),
                    right,
                )
            # Both dynamic or both constant: leave Leibniz alone
            if inner is not expr.expr:
                return PartialDeriv(inner, expr.deriv_index)
            return expr
        if inner is not expr.expr:
            return PartialDeriv(inner, expr.deriv_index)
        return expr

    if isinstance(expr, TensorProduct):
        new_l = commute_partial_through_constants(expr.left)
        new_r = commute_partial_through_constants(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    if isinstance(expr, TensorSum):
        new_l = commute_partial_through_constants(expr.left)
        new_r = commute_partial_through_constants(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = commute_partial_through_constants(expr.expr)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    return expr


def _find_sigma_gamma_pair(
    factors: list[TensorExpr],
    sigma_name: str,
    gamma_name: str,
) -> tuple[int, int] | None:
    """factor 리스트에서 Σ.col ↔ γ.row 가 spinor dummy로 contract된 쌍을 찾는다.

    Σ convention: slots = (a↑frame, b↑frame, row↑spinor, col↓spinor), 4 indices.
    γ convention: slots = (μ↑frame, row↑spinor, col↓spinor), 3 indices.

    Σ.col (slot 3, lower) 의 name 이 γ.row (slot 1, upper) 의 name 과 같고
    같은 IndexSpace 면 contraction 으로 본다. 못 찾으면 ``None``.
    """
    for i, fi in enumerate(factors):
        if not (
            isinstance(fi, Tensor)
            and fi.name == sigma_name
            and len(fi.indices) == 4
        ):
            continue
        sigma_col = fi.indices[3]
        if sigma_col.position != "lower":
            continue
        for j, fj in enumerate(factors):
            if i == j:
                continue
            if not (
                isinstance(fj, Tensor)
                and fj.name == gamma_name
                and len(fj.indices) == 3
            ):
                continue
            gamma_row = fj.indices[1]
            if gamma_row.position != "upper":
                continue
            if (
                gamma_row.name == sigma_col.name
                and gamma_row.space == sigma_col.space
            ):
                return (i, j)
    return None


def _build_sigma_gamma_swap(
    factors: list[TensorExpr],
    i_sigma: int,
    j_gamma: int,
) -> list[TensorExpr]:
    """[Σ,γ] identity 의 첫째 term: γ·Σ swapped ordering.

    Original: ψ̄.col → Σ.row(slot2); Σ.col(slot3) ↔ γ.row(slot1); γ.col(slot2) → ∂ψ.
    Swap:     ψ̄.col → γ.row(slot1); γ.col(slot2) ↔ Σ.row(slot2); Σ.col(slot3) → ∂ψ.

    γ.row name = ψ̄'s contraction name (was Σ.row name);
    γ.col name = NEW dummy Y;
    Σ.row name = Y;
    Σ.col name = ∂ψ's contraction name (was γ.col name).
    """
    sigma = factors[i_sigma]
    gamma = factors[j_gamma]
    sigma_row_name = sigma.indices[2].name
    sigma_col_name = sigma.indices[3].name
    gamma_row_name = gamma.indices[1].name
    gamma_col_name = gamma.indices[2].name
    spinor_space = sigma.indices[2].space

    new_dummy = _fresh_swap_dummy()

    new_gamma = Tensor(
        gamma.name,
        [
            gamma.indices[0],
            Index(sigma_row_name, spinor_space, "upper"),
            Index(new_dummy, spinor_space, "lower"),
        ],
        antisymmetric_pairs=list(gamma.antisymmetric_pairs),
        reps=dict(gamma.reps),
        statistics=gamma.statistics,
    )
    new_sigma = Tensor(
        sigma.name,
        [
            sigma.indices[0],
            sigma.indices[1],
            Index(new_dummy, spinor_space, "upper"),
            Index(gamma_col_name, spinor_space, "lower"),
        ],
        antisymmetric_pairs=list(sigma.antisymmetric_pairs),
        reps=dict(sigma.reps),
        statistics=sigma.statistics,
    )

    new_factors = list(factors)
    new_factors[i_sigma] = new_sigma
    new_factors[j_gamma] = new_gamma
    return new_factors


def _build_m_vec_gamma_collapse(
    factors: list[TensorExpr],
    i_sigma: int,
    j_gamma: int,
    m_vec_name: str,
) -> list[TensorExpr]:
    """[Σ,γ] identity 의 둘째 term: $-2i \\cdot (M^{ab})^c{}_d \\gamma^d$.

    Σ 제거, γ 의 vector 인덱스를 dummy로 바꾸고 그 자리에 M_vec contraction 추가.
    M_vec.row(slot 2, ↑frame) = 원 γ의 vector 인덱스 c name (free index, 외부 chain
    유지);  M_vec.col(slot 3, ↓frame) = NEW dummy Z;  γ_new.vector(slot 0, ↑frame)
    = Z (M_vec와 contract).

    spinor row/col 은 γ original 그대로 — Σ가 사라지면서 그 spinor 연결이 ψ̄·γ_new
    로 직접 이어진다 (γ_new의 row 가 원래 Σ의 row name = ψ̄ contraction name).
    """
    sigma = factors[i_sigma]
    gamma = factors[j_gamma]
    sigma_row_name = sigma.indices[2].name  # → ψ̄'s contraction
    gamma_vec = gamma.indices[0]            # frame upper, e.g. μ
    gamma_col_name = gamma.indices[2].name  # → ∂ψ's contraction
    spinor_space = sigma.indices[2].space
    frame_space = sigma.indices[0].space

    new_dummy = _fresh_swap_dummy()

    new_gamma = Tensor(
        gamma.name,
        [
            Index(new_dummy, frame_space, "upper"),  # vector → contracts with M_vec.col
            Index(sigma_row_name, spinor_space, "upper"),  # row → ψ̄
            Index(gamma_col_name, spinor_space, "lower"),  # col → ∂ψ
        ],
        antisymmetric_pairs=list(gamma.antisymmetric_pairs),
        reps=dict(gamma.reps),
        statistics=gamma.statistics,
    )
    M_vec = Tensor(
        m_vec_name,
        [
            sigma.indices[0],  # a↑ (Lorentz parameter from Σ)
            sigma.indices[1],  # b↑
            Index(gamma_vec.name, frame_space, "upper"),  # c↑ — original γ vector index
            Index(new_dummy, frame_space, "lower"),       # d↓ — contracts with γ_new vector
        ],
        antisymmetric_pairs=[(0, 1)],
    )

    new_factors = [
        f for k, f in enumerate(factors)
        if k != i_sigma and k != j_gamma
    ]
    new_factors.extend([M_vec, new_gamma])
    return new_factors


# Fresh-dummy counter for Clifford rewriter (separate prefix from generator's _act).
import itertools as _itertools_clifford
_clifford_dummy_counter = _itertools_clifford.count()


def _fresh_swap_dummy(base: str = "_clf") -> str:
    return f"{base}{next(_clifford_dummy_counter)}"


def apply_clifford_sigma_gamma(
    expr: TensorExpr,
    sigma_name: str = "Sigma",
    gamma_name: str = "gamma",
    m_vec_name: str = "M_vec",
) -> TensorExpr:
    """Forward rewrite of $[\\Sigma^{ab}, \\gamma^c]$ using Clifford identity.

    .. math::
        \\Sigma^{ab,\\alpha}{}_\\beta\\, \\gamma^{c,\\beta}{}_\\delta
        \\;\\to\\;
        \\gamma^{c,\\alpha}{}_\\beta\\, \\Sigma^{ab,\\beta}{}_\\delta
        + (-2i)\\, (M^{ab})^c{}_d\\, \\gamma^{d,\\alpha}{}_\\delta

    M4 IR convention 에 맞춘 coefficient $-2i$. 외부 factor (ψ̄, ∂ψ, scalar 등)
    는 그대로 유지되며, Σ-γ 페어가 차지하던 두 자리를 위 두 결과 term 으로 분기.

    한 번에 한 쌍만 처리 — 더 많은 Σ-γ 쌍이 있으면 fixed-point loop 에서 추가 호출.
    """
    if isinstance(expr, TensorSum):
        new_l = apply_clifford_sigma_gamma(expr.left, sigma_name, gamma_name, m_vec_name)
        new_r = apply_clifford_sigma_gamma(expr.right, sigma_name, gamma_name, m_vec_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = apply_clifford_sigma_gamma(expr.expr, sigma_name, gamma_name, m_vec_name)
        if isinstance(new_inner, TensorSum):
            return TensorSum(
                ScalarMul(expr.scalar, new_inner.left),
                ScalarMul(expr.scalar, new_inner.right),
            )
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    pair = _find_sigma_gamma_pair(factors, sigma_name, gamma_name)
    if pair is None:
        # No pattern; recurse into substructure (rarely needed since
        # collect_factors flattens, but PartialDeriv internals stay nested).
        new_l = apply_clifford_sigma_gamma(expr.left, sigma_name, gamma_name, m_vec_name)
        new_r = apply_clifford_sigma_gamma(expr.right, sigma_name, gamma_name, m_vec_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    i_sigma, j_gamma = pair

    # Term 1: γ-Σ swap (factor multiset commutes; rebuild left-associated).
    swap_factors = _build_sigma_gamma_swap(factors, i_sigma, j_gamma)
    term1 = _product_of_factors(swap_factors)

    # Term 2: -2i · M_vec · γ (Σ removed, γ rebuilt with vector contraction).
    collapse_factors = _build_m_vec_gamma_collapse(
        factors, i_sigma, j_gamma, m_vec_name
    )
    term2 = ScalarMul(-2j, _product_of_factors(collapse_factors))

    return TensorSum(term1, term2)


def _find_T_epsilon_pair(
    factors: list[TensorExpr],
    rep_matrix_name: str,
    epsilon_name: str,
) -> tuple[int, int] | None:
    """T (3-index rep matrix) 와 ε (2-index totally antisym invariant) 가 같은
    fund space 의 dummy 한 개로 contract 된 쌍을 찾는다.

    T convention: ``(adj_param ↑, fund_row ↑, fund_col ↓)``.
    ε convention: ``(fund ↓, fund ↓)`` totally antisymmetric.
    T.row(↑) 와 ε 의 두 slot 중 하나(↓) 가 같은 이름 + 같은 IndexSpace 면 매칭.
    """
    for i, T in enumerate(factors):
        if not (
            isinstance(T, Tensor)
            and T.name == rep_matrix_name
            and len(T.indices) == 3
        ):
            continue
        T_row = T.indices[1]
        if T_row.position != "upper":
            continue
        for j, E in enumerate(factors):
            if i == j or not isinstance(E, Tensor):
                continue
            if E.name != epsilon_name or len(E.indices) != 2:
                continue
            # ε 가 antisymmetric 인지 확인 (둘 다 ↓)
            if E.indices[0].position != "lower" or E.indices[1].position != "lower":
                continue
            if not E.antisymmetric_pairs:
                continue
            if E.indices[0].space != T_row.space:
                continue
            if T_row.name == E.indices[0].name or T_row.name == E.indices[1].name:
                return (i, j)
    return None


def apply_epsilon_su_n_invariance(
    expr: TensorExpr,
    rep_matrix_name: str = "T",
    epsilon_name: str = "epsilon",
) -> TensorExpr:
    """SU(N) ε invariance identity 를 normalizer 로 사용해 Leibniz 항을 통일된
    contraction graph 로 변환한다.

    배경:
        ε_{ij...} 는 SU(N) fund rep 의 totally antisymmetric invariant.
        $(T^a)^p{}_q \\epsilon_{p j} + (T^a)^p{}_j \\epsilon_{q p} = 0$
        $\\Leftrightarrow (T^a)^p{}_q \\epsilon_{pj} = (T^a)^p{}_j \\epsilon_{pq}$
        (using $\\epsilon$ antisymmetry).

    이 identity 와 ε antisymmetry 만으로, $\\delta_a(\\bar L^i H^j \\epsilon_{ij} e_R)$
    의 두 Leibniz 항 ((T 가 \\bar L 에 작용한 항) + (T 가 H 에 작용한 항))을
    같은 canonical contraction graph 로 정규화 가능 → ``collect_scalar_terms`` 가
    +i, -i 합산 → 0.

    Normalization 2 단계:
        1. **ε antisym normalize**: T.row 의 이름과 매칭되는 ε slot 이 slot1
           이면, ε 의 두 slot 을 swap 하고 containing ``ScalarMul`` 의 부호를 -1
           곱한다. 결과: T.row ↔ ε.slot0 가 항상 성립.
        2. **Lexicographic identity rewrite**: T.col 과 contract 되는 외부 factor
           X (이름) 와 ε.slot1 과 contract 되는 외부 factor Y (이름) 를 식별.
           ``X.name > Y.name`` 이면 identity 를 적용해 T.col 이름 ↔ ε.slot1 이름을
           swap (factor X, Y 자체는 그대로). 결과적으로 contraction graph 가
           lexicographically 가장 작은 이름이 T.col 에 contract 되는 form 으로
           통일.

    Tradeoff: 두 단계 모두 적용해도 한 ε 와 한 T 가 있는 단일 항에 대해서만
    작동. 여러 T (multi-Leibniz) 또는 여러 ε 가 있는 경우엔 더 일반화 필요 —
    M8 사용 케이스 ($\\bar L H \\epsilon e_R$) 에는 충분.
    """
    if isinstance(expr, TensorSum):
        new_l = apply_epsilon_su_n_invariance(expr.left, rep_matrix_name, epsilon_name)
        new_r = apply_epsilon_su_n_invariance(expr.right, rep_matrix_name, epsilon_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = apply_epsilon_su_n_invariance(expr.expr, rep_matrix_name, epsilon_name)
        if isinstance(new_inner, ScalarMul):
            # 부호 흡수: 외부 scalar * 내부 scalar
            return ScalarMul(expr.scalar * new_inner.scalar, new_inner.expr)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    pair = _find_T_epsilon_pair(factors, rep_matrix_name, epsilon_name)
    if pair is None:
        return expr
    i_T, j_eps = pair
    T = factors[i_T]
    eps = factors[j_eps]
    fund_space = T.indices[2].space

    # Step 1: ε antisym normalize — T.row ↔ ε.slot0
    sign = 1
    T_row_name = T.indices[1].name
    if T_row_name == eps.indices[1].name:
        new_eps = Tensor(
            eps.name,
            [eps.indices[1], eps.indices[0]],
            antisymmetric_pairs=list(eps.antisymmetric_pairs),
            reps=dict(eps.reps),
            statistics=eps.statistics,
        )
        factors[j_eps] = new_eps
        eps = new_eps
        sign = -1
    # else: already T.row ↔ ε.slot0

    # Step 2: Identify X (T.col contraction partner) and Y (ε.slot1 partner).
    # M8.1: partners may be wrapped in PartialDeriv/CovariantDeriv (e.g. ∂L
    # in a chiral kinetic term). _innermost_tensor unwraps; the partner's
    # SU(N) fund index lives on the inner Tensor's slot list, and the inner
    # Tensor's name is what we use for lex comparison.
    T_col_name = T.indices[2].name
    eps_s1_name = eps.indices[1].name

    X_idx = Y_idx = None
    X_inner = Y_inner = None
    for k, f in enumerate(factors):
        if k == i_T or k == j_eps:
            continue
        inner = f if isinstance(f, Tensor) else _innermost_tensor(f)
        if inner is None:
            continue
        for idx in inner.indices:
            if idx.space == fund_space:
                if idx.name == T_col_name and idx.position == "upper":
                    X_idx = k
                    X_inner = inner
                if idx.name == eps_s1_name and idx.position == "upper":
                    Y_idx = k
                    Y_inner = inner

    # 두 partner 가 모두 식별되고 서로 다를 때만 lex normalize 가능.
    if X_idx is not None and Y_idx is not None and X_idx != Y_idx:
        X = X_inner if X_inner is not None else factors[X_idx]
        Y = Y_inner if Y_inner is not None else factors[Y_idx]
        if X.name > Y.name:
            # Identity: swap T.col name <-> ε.slot1 name.
            new_T = Tensor(
                T.name,
                [
                    T.indices[0],
                    T.indices[1],
                    Index(eps_s1_name, fund_space, "lower"),
                ],
                antisymmetric_pairs=list(T.antisymmetric_pairs),
                reps=dict(T.reps),
                statistics=T.statistics,
            )
            new_eps2 = Tensor(
                eps.name,
                [
                    eps.indices[0],
                    Index(T_col_name, fund_space, "lower"),
                ],
                antisymmetric_pairs=list(eps.antisymmetric_pairs),
                reps=dict(eps.reps),
                statistics=eps.statistics,
            )
            factors[i_T] = new_T
            factors[j_eps] = new_eps2

    result = _product_of_factors(factors)
    if sign == -1:
        result = ScalarMul(-1, result)
    return result


# ─── M9: chiral projectors + γ_5 anticommute ──────────────────


# (left.col → right.row) → (sign, result_name) or None (= 0).
# "delta_spinor" is a sentinel: γ_5·γ_5 → 1 (rebuilt as `delta` invariant).
_PROJECTOR_IDENTITY_TABLE: dict[tuple[str, str], tuple[int, str] | None] = {
    ("P_L", "P_L"): (1, "P_L"),
    ("P_R", "P_R"): (1, "P_R"),
    ("P_L", "P_R"): None,
    ("P_R", "P_L"): None,
    ("gamma_5", "P_L"): (1, "P_L"),
    ("P_L", "gamma_5"): (1, "P_L"),
    ("gamma_5", "P_R"): (-1, "P_R"),
    ("P_R", "gamma_5"): (-1, "P_R"),
    ("gamma_5", "gamma_5"): (1, "delta_spinor"),
}

_PROJECTOR_NAMES = {"P_L", "P_R", "gamma_5"}


def _is_projector_factor(f: TensorExpr) -> bool:
    return (
        isinstance(f, Tensor)
        and f.name in _PROJECTOR_NAMES
        and len(f.indices) == 2
        and f.indices[0].position == "upper"
        and f.indices[1].position == "lower"
    )


def _find_projector_pair(
    factors: list[TensorExpr],
) -> tuple[int, int] | None:
    """Adjacent (chain-wise) pair of projector/γ_5 factors.

    Match: factors[i].col(slot 1, lower) name == factors[j].row(slot 0, upper)
    name, same spinor space. (i, j) ordering is left→right in the spinor chain.
    """
    for i, fi in enumerate(factors):
        if not _is_projector_factor(fi):
            continue
        col_i = fi.indices[1]
        for j, fj in enumerate(factors):
            if i == j or not _is_projector_factor(fj):
                continue
            if (fi.name, fj.name) not in _PROJECTOR_IDENTITY_TABLE:
                continue
            row_j = fj.indices[0]
            if col_i.name == row_j.name and col_i.space == row_j.space:
                return (i, j)
    return None


def apply_chiral_projector_identities(expr: TensorExpr) -> TensorExpr:
    """Adjacent P_L/P_R/γ_5 contraction → simplification.

    Patterns (left.col ↔ right.row on spinor space, see
    ``_PROJECTOR_IDENTITY_TABLE``):

    - ``P_L · P_L → P_L``, ``P_R · P_R → P_R``
    - ``P_L · P_R → 0``, ``P_R · P_L → 0``
    - ``γ_5 · P_L → P_L``, ``P_L · γ_5 → P_L``
    - ``γ_5 · P_R → -P_R``, ``P_R · γ_5 → -P_R``
    - ``γ_5 · γ_5 → 1`` (rebuilt as ``δ_spinor`` invariant tying the two
      outside indices).

    한 번에 한 쌍만 처리 — fixed-point loop 에서 반복.
    """
    if isinstance(expr, TensorSum):
        new_l = apply_chiral_projector_identities(expr.left)
        new_r = apply_chiral_projector_identities(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = apply_chiral_projector_identities(expr.expr)
        if isinstance(new_inner, ZeroTensor):
            return new_inner
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    pair = _find_projector_pair(factors)
    if pair is None:
        new_l = apply_chiral_projector_identities(expr.left)
        new_r = apply_chiral_projector_identities(expr.right)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    i, j = pair
    left, right = factors[i], factors[j]
    rule = _PROJECTOR_IDENTITY_TABLE[(left.name, right.name)]

    if rule is None:
        # P_L · P_R or P_R · P_L → 0
        return ZeroTensor(expr.free_indices)

    sign, result_name = rule
    outer_row = left.indices[0]   # upper, attaches to left neighbour in chain
    outer_col = right.indices[1]  # lower, attaches to right neighbour

    new_factors = [f for k, f in enumerate(factors) if k != i and k != j]

    if result_name == "delta_spinor":
        merged = Tensor("delta", [outer_row, outer_col], reps={})
    else:
        merged = Tensor(result_name, [outer_row, outer_col], reps={})
    new_factors.append(merged)

    rebuilt = _product_of_factors(new_factors)
    if sign == -1:
        return ScalarMul(-1, rebuilt)
    return rebuilt


def _find_gamma5_gamma_pair(
    factors: list[TensorExpr],
    gamma5_name: str,
    gamma_name: str,
) -> tuple[int, int] | None:
    """γ_5 (2-index spinor) directly followed by γ (3-index, frame+spinor) in
    the spinor chain. γ_5.col(slot 1, lower) ↔ γ.row(slot 1, upper)."""
    for i, fi in enumerate(factors):
        if not (
            isinstance(fi, Tensor)
            and fi.name == gamma5_name
            and len(fi.indices) == 2
        ):
            continue
        g5_col = fi.indices[1]
        if g5_col.position != "lower":
            continue
        for j, fj in enumerate(factors):
            if i == j:
                continue
            if not (
                isinstance(fj, Tensor)
                and fj.name == gamma_name
                and len(fj.indices) == 3
            ):
                continue
            g_row = fj.indices[1]
            if g_row.position != "upper":
                continue
            if g_row.name == g5_col.name and g_row.space == g5_col.space:
                return (i, j)
    return None


def apply_gamma5_gamma_anticommute(
    expr: TensorExpr,
    gamma5_name: str = "gamma_5",
    gamma_name: str = "gamma",
) -> TensorExpr:
    """{γ_5, γ^μ} = 0 → push γ_5 to the right of every γ.

    .. math::
        \\gamma_5^\\alpha{}_\\beta\\, \\gamma^{\\mu,\\beta}{}_\\delta
        \\;\\to\\;
        -\\,\\gamma^{\\mu,\\alpha}{}_\\rho\\, \\gamma_5^\\rho{}_\\delta

    Outer spinor indices (α, δ) are preserved so the chain reconnects to ψ̄, ψ
    unchanged; a fresh dummy ρ replaces the original contraction.

    한 번에 한 쌍만 처리 — fixed-point loop 에서 반복.
    """
    if isinstance(expr, TensorSum):
        new_l = apply_gamma5_gamma_anticommute(expr.left, gamma5_name, gamma_name)
        new_r = apply_gamma5_gamma_anticommute(expr.right, gamma5_name, gamma_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = apply_gamma5_gamma_anticommute(expr.expr, gamma5_name, gamma_name)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    pair = _find_gamma5_gamma_pair(factors, gamma5_name, gamma_name)
    if pair is None:
        new_l = apply_gamma5_gamma_anticommute(expr.left, gamma5_name, gamma_name)
        new_r = apply_gamma5_gamma_anticommute(expr.right, gamma5_name, gamma_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    i_g5, j_g = pair
    g5 = factors[i_g5]
    g = factors[j_g]
    spinor_space = g5.indices[0].space
    g5_row_name = g5.indices[0].name   # α  (outer left, → ψ̄)
    g_col_name = g.indices[2].name     # δ  (outer right, → ψ)
    new_dummy = _fresh_swap_dummy()

    new_g = Tensor(
        g.name,
        [
            g.indices[0],                                    # μ↑frame
            Index(g5_row_name, spinor_space, "upper"),       # α (was γ_5.row)
            Index(new_dummy, spinor_space, "lower"),         # ρ (new dummy)
        ],
        antisymmetric_pairs=list(g.antisymmetric_pairs),
        reps=dict(g.reps),
        statistics=g.statistics,
    )
    new_g5 = Tensor(
        g5.name,
        [
            Index(new_dummy, spinor_space, "upper"),         # ρ
            Index(g_col_name, spinor_space, "lower"),        # δ (was γ.col)
        ],
        antisymmetric_pairs=list(g5.antisymmetric_pairs),
        reps=dict(g5.reps),
        statistics=g5.statistics,
    )

    new_factors = list(factors)
    new_factors[i_g5] = new_g
    new_factors[j_g] = new_g5
    return ScalarMul(-1, _product_of_factors(new_factors))


def _find_sigma_projector_pair(
    factors: list[TensorExpr],
    sigma_name: str,
) -> tuple[int, int] | None:
    """Σ (4-index: a↑,b↑,row↑,col↓) 가 인접 spinor invariant projector
    (`P_L`/`P_R`/`gamma_5`, 2-index) 의 row 와 contract 된 쌍 (Σ, P).

    Σ.col(slot 3, ↓spinor) ↔ P.row(slot 0, ↑spinor), 같은 spinor space.
    """
    for i, fi in enumerate(factors):
        if not (
            isinstance(fi, Tensor)
            and fi.name == sigma_name
            and len(fi.indices) == 4
        ):
            continue
        sigma_col = fi.indices[3]
        if sigma_col.position != "lower":
            continue
        for j, fj in enumerate(factors):
            if i == j:
                continue
            if not _is_projector_factor(fj):
                continue
            p_row = fj.indices[0]
            if (
                p_row.name == sigma_col.name
                and p_row.space == sigma_col.space
            ):
                return (i, j)
    return None


def apply_sigma_projector_commute(
    expr: TensorExpr,
    sigma_name: str = "Sigma",
) -> TensorExpr:
    """[Σ^{ab}, P_L] = [Σ^{ab}, P_R] = [Σ^{ab}, γ_5] = 0 →
    push Σ past chiral projectors / γ_5 to the right.

    .. math::
        \\Sigma^{ab,\\alpha}{}_\\beta\\, P^\\beta{}_\\gamma
        \\;\\to\\;
        P^\\alpha{}_\\rho\\, \\Sigma^{ab,\\rho}{}_\\gamma

    Σ 가 짝수 개의 γ → γ_5 (그리고 P_{L,R}) 와 commute. 한 쌍씩 처리,
    fixed-point loop 에서 반복.
    """
    if isinstance(expr, TensorSum):
        new_l = apply_sigma_projector_commute(expr.left, sigma_name)
        new_r = apply_sigma_projector_commute(expr.right, sigma_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = apply_sigma_projector_commute(expr.expr, sigma_name)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    pair = _find_sigma_projector_pair(factors, sigma_name)
    if pair is None:
        new_l = apply_sigma_projector_commute(expr.left, sigma_name)
        new_r = apply_sigma_projector_commute(expr.right, sigma_name)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    i_sigma, j_p = pair
    sigma = factors[i_sigma]
    p = factors[j_p]
    spinor_space = sigma.indices[2].space
    sigma_row_name = sigma.indices[2].name  # α (outer left, → ψ̄)
    p_col_name = p.indices[1].name           # γ (outer right, → ψ)
    new_dummy = _fresh_swap_dummy()

    new_p = Tensor(
        p.name,
        [
            Index(sigma_row_name, spinor_space, "upper"),  # α (was Σ.row)
            Index(new_dummy, spinor_space, "lower"),        # ρ (new dummy)
        ],
        antisymmetric_pairs=list(p.antisymmetric_pairs),
        reps=dict(p.reps),
        statistics=p.statistics,
    )
    new_sigma = Tensor(
        sigma.name,
        [
            sigma.indices[0],                                  # a↑frame
            sigma.indices[1],                                  # b↑frame
            Index(new_dummy, spinor_space, "upper"),           # ρ
            Index(p_col_name, spinor_space, "lower"),          # γ (was P.col)
        ],
        antisymmetric_pairs=list(sigma.antisymmetric_pairs),
        reps=dict(sigma.reps),
        statistics=sigma.statistics,
    )

    new_factors = list(factors)
    new_factors[i_sigma] = new_p
    new_factors[j_p] = new_sigma
    return _product_of_factors(new_factors)


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


# ─── M9.6: metric absorption (Einstein raise/lower) ────────


def _is_metric_factor_for_absorption(M: TensorExpr, mreg=None) -> bool:
    """Einstein 흡수 가능 metric factor 식별.

    조건: 두 인덱스가 (i) 같은 IndexSpace, (ii) 그 공간이 metric을 가짐,
    (iii) **같은 position** (둘 다 upper 또는 둘 다 lower — η^{αβ} or η_{αβ}).
    Mixed-position Kronecker δ^α_β는 흡수 대상 아님 (Einstein convention의
    direct contraction은 simplify의 다른 path에서 이미 처리).

    mreg가 주어지면 그것을 우선 사용; 없으면 heuristic (name == "eta"
    + symmetric_pair (0,1) + reps == {})로 fallback.
    """
    if not isinstance(M, Tensor):
        return False
    if len(M.indices) != 2:
        return False
    if M.indices[0].position != M.indices[1].position:
        return False
    if M.indices[0].space != M.indices[1].space:
        return False
    if not M.indices[0].space.metric:
        return False
    if mreg is not None and mreg.is_metric(M):
        return True
    # Heuristic fallback (LIONS builders.make_eta convention)
    if M.name != "eta":
        return False
    if M.reps:
        return False
    sym_set = {tuple(sorted(p)) for p in M.symmetric_pairs}
    if (0, 1) not in sym_set:
        return False
    return True


def _find_slot_with_name(F: TensorExpr, name: str) -> int | None:
    """Tensor F의 인덱스 슬롯 중 이름이 ``name``인 것의 위치 반환.

    PartialDeriv/CovariantDeriv wrapper의 deriv_index는 별도 path —
    여기선 metric 흡수가 inner Tensor에 대해서만이라 None 반환 (deferred).
    """
    if isinstance(F, Tensor):
        for s, idx in enumerate(F.indices):
            if idx.name == name:
                return s
    return None


def _rebuild_with_renamed_slot(
    F: Tensor, slot: int, new_name: str, new_position: str,
) -> Tensor:
    """Tensor의 특정 슬롯 인덱스를 (new_name, new_position)로 교체한 새 Tensor."""
    new_indices = list(F.indices)
    old = new_indices[slot]
    new_indices[slot] = Index(new_name, old.space, new_position)
    return Tensor(
        F.name, new_indices,
        antisymmetric_pairs=list(F.antisymmetric_pairs),
        symmetric_pairs=list(F.symmetric_pairs),
        traceless=list(F.traceless),
        transverse=list(F.transverse),
        reps=dict(F.reps),
        statistics=F.statistics,
    )


def absorb_einstein_metric(expr: TensorExpr, mreg=None) -> TensorExpr:
    """metric η^{αβ} 또는 η_{αβ}를 흡수해 host 텐서 슬롯을 raise/lower.

    Pattern:
        η^{αβ} T_β … = T^α …      (β dummy contracted to T's lower slot)
        η_{αβ} T^β … = T_α …      (β dummy contracted to T's upper slot)

    Preconditions (한 번에 하나 적용; outer simplify fixed-point가 반복):
      - 두 metric 인덱스 α, β 모두 dummy: count == 2 globally, free_indices
        에 없음, distinct.
      - α, β 각각이 다른 (metric 아닌) factor에서 발견 — **cross-factor**.
        Self-trace η^{μν} T_{μν} = T^μ_μ는 host_0 == host_1 → skip (deferred).
      - Host factor가 Tensor (PartialDeriv wrapper는 deferred).

    Effect:
      - metric 제거.
      - host_α의 슬롯을 (name n_β, position = metric의 position)으로 교체.

    이 흡수 후 strict canonical에서 W^A W_A 형식이 dummy 이름만 다른 등가
    factor multiset으로 보여 ``is_zero_by_antisym_swap``이 antisym × sym
    cancellation을 잡을 수 있다 (M9.6 핵심 use case).
    """
    if isinstance(expr, TensorSum):
        new_l = absorb_einstein_metric(expr.left, mreg)
        new_r = absorb_einstein_metric(expr.right, mreg)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = absorb_einstein_metric(expr.expr, mreg)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    if not isinstance(expr, TensorProduct):
        return expr

    factors = collect_factors(expr)
    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}

    for k, M in enumerate(factors):
        if not _is_metric_factor_for_absorption(M, mreg):
            continue
        n_0 = M.indices[0].name
        n_1 = M.indices[1].name
        if n_0 == n_1:
            continue
        if n_0 in free_names or n_1 in free_names:
            continue
        if name_counts.get(n_0, 0) != 2 or name_counts.get(n_1, 0) != 2:
            continue

        host_0 = None
        host_1 = None
        for j, F in enumerate(factors):
            if j == k:
                continue
            if host_0 is None:
                s = _find_slot_with_name(F, n_0)
                if s is not None:
                    host_0 = (j, s)
            if host_1 is None:
                s = _find_slot_with_name(F, n_1)
                if s is not None:
                    host_1 = (j, s)
        if host_0 is None or host_1 is None:
            continue
        if host_0[0] == host_1[0]:
            continue  # self-trace (deferred)

        j_a, slot_a = host_0
        F_a = factors[j_a]
        if not isinstance(F_a, Tensor):
            continue
        new_pos = M.indices[0].position
        F_a_new = _rebuild_with_renamed_slot(F_a, slot_a, n_1, new_pos)

        new_list: list[TensorExpr] = []
        for i, f in enumerate(factors):
            if i == k:
                continue
            new_list.append(F_a_new if i == j_a else f)
        return _product_of_factors(new_list)

    return expr


# ─── simplify (top-level) ───────────────────────────────────


def simplify(expr: TensorExpr, mreg=None) -> TensorExpr:
    """모든 정규화 규칙을 fixed-point까지 적용한다.

    적용 규칙 (M2.5 + M3 + G5):
        - 재귀: TensorSum / ScalarMul / TensorProduct.
        - is_zero_by_antisym_swap (M2: antisym × sym = 0).
          ``Tensor.symmetric_pairs`` slot이 _factor_key_no_swap에서 정렬되므로
          symmetric_pairs 속성도 자동으로 패턴에 사용된다.
        - is_zero_by_traceless_metric (G5a, ``mreg`` 필요).
        - is_zero_by_transverse_deriv (G5b, ``mreg`` 있으면 metric-경유 case도).
        - pull_scalars (M3: ScalarMul hoist).
        - collect_scalar_terms (M3: TensorSum 같은 body 합산).
        - ZeroTensor 흡수 (variation.py).

    Parameters
    ----------
    expr : TensorExpr
    mreg : MetricRegistry, optional
        Provided 시 ``traceless × metric → 0``과 ``transverse × ∂ via metric →
        0`` rule들이 활성화된다.
    """
    from indexcalc.core.variation import _simplify_zeros

    cur = expr
    for _ in range(20):  # max 20 passes — guards against rule-cycle
        prev = cur
        cur = _simplify_once(cur, mreg)
        cur = distribute_products(cur)
        cur = pull_scalars(cur)
        cur = absorb_einstein_metric(cur, mreg)
        cur = commute_partial_through_constants(cur)
        cur = apply_clifford_sigma_gamma(cur)
        cur = apply_gamma5_gamma_anticommute(cur)
        cur = apply_sigma_projector_commute(cur)
        cur = apply_chiral_projector_identities(cur)
        cur = apply_epsilon_su_n_invariance(cur)
        cur = collect_scalar_terms(cur)
        cur = _simplify_zeros(cur)
        if cur is prev:
            break
    return cur


def _simplify_once(expr: TensorExpr, mreg=None) -> TensorExpr:
    """한 번의 simplify pass (재귀 + 한정된 규칙 적용)."""
    if isinstance(expr, TensorProduct):
        new_l = _simplify_once(expr.left, mreg)
        new_r = _simplify_once(expr.right, mreg)
        prod = TensorProduct(new_l, new_r) if (new_l is not expr.left or new_r is not expr.right) else expr
        # antisym swap 0-detection을 product에 적용
        zero_check = is_zero_by_antisym_swap(prod)
        if isinstance(zero_check, ZeroTensor):
            return zero_check
        # G5 rules
        zero_check = is_zero_by_traceless_metric(zero_check, mreg)
        if isinstance(zero_check, ZeroTensor):
            return zero_check
        zero_check = is_zero_by_transverse_deriv(zero_check, mreg)
        return zero_check
    if isinstance(expr, TensorSum):
        new_l = _simplify_once(expr.left, mreg)
        new_r = _simplify_once(expr.right, mreg)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr
    if isinstance(expr, ScalarMul):
        new_inner = _simplify_once(expr.expr, mreg)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr
    if isinstance(expr, (PartialDeriv, CovariantDeriv)):
        # Direct case (A): unwrapped Deriv(T) with transverse slot
        zero_check = is_zero_by_transverse_deriv(expr, mreg)
        return zero_check
    return expr
