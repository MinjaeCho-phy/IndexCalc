"""
Metric raise/lower 모듈.

MetricRegistry로 metric/inverse metric 쌍을 관리하고,
raise_index, lower_index, absorb_metric, expand_metric 연산을 제공한다.
"""

from __future__ import annotations
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.contract import collect_tensors, collect_all_indices


# ─── MetricRegistry ──────────────────────────────────────────

class MetricRegistry:
    """Metric과 inverse metric 쌍을 IndexSpace에 매핑하여 관리한다.

    Parameters
    ----------
    (none — register()로 등록)

    Examples
    --------
    >>> metrics = MetricRegistry()
    >>> metrics.register(g, g_inv, spacetime)
    >>> metrics.get_metric(spacetime)
    g_μ_ν
    """

    def __init__(self):
        self._registry: dict[str, tuple[Tensor, Tensor]] = {}

    def register(self, metric: Tensor, inverse: Tensor, space: IndexSpace) -> None:
        """Metric과 inverse metric 쌍을 등록한다.

        Parameters
        ----------
        metric : Tensor
            Lower index metric (e.g., g_{μν}).
        inverse : Tensor
            Upper index inverse metric (e.g., g^{μν}).
        space : IndexSpace
            이 metric이 속하는 공간.
        """
        self._registry[space.name] = (metric, inverse)

    def get_metric(self, space: IndexSpace) -> Tensor:
        """공간의 metric 텐서를 반환한다."""
        return self._registry[space.name][0]

    def get_inverse(self, space: IndexSpace) -> Tensor:
        """공간의 inverse metric 텐서를 반환한다."""
        return self._registry[space.name][1]

    def has_space(self, space: IndexSpace) -> bool:
        return space.name in self._registry

    def is_metric(self, tensor: Tensor) -> IndexSpace | None:
        """이 텐서가 등록된 metric 또는 inverse metric인지 확인한다.

        Returns
        -------
        IndexSpace | None
            metric이면 해당 공간, 아니면 None.
        """
        for space_name, (met, inv) in self._registry.items():
            if tensor.name == met.name or tensor.name == inv.name:
                # 인덱스 공간도 확인
                for idx in tensor.indices:
                    if idx.space.name == space_name:
                        return idx.space
        return None


# ─── Dummy index 생성 ────────────────────────────────────────

def _generate_dummy(space: IndexSpace, existing: set[str], position: str) -> Index:
    """충돌 없는 내부용 dummy index를 생성한다.

    형식: space.indices[0] + "_" + 숫자 (e.g., μ_1, μ_2, ...)

    Parameters
    ----------
    space : IndexSpace
        인덱스가 속할 공간.
    existing : set[str]
        현재 표현식에서 이미 사용 중인 인덱스 이름들.
    position : str
        "upper" or "lower".

    Returns
    -------
    Index
        충돌 없는 새 dummy index.
    """
    base = space.indices[0] if space.indices else "i"
    counter = 1
    while f"{base}_{counter}" in existing:
        counter += 1
    name = f"{base}_{counter}"
    return Index(name, space, position)


def _collect_existing_names(expr: TensorExpr) -> set[str]:
    """표현식에서 사용 중인 모든 인덱스 이름을 수집한다."""
    return {idx.name for idx in collect_all_indices(expr)}


# ─── raise / lower ───────────────────────────────────────────

def raise_index(expr: TensorExpr, index_name: str, metrics: MetricRegistry) -> TensorExpr:
    """Lower index를 upper로 올린다 (inverse metric 삽입).

    V_{μ} → g^{μ, μ_1} V_{μ_1}

    대상 인덱스가 이미 upper이면 ValueError.

    Parameters
    ----------
    expr : TensorExpr
        대상 표현식.
    index_name : str
        올릴 인덱스의 이름.
    metrics : MetricRegistry
        Metric 레지스트리.

    Returns
    -------
    TensorExpr
        Inverse metric이 삽입된 새 표현식.
    """
    return _shift_index(expr, index_name, metrics, direction="raise")


def lower_index(expr: TensorExpr, index_name: str, metrics: MetricRegistry) -> TensorExpr:
    """Upper index를 lower로 내린다 (metric 삽입).

    V^{μ} → g_{μ, μ_1} V^{μ_1}

    대상 인덱스가 이미 lower이면 ValueError.

    Parameters
    ----------
    expr : TensorExpr
        대상 표현식.
    index_name : str
        내릴 인덱스의 이름.
    metrics : MetricRegistry
        Metric 레지스트리.

    Returns
    -------
    TensorExpr
        Metric이 삽입된 새 표현식.
    """
    return _shift_index(expr, index_name, metrics, direction="lower")


def _shift_index(
    expr: TensorExpr,
    index_name: str,
    metrics: MetricRegistry,
    direction: str,  # "raise" or "lower"
) -> TensorExpr:
    """raise_index / lower_index의 공통 구현."""

    # 1. 대상 인덱스 찾기
    free = expr.free_indices
    target = None
    for idx in free:
        if idx.name == index_name:
            target = idx
            break

    if target is None:
        raise ValueError(
            f"Index '{index_name}' not found in free indices: {free}"
        )

    if direction == "raise" and target.position == "upper":
        raise ValueError(f"Index '{index_name}' is already upper, cannot raise")
    if direction == "lower" and target.position == "lower":
        raise ValueError(f"Index '{index_name}' is already lower, cannot lower")

    space = target.space
    if not metrics.has_space(space):
        raise ValueError(f"No metric registered for space '{space.name}'")

    # 2. Dummy index 생성
    existing = _collect_existing_names(expr)
    dummy_position = target.position  # dummy는 원래 위치를 이어받음
    dummy = _generate_dummy(space, existing, dummy_position)

    # 3. 표현식에서 대상 인덱스를 dummy로 교체
    new_expr = _replace_index(expr, target, dummy)

    # 4. Metric 텐서 생성
    if direction == "raise":
        # g^{μ, dummy} — 둘 다 upper가 아니라, μ가 upper(올려진 결과), dummy가 upper(contraction용)
        raised_idx = Index(index_name, space, "upper")
        met_dummy = Index(dummy.name, space, "upper")
        met_tensor = Tensor(
            metrics.get_inverse(space).name,
            [raised_idx, met_dummy],
        )
    else:  # lower
        lowered_idx = Index(index_name, space, "lower")
        met_dummy = Index(dummy.name, space, "lower")
        met_tensor = Tensor(
            metrics.get_metric(space).name,
            [lowered_idx, met_dummy],
        )

    # 5. metric * expr
    return TensorProduct(met_tensor, new_expr)


def _replace_index(expr: TensorExpr, old: Index, new: Index) -> TensorExpr:
    """표현식 트리에서 특정 인덱스를 교체한다 (새 트리 반환)."""

    if isinstance(expr, Tensor):
        new_indices = [
            new if (idx.name == old.name and idx.space == old.space
                    and idx.position == old.position)
            else idx
            for idx in expr.indices
        ]
        return Tensor(expr.name, new_indices)

    if isinstance(expr, TensorProduct):
        return TensorProduct(
            _replace_index(expr.left, old, new),
            _replace_index(expr.right, old, new),
        )

    if isinstance(expr, TensorSum):
        return TensorSum(
            _replace_index(expr.left, old, new),
            _replace_index(expr.right, old, new),
        )

    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, _replace_index(expr.expr, old, new))

    return expr


# ─── absorb_metric ───────────────────────────────────────────

def absorb_metric(expr: TensorExpr, metrics: MetricRegistry) -> TensorExpr:
    """표현식에서 metric * tensor 패턴을 찾아 metric을 흡수한다.

    g_{μν} V^{ν} → V_{μ}
    g^{μν} T_{νλ} → T^{μ}_{λ}

    Option B 방식: 인덱스 룩업 맵을 빌드하고 O(1)로 대상을 찾는다.

    Parameters
    ----------
    expr : TensorExpr
        대상 표현식.
    metrics : MetricRegistry
        Metric 레지스트리.

    Returns
    -------
    TensorExpr
        Metric이 흡수된 새 표현식.
    """
    return _absorb_recursive(expr, metrics)


def _absorb_recursive(expr: TensorExpr, metrics: MetricRegistry) -> TensorExpr:
    """재귀적으로 absorb를 적용한다."""

    if isinstance(expr, Tensor):
        return expr

    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, _absorb_recursive(expr.expr, metrics))

    if isinstance(expr, TensorSum):
        return TensorSum(
            _absorb_recursive(expr.left, metrics),
            _absorb_recursive(expr.right, metrics),
        )

    if isinstance(expr, TensorProduct):
        # 먼저 하위 트리를 재귀 처리
        left = _absorb_recursive(expr.left, metrics)
        right = _absorb_recursive(expr.right, metrics)

        # 평탄화: 곱에 참여하는 모든 텐서와 스칼라를 수집
        factors, scalar = _flatten_product(TensorProduct(left, right))

        # 인덱스 룩업 맵 빌드
        # {(index_name, space_name): [(factor_idx, slot_idx, index)]}
        index_map: dict[tuple[str, str], list[tuple[int, int, Index]]] = {}
        for fi, tensor in enumerate(factors):
            for si, idx in enumerate(tensor.indices):
                key = (idx.name, idx.space.name)
                index_map.setdefault(key, []).append((fi, si, idx))

        # Metric 텐서 찾기 & 흡수
        absorbed = set()  # 흡수된 factor 인덱스
        new_factors = list(factors)

        for fi, tensor in enumerate(factors):
            if fi in absorbed:
                continue

            space = metrics.is_metric(tensor)
            if space is None:
                continue
            if len(tensor.indices) != 2:
                continue

            idx_a, idx_b = tensor.indices

            # metric의 두 인덱스 중 다른 텐서와 contracted되는 쪽 찾기
            contracted_slot, free_slot = _find_metric_contraction(
                fi, tensor, factors, index_map, absorbed,
            )

            if contracted_slot is None:
                continue

            # 흡수 실행
            met_free_idx = tensor.indices[free_slot]
            met_contracted_idx = tensor.indices[contracted_slot]

            # contracted 대상 텐서 찾기
            key = (met_contracted_idx.name, met_contracted_idx.space.name)
            for target_fi, target_si, target_idx in index_map[key]:
                if target_fi != fi and target_fi not in absorbed:
                    if target_idx.position != met_contracted_idx.position:
                        # 대상 텐서의 해당 slot을 metric의 free index로 교체
                        old_tensor = new_factors[target_fi]
                        new_indices = list(old_tensor.indices)
                        # 새 인덱스: metric의 free index 이름 + 반대 위치
                        new_indices[target_si] = Index(
                            met_free_idx.name,
                            met_free_idx.space,
                            met_free_idx.position,
                        )
                        new_factors[target_fi] = Tensor(old_tensor.name, new_indices)
                        absorbed.add(fi)
                        break

        # 흡수되지 않은 factor들로 다시 곱 구성
        remaining = [f for i, f in enumerate(new_factors) if i not in absorbed]

        if not remaining:
            # 모든 게 흡수됨 (이런 경우는 거의 없지만 안전장치)
            from indexcalc.parse.latex import _ScalarOne
            result = _ScalarOne()
        else:
            result = remaining[0]
            for f in remaining[1:]:
                result = TensorProduct(result, f)

        if scalar != 1:
            result = ScalarMul(scalar, result)

        return result

    return expr


def _flatten_product(expr: TensorExpr) -> tuple[list[Tensor], float]:
    """중첩된 TensorProduct를 평탄화하여 (텐서 리스트, 스칼라) 반환."""
    tensors = []
    scalar = 1.0

    def _walk(e: TensorExpr):
        nonlocal scalar
        if isinstance(e, Tensor):
            tensors.append(e)
        elif isinstance(e, TensorProduct):
            _walk(e.left)
            _walk(e.right)
        elif isinstance(e, ScalarMul):
            scalar *= e.scalar
            _walk(e.expr)
        # TensorSum, Trace 등은 평탄화하지 않음 — 그대로 둠

    _walk(expr)
    return tensors, scalar


def _find_metric_contraction(
    metric_fi: int,
    metric_tensor: Tensor,
    factors: list[Tensor],
    index_map: dict,
    absorbed: set[int],
) -> tuple[int | None, int | None]:
    """Metric 텐서의 두 인덱스 중 다른 텐서와 contracted되는 slot을 찾는다.

    Returns
    -------
    (contracted_slot, free_slot) or (None, None)
    """
    for slot in (0, 1):
        idx = metric_tensor.indices[slot]
        key = (idx.name, idx.space.name)

        if key not in index_map:
            continue

        for target_fi, target_si, target_idx in index_map[key]:
            if target_fi != metric_fi and target_fi not in absorbed:
                if target_idx.position != idx.position:
                    # 이 slot이 contracted됨
                    free_slot = 1 - slot
                    return slot, free_slot

    return None, None


# ─── expand_metric ───────────────────────────────────────────

def expand_metric(
    expr: TensorExpr,
    index_name: str,
    metrics: MetricRegistry,
) -> TensorExpr:
    """흡수된 인덱스를 다시 metric으로 펼친다.

    V_{μ} → g_{μ, μ_1} V^{μ_1}   (absorb의 역연산)

    내부적으로 lower index는 metric + raise,
    upper index는 inverse metric + lower로 처리한다.

    Parameters
    ----------
    expr : TensorExpr
        대상 표현식.
    index_name : str
        펼칠 인덱스의 이름.
    metrics : MetricRegistry
        Metric 레지스트리.

    Returns
    -------
    TensorExpr
        Metric이 명시된 새 표현식.
    """
    # expand는 raise/lower의 역: lower index면 metric 삽입 + flip
    free = expr.free_indices
    target = None
    for idx in free:
        if idx.name == index_name:
            target = idx
            break

    if target is None:
        raise ValueError(
            f"Index '{index_name}' not found in free indices: {free}"
        )

    # lower index → metric 삽입하고 원래 텐서의 index를 upper로
    # upper index → inverse metric 삽입하고 원래 텐서의 index를 lower로
    if target.position == "lower":
        return _shift_index(expr, index_name, metrics, direction="raise")
    else:
        return _shift_index(expr, index_name, metrics, direction="lower")
