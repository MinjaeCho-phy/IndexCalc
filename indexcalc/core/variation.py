"""
변분 연산자: Variation, VariationRegistry, expand_variation.

δ(expr)을 symbolic하게 표현하고 Leibniz rule로 전개한다.
δ(A + B) → δA + δB, δ(c*A) → c*δA, δ(A*B) → (δA)*B + A*(δB).
Tensor leaf에서는 VariationRegistry를 참조하여 δT → "δT" 텐서로 치환하거나
background이면 ZeroTensor로 소거한다.
"""

from __future__ import annotations
from indexcalc.core.index import Index
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)


def _strip_delta_prefix(name: str) -> str:
    """'δδA' → 'A'. 선행 'δ'를 전부 제거한다."""
    i = 0
    while i < len(name) and name[i] == "δ":
        i += 1
    return name[i:]


class ZeroTensor(TensorExpr):
    """변분이 0인 background 텐서의 결과. free index 구조를 보존한다."""

    def __init__(self, free: list[Index]):
        self._free = list(free)

    @property
    def free_indices(self) -> list[Index]:
        return list(self._free)

    def __repr__(self) -> str:
        return "0"


class Variation(TensorExpr):
    """δ(expr) — symbolic 변분 연산자. free_indices는 inner와 동일."""

    def __init__(self, expr: TensorExpr, order: int = 1):
        self.expr = expr
        self.order = order

    @property
    def free_indices(self) -> list[Index]:
        return self.expr.free_indices

    def __repr__(self) -> str:
        return f"δ({self.expr})"


class VariationRegistry:
    """어떤 텐서가 varying이고 어떤 것이 background(δ=0)인지 관리한다.

    Examples
    --------
    >>> vreg = VariationRegistry()
    >>> vreg.declare_varying("e")       # δe 자동 생성
    >>> vreg.declare_background("η")    # δη = 0
    """

    def __init__(self):
        self._rules: dict[str, TensorExpr | None] = {}
        self._background: set[str] = set()
        self._varying_connections: set[str] = set()

    def declare_varying_connection(self, symbol: str):
        """Connection symbol을 varying으로 선언한다 (Palatini: δΓ ≠ 0).

        ``δ(∇_μ T)`` 전개 시 ``∇_μ(δT) + δΓ·T`` 에서 δΓ 항이 살아남는다.
        """
        self._varying_connections.add(symbol)

    def declare_varying(self, name: str, replacement: TensorExpr | None = None):
        """name을 varying으로 선언한다.

        replacement=None이면 기본으로 "δ" + name prefix 텐서를 생성한다.
        """
        self._rules[name] = replacement

    def declare_background(self, name: str):
        """name을 background(δ=0)로 선언한다."""
        self._background.add(name)

    def delta_of(self, tensor: Tensor) -> TensorExpr:
        """단일 Tensor에 δ를 적용한다.

        Returns
        -------
        TensorExpr
            varying이면 δ-prefixed Tensor, background이면 ZeroTensor.

        Raises
        ------
        ValueError
            선언되지 않은 텐서에 δ를 적용하려 할 때.
        """
        if tensor.name in self._background:
            return ZeroTensor(tensor.free_indices)
        if tensor.name in self._rules:
            template = self._rules[tensor.name]
            if template is None:
                return Tensor("δ" + tensor.name, list(tensor.indices))
            return _substitute_indices(template, tensor.indices)
        # 자동 규칙: 이름이 "δ"로 시작하고 본체가 이미 선언되어 있으면
        # δⁿ⁺¹ 로 한 레벨 더 prefix. (P5: 2차 이상 변분 지원)
        stripped = _strip_delta_prefix(tensor.name)
        if stripped != tensor.name:
            if stripped in self._background:
                return ZeroTensor(tensor.free_indices)
            if stripped in self._rules:
                return Tensor("δ" + tensor.name, list(tensor.indices))
        raise ValueError(
            f"'{tensor.name}' not declared in VariationRegistry. "
            f"Use declare_varying() or declare_background() first."
        )


def _substitute_indices(
    template: TensorExpr, indices: tuple[Index, ...],
) -> TensorExpr:
    """template의 인덱스를 실제 인덱스로 치환한다 (P8용)."""
    raise NotImplementedError("Custom replacement templates not yet supported")


# ─── expand_variation ────────────────────────────────────────


def expand_variation(
    expr: TensorExpr, registry: VariationRegistry,
) -> TensorExpr:
    """Variation 노드를 Leibniz rule로 전개한다.

    Parameters
    ----------
    expr : TensorExpr
        전개할 표현식. Variation 노드를 찾아서 전개한다.
    registry : VariationRegistry
        각 텐서의 varying/background 선언.

    Returns
    -------
    TensorExpr
        δ가 전개된 표현식. Variation 노드가 Tensor leaf로 치환된다.
    """
    if isinstance(expr, Variation):
        return _apply_delta(expr.expr, registry)

    if isinstance(expr, TensorProduct):
        return TensorProduct(
            expand_variation(expr.left, registry),
            expand_variation(expr.right, registry),
        )
    if isinstance(expr, TensorSum):
        return TensorSum(
            expand_variation(expr.left, registry),
            expand_variation(expr.right, registry),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, expand_variation(expr.expr, registry))

    return expr


def _apply_delta(
    inner: TensorExpr, registry: VariationRegistry,
) -> TensorExpr:
    """Variation 내부 표현식에 δ를 적용한다 (Leibniz rule 재귀)."""
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
    from indexcalc.core.contract import Trace

    # δ(0) = 0
    if isinstance(inner, ZeroTensor):
        return inner

    # δ(Tensor) → registry lookup
    if isinstance(inner, Tensor):
        return _simplify_zeros(registry.delta_of(inner))

    # δ(A + B) → δA + δB
    if isinstance(inner, TensorSum):
        left = _apply_delta(inner.left, registry)
        right = _apply_delta(inner.right, registry)
        return _simplify_zeros(TensorSum(left, right))

    # δ(A * B) → (δA)*B + A*(δB)
    if isinstance(inner, TensorProduct):
        dA_B = TensorProduct(_apply_delta(inner.left, registry), inner.right)
        A_dB = TensorProduct(inner.left, _apply_delta(inner.right, registry))
        return _simplify_zeros(TensorSum(dA_B, A_dB))

    # δ(c * A) → c * δA
    if isinstance(inner, ScalarMul):
        return _simplify_zeros(
            ScalarMul(inner.scalar, _apply_delta(inner.expr, registry))
        )

    # δ(Tr T) → Tr(δT)  (trace 인덱스는 변분과 무관하게 보존)
    if isinstance(inner, Trace):
        inner_delta = _apply_delta(inner.tensor, registry)
        if isinstance(inner_delta, ZeroTensor):
            return ZeroTensor(inner.free_indices)
        if isinstance(inner_delta, Tensor):
            return Trace(inner_delta, inner.index_name)
        # 전개 후 Tensor가 아니게 되면(드물다) 그대로 반환
        return inner_delta

    # δ(∂_μ X) → ∂_μ(δX)
    if isinstance(inner, PartialDeriv):
        return PartialDeriv(
            _apply_delta(inner.expr, registry), inner.deriv_index,
        )

    # δ(∇_μ X) → ∇_μ(δX) + (δΓ·X)  (P6: Palatini)
    if isinstance(inner, CovariantDeriv):
        return _apply_delta_covariant(inner, registry)

    # δ(δX) → δ를 두 번 적용 (P5: 완전 전개)
    if isinstance(inner, Variation):
        once = _apply_delta(inner.expr, registry)
        return _apply_delta(once, registry)

    # 알 수 없는 노드 → Variation으로 감싸서 유지
    return Variation(inner)


def _apply_delta_covariant(
    cov, registry: VariationRegistry,
) -> TensorExpr:
    """δ(∇_μ expr) 전개. varying connection이 있으면 Palatini δΓ 항 추가."""
    from indexcalc.core.deriv import CovariantDeriv

    inner = cov.expr
    mu = cov.deriv_index
    conns = cov.connections

    # 복합 내부면 먼저 ∇ Leibniz로 분배 후 재귀
    if not isinstance(inner, Tensor):
        distributed = _distribute_nabla_once(cov)
        if distributed is cov:
            # 더 분배할 수 없는 노드 — background connection으로 가정하고 δ 이동
            return type(cov)(_apply_delta(inner, registry), mu, conns)
        return _apply_delta(distributed, registry)

    T = inner
    dT = _apply_delta(T, registry)

    cov_cls = type(cov)
    if isinstance(dT, ZeroTensor):
        base: TensorExpr = ZeroTensor([mu] + list(T.free_indices))
    else:
        base = cov_cls(dT, mu, conns)

    correction = _palatini_correction(T, mu, conns, registry._varying_connections)
    if correction is None:
        return _simplify_zeros(base)
    if isinstance(base, ZeroTensor):
        return _simplify_zeros(correction)
    return _simplify_zeros(TensorSum(base, correction))


def _distribute_nabla_once(cov) -> TensorExpr:
    """∇_μ(A*B) → ∇A·B + A·∇B 등의 한 단계 Leibniz 분배.

    SpatialCovariantDeriv 등 subclass가 유지되도록 type(cov)로 재생성한다.
    """
    cov_cls = type(cov)
    inner = cov.expr
    mu = cov.deriv_index
    conns = cov.connections

    if isinstance(inner, TensorProduct):
        return TensorSum(
            TensorProduct(cov_cls(inner.left, mu, conns), inner.right),
            TensorProduct(inner.left, cov_cls(inner.right, mu, conns)),
        )
    if isinstance(inner, TensorSum):
        return TensorSum(
            cov_cls(inner.left, mu, conns),
            cov_cls(inner.right, mu, conns),
        )
    if isinstance(inner, ScalarMul):
        return ScalarMul(inner.scalar, cov_cls(inner.expr, mu, conns))
    return cov


def _palatini_correction(
    tensor: Tensor,
    mu: Index,
    connections: dict,
    varying_syms: set[str],
) -> TensorExpr | None:
    """δΓ·T 형태의 Palatini 보정 항을 생성한다.

    각 slot s에 대해 해당 공간에 varying connection이 있으면:
      - upper index a: + δΓ^a_{μ, ρ} · T(slot s ← ρ_upper)
      - lower index b: - δΓ^ρ_{μ, b} · T(slot s ← ρ_lower)
    """
    existing = {idx.name for idx in tensor.indices} | {mu.name}
    terms: list[TensorExpr] = []

    for slot, idx in enumerate(tensor.indices):
        space_name = idx.space.name
        if space_name not in connections:
            continue
        conn = connections[space_name]
        if conn.symbol not in varying_syms:
            continue

        base_char = idx.space.indices[0] if idx.space.indices else "i"
        counter = 1
        while f"{base_char}_{counter}" in existing:
            counter += 1
        dummy_name = f"{base_char}_{counter}"
        existing.add(dummy_name)

        dsym = "δ" + conn.symbol
        new_indices = list(tensor.indices)

        if idx.position == "upper":
            dgamma = Tensor(dsym, [
                Index(idx.name, idx.space, "upper"),
                Index(mu.name, mu.space, "lower"),
                Index(dummy_name, idx.space, "lower"),
            ])
            new_indices[slot] = Index(dummy_name, idx.space, "upper")
            Tp = Tensor(tensor.name, new_indices)
            terms.append(TensorProduct(dgamma, Tp))
        else:
            dgamma = Tensor(dsym, [
                Index(dummy_name, idx.space, "upper"),
                Index(mu.name, mu.space, "lower"),
                Index(idx.name, idx.space, "lower"),
            ])
            new_indices[slot] = Index(dummy_name, idx.space, "lower")
            Tp = Tensor(tensor.name, new_indices)
            terms.append(ScalarMul(-1, TensorProduct(dgamma, Tp)))

    if not terms:
        return None
    result = terms[0]
    for t in terms[1:]:
        result = TensorSum(result, t)
    return result


# ─── ZeroTensor 정리 ─────────────────────────────────────────


def _simplify_zeros(expr: TensorExpr) -> TensorExpr:
    """ZeroTensor가 포함된 합/곱을 정리한다.

    0 + A → A,  A + 0 → A
    0 * A → 0,  A * 0 → 0
    c * 0 → 0
    """
    if isinstance(expr, TensorSum):
        left = _simplify_zeros(expr.left)
        right = _simplify_zeros(expr.right)
        if isinstance(left, ZeroTensor):
            return right
        if isinstance(right, ZeroTensor):
            return left
        if left is not expr.left or right is not expr.right:
            return TensorSum(left, right)
        return expr

    if isinstance(expr, TensorProduct):
        left = _simplify_zeros(expr.left)
        right = _simplify_zeros(expr.right)
        if isinstance(left, ZeroTensor) or isinstance(right, ZeroTensor):
            return ZeroTensor(expr.free_indices)
        if left is not expr.left or right is not expr.right:
            return TensorProduct(left, right)
        return expr

    if isinstance(expr, ScalarMul):
        inner = _simplify_zeros(expr.expr)
        if isinstance(inner, ZeroTensor):
            return inner
        if inner is not expr.expr:
            return ScalarMul(expr.scalar, inner)
        return expr

    return expr
