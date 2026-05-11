"""
Antisymmetric/symmetric tensor 속성의 canonicalization + 명시적
대칭화/반대칭화/traceless-symmetric 연산자 노드.

Tensor에 ``antisymmetric_pairs``가 선언되어 있으면, 해당 slot 쌍의 인덱스가
canonical 순서(이름 오름차순, 단 같은 이름일 땐 position upper 먼저)로
정렬되도록 재배치하고, 교환으로 인한 부호를 ScalarMul로 누적한다.

표현식 트리 노드:
    Sym(expr, [i, j])           — T_{(ij)} = ½(T_{ij} + T_{ji})
    Antisym(expr, [i, j])       — T_{[ij]} = ½(T_{ij} − T_{ji})
    TraceFreeSym(expr, [i, j])  — T_{⟨ij⟩} = T_{(ij)} − (1/n)γ_{ij} γ^{kl} T_{(kl)}

n=2 swap만 우선 지원 (cosmological perturbation의 D_(i E_j), D_⟨i D_j⟩ E 가
모두 n=2). 일반 n!은 후속.
"""

from __future__ import annotations
from indexcalc.core.index import Index
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


# ─── 명시적 대칭화/반대칭화/traceless-sym 노드 ─────────────────

class _SymmetrizerBase(TensorExpr):
    """Sym/Antisym/TraceFreeSym 공통 베이스. n=2 indices만 지원."""

    _latex_left: str = "?"
    _latex_right: str = "?"

    def __init__(self, expr: TensorExpr, indices: list[Index]):
        if len(indices) != 2:
            raise NotImplementedError(
                f"{type(self).__name__}: n=2 indices only (got {len(indices)}). "
                "일반 n!은 후속에서 지원 예정."
            )
        # 두 인덱스가 같은 IndexSpace에 있어야 swap 의미가 있음
        if indices[0].space != indices[1].space:
            raise ValueError(
                f"{type(self).__name__} indices must share IndexSpace; got "
                f"{indices[0].space.name!r} and {indices[1].space.name!r}"
            )
        # 두 인덱스가 expr의 free indices에 모두 등장해야 함
        free_names = {i.name for i in expr.free_indices}
        for idx in indices:
            if idx.name not in free_names:
                raise ValueError(
                    f"{type(self).__name__}: index {idx.name!r} not free in expr"
                )
        self.expr = expr
        self.sym_indices: tuple[Index, ...] = tuple(indices)

    @property
    def free_indices(self) -> list[Index]:
        return list(self.expr.free_indices)

    def __repr__(self) -> str:
        names = "".join(i.name for i in self.sym_indices)
        return f"{type(self).__name__}_{{{names}}}[{self.expr!r}]"


class Sym(_SymmetrizerBase):
    """T_{(ij)} = ½(T_{ij} + T_{ji})."""
    _latex_left = "("
    _latex_right = ")"


class Antisym(_SymmetrizerBase):
    """T_{[ij]} = ½(T_{ij} − T_{ji})."""
    _latex_left = "["
    _latex_right = "]"


class TraceFreeSym(_SymmetrizerBase):
    """T_{⟨ij⟩} = T_{(ij)} − (1/n) γ_{ij} γ^{kl} T_{(kl)}.

    expand 시 metric을 알아야 trace 항을 만들 수 있다. 첫 패스에서는 노드 자체만
    제공하고, expand_symmetrization에서 metric registry를 함께 받아 처리한다.
    """
    _latex_left = "⟨"
    _latex_right = "⟩"


# ─── swap & expand ───────────────────────────────────────────────

def _swap_index_in_tensor(t: Tensor, a: Index, b: Index) -> Tensor:
    """Tensor의 indices 중 a와 b를 자리바꿈해서 새 Tensor를 반환.

    동일 IndexSpace 가정. a와 b는 position이 같다고 가정(보통 둘 다 lower).
    """
    new = []
    for idx in t.indices:
        if idx == a:
            new.append(b)
        elif idx == b:
            new.append(a)
        else:
            new.append(idx)
    return Tensor(
        t.name, new,
        antisymmetric_pairs=[tuple(p) for p in t.antisymmetric_pairs],
        symmetric_pairs=[tuple(p) for p in t.symmetric_pairs],
        traceless=[tuple(p) for p in t.traceless],
        transverse=list(t.transverse),
        reps=dict(t.reps),
        statistics=t.statistics,
    )


def _swap_in_expr(expr: TensorExpr, a: Index, b: Index) -> TensorExpr:
    """expr 전체에서 free index a ↔ b 자리바꿈한 새 expr."""
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv

    if isinstance(expr, Tensor):
        return _swap_index_in_tensor(expr, a, b)
    if isinstance(expr, TensorSum):
        return TensorSum(
            _swap_in_expr(expr.left, a, b),
            _swap_in_expr(expr.right, a, b),
        )
    if isinstance(expr, TensorProduct):
        return TensorProduct(
            _swap_in_expr(expr.left, a, b),
            _swap_in_expr(expr.right, a, b),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, _swap_in_expr(expr.expr, a, b))
    if isinstance(expr, PartialDeriv):
        new_inner = _swap_in_expr(expr.expr, a, b)
        new_di = expr.deriv_index
        if new_di == a:
            new_di = b
        elif new_di == b:
            new_di = a
        return PartialDeriv(new_inner, new_di)
    if isinstance(expr, CovariantDeriv):
        new_inner = _swap_in_expr(expr.expr, a, b)
        new_di = expr.deriv_index
        if new_di == a:
            new_di = b
        elif new_di == b:
            new_di = a
        return type(expr)(new_inner, new_di, expr.connections)
    # 기타 노드는 그대로
    return expr


def expand_symmetrization(expr: TensorExpr, metrics=None) -> TensorExpr:
    """Sym/Antisym/TraceFreeSym 노드를 explicit form으로 전개.

    Parameters
    ----------
    expr : TensorExpr
    metrics : MetricRegistry, optional
        TraceFreeSym 전개에 필요. 없으면 TraceFreeSym 마주칠 시 에러.
    """
    if isinstance(expr, Sym):
        inner = expand_symmetrization(expr.expr, metrics)
        a, b = expr.sym_indices
        swapped = _swap_in_expr(inner, a, b)
        return ScalarMul(0.5, TensorSum(inner, swapped))

    if isinstance(expr, Antisym):
        inner = expand_symmetrization(expr.expr, metrics)
        a, b = expr.sym_indices
        swapped = _swap_in_expr(inner, a, b)
        return ScalarMul(0.5, TensorSum(inner, ScalarMul(-1, swapped)))

    if isinstance(expr, TraceFreeSym):
        if metrics is None:
            raise ValueError(
                "TraceFreeSym expansion needs a MetricRegistry "
                "to subtract the trace."
            )
        # T_{⟨ij⟩} = T_{(ij)} − (1/n) γ_{ij} γ^{kl} T_{(kl)}
        inner = expand_symmetrization(expr.expr, metrics)
        a, b = expr.sym_indices
        space = a.space
        n = space.dim
        # symmetric part
        swapped_ab = _swap_in_expr(inner, a, b)
        sym_part = ScalarMul(0.5, TensorSum(inner, swapped_ab))
        # trace part: 1/n · γ_{ij} · γ^{kl} · Sym(T)_{kl}
        # k, l 더미 이름 — 기존 free와 충돌 회피
        existing = {i.name for i in inner.free_indices}
        k_name = next(c for c in "klmnopqrstuvwxyz" if c not in existing)
        l_name = next(c for c in "lmnopqrstuvwxyz" if c not in existing and c != k_name)
        k = Index(k_name, space, "lower")
        l = Index(l_name, space, "lower")
        # T_{(kl)}: a→k, b→l 로 swap 후 sym
        T_kl = _swap_in_expr(_swap_in_expr(inner, a, k), b, l)
        T_lk = _swap_in_expr(T_kl, k, l)
        γ_ij_at = Tensor(
            metrics.get_metric(space).name,
            [a, b],
            symmetric_pairs=[(0, 1)],
        )
        γ_inv_kl = Tensor(
            metrics.get_inverse(space).name,
            [k.flip(), l.flip()],
            symmetric_pairs=[(0, 1)],
        )
        # outer scalar 결합: 1/n · 1/2 = 1/(2n). 안쪽은 ScalarMul 없이 TensorSum만.
        trace_part = ScalarMul(
            0.5 / n,
            TensorProduct(TensorProduct(γ_ij_at, γ_inv_kl), TensorSum(T_kl, T_lk)),
        )
        return TensorSum(sym_part, ScalarMul(-1, trace_part))

    if isinstance(expr, TensorSum):
        return TensorSum(
            expand_symmetrization(expr.left, metrics),
            expand_symmetrization(expr.right, metrics),
        )
    if isinstance(expr, TensorProduct):
        return TensorProduct(
            expand_symmetrization(expr.left, metrics),
            expand_symmetrization(expr.right, metrics),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(
            expr.scalar, expand_symmetrization(expr.expr, metrics),
        )
    return expr
