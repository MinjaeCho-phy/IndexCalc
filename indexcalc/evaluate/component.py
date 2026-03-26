"""
Component evaluation: TensorExpr → numeric array via einsum.

표현식 트리를 재귀적으로 순회하면서:
  1. 각 Tensor 잎 노드의 component 배열을 components dict에서 조회
  2. TensorProduct → einsum으로 contraction 수행
  3. TensorSum → axis 순서 맞춘 뒤 덧셈
  4. ScalarMul → 스칼라 곱셈
  5. Trace → einsum으로 대각합
  6. PartialDeriv → "∂V" 키로 조회, 또는 JAX autodiff로 자동 계산

Functional components (Phase 6e):
  - components의 값이 callable이면, coords를 인자로 호출하여 배열을 얻는다.
  - PartialDeriv에서 "∂V" 키가 없고, "V"가 callable이면
    jax.jacfwd로 자동미분하여 ∂_μ V^ν를 계산한다.
"""

from __future__ import annotations
import numpy as np

from indexcalc.core.index import Index
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.contract import Trace
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv


# ─── Public API ───────────────────────────────────────────────

def evaluate(
    expr: TensorExpr,
    components: dict,
    backend: str = "numpy",
    coords=None,
):
    """텐서 표현식을 숫자 배열로 평가한다.

    Parameters
    ----------
    expr : TensorExpr
        평가할 표현식. CovariantDeriv가 포함되어 있으면
        expand_covariant()로 먼저 전개해야 한다.
    components : dict
        텐서 component 매핑. 값은 ndarray 또는 callable.
        - ndarray: 직접 사용
        - callable: coords를 인자로 호출하여 ndarray를 얻는다.
        키 형식:
          - str: 텐서 이름 (e.g., "V", "Γ")
          - (str, str): (텐서 이름, 위치문자열) (e.g., ("g", "dd"))
          - "∂V": PartialDeriv의 explicit component (autodiff보다 우선)
    backend : str
        "numpy" (기본) 또는 "jax".
    coords : ndarray or None
        좌표값. functional component 평가 및 autodiff에 사용.
        None이면 모든 component가 배열이어야 한다.

    Returns
    -------
    ndarray
        결과 배열. axis 순서는 expr.free_indices 순서와 동일.

    Examples
    --------
    >>> # 배열 모드 (Phase 6d)
    >>> evaluate(expr, {"η": eta_arr, "V": v_arr})
    >>> # 함수 모드 (Phase 6e) — V를 좌표 함수로 제공
    >>> evaluate(expr, {"V": lambda x: jnp.array([x[0]**2, x[1], 0., 0.])},
    ...          backend="jax", coords=jnp.array([1., 0.5, 0., 0.]))
    """
    xp = _get_backend(backend)
    return _eval(expr, components, xp, coords)


# ─── Backend ──────────────────────────────────────────────────

def _get_backend(backend: str):
    if backend == "jax":
        import jax.numpy as jnp
        return jnp
    return np


# ─── Component 조회 ──────────────────────────────────────────

def _find_in_comp(name: str, indices: tuple, comp: dict):
    """components dict에서 값을 찾는다 (조회만, 평가 안함)."""
    pos = "".join("u" if i.position == "upper" else "d" for i in indices)
    key_full = (name, pos)
    if key_full in comp:
        return comp[key_full]
    if name in comp:
        return comp[name]
    return None


def _resolve_value(val, xp, coords):
    """값이 callable이면 coords로 호출, 아니면 그대로 반환."""
    if callable(val):
        if coords is None:
            raise ValueError(
                "함수형 component를 사용하려면 coords를 제공해야 합니다."
            )
        return xp.asarray(val(coords))
    return xp.asarray(val)


def _lookup_tensor(tensor: Tensor, comp: dict, xp, coords):
    """Tensor의 component 배열을 조회한다."""
    val = _find_in_comp(tensor.name, tensor.indices, comp)
    if val is None:
        pos = "".join("u" if i.position == "upper" else "d" for i in tensor.indices)
        raise KeyError(
            f"'{tensor.name}' (indices: {pos})에 대한 component를 찾을 수 없습니다. "
            f"사용 가능한 키: {list(comp.keys())}"
        )
    return _resolve_value(val, xp, coords)


def _lookup_partial(pderiv: PartialDeriv, comp: dict, xp, coords):
    """PartialDeriv의 component 배열을 조회하거나 autodiff로 계산한다.

    우선순위:
      1. "∂V" (또는 ("∂V", pos)) 키로 explicit component 조회
      2. "V"가 callable이면 jax.jacfwd로 자동미분
      3. 실패 시 KeyError
    """
    depth, inner_tensor = _unwrap_partial(pderiv)

    # 1) Explicit key 시도
    deriv_name = "∂" * depth + inner_tensor.name
    all_free = pderiv.free_indices
    val = _find_in_comp(deriv_name, tuple(all_free), comp)
    if val is not None:
        return _resolve_value(val, xp, coords)

    # 2) Autodiff 시도
    base_val = _find_in_comp(inner_tensor.name, inner_tensor.indices, comp)
    if base_val is not None and callable(base_val):
        return _autodiff_partial(base_val, inner_tensor, depth, xp, coords)

    # 3) 실패
    pos = "".join("u" if i.position == "upper" else "d" for i in all_free)
    raise KeyError(
        f"'{deriv_name}' (indices: {pos})에 대한 component를 찾을 수 없습니다. "
        f"'{inner_tensor.name}'이 callable이면 autodiff를 사용합니다. "
        f"사용 가능한 키: {list(comp.keys())}"
    )


def _unwrap_partial(pderiv: PartialDeriv):
    """PartialDeriv 중첩을 풀어서 (depth, inner_tensor)를 반환한다."""
    depth = 0
    inner = pderiv
    while isinstance(inner, PartialDeriv):
        depth += 1
        inner = inner.expr

    if not isinstance(inner, Tensor):
        raise ValueError(
            f"PartialDeriv 내부가 {type(inner).__name__}입니다. "
            f"expand_partial()로 먼저 전개하세요."
        )
    return depth, inner


def _autodiff_partial(func, inner_tensor: Tensor, depth: int, xp, coords):
    """JAX jacfwd로 편미분을 자동 계산한다.

    jacfwd(f)(x) 결과 shape: (*tensor_shape, coord_dim)
      → PartialDeriv 순서에 맞게 축 재배열 필요.

    PartialDeriv(T, _μ).free_indices = [_μ, ...T_indices]
    jacfwd(T)(x) shape = (*T_shape, coord_dim)
      → 마지막 축을 맨 앞으로 이동.

    depth=2: jacfwd(jacfwd(T))(x) shape = (*T_shape, inner_dim, outer_dim)
      → 마지막 depth개 축을 역순으로 앞에 배치.
    """
    if coords is None:
        raise ValueError(
            "autodiff를 사용하려면 coords를 제공해야 합니다."
        )

    import jax

    f = func
    for _ in range(depth):
        f = jax.jacfwd(f)

    result = f(coords)
    tensor_ndim = len(inner_tensor.indices)

    return xp.asarray(_reorder_deriv_axes(result, tensor_ndim, depth))


def _reorder_deriv_axes(arr, tensor_ndim: int, depth: int):
    """jacfwd 결과의 축 순서를 PartialDeriv.free_indices 순서에 맞춘다.

    jacfwd 결과: (*tensor_axes, deriv_inner, ..., deriv_outer)
    PartialDeriv:  (deriv_outer, ..., deriv_inner, *tensor_axes)

    Parameters
    ----------
    arr : ndarray
        jacfwd 결과.
    tensor_ndim : int
        원래 텐서의 인덱스 수 (0 = scalar, 1 = vector, 2 = matrix, ...).
    depth : int
        미분 깊이 (1 = 1차, 2 = 2차, ...).
    """
    k = tensor_ndim
    # deriv axes: [k, k+1, ..., k+depth-1] (inner first)
    # target: [k+depth-1, ..., k+1, k, 0, 1, ..., k-1] (outer first, then tensor)
    perm = list(range(k + depth - 1, k - 1, -1)) + list(range(k))
    import numpy as _np
    return _np.transpose(arr, perm) if hasattr(arr, '__array__') else arr.transpose(perm)


# ─── Einsum label 할당 ───────────────────────────────────────

def _assign_labels(
    indices: list[Index],
    label_map: dict[tuple[str, str], str],
    counter: list[int],
) -> str:
    """인덱스 리스트에 einsum label(a,b,c,...)을 할당한다.

    같은 (name, space) 쌍은 같은 label을 받는다 → contraction 자동 처리.
    """
    result = []
    for idx in indices:
        key = (idx.name, idx.space.name)
        if key not in label_map:
            if counter[0] >= 26:
                raise ValueError("인덱스가 26개를 초과합니다 (einsum label 부족).")
            label_map[key] = chr(ord('a') + counter[0])
            counter[0] += 1
        result.append(label_map[key])
    return "".join(result)


# ─── Axis 순서 맞추기 ────────────────────────────────────────

def _match_axes(target_free: list[Index], source_free: list[Index]):
    """source의 axis 순서를 target에 맞추는 permutation을 반환한다.

    이미 같은 순서이면 None을 반환.
    """
    if len(target_free) != len(source_free):
        return None

    same = all(
        t.name == s.name and t.space == s.space
        for t, s in zip(target_free, source_free)
    )
    if same:
        return None

    perm = []
    for t_idx in target_free:
        for j, s_idx in enumerate(source_free):
            if t_idx.name == s_idx.name and t_idx.space == s_idx.space:
                perm.append(j)
                break
        else:
            raise ValueError(
                f"TensorSum의 free index 구조가 일치하지 않습니다: "
                f"{t_idx}가 오른쪽 항에 없습니다."
            )
    return tuple(perm)


# ─── 재귀 평가 ────────────────────────────────────────────────

def _eval(expr: TensorExpr, comp: dict, xp, coords=None) -> np.ndarray:
    """표현식 트리를 재귀적으로 평가한다."""

    # ── Tensor (잎 노드) ──
    if isinstance(expr, Tensor):
        return _lookup_tensor(expr, comp, xp, coords)

    # ── ScalarMul ──
    if isinstance(expr, ScalarMul):
        return expr.scalar * _eval(expr.expr, comp, xp, coords)

    # ── TensorSum ──
    if isinstance(expr, TensorSum):
        left_val = _eval(expr.left, comp, xp, coords)
        right_val = _eval(expr.right, comp, xp, coords)

        perm = _match_axes(expr.left.free_indices, expr.right.free_indices)
        if perm is not None:
            right_val = xp.transpose(right_val, perm)

        return left_val + right_val

    # ── TensorProduct → einsum ──
    if isinstance(expr, TensorProduct):
        left_val = _eval(expr.left, comp, xp, coords)
        right_val = _eval(expr.right, comp, xp, coords)

        label_map: dict[tuple[str, str], str] = {}
        counter = [0]

        left_str = _assign_labels(expr.left.free_indices, label_map, counter)
        right_str = _assign_labels(expr.right.free_indices, label_map, counter)
        output_str = _assign_labels(expr.free_indices, label_map, counter)

        einsum_str = f"{left_str},{right_str}->{output_str}"
        return xp.einsum(einsum_str, left_val, right_val)

    # ── Trace → einsum ──
    if isinstance(expr, Trace):
        inner_val = _lookup_tensor(expr.tensor, comp, xp, coords)

        label_map: dict[tuple[str, str], str] = {}
        counter = [0]

        input_str = _assign_labels(list(expr.tensor.indices), label_map, counter)
        output_str = _assign_labels(expr.free_indices, label_map, counter)

        einsum_str = f"{input_str}->{output_str}"
        return xp.einsum(einsum_str, inner_val)

    # ── PartialDeriv ──
    if isinstance(expr, PartialDeriv):
        return _lookup_partial(expr, comp, xp, coords)

    # ── CovariantDeriv (전개 필요) ──
    if isinstance(expr, CovariantDeriv):
        raise ValueError(
            "CovariantDeriv를 직접 평가할 수 없습니다. "
            "expand_covariant()로 먼저 전개하세요."
        )

    raise TypeError(f"평가할 수 없는 표현식 타입: {type(expr).__name__}")
