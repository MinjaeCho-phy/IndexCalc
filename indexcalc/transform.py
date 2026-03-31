"""
좌표 변환 (Coordinate Transformations).

CoordinateTransform 클래스로 두 좌표계 간 변환을 정의하고,
Jacobian 자동 계산 → metric 변환 → 일반 텐서 변환을 수행한다.
Symbolic (SymPy) 및 numeric (JAX) 모두 지원.

사용법
------
>>> import sympy as sp
>>> from indexcalc import CoordinateTransform, Metric
>>>
>>> r, θ, φ = sp.symbols('r θ φ', positive=True)
>>> T = CoordinateTransform(
...     source="spherical", target="cartesian",
...     forward=[r*sp.sin(θ)*sp.cos(φ), r*sp.sin(θ)*sp.sin(φ), r*sp.cos(θ)],
...     source_symbols=[r, θ, φ],
... )
>>> T.jacobian()   # ∂x'/∂x, 3×3 SymPy Matrix
"""

from __future__ import annotations
from typing import Callable

from indexcalc.coordinates import Coordinates


class CoordinateTransform:
    """두 좌표계 간의 변환.

    Parameters
    ----------
    source : str | list[str] | Coordinates
        원래 좌표계.
    target : str | list[str] | Coordinates
        목표 좌표계.
    forward : list | callable
        변환 함수 x → x'.
        - list of SymPy expressions: symbolic 변환.
        - callable: numeric 변환 (JAX 호환).
    source_symbols : list[sp.Symbol] or None
        symbolic 변환 시 원래 좌표의 SymPy 심볼.
        forward가 list일 때 필수.
    inverse : list | callable or None
        역변환 함수 x' → x. 생략하면 symbolic일 때 자동 계산 시도.

    Examples
    --------
    >>> # Symbolic
    >>> T = CoordinateTransform("spherical", "cartesian",
    ...     forward=[r*sp.sin(θ)*sp.cos(φ), ...], source_symbols=[r,θ,φ])
    >>>
    >>> # Numeric
    >>> T = CoordinateTransform("spherical", "cartesian",
    ...     forward=lambda x: jnp.array([...]))
    """

    def __init__(
        self,
        source,
        target,
        forward,
        source_symbols=None,
        inverse=None,
    ):
        self.source = _to_coords(source)
        self.target = _to_coords(target)

        if self.source.dim != self.target.dim:
            raise ValueError(
                f"Dimension mismatch: source dim={self.source.dim}, "
                f"target dim={self.target.dim}"
            )
        self._dim = self.source.dim

        # ── Forward map 분류 ──
        self._is_symbolic = not callable(forward)

        if self._is_symbolic:
            import sympy as sp
            if source_symbols is None:
                raise ValueError(
                    "source_symbols 필수 (symbolic 변환 시)."
                )
            if len(source_symbols) != self._dim:
                raise ValueError(
                    f"source_symbols 길이 {len(source_symbols)} ≠ dim {self._dim}"
                )
            self._source_symbols = list(source_symbols)
            self._forward_sym = sp.Matrix(forward)
            self._forward_func = None

            # inverse
            if inverse is not None and not callable(inverse):
                self._inverse_sym = sp.Matrix(inverse)
            else:
                self._inverse_sym = None
        else:
            self._source_symbols = None
            self._forward_sym = None
            self._forward_func = forward
            self._inverse_sym = None

        self._inverse_func = inverse if callable(inverse) else None

        # Jacobian 캐시
        self._jac_cache = None
        self._jac_inv_cache = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_symbolic(self) -> bool:
        return self._is_symbolic

    # ─── Jacobian ────────────────────────────────────────────

    def jacobian(self, at=None):
        """Jacobian ∂x'/∂x 를 계산한다.

        Parameters
        ----------
        at : array-like or None
            Numeric 변환이면 이 좌표에서 Jacobian을 계산.
            Symbolic이면 무시.

        Returns
        -------
        sp.Matrix or ndarray
            (dim × dim) Jacobian 행렬. J[i,j] = ∂x'_i/∂x_j.
        """
        if self._is_symbolic:
            if self._jac_cache is None:
                import sympy as sp
                self._jac_cache = self._forward_sym.jacobian(self._source_symbols)
            return self._jac_cache

        # numeric
        import jax
        if at is None:
            raise ValueError("Numeric 변환은 at= 좌표가 필요합니다.")
        return jax.jacfwd(self._forward_func)(at)

    def jacobian_inv(self, at=None):
        """Inverse Jacobian ∂x/∂x' 를 계산한다.

        Returns
        -------
        sp.Matrix or ndarray
        """
        if self._is_symbolic:
            if self._jac_inv_cache is None:
                J = self.jacobian()
                self._jac_inv_cache = J.inv()
            return self._jac_inv_cache

        # numeric
        import jax.numpy as jnp
        J = self.jacobian(at=at)
        return jnp.linalg.inv(J)

    # ─── Metric 변환 ─────────────────────────────────────────

    def transform_metric(self, metric):
        """Metric을 target 좌표계로 변환한다.

        g'_{μν} = (J^{-1})^α_μ (J^{-1})^β_ν g_{αβ}
        즉, g' = (J^{-1})^T g (J^{-1})

        Parameters
        ----------
        metric : Metric
            원래 좌표계의 Metric.

        Returns
        -------
        Metric
            target 좌표계의 새 Metric.
        """
        from indexcalc.curvature import Metric

        if self._is_symbolic and metric._is_symbolic:
            return self._transform_metric_symbolic(metric)
        elif not self._is_symbolic and not metric._is_symbolic:
            return self._transform_metric_numeric(metric)
        else:
            raise TypeError(
                "Metric과 CoordinateTransform이 같은 모드여야 합니다 "
                "(둘 다 symbolic 또는 둘 다 numeric)."
            )

    def _transform_metric_symbolic(self, metric):
        """Symbolic metric 변환."""
        import sympy as sp
        from indexcalc.curvature import Metric

        J_inv = self.jacobian_inv()  # ∂x/∂x'
        g = metric._sympy_metric

        # g' = J_inv^T · g · J_inv
        # 여기서 g는 source 좌표의 함수 → source 좌표를 target으로 치환 필요
        # 하지만 역변환 함수가 필요: x = x(x')
        # J_inv^T · g(x(x')) · J_inv

        # 역변환이 있으면 좌표 치환
        g_prime = sp.simplify(J_inv.T * g * J_inv)

        return Metric(
            g_prime,
            self.target,
            signature=metric._signs,
        )

    def _transform_metric_numeric(self, metric):
        """Numeric metric 변환."""
        from indexcalc.curvature import Metric

        forward_func = self._forward_func
        inverse_func = self._inverse_func
        metric_func = metric.func

        if inverse_func is None:
            raise ValueError(
                "Numeric metric 변환에는 inverse 함수가 필요합니다."
            )

        def new_metric_func(x_prime):
            import jax
            import jax.numpy as jnp

            # x' → x (역변환)
            x = inverse_func(x_prime)

            # Jacobian at x
            J = jax.jacfwd(forward_func)(x)
            J_inv = jnp.linalg.inv(J)

            # g'_{μν} = J_inv^T · g(x) · J_inv
            g = metric_func(x)
            return J_inv.T @ g @ J_inv

        return Metric(
            new_metric_func,
            self.target,
            signature=metric._signs,
        )

    # ─── 일반 텐서 변환 ──────────────────────────────────────

    def transform_components(self, components, rank, at=None):
        """일반 텐서 성분을 변환한다.

        T'^{μ'₁...}_{ν'₁...} = J^{μ'₁}_{μ₁}...
                                (J^{-1})^{ν₁}_{ν'₁}...
                                T^{μ₁...}_{ν₁...}

        Parameters
        ----------
        components : ndarray | sp.Array | sp.Matrix
            원래 좌표계의 텐서 성분.
        rank : tuple[int, int]
            (n_upper, n_lower) — contravariant, covariant 인덱스 수.
        at : array-like or None
            Numeric 변환 시 좌표.

        Returns
        -------
        ndarray | sp.Array | sp.Matrix
            변환된 텐서 성분.
        """
        n_up, n_down = rank
        total_rank = n_up + n_down

        if self._is_symbolic:
            return self._transform_symbolic(components, n_up, n_down)
        else:
            return self._transform_numeric(components, n_up, n_down, at)

    def _transform_symbolic(self, T, n_up, n_down):
        """Symbolic 텐서 변환."""
        import sympy as sp

        dim = self._dim
        J = self.jacobian()       # ∂x'/∂x → upper index 변환
        J_inv = self.jacobian_inv()  # ∂x/∂x' → lower index 변환

        total = n_up + n_down

        if total == 0:
            # scalar
            return T

        if total == 1 and n_up == 1:
            # vector V'^μ = J^μ_α V^α
            V = sp.Matrix(T)
            return sp.simplify(J * V)

        if total == 1 and n_down == 1:
            # covector ω'_μ = (J^{-1})^α_μ ω_α = (J_inv^T · ω)
            w = sp.Matrix(T)
            return sp.simplify(J_inv.T * w)

        if total == 2 and n_down == 2:
            # (0,2) tensor: T'_{μν} = J_inv^T · T · J_inv
            M = sp.Matrix(T)
            return sp.simplify(J_inv.T * M * J_inv)

        if total == 2 and n_up == 2:
            # (2,0) tensor: T'^{μν} = J · T · J^T
            M = sp.Matrix(T)
            return sp.simplify(J * M * J.T)

        if total == 2 and n_up == 1 and n_down == 1:
            # (1,1) tensor: T'^μ_ν = J^μ_α (J^{-1})^β_ν T^α_β
            # = J · T · J_inv
            M = sp.Matrix(T)
            return sp.simplify(J * M * J_inv)

        # 일반 rank: 명시적 루프
        return self._transform_general_symbolic(T, n_up, n_down, J, J_inv)

    def _transform_general_symbolic(self, T, n_up, n_down, J, J_inv):
        """임의 rank 텐서의 symbolic 변환 (명시적 합산)."""
        import sympy as sp
        from itertools import product as iterproduct

        dim = self._dim
        total = n_up + n_down
        shape = (dim,) * total
        result = sp.MutableDenseNDimArray.zeros(*shape)

        for new_idx in iterproduct(range(dim), repeat=total):
            val = sp.Rational(0)
            for old_idx in iterproduct(range(dim), repeat=total):
                # 변환 행렬 곱
                factor = sp.Rational(1)
                for k in range(n_up):
                    factor *= J[new_idx[k], old_idx[k]]
                for k in range(n_down):
                    factor *= J_inv[old_idx[n_up + k], new_idx[n_up + k]]

                factor *= T[old_idx]
                val += factor

            result[new_idx] = sp.simplify(val)

        return result

    def _transform_numeric(self, T, n_up, n_down, at):
        """Numeric 텐서 변환."""
        import jax.numpy as jnp

        J = self.jacobian(at=at)
        J_inv = self.jacobian_inv(at=at)

        total = n_up + n_down

        if total == 0:
            return T

        if total == 1 and n_up == 1:
            return J @ jnp.asarray(T)

        if total == 1 and n_down == 1:
            return J_inv.T @ jnp.asarray(T)

        if total == 2 and n_down == 2:
            return J_inv.T @ jnp.asarray(T) @ J_inv

        if total == 2 and n_up == 2:
            return J @ jnp.asarray(T) @ J.T

        if total == 2 and n_up == 1 and n_down == 1:
            return J @ jnp.asarray(T) @ J_inv

        # 일반 rank: einsum 동적 생성
        return self._transform_general_numeric(T, n_up, n_down, J, J_inv)

    def _transform_general_numeric(self, T, n_up, n_down, J, J_inv):
        """임의 rank 텐서의 numeric 변환."""
        import jax.numpy as jnp

        total = n_up + n_down
        T = jnp.asarray(T)

        # 순차적으로 각 축에 변환 적용
        result = T
        for k in range(n_up):
            # k번째 축에 J 적용: result = J @ result (along axis k)
            result = jnp.tensordot(J, result, axes=([1], [k]))
            # tensordot 후 새 축이 앞에 옴 → 원래 위치로 이동
            result = jnp.moveaxis(result, 0, k)

        for k in range(n_down):
            axis = n_up + k
            # J_inv^T 적용: (J_inv)^old_new → contract old
            result = jnp.tensordot(J_inv, result, axes=([0], [axis]))
            result = jnp.moveaxis(result, 0, axis)

        return result

    # ─── 역변환 ──────────────────────────────────────────────

    def inverse(self) -> CoordinateTransform:
        """역변환을 반환한다.

        Returns
        -------
        CoordinateTransform
            source ↔ target이 교환된 변환.
        """
        if self._is_symbolic:
            if self._inverse_sym is not None:
                return CoordinateTransform(
                    source=self.target,
                    target=self.source,
                    forward=list(self._inverse_sym),
                    source_symbols=None,  # target의 심볼 필요
                )
            raise ValueError(
                "Symbolic 역변환이 정의되지 않았습니다. "
                "inverse= 파라미터로 역변환을 지정하세요."
            )

        if self._inverse_func is not None:
            return CoordinateTransform(
                source=self.target,
                target=self.source,
                forward=self._inverse_func,
                inverse=self._forward_func,
            )

        raise ValueError("역변환 함수가 정의되지 않았습니다.")

    # ─── LaTeX ───────────────────────────────────────────────

    def latex(self) -> str:
        """변환 공식과 Jacobian을 LaTeX로 반환한다."""
        if not self._is_symbolic:
            return r"\text{(numeric transform — no symbolic form)}"

        import sympy as sp

        src_names = self.source.names
        tgt_names = self.target.names
        syms = self._source_symbols

        # 변환 공식
        eqs = []
        for i, name in enumerate(tgt_names):
            lhs = _latex_coord(name)
            rhs = sp.latex(self._forward_sym[i])
            eqs.append(f"{lhs} &= {rhs}")

        transform_block = r" \\ ".join(eqs)

        # Jacobian
        J = self.jacobian()
        jac_latex = sp.latex(J)

        return (
            rf"\begin{{aligned}} {transform_block} \end{{aligned}}"
            rf"\quad J = {jac_latex}"
        )

    def __repr__(self) -> str:
        mode = "symbolic" if self._is_symbolic else "numeric"
        return f"CoordinateTransform({self.source} → {self.target}, {mode})"


# ─── Helpers ─────────────────────────────────────────────────

def _to_coords(x) -> Coordinates:
    """문자열, 리스트, 또는 Coordinates를 Coordinates로 변환."""
    if isinstance(x, Coordinates):
        return x
    if isinstance(x, str):
        # dim 모르므로 나중에 검증 — preset 이름으로 저장
        # 일단 지원하는 것 중 default dim 사용 불가 → 에러 방지를 위해
        # 여기서는 변환 생성 시 dim을 forward에서 추론
        raise ValueError(
            f"preset 이름 '{x}'만으로는 dim을 알 수 없습니다. "
            f"Coordinates.preset('{x}', dim=N) 또는 좌표 이름 리스트를 사용하세요."
        )
    if isinstance(x, (list, tuple)) and all(isinstance(c, str) for c in x):
        return Coordinates(x)
    raise TypeError(f"좌표 타입이 올바르지 않습니다: {type(x).__name__}")


_LATEX_NAMES = {
    "t": "t", "r": "r", "x": "x", "y": "y", "z": "z",
    "θ": r"\theta", "theta": r"\theta",
    "φ": r"\varphi", "phi": r"\varphi",
    "ρ": r"\rho", "rho": r"\rho",
}


def _latex_coord(name: str) -> str:
    return _LATEX_NAMES.get(name, name)
