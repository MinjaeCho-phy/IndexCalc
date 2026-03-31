"""
Curvature tensor auto-derivation (Phase 6f).

Metric 클래스로 metric 함수 + 좌표계를 정의하고,
JAX autodiff로 Christoffel, Riemann, Ricci, Einstein 텐서를 자동 계산한다.

사용법
------
>>> import jax.numpy as jnp
>>> from indexcalc import Metric
>>>
>>> def schwarzschild(x):
...     r, theta = x[1], x[2]
...     f = 1 - 1.0 / r
...     return jnp.diag(jnp.array([-f, 1/f, r**2, r**2 * jnp.sin(theta)**2]))
>>>
>>> g = Metric(schwarzschild, "spherical", signature=(3,1))
>>> result = g.at(jnp.array([0., 5., jnp.pi/2, 0.]))
>>> result.R   # ≈ 0 (vacuum)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as _np

from indexcalc.coordinates import Coordinates, parse_signature


def _latex_align(equations: list[str]) -> str:
    """LaTeX aligned 블록 생성."""
    body = r" \\ ".join(equations)
    return rf"\begin{{aligned}} {body} \end{{aligned}}"


# ─── Metric ──────────────────────────────────────────────────

class Metric:
    """Metric 텐서 정의: 함수 + 좌표계 + signature.

    Parameters
    ----------
    func : callable
        x → g_{μν}(x), shape (dim, dim). JAX 호환 필요.
    coords : list[str] | str | Coordinates
        좌표 지정 방법 3가지:
        - list[str]: custom 좌표 이름.  e.g., ["t", "x", "y", "z"]
        - str: preset 이름.  e.g., "spherical", "cartesian"
        - Coordinates: 이미 생성된 Coordinates 객체.
    signature : tuple or None
        (p, q) 축약: p개의 +1, q개의 -1.  e.g., (3,1) → (-,+,+,+)
        explicit: (-1, 1, 1, 1).
        None → 전부 +1 (Riemannian).
    dim : int or None
        preset 사용 시 공간 차원수. 생략하면 signature에서 추론.

    Examples
    --------
    >>> # Custom 좌표
    >>> g = Metric(func, ["t", "x", "y", "z"], signature=(3,1))
    >>>
    >>> # Preset 좌표계
    >>> g = Metric(func, "spherical", signature=(3,1))  # t, r, θ, φ
    >>> g = Metric(func, "cartesian", dim=3)             # x, y, z
    >>>
    >>> # 곡률 계산
    >>> result = g.at(coords)
    >>> result.R           # Ricci scalar
    >>> result.Γ           # Christoffel
    >>> result.show()      # named output
    """

    def __init__(
        self,
        func,
        coords: list[str] | str | Coordinates,
        signature: tuple | None = None,
        dim: int | None = None,
    ):
        self.func = func

        # ── SymPy Matrix 감지 ──
        self._is_symbolic = False
        self._sympy_metric = None
        try:
            import sympy as sp
            if isinstance(func, sp.MatrixBase):
                self._is_symbolic = True
                self._sympy_metric = func
                # dim을 matrix에서 추론
                if dim is None:
                    dim = func.shape[0]
        except ImportError:
            pass

        # ── 좌표계 해석 ──
        if isinstance(coords, Coordinates):
            self._coords = coords
        elif isinstance(coords, str):
            # preset 이름
            self._coords = Coordinates.preset(coords, dim=dim, signature=signature)
        elif isinstance(coords, (list, tuple)) and all(isinstance(c, str) for c in coords):
            # custom 좌표 이름
            self._coords = Coordinates(coords)
        else:
            raise TypeError(
                f"coords must be list[str], preset name (str), or Coordinates. "
                f"Got {type(coords).__name__}."
            )

        # ── dim 정합성 검사 ──
        if dim is not None and dim != self._coords.dim:
            raise ValueError(
                f"dim={dim} conflicts with coordinates dimension "
                f"{self._coords.dim} ({self._coords.names})."
            )

        # ── Signature ──
        self._signs = parse_signature(signature, self._coords.dim)

        # ── 내부 curvature computer (numeric only) ──
        if not self._is_symbolic:
            self._computer = _CurvatureComputer(func, self._coords.dim)
        else:
            self._computer = None

    @property
    def coords(self) -> Coordinates:
        return self._coords

    @property
    def dim(self) -> int:
        return self._coords.dim

    @property
    def signature(self) -> tuple[int, ...]:
        return self._signs

    def at(self, x, params: dict | None = None) -> CurvatureResult:
        """주어진 좌표에서 모든 곡률 텐서를 수치 계산한다.

        Parameters
        ----------
        x : array-like
            좌표 벡터, shape (dim,).
        params : dict or None
            SymPy metric일 때 매개변수 값.  e.g., {M: 1.0, H: 0.5}.
            callable metric이면 무시된다.

        Returns
        -------
        CurvatureResult
        """
        if self._is_symbolic:
            return self._at_from_sympy(x, params)
        return self._computer.at(x, self._coords)

    def symbolic(self, coord_symbols: list | None = None) -> SymbolicCurvatureResult:
        """모든 곡률 텐서를 symbolic으로 계산한다.

        Parameters
        ----------
        coord_symbols : list of sympy.Symbol
            좌표 심볼.  e.g., [t, r, θ, φ].

        Returns
        -------
        SymbolicCurvatureResult
        """
        if not self._is_symbolic:
            raise TypeError(
                "symbolic()은 SymPy Matrix로 생성된 Metric에서만 사용 가능합니다. "
                "수치 계산은 .at(x)를 사용하세요."
            )
        if coord_symbols is None:
            raise ValueError(
                "coord_symbols를 지정해야 합니다.  e.g., g.symbolic([t, r, θ, φ])"
            )
        if len(coord_symbols) != self.dim:
            raise ValueError(
                f"coord_symbols 길이 {len(coord_symbols)} ≠ dim {self.dim}"
            )

        computer = _SymbolicCurvatureComputer(
            self._sympy_metric, coord_symbols, self.dim
        )
        return computer.compute(self._coords)

    def _at_from_sympy(self, x, params: dict | None) -> CurvatureResult:
        """SymPy metric을 lambdify하여 수치 계산."""
        import sympy as sp
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        # metric의 free symbols에서 좌표 심볼 추출
        free = self._sympy_metric.free_symbols
        # params에서 좌표가 아닌 매개변수 대입
        g_substituted = self._sympy_metric
        if params:
            g_substituted = g_substituted.subs(params)

        # 남은 free symbols를 좌표로 간주
        coord_syms = sorted(g_substituted.free_symbols, key=lambda s: str(s))
        if len(coord_syms) != self.dim:
            raise ValueError(
                f"대입 후 남은 심볼 {coord_syms}의 수가 dim={self.dim}과 다릅니다. "
                f"params로 매개변수를 지정하세요."
            )

        # lambdify → JAX 함수
        g_func = sp.lambdify(
            [coord_syms], g_substituted, modules=["numpy"]
        )

        def metric_func(coords):
            return jnp.array(g_func(coords), dtype=jnp.float64)

        computer = _CurvatureComputer(metric_func, self.dim)
        return computer.at(x, self._coords)

    def __repr__(self) -> str:
        sig_str = "".join("−" if s == -1 else "+" for s in self._signs)
        return f"Metric({self._coords}, signature={sig_str})"


# ─── CurvatureResult ─────────────────────────────────────────

@dataclass
class CurvatureResult:
    """곡률 텐서 계산 결과.

    Attributes
    ----------
    metric : ndarray, shape (dim, dim)
        g_{μν}.
    inverse_metric : ndarray, shape (dim, dim)
        g^{μν}.
    christoffel : ndarray, shape (dim, dim, dim)
        Γ^σ_{μν}. 축: (σ, μ, ν).
    riemann : ndarray, shape (dim, dim, dim, dim)
        R^ρ_{σμν}. 축: (ρ, σ, μ, ν).
    ricci_tensor : ndarray, shape (dim, dim)
        R_{μν}.
    ricci_scalar : float
        R = g^{μν} R_{μν}.
    einstein_tensor : ndarray, shape (dim, dim)
        G_{μν} = R_{μν} - ½ g_{μν} R.
    kretschner : float
        K = R_{ρσμν} R^{ρσμν}.
    coords_values : ndarray
        평가 지점 좌표값.
    coords_names : tuple[str, ...]
        좌표 이름.
    """
    metric: _np.ndarray
    inverse_metric: _np.ndarray
    christoffel: _np.ndarray
    riemann: _np.ndarray
    ricci_tensor: _np.ndarray
    ricci_scalar: float
    einstein_tensor: _np.ndarray
    kretschner: float
    coords_values: _np.ndarray
    coords_names: tuple[str, ...]

    # ── 별칭 (property) ──
    @property
    def g(self) -> _np.ndarray:
        return self.metric

    @property
    def g_inv(self) -> _np.ndarray:
        return self.inverse_metric

    @property
    def Γ(self) -> _np.ndarray:
        return self.christoffel

    @property
    def R(self) -> float:
        return self.ricci_scalar

    @property
    def Ric(self) -> _np.ndarray:
        return self.ricci_tensor

    @property
    def G(self) -> _np.ndarray:
        return self.einstein_tensor

    @property
    def K(self) -> float:
        return self.kretschner

    def summary(self) -> str:
        return (
            f"R = {float(self.ricci_scalar):.6g}, "
            f"K = {float(self.kretschner):.6g}, "
            f"|G|_max = {float(_np.max(_np.abs(self.einstein_tensor))):.6g}"
        )

    def show(self, tensor: str = "christoffel", threshold: float = 1e-10) -> str:
        """곡률 텐서를 좌표 이름으로 출력한다.

        Parameters
        ----------
        tensor : str
            "christoffel", "riemann", "ricci", "einstein" 중 하나.
        threshold : float
            이 값보다 작은 성분은 생략.

        Returns
        -------
        str
        """
        names = self.coords_names
        lines = []

        if tensor == "christoffel":
            arr = self.christoffel
            dim = len(names)
            for s in range(dim):
                for m in range(dim):
                    for n in range(m, dim):  # 대칭
                        val = float(arr[s, m, n])
                        if abs(val) > threshold:
                            lines.append(
                                f"  Γ^{names[s]}_{{{names[m]}{names[n]}}} "
                                f"= {val:.8f}"
                            )

        elif tensor == "riemann":
            arr = self.riemann
            dim = len(names)
            for r in range(dim):
                for s in range(dim):
                    for m in range(dim):
                        for n in range(m + 1, dim):  # 반대칭
                            val = float(arr[r, s, m, n])
                            if abs(val) > threshold:
                                lines.append(
                                    f"  R^{names[r]}_{{{names[s]}{names[m]}{names[n]}}} "
                                    f"= {val:.8f}"
                                )

        elif tensor == "ricci":
            arr = self.ricci_tensor
            dim = len(names)
            for m in range(dim):
                for n in range(m, dim):
                    val = float(arr[m, n])
                    if abs(val) > threshold:
                        lines.append(
                            f"  R_{{{names[m]}{names[n]}}} = {val:.8f}"
                        )

        elif tensor == "einstein":
            arr = self.einstein_tensor
            dim = len(names)
            for m in range(dim):
                for n in range(m, dim):
                    val = float(arr[m, n])
                    if abs(val) > threshold:
                        lines.append(
                            f"  G_{{{names[m]}{names[n]}}} = {val:.8f}"
                        )

        else:
            raise ValueError(
                f"Unknown tensor '{tensor}'. "
                f"Use 'christoffel', 'riemann', 'ricci', or 'einstein'."
            )

        result = "\n".join(lines) if lines else "  (all components zero)"
        print(result)
        return result


# ─── Internal compute engine ─────────────────────────────────

class _CurvatureComputer:
    """내부 계산 엔진."""

    def __init__(self, metric_func: Callable, dim: int):
        self.metric_func = metric_func
        self.dim = dim

    def at(self, coords, coord_obj: Coordinates) -> CurvatureResult:
        import jax
        import jax.numpy as jnp

        coords = jnp.asarray(coords, dtype=jnp.float64)

        g = self.metric_func(coords)
        g_inv = jnp.linalg.inv(g)

        gamma = self._christoffel(coords, g_inv)
        riem = self._riemann(coords, gamma)

        ricci = jnp.einsum('msmn->sn', riem)
        R = jnp.einsum('sn,sn->', g_inv, ricci)
        einstein = ricci - 0.5 * g * R

        riemann_lower = jnp.einsum('ra,asmn->rsmn', g, riem)
        kretschner = jnp.einsum(
            'rsmn,ra,sb,mc,nd,abcd->',
            riemann_lower, g_inv, g_inv, g_inv, g_inv, riemann_lower,
        )

        return CurvatureResult(
            metric=g,
            inverse_metric=g_inv,
            christoffel=gamma,
            riemann=riem,
            ricci_tensor=ricci,
            ricci_scalar=R,
            einstein_tensor=einstein,
            kretschner=kretschner,
            coords_values=coords,
            coords_names=coord_obj.names,
        )

    def _christoffel(self, coords, g_inv):
        """Γ^σ_{μν} = ½ g^{σρ}(g_{ρμ,ν} + g_{ρν,μ} - g_{μν,ρ})."""
        import jax
        import jax.numpy as jnp

        # dg[a,b,c] = ∂g_{ab}/∂x^c
        dg = jax.jacfwd(self.metric_func)(coords)

        # bracket[ρ,μ,ν] = g_{ρμ,ν} + g_{ρν,μ} - g_{μν,ρ}
        #
        # dg[ρ,μ,ν]                    → dg 그대로
        # dg[ρ,ν,μ]  (μ↔ν 교환)        → transpose(dg, (0,2,1))
        # dg[μ,ν,ρ]  (자유변수 ρ,μ,ν 기준  → transpose(dg, (2,0,1))
        #             result[i,j,k] = dg[j,k,i])
        bracket = (
            dg
            + jnp.transpose(dg, (0, 2, 1))
            - jnp.transpose(dg, (2, 0, 1))
        )

        return 0.5 * jnp.einsum('sr,rmn->smn', g_inv, bracket)

    def _christoffel_func(self, coords):
        """Christoffel을 좌표의 함수로 (Riemann jacfwd용)."""
        import jax.numpy as jnp
        g_inv = jnp.linalg.inv(self.metric_func(coords))
        return self._christoffel(coords, g_inv)

    def _riemann(self, coords, gamma):
        """R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ}."""
        import jax
        import jax.numpy as jnp

        # dgamma[a,b,c,d] = ∂Γ^a_{bc}/∂x^d
        dgamma = jax.jacfwd(self._christoffel_func)(coords)

        # ∂_μ Γ^ρ_{νσ} as R[ρ,σ,μ,ν]:
        #   dgamma indices: (ρ,ν,σ,μ) → target (ρ,σ,μ,ν) → (0,2,3,1)
        term1 = jnp.transpose(dgamma, (0, 2, 3, 1))

        # ∂_ν Γ^ρ_{μσ} as R[ρ,σ,μ,ν]:
        #   dgamma indices: (ρ,μ,σ,ν) → target (ρ,σ,μ,ν) → (0,2,1,3)
        term2 = jnp.transpose(dgamma, (0, 2, 1, 3))

        # Γ^ρ_{μλ} Γ^λ_{νσ} as R[ρ,σ,μ,ν]
        term3 = jnp.einsum('rml,lns->rsmn', gamma, gamma)

        # Γ^ρ_{νλ} Γ^λ_{μσ} as R[ρ,σ,μ,ν]
        term4 = jnp.einsum('rnl,lms->rsmn', gamma, gamma)

        return term1 - term2 + term3 - term4


# ─── Symbolic compute engine ─────────────────────────────────

class SymbolicCurvatureResult:
    """Symbolic 곡률 텐서 계산 결과.

    모든 성분이 SymPy 수식으로 표현된다.
    .subs()로 값을 대입하여 수치 결과를 얻을 수 있다.

    Attributes
    ----------
    metric : sp.Matrix
        g_{μν}.
    inverse_metric : sp.Matrix
        g^{μν}.
    christoffel : sp.Array
        Γ^σ_{μν}. 축: (σ, μ, ν).
    riemann : sp.Array
        R^ρ_{σμν}. 축: (ρ, σ, μ, ν).
    ricci_tensor : sp.Matrix
        R_{μν}.
    ricci_scalar : sp.Expr
        R = g^{μν} R_{μν}.
    einstein_tensor : sp.Matrix
        G_{μν} = R_{μν} - ½ g_{μν} R.
    kretschner : sp.Expr
        K = R_{ρσμν} R^{ρσμν}.
    coord_symbols : list[sp.Symbol]
        좌표 SymPy 심볼.
    coords_names : tuple[str, ...]
        좌표 이름.
    """

    def __init__(
        self,
        metric,
        inverse_metric,
        christoffel,
        riemann,
        ricci_tensor,
        ricci_scalar,
        einstein_tensor,
        kretschner,
        coord_symbols,
        coords_names,
    ):
        self.metric = metric
        self.inverse_metric = inverse_metric
        self.christoffel = christoffel
        self.riemann = riemann
        self.ricci_tensor = ricci_tensor
        self.ricci_scalar = ricci_scalar
        self.einstein_tensor = einstein_tensor
        self.kretschner = kretschner
        self.coord_symbols = coord_symbols
        self.coords_names = coords_names

    # ── 별칭 ──
    @property
    def g(self):
        return self.metric

    @property
    def g_inv(self):
        return self.inverse_metric

    @property
    def Γ(self):
        return self.christoffel

    @property
    def R(self):
        return self.ricci_scalar

    @property
    def Ric(self):
        return self.ricci_tensor

    @property
    def G(self):
        return self.einstein_tensor

    @property
    def K(self):
        return self.kretschner

    def summary(self) -> str:
        import sympy as sp
        return (
            f"R = {sp.simplify(self.ricci_scalar)}, "
            f"K = {sp.simplify(self.kretschner)}"
        )

    # ── LaTeX 좌표 이름 매핑 ──
    _LATEX_NAMES = {
        "t": "t", "r": "r", "x": "x", "y": "y", "z": "z", "w": "w",
        "θ": r"\theta", "theta": r"\theta",
        "φ": r"\varphi", "phi": r"\varphi",
        "ρ": r"\rho", "rho": r"\rho",
        "ψ": r"\psi", "psi": r"\psi",
    }

    def _latex_name(self, name: str) -> str:
        """좌표 이름을 LaTeX로 변환."""
        return self._LATEX_NAMES.get(name, name)

    def latex(self, tensor: str = "christoffel") -> str:
        """곡률 텐서의 비영 성분을 LaTeX 수식으로 반환한다.

        Parameters
        ----------
        tensor : str
            "christoffel", "riemann", "ricci", "einstein",
            "ricci_scalar", "kretschner", "metric" 중 하나.

        Returns
        -------
        str
            LaTeX aligned equation block.
        """
        import sympy as sp

        names = [self._latex_name(n) for n in self.coords_names]
        dim = len(names)
        eqs = []

        if tensor == "ricci_scalar":
            return f"R = {sp.latex(sp.simplify(self.ricci_scalar))}"

        if tensor == "kretschner":
            return f"K = {sp.latex(sp.simplify(self.kretschner))}"

        if tensor == "metric":
            for m in range(dim):
                for n in range(m, dim):
                    val = sp.simplify(self.metric[m, n])
                    if val != 0:
                        eqs.append(
                            f"g_{{{names[m]}{names[n]}}} &= {sp.latex(val)}"
                        )
            return _latex_align(eqs) if eqs else r"\text{(zero metric)}"

        if tensor == "christoffel":
            arr = self.christoffel
            for s in range(dim):
                for m in range(dim):
                    for n in range(m, dim):
                        val = sp.simplify(arr[s, m, n])
                        if val != 0:
                            eqs.append(
                                rf"\Gamma^{{{names[s]}}}_{{{names[m]}{names[n]}}} "
                                rf"&= {sp.latex(val)}"
                            )

        elif tensor == "riemann":
            arr = self.riemann
            for r in range(dim):
                for s in range(dim):
                    for m in range(dim):
                        for n in range(m + 1, dim):
                            val = sp.simplify(arr[r, s, m, n])
                            if val != 0:
                                eqs.append(
                                    rf"R^{{{names[r]}}}_{{{names[s]}{names[m]}{names[n]}}} "
                                    rf"&= {sp.latex(val)}"
                                )

        elif tensor == "ricci":
            arr = self.ricci_tensor
            for m in range(dim):
                for n in range(m, dim):
                    val = sp.simplify(arr[m, n])
                    if val != 0:
                        eqs.append(
                            rf"R_{{{names[m]}{names[n]}}} &= {sp.latex(val)}"
                        )

        elif tensor == "einstein":
            arr = self.einstein_tensor
            for m in range(dim):
                for n in range(m, dim):
                    val = sp.simplify(arr[m, n])
                    if val != 0:
                        eqs.append(
                            rf"G_{{{names[m]}{names[n]}}} &= {sp.latex(val)}"
                        )

        else:
            raise ValueError(
                f"Unknown tensor '{tensor}'. Use 'christoffel', 'riemann', "
                f"'ricci', 'einstein', 'ricci_scalar', 'kretschner', or 'metric'."
            )

        return _latex_align(eqs) if eqs else r"\text{(all components zero)}"

    def show(self, tensor: str = "christoffel", output: str = "text") -> str:
        """곡률 텐서의 비영 성분을 출력한다.

        Parameters
        ----------
        tensor : str
            "christoffel", "riemann", "ricci", "einstein" 등.
        output : str
            "text" 또는 "latex".

        Returns
        -------
        str
        """
        if output == "latex":
            result = self.latex(tensor)
            # Jupyter에서 display 시도
            try:
                from IPython.display import display, Math
                display(Math(result))
            except ImportError:
                print(result)
            return result

        # text 모드
        import sympy as sp
        names = self.coords_names
        dim = len(names)
        lines = []

        def _fmt(val):
            val = sp.simplify(val)
            return None if val == 0 else str(val)

        if tensor == "christoffel":
            arr = self.christoffel
            for s in range(dim):
                for m in range(dim):
                    for n in range(m, dim):
                        v = _fmt(arr[s, m, n])
                        if v is not None:
                            lines.append(
                                f"  Γ^{names[s]}_{{{names[m]}{names[n]}}} = {v}"
                            )

        elif tensor == "riemann":
            arr = self.riemann
            for r in range(dim):
                for s in range(dim):
                    for m in range(dim):
                        for n in range(m + 1, dim):
                            v = _fmt(arr[r, s, m, n])
                            if v is not None:
                                lines.append(
                                    f"  R^{names[r]}_{{{names[s]}{names[m]}{names[n]}}} = {v}"
                                )

        elif tensor == "ricci":
            arr = self.ricci_tensor
            for m in range(dim):
                for n in range(m, dim):
                    v = _fmt(arr[m, n])
                    if v is not None:
                        lines.append(f"  R_{{{names[m]}{names[n]}}} = {v}")

        elif tensor == "einstein":
            arr = self.einstein_tensor
            for m in range(dim):
                for n in range(m, dim):
                    v = _fmt(arr[m, n])
                    if v is not None:
                        lines.append(f"  G_{{{names[m]}{names[n]}}} = {v}")

        else:
            raise ValueError(
                f"Unknown tensor '{tensor}'. "
                f"Use 'christoffel', 'riemann', 'ricci', or 'einstein'."
            )

        result = "\n".join(lines) if lines else "  (all components zero)"
        print(result)
        return result

    # ── Jupyter repr ──
    def _repr_latex_(self) -> str:
        """Jupyter에서 자동 LaTeX 렌더링."""
        import sympy as sp
        R_latex = sp.latex(sp.simplify(self.ricci_scalar))
        K_latex = sp.latex(sp.simplify(self.kretschner))
        return rf"$R = {R_latex}, \quad K = {K_latex}$"


class _SymbolicCurvatureComputer:
    """SymPy 기반 symbolic 곡률 계산 엔진."""

    def __init__(self, g_matrix, coord_symbols, dim: int):
        import sympy as sp
        self.g = g_matrix
        self.g_inv = g_matrix.inv()
        self.x = coord_symbols
        self.dim = dim
        self.sp = sp

    def compute(self, coord_obj: Coordinates) -> SymbolicCurvatureResult:
        sp = self.sp
        dim = self.dim

        gamma = self._christoffel()
        riem = self._riemann(gamma)
        ricci = self._ricci(riem)
        R = self._ricci_scalar(ricci)
        einstein = self._einstein(ricci, R)
        kretschner = self._kretschner(riem)

        return SymbolicCurvatureResult(
            metric=self.g,
            inverse_metric=self.g_inv,
            christoffel=gamma,
            riemann=riem,
            ricci_tensor=ricci,
            ricci_scalar=R,
            einstein_tensor=einstein,
            kretschner=kretschner,
            coord_symbols=self.x,
            coords_names=coord_obj.names,
        )

    def _christoffel(self):
        """Γ^σ_{μν} = ½ g^{σρ}(g_{ρμ,ν} + g_{ρν,μ} - g_{μν,ρ})."""
        sp = self.sp
        dim, g, g_inv, x = self.dim, self.g, self.g_inv, self.x

        gamma = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
        for s in range(dim):
            for m in range(dim):
                for n in range(m, dim):  # 대칭: Γ^s_{mn} = Γ^s_{nm}
                    val = sp.Rational(0)
                    for r in range(dim):
                        val += g_inv[s, r] * (
                            sp.diff(g[r, m], x[n])
                            + sp.diff(g[r, n], x[m])
                            - sp.diff(g[m, n], x[r])
                        )
                    val = sp.simplify(sp.Rational(1, 2) * val)
                    gamma[s, m, n] = val
                    gamma[s, n, m] = val  # 대칭
        return gamma

    def _riemann(self, gamma):
        """R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} − Γ^ρ_{νλ}Γ^λ_{μσ}."""
        sp = self.sp
        dim, x = self.dim, self.x

        riem = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
        for rho in range(dim):
            for sig in range(dim):
                for mu in range(dim):
                    for nu in range(mu + 1, dim):  # 반대칭: R^ρ_{σμν} = -R^ρ_{σνμ}
                        val = (
                            sp.diff(gamma[rho, nu, sig], x[mu])
                            - sp.diff(gamma[rho, mu, sig], x[nu])
                        )
                        for lam in range(dim):
                            val += (
                                gamma[rho, mu, lam] * gamma[lam, nu, sig]
                                - gamma[rho, nu, lam] * gamma[lam, mu, sig]
                            )
                        val = sp.simplify(val)
                        riem[rho, sig, mu, nu] = val
                        riem[rho, sig, nu, mu] = -val  # 반대칭
        return riem

    def _ricci(self, riem):
        """R_{σν} = R^μ_{σμν} (첫째-셋째 인덱스 축약)."""
        sp = self.sp
        dim = self.dim

        ricci = sp.zeros(dim, dim)
        for s in range(dim):
            for n in range(s, dim):  # 대칭: R_{sn} = R_{ns}
                val = sum(riem[m, s, m, n] for m in range(dim))
                val = sp.simplify(val)
                ricci[s, n] = val
                ricci[n, s] = val
        return ricci

    def _ricci_scalar(self, ricci):
        """R = g^{μν} R_{μν}."""
        sp = self.sp
        dim, g_inv = self.dim, self.g_inv

        val = sum(
            g_inv[m, n] * ricci[m, n]
            for m in range(dim)
            for n in range(dim)
        )
        return sp.simplify(val)

    def _einstein(self, ricci, R):
        """G_{μν} = R_{μν} - ½ g_{μν} R."""
        sp = self.sp
        dim, g = self.dim, self.g

        einstein = sp.zeros(dim, dim)
        for m in range(dim):
            for n in range(m, dim):
                val = sp.simplify(ricci[m, n] - sp.Rational(1, 2) * g[m, n] * R)
                einstein[m, n] = val
                einstein[n, m] = val
        return einstein

    def _kretschner(self, riem):
        """K = R_{ρσμν} R^{ρσμν}."""
        sp = self.sp
        dim, g, g_inv = self.dim, self.g, self.g_inv

        # R_{ρσμν} = g_{ρα} R^α_{σμν}
        # R^{ρσμν} = g^{ρα} g^{σβ} g^{μγ} g^{νδ} R_{αβγδ}
        # K = Σ R_{ρσμν} R^{ρσμν}
        #   = Σ g_{ρα} R^α_{σμν} · g^{ρβ} g^{σγ} g^{μδ} g^{νε} g_{βφ} R^φ_{γδε}
        # 더 간단하게: K = Σ R^a_{bcd} R^e_{fgh} g_{ae} g^{bf} g^{cg} g^{dh}
        val = sp.Rational(0)
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    for d in range(c + 1, dim):  # 반대칭 이용: *2
                        if riem[a, b, c, d] == 0:
                            continue
                        for e in range(dim):
                            for f in range(dim):
                                for h in range(dim):  # g, h → 반대칭
                                    r_up = sum(
                                        g_inv[c, gg] * g_inv[d, h]
                                        * riem[e, f, gg, h]
                                        for gg in range(dim)
                                    )
                                    if r_up == 0:
                                        continue
                                    val += (
                                        g[a, e] * g_inv[b, f]
                                        * riem[a, b, c, d] * r_up
                                    )
        val = 2 * val  # 반대칭 (c,d) 보정
        return sp.simplify(val)
