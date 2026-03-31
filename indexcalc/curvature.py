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
        func: Callable,
        coords: list[str] | str | Coordinates,
        signature: tuple | None = None,
        dim: int | None = None,
    ):
        self.func = func

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

        # ── 내부 curvature computer ──
        self._computer = _CurvatureComputer(func, self._coords.dim)

    @property
    def coords(self) -> Coordinates:
        return self._coords

    @property
    def dim(self) -> int:
        return self._coords.dim

    @property
    def signature(self) -> tuple[int, ...]:
        return self._signs

    def at(self, x) -> CurvatureResult:
        """주어진 좌표에서 모든 곡률 텐서를 계산한다.

        Parameters
        ----------
        x : array-like
            좌표 벡터, shape (dim,).

        Returns
        -------
        CurvatureResult
        """
        return self._computer.at(x, self._coords)

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
