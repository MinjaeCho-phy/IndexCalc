"""
InvariantTensor: 그룹 변환 하에서 invariant한 텐서의 메타데이터.

예시: $\\delta^I_J$ ($SU(N)$ fund-antifund), $f^{abc}, d^{abc}$ (adjoint),
$\\epsilon_{i_1\\cdots i_N}$ ($SU(N)$ fund), $\\eta_{ab}$ (Lorentz frame).

이 모듈은 invariance 검증기에서 "이 텐서는 $G$ 변환 하에 자기 자신으로 간다 ⇒
generator 작용 시 0이다"를 lookup하는 데 사용된다.

Examples
--------
>>> reg = InvariantTensorRegistry()
>>> reg.declare("delta", group_name="SU(3)",
...             index_pattern=("fund_upper", "fund_lower"),
...             symmetry=None)
>>> reg.declare("f", group_name="SU(3)",
...             index_pattern=("adj", "adj", "adj"),
...             symmetry="totally_antisymmetric")
>>> reg.is_invariant("delta", "SU(3)")
True
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InvariantTensor:
    """그룹 invariant 텐서의 메타데이터.

    Parameters
    ----------
    name : str
        텐서 이름 (e.g., "delta", "f", "d", "epsilon", "eta", "g").
    group_name : str
        invariant인 그룹의 이름. spacetime metric이면 "spacetime" 같은 dummy 사용.
    index_pattern : tuple[str, ...]
        각 슬롯의 rep 라벨 — 그룹 내부 표현 + 위/아래 표시.
        non-abelian 그룹: ("fund_upper", "fund_lower"), ("adj", "adj", "adj") 등.
        spacetime metric: ("spacetime_lower", "spacetime_lower").
    symmetry : str | None
        "symmetric" / "antisymmetric" / "totally_symmetric" /
        "totally_antisymmetric" / None.
    """

    name: str
    group_name: str
    index_pattern: tuple[str, ...]
    symmetry: Optional[str] = None


class InvariantTensorRegistry:
    """그룹별 invariant 텐서 등록소.

    한 그룹에 여러 invariant 텐서가 등록될 수 있다 ($SU(N)$이면 $\\delta, f, d, \\epsilon$).
    같은 이름이 다른 그룹에 동시에 등록되는 것도 허용된다 (e.g., $\\epsilon$).

    Examples
    --------
    >>> reg = InvariantTensorRegistry()
    >>> reg.declare("delta", "SU(2)", ("fund_upper", "fund_lower"))
    >>> reg.declare("epsilon", "SU(2)", ("fund_lower", "fund_lower"),
    ...             symmetry="totally_antisymmetric")
    >>> reg.list_for_group("SU(2)")
    ['delta', 'epsilon']
    """

    def __init__(self):
        # key: (name, group_name) → InvariantTensor
        self._tensors: dict[tuple[str, str], InvariantTensor] = {}

    def declare(
        self,
        name: str,
        group_name: str,
        index_pattern: tuple[str, ...],
        symmetry: Optional[str] = None,
    ) -> InvariantTensor:
        """invariant 텐서를 등록한다."""
        key = (name, group_name)
        if key in self._tensors:
            raise ValueError(
                f"Invariant tensor {name!r} for group {group_name!r} already declared"
            )
        if symmetry not in (
            None,
            "symmetric",
            "antisymmetric",
            "totally_symmetric",
            "totally_antisymmetric",
        ):
            raise ValueError(f"unknown symmetry {symmetry!r}")
        inv = InvariantTensor(
            name=name,
            group_name=group_name,
            index_pattern=tuple(index_pattern),
            symmetry=symmetry,
        )
        self._tensors[key] = inv
        return inv

    def get(self, name: str, group_name: str) -> InvariantTensor:
        key = (name, group_name)
        if key not in self._tensors:
            raise KeyError(
                f"Invariant tensor ({name!r}, {group_name!r}) not declared"
            )
        return self._tensors[key]

    def is_invariant(self, name: str, group_name: str) -> bool:
        return (name, group_name) in self._tensors

    def list_for_group(self, group_name: str) -> list[str]:
        return [n for (n, g) in self._tensors if g == group_name]

    @property
    def all_tensors(self) -> dict[tuple[str, str], InvariantTensor]:
        return dict(self._tensors)


# ─── 표준 패키지 헬퍼 ─────────────────────────────────────────


def standard_su_n_invariants(N: int) -> list[InvariantTensor]:
    """$SU(N)$의 표준 invariant 텐서 목록을 메타데이터로 반환한다.

    Returns
    -------
    [delta (fund-antifund), f (adj), d (adj), epsilon_N (fund×N)]
    """
    g = f"SU({N})"
    items = [
        InvariantTensor("delta", g, ("fund_upper", "fund_lower"), None),
        InvariantTensor("f", g, ("adj",) * 3, "totally_antisymmetric"),
        InvariantTensor("d", g, ("adj",) * 3, "totally_symmetric"),
        InvariantTensor(
            "epsilon", g, ("fund_lower",) * N, "totally_antisymmetric"
        ),
    ]
    return items


def standard_u_n_invariants(N: int) -> list[InvariantTensor]:
    """$U(N) = SU(N) \\times U(1)$의 표준 invariants. $\\epsilon$은 fund 한정."""
    g = f"U({N})"
    return [
        InvariantTensor("delta", g, ("fund_upper", "fund_lower"), None),
        InvariantTensor("f", g, ("adj",) * 3, "totally_antisymmetric"),
        InvariantTensor("d", g, ("adj",) * 3, "totally_symmetric"),
    ]


def standard_lorentz_invariants() -> list[InvariantTensor]:
    """Lorentz frame ($SO(1,3)$ vector rep)의 invariants — $\\eta_{ab}$, $\\epsilon_{abcd}$."""
    g = "Lorentz"
    return [
        InvariantTensor(
            "eta", g, ("frame_lower", "frame_lower"), "symmetric"
        ),
        InvariantTensor(
            "epsilon4", g, ("frame_lower",) * 4, "totally_antisymmetric"
        ),
    ]


def standard_o_n_invariants(N: int) -> list[InvariantTensor]:
    """$O(N)$ vector rep의 standard invariants — $\\delta_{ij}$ (symmetric metric)
    그리고 mixed identity $\\delta^i{}_j$.

    Note: $\\epsilon_{i_1\\cdots i_N}$은 $O(N)$이 아닌 $SO(N)$ 한정 invariant
    (improper rotation에서 sign flip). ``standard_so_n_invariants``에 등록.
    """
    g = f"O({N})"
    return [
        InvariantTensor(
            "delta", g, ("vector_lower", "vector_lower"), "symmetric"
        ),
        InvariantTensor(
            "delta_mixed", g, ("vector_upper", "vector_lower"), None
        ),
    ]


def standard_sp_2n_invariants(N: int) -> list[InvariantTensor]:
    """$Sp(2N)$ fundamental rep의 standard invariants — **반대칭** symplectic
    form $\\Omega_{ij} = -\\Omega_{ji}$ 그리고 mixed identity $\\Omega^i{}_j$.

    $O(N)$의 대칭 $\\delta_{ij}$ 와 대비된다 (``standard_o_n_invariants``).
    Ω가 반대칭이므로 단일 boson bilinear $\\Omega_{ij}\\phi^i\\phi^j = 0$ 으로
    사라진다 — 살아남는 invariant는 두 다른 field $\\Omega_{ij}\\phi^i\\chi^j$,
    Grassmann $\\Omega_{ij}\\psi^i\\psi^j$, 또는 미분 $\\Omega_{ij}\\phi^i\\partial\\phi^j$.

    Parameters
    ----------
    N : int
        symplectic rank. fundamental rep의 차원은 $2N$ (= group의 작용 차원).
    """
    g = f"Sp({2 * N})"
    return [
        InvariantTensor(
            "omega", g, ("vector_lower", "vector_lower"), "antisymmetric"
        ),
        InvariantTensor(
            "omega_mixed", g, ("vector_upper", "vector_lower"), None
        ),
    ]


def standard_conformal_invariants(d: int) -> list[InvariantTensor]:
    """Conformal group $SO(d,2)$ in the embedding-space formalism.

    Treated as an orthogonal group acting on a $(d+2)$-dimensional space
    with an indefinite $(d,2)$-signature metric $\\eta^{\\mathrm{conf}}_{AB}$
    — exactly as the Lorentz group is $SO(1,d-1)$ with $\\eta_{ab}$. The
    signature does not change the IR contraction structure (a symmetric
    2-index invariant), so this mirrors ``standard_so_n_invariants`` with a
    distinct tensor/metric name to separate it from Euclidean δ and Lorentz η.

    Parameters
    ----------
    d : int
        Spacetime dimension; the conformal group is $SO(d,2)$ acting on
        the $(d+2)$-dim embedding space.
    """
    g = f"SO({d},2)"
    return [
        InvariantTensor(
            "eta_conf", g, ("vector_lower", "vector_lower"), "symmetric"
        ),
        InvariantTensor(
            "eta_conf_mixed", g, ("vector_upper", "vector_lower"), None
        ),
        InvariantTensor(
            "epsilon", g, ("vector_lower",) * (d + 2), "totally_antisymmetric"
        ),
    ]


def standard_o_dd_invariants(D: int) -> list[InvariantTensor]:
    """T-duality group $O(D,D)$ in the doubled / embedding-space formalism.

    The continuous duality group of NS-NS string theory on a $D$-torus
    acts on a $2D$-dimensional doubled space preserving a symmetric
    $O(D,D)$ metric $\\eta^{\\mathrm{dd}}_{MN}$ of split signature $(D,D)$
    (the off-diagonal $\\eta=\\bigl(\\begin{smallmatrix}0&1\\\\1&0\\end{smallmatrix}\\bigr)$).
    Like the conformal and Lorentz cases, the indefinite signature does not
    change the IR contraction structure (a symmetric 2-index invariant), so
    this mirrors ``standard_so_n_invariants`` *without* the Levi-Civita —
    $O(D,D)$ (the full T-duality group, including orientation-reversing
    $\\det=-1$ elements) carries only the metric, like $O(N)$. A distinct
    tensor/metric name separates it from Euclidean $\\delta$, conformal
    $\\eta^{\\mathrm{conf}}$, and symplectic $\\Omega$ at equal dimension.

    Parameters
    ----------
    D : int
        Number of doubled dimensions; the group is $O(D,D)$ acting on the
        $2D$-dim doubled space.
    """
    g = f"O({D},{D})"
    return [
        InvariantTensor(
            "eta_dd", g, ("vector_lower", "vector_lower"), "symmetric"
        ),
        InvariantTensor(
            "eta_dd_mixed", g, ("vector_upper", "vector_lower"), None
        ),
    ]


def standard_so_n_invariants(N: int) -> list[InvariantTensor]:
    """$SO(N)$ vector rep — $O(N)$의 invariants + $\\epsilon_{i_1\\cdots i_N}$.

    Parameters
    ----------
    N : int
        벡터 rep의 차원 (= group의 fundamental action 차원).
    """
    g = f"SO({N})"
    return [
        InvariantTensor(
            "delta", g, ("vector_lower", "vector_lower"), "symmetric"
        ),
        InvariantTensor(
            "delta_mixed", g, ("vector_upper", "vector_lower"), None
        ),
        InvariantTensor(
            "epsilon", g, ("vector_lower",) * N, "totally_antisymmetric"
        ),
    ]
