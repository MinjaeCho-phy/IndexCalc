"""ADM 3+1 분해 setup + extrinsic curvature.

4D spacetime을 3D spatial slice + time direction으로 분해할 때 등장하는
대상들 (lapse N, shift N^i, spatial metric h_{ij}, extrinsic curvature K_{ij})
을 IndexCalc 텐서로 빌드해 주는 helper.

설계:
    - ``ADMSetup``: 4D / 3D IndexSpace 페어 + 이름 컨벤션 보유. 각종 leaf
      텐서 (``lapse``, ``shift``, ``spatial_metric``, ``extrinsic_curvature``)
      를 builder 메서드로 노출.
    - ``TimeDeriv``: ∂_t T. 4D μ 인덱스를 free에 추가하지 않는다 — t는
      coordinate symbol로 취급 (3D-context의 ḣ_{ij}는 (0,2) tensor로 본다).
    - ``extrinsic_curvature_definition``: K_{ij} = (1/(2N))(∂_t h_{ij} - D_i N_j - D_j N_i).
    - ``K_trace_definition``: K = h^{ij} K_{ij}.
    - ``metric_lower_components`` / ``metric_upper_components``: g_{μν}, g^{μν}을
      ('tt', 'ti', 'ij')로 키된 dict. 각 entry는 TensorExpr (free index 구조에
      맞춰 0, 1, 2 free).

이 모듈은 4D abstract index의 0-component projection을 IndexCalc 안에 직접
구현하지 않는다. component dict는 사용자가 보고 추후 evaluation에 쓰거나
독립 검산용으로 사용.
"""

from __future__ import annotations

import sympy as sp

from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, Connection
from indexcalc.core.spatial_deriv import SpatialCovariantDeriv


# ─── Lie derivative ────────────────────────────────────────


class LieDeriv(TensorExpr):
    """``L_X T`` — vector ``X``에 대한 Lie 미분.

    ``X``는 1-index Tensor (upper). free_indices = inner ``T``의 free.

    Note: Lie derivative는 connection-independent; expansion 시 ``∂``만 사용.
    """

    def __init__(self, vector: Tensor, expr: TensorExpr):
        if not isinstance(vector, Tensor) or len(vector.indices) != 1:
            raise ValueError("LieDeriv vector must be a 1-index Tensor")
        if vector.indices[0].position != "upper":
            raise ValueError("LieDeriv vector index must be upper")
        self.vector = vector
        self.expr = expr

    @property
    def free_indices(self) -> list[Index]:
        return self.expr.free_indices

    def __repr__(self) -> str:
        return f"L_{self.vector.name}({self.expr})"


def _next_lie_dummy(base: str, taken: set[str]) -> str:
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate not in taken:
            return candidate
        i += 1


def _replace_slot_preserve(T: Tensor, slot: int, new_idx: Index) -> Tensor:
    """T의 slot 인덱스만 new_idx로 교체 (속성 보존)."""
    new_indices = list(T.indices)
    new_indices[slot] = new_idx
    return Tensor(
        T.name, new_indices,
        antisymmetric_pairs=list(T.antisymmetric_pairs),
        symmetric_pairs=list(T.symmetric_pairs),
        traceless=list(T.traceless),
        transverse=list(T.transverse),
        reps=dict(T.reps),
        statistics=T.statistics,
    )


def _expand_single_lie(lie: LieDeriv) -> TensorExpr:
    """단일 LieDeriv 노드 전개 (T가 leaf Tensor일 때)."""
    X = lie.vector
    T = lie.expr

    if not isinstance(T, Tensor):
        # Leibniz 분배
        if isinstance(T, TensorProduct):
            return _expand_lie_walk(
                TensorSum(
                    TensorProduct(LieDeriv(X, T.left), T.right),
                    TensorProduct(T.left, LieDeriv(X, T.right)),
                )
            )
        if isinstance(T, TensorSum):
            return _expand_lie_walk(
                TensorSum(LieDeriv(X, T.left), LieDeriv(X, T.right))
            )
        if isinstance(T, ScalarMul):
            return ScalarMul(T.scalar, _expand_lie_walk(LieDeriv(X, T.expr)))
        # 기타 노드는 그대로 LieDeriv 유지
        return lie

    X_idx = X.indices[0]
    rho = X_idx.name
    rho_lo = Index(rho, X_idx.space, "lower")

    # Advection: X^ρ ∂_ρ T (G8 자동 contract; X와 ∂_ρ deriv_index가 같은 ρ)
    advection = TensorProduct(X, PartialDeriv(T, rho_lo))

    correction_terms: list[TensorExpr] = []
    taken = {idx.name for idx in T.indices} | {rho, X.name}

    for slot, idx in enumerate(T.indices):
        base = idx.space.indices[0] if idx.space.indices else "i"
        dummy = _next_lie_dummy(base, taken)
        taken.add(dummy)

        if idx.position == "upper":
            # - T^{ρ-replaced ...} ∂_ρ X^{idx.name}
            T_replaced = _replace_slot_preserve(
                T, slot, Index(dummy, idx.space, "upper"),
            )
            X_a = Tensor(X.name, [Index(idx.name, idx.space, "upper")])
            d_X = PartialDeriv(X_a, Index(dummy, idx.space, "lower"))
            correction_terms.append(
                ScalarMul(-1, TensorProduct(T_replaced, d_X))
            )
        else:
            # + T^{... ρ-replaced ...} ∂_{idx.name} X^ρ
            T_replaced = _replace_slot_preserve(
                T, slot, Index(dummy, idx.space, "lower"),
            )
            X_rho = Tensor(X.name, [Index(dummy, idx.space, "upper")])
            d_X = PartialDeriv(X_rho, Index(idx.name, idx.space, "lower"))
            correction_terms.append(TensorProduct(T_replaced, d_X))

    result: TensorExpr = advection
    for term in correction_terms:
        result = TensorSum(result, term)
    return result


def _expand_lie_walk(expr: TensorExpr) -> TensorExpr:
    if isinstance(expr, LieDeriv):
        return _expand_single_lie(expr)
    if isinstance(expr, TensorSum):
        return TensorSum(
            _expand_lie_walk(expr.left), _expand_lie_walk(expr.right),
        )
    if isinstance(expr, TensorProduct):
        return TensorProduct(
            _expand_lie_walk(expr.left), _expand_lie_walk(expr.right),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, _expand_lie_walk(expr.expr))
    if isinstance(expr, PartialDeriv):
        return PartialDeriv(_expand_lie_walk(expr.expr), expr.deriv_index)
    return expr


def expand_lie_deriv(expr: TensorExpr) -> TensorExpr:
    """``LieDeriv`` 노드를 explicit ``∂`` form으로 전개.

    공식:
        L_X T = X^ρ ∂_ρ T
              - Σ_{upper s} T^{...ρ_s...} ∂_ρ X^{idx_s}
              + Σ_{lower s} T^{...ρ_s...} ∂_{idx_s} X^ρ

    Connection-independent — ∂만 사용 (공식 자체가 Christoffel로 환원되어
    cancel되도록 설계되어 있다).
    """
    return _expand_lie_walk(expr)


# ─── TimeDeriv ──────────────────────────────────────────────


class TimeDeriv(TensorExpr):
    """∂_t T — 시간 미분. free_indices는 inner와 동일 (t는 free 추가 안 됨).

    ADM 3+1 셋업에서 ḣ_{ij}, Ṅ 같은 시간 미분을 (0, n) 텐서로 다루기 위해
    도입. PartialDeriv와 달리 deriv_index가 없고, t는 coordinate symbol로
    취급된다.

    Examples
    --------
    >>> h_ij = Tensor("h", [i_lo, j_lo], symmetric_pairs=[(0,1)])
    >>> dt_h = TimeDeriv(h_ij)
    >>> [i.name for i in dt_h.free_indices]
    ['i', 'j']
    """

    def __init__(self, expr: TensorExpr):
        self.expr = expr

    @property
    def free_indices(self) -> list[Index]:
        return self.expr.free_indices

    def __repr__(self) -> str:
        return f"∂_t({self.expr})"


# ─── ADMSetup ──────────────────────────────────────────────


class ADMSetup:
    """ADM 3+1 분해 셋업.

    Parameters
    ----------
    st : IndexSpace
        4D spacetime (dim=4).
    sp : IndexSpace
        3D spatial slice (dim=3).
    lapse_name, shift_name, spatial_name, K_name : str
        Builder가 만들 텐서의 이름. 기본 'N', 'N', 'h', 'K'.
    """

    def __init__(
        self,
        st: IndexSpace,
        sp: IndexSpace,
        *,
        lapse_name: str = "N",
        shift_name: str = "N",
        spatial_name: str = "h",
        K_name: str = "K",
    ):
        if st.dim != 4:
            raise ValueError(f"ADM expects 4D spacetime, got dim={st.dim}")
        if sp.dim != 3:
            raise ValueError(f"ADM expects 3D spatial slice, got dim={sp.dim}")
        self.st = st
        self.sp = sp
        self.lapse_name = lapse_name
        self.shift_name = shift_name
        self.spatial_name = spatial_name
        self.K_name = K_name

    def lapse(self) -> Tensor:
        """``N`` — scalar Tensor."""
        return Tensor(self.lapse_name, [])

    def shift(self, name: str = "i", position: str = "upper") -> Tensor:
        """``N^i`` (default) 또는 ``N_i``."""
        return Tensor(self.shift_name, [Index(name, self.sp, position)])

    def spatial_metric_lower(self, i: str = "i", j: str = "j") -> Tensor:
        """``h_{ij}`` (symmetric)."""
        return Tensor(
            self.spatial_name,
            [Index(i, self.sp, "lower"), Index(j, self.sp, "lower")],
            symmetric_pairs=[(0, 1)],
        )

    def spatial_metric_upper(self, i: str = "i", j: str = "j") -> Tensor:
        """``h^{ij}`` (symmetric)."""
        return Tensor(
            self.spatial_name,
            [Index(i, self.sp, "upper"), Index(j, self.sp, "upper")],
            symmetric_pairs=[(0, 1)],
        )

    def extrinsic_curvature(self, i: str = "i", j: str = "j") -> Tensor:
        """``K_{ij}`` (symmetric)."""
        return Tensor(
            self.K_name,
            [Index(i, self.sp, "lower"), Index(j, self.sp, "lower")],
            symmetric_pairs=[(0, 1)],
        )

    def extrinsic_curvature_upper(self, i: str = "i", j: str = "j") -> Tensor:
        """``K^{ij}`` (symmetric)."""
        return Tensor(
            self.K_name,
            [Index(i, self.sp, "upper"), Index(j, self.sp, "upper")],
            symmetric_pairs=[(0, 1)],
        )

    def extrinsic_curvature_mixed(
        self, upper: str = "j", lower: str = "i",
    ) -> Tensor:
        """``K^{j}{}_{i}`` (mixed; symmetric_pairs 안 부여 — 위치 mismatch).

        K_{ij} 대칭 가정 하에 K^j_i = K_i^j이지만, IndexCalc Tensor 모델에서는
        slot position이 다르면 자동 swap 무효 — sym 속성 생략.
        """
        return Tensor(
            self.K_name,
            [Index(upper, self.sp, "upper"), Index(lower, self.sp, "lower")],
        )

    # ─── 3D 내재 곡률 (R^(3)) ─────────────────────────────────

    def ricci3_lower(
        self,
        i: str = "i", j: str = "j",
        *,
        name: str = "R^{(3)}",
    ) -> Tensor:
        """``R^{(3)}_{ij}`` (3D Ricci tensor, symmetric)."""
        return Tensor(
            name,
            [Index(i, self.sp, "lower"), Index(j, self.sp, "lower")],
            symmetric_pairs=[(0, 1)],
        )

    def ricci3_scalar(self, *, name: str = "R^{(3)}") -> Tensor:
        """``R^{(3)}`` (3D Ricci scalar, no indices)."""
        return Tensor(name, [])

    def riemann3(
        self,
        i: str = "i", j: str = "j", k: str = "k", l: str = "l",
        *,
        name: str = "R^{(3)}",
    ) -> Tensor:
        """``R^{(3)}_{ijkl}`` (3D Riemann, antisym (0,1) and (2,3)).

        대각선 페어 swap 대칭 R_{ijkl}=R_{klij}는 Tensor 모델에서 직접
        표현되지 않음 (현 패스에서는 미지원).
        """
        return Tensor(
            name,
            [
                Index(i, self.sp, "lower"),
                Index(j, self.sp, "lower"),
                Index(k, self.sp, "lower"),
                Index(l, self.sp, "lower"),
            ],
            antisymmetric_pairs=[(0, 1), (2, 3)],
        )


# ─── K definitions ─────────────────────────────────────────


def extrinsic_curvature_definition(
    adm: ADMSetup,
    conn3: Connection,
    *,
    i: str = "i",
    j: str = "j",
) -> TensorExpr:
    """``K_{ij} = (1/(2N))(∂_t h_{ij} - D_i N_j - D_j N_i)``.

    Parameters
    ----------
    adm : ADMSetup
    conn3 : Connection
        3D spatial Levi-Civita connection (h_{ij}의).
    i, j : str
        Free index 이름 (기본 'i', 'j').
    """
    h_ij = adm.spatial_metric_lower(i, j)
    Ni = adm.shift(i, "lower")
    Nj = adm.shift(j, "lower")
    i_lo = Index(i, adm.sp, "lower")
    j_lo = Index(j, adm.sp, "lower")

    dt_h = TimeDeriv(h_ij)
    Di_Nj = SpatialCovariantDeriv(Nj, i_lo, conn3)
    Dj_Ni = SpatialCovariantDeriv(Ni, j_lo, conn3)

    inner = dt_h - Di_Nj - Dj_Ni  # TensorSum
    N_sym = sp.Symbol(adm.lapse_name)
    coeff = sp.Rational(1, 2) / N_sym
    return ScalarMul(coeff, inner)


def K_trace_definition(adm: ADMSetup, *, i: str = "i", j: str = "j") -> TensorExpr:
    """``K = h^{ij} K_{ij}``."""
    h_inv = adm.spatial_metric_upper(i, j)
    K = adm.extrinsic_curvature(i, j)
    return TensorProduct(h_inv, K)


# ─── Constraints (Hamiltonian / Momentum) ──────────────────


def hamiltonian_constraint(adm: ADMSetup) -> TensorExpr:
    """진공 Hamiltonian 제약 ``ℋ = R^{(3)} + K^2 - K_{ij} K^{ij} = 0``의 LHS.

    Returns
    -------
    TensorExpr
        free=[]. ``R^{(3)} + (h^{ij}K_{ij})(h^{kl}K_{kl}) - K_{ij}K^{ij}``.
    """
    R3 = adm.ricci3_scalar()
    # K^2 = K_trace × K_trace (different dummy names)
    K_tr_a = K_trace_definition(adm, i="i", j="j")
    K_tr_b = K_trace_definition(adm, i="k", j="l")
    K_sq = TensorProduct(K_tr_a, K_tr_b)
    # K_{ij} K^{ij}: 다른 dummy로 K_sq와 충돌 회피
    K_lo = adm.extrinsic_curvature("m", "n")
    K_up = adm.extrinsic_curvature_upper("m", "n")
    K_ij_sq = TensorProduct(K_lo, K_up)
    return R3 + K_sq - K_ij_sq


def momentum_constraint(
    adm: ADMSetup,
    conn3: Connection,
    *,
    i: str = "i",
) -> TensorExpr:
    """진공 momentum 제약 ``ℋ_i = D_j K^{j}{}_{i} - D_i K = 0``의 LHS.

    Parameters
    ----------
    adm : ADMSetup
    conn3 : Connection
        3D spatial Levi-Civita connection.
    i : str
        Free index 이름 (기본 'i').

    Returns
    -------
    TensorExpr
        free=[i]. ``D_j K^j_i - D_i K``.
    """
    K_mixed = adm.extrinsic_curvature_mixed(upper="j", lower=i)
    j_lo = Index("j", adm.sp, "lower")
    Dj_K = SpatialCovariantDeriv(K_mixed, j_lo, conn3)

    K_tr = K_trace_definition(adm, i="k", j="l")
    i_lo = Index(i, adm.sp, "lower")
    Di_K = SpatialCovariantDeriv(K_tr, i_lo, conn3)

    return Dj_K - Di_K


# ─── Evolution equations RHS ───────────────────────────────


def h_evolution_rhs(
    adm: ADMSetup,
    conn3: Connection,
    *,
    i: str = "i",
    j: str = "j",
) -> TensorExpr:
    """``∂_t h_{ij} = -2N K_{ij} + D_i N_j + D_j N_i`` 의 우변.

    extrinsic_curvature_definition을 ∂_t h에 대해 푼 형태 (등가).
    """
    N = adm.lapse()
    K_ij = adm.extrinsic_curvature(i, j)
    Ni = adm.shift(i, "lower")
    Nj = adm.shift(j, "lower")
    i_lo = Index(i, adm.sp, "lower")
    j_lo = Index(j, adm.sp, "lower")

    term1 = ScalarMul(-2, TensorProduct(N, K_ij))
    term2 = SpatialCovariantDeriv(Nj, i_lo, conn3)
    term3 = SpatialCovariantDeriv(Ni, j_lo, conn3)
    return term1 + term2 + term3


def K_evolution_rhs(
    adm: ADMSetup,
    conn3: Connection,
    *,
    i: str = "i",
    j: str = "j",
    include_shift_advection: bool = False,
) -> TensorExpr:
    """``∂_t K_{ij}`` RHS.

    .. math::
        \\partial_t K_{ij} \\;=\\; -D_i D_j N \\;+\\; N\\bigl(R^{(3)}_{ij}
        + K\\, K_{ij} - 2 K_{ik} K^k{}_j\\bigr)
        \\;[\\;+\\; \\mathcal{L}_{\\vec N} K_{ij}\\;]

    ``include_shift_advection=True``이면 마지막 ``L_N K_ij`` 항 추가 — 일반
    (non-static shift) 케이스. 진공·정지 shift는 기본값.
    Matter 항은 미포함.
    """
    N = adm.lapse()
    R3_ij = adm.ricci3_lower(i, j)
    K_ij = adm.extrinsic_curvature(i, j)
    K_tr = K_trace_definition(adm, i="m", j="n")
    K_ik = adm.extrinsic_curvature(i, "k")
    K_kj = adm.extrinsic_curvature_mixed(upper="k", lower=j)

    i_lo = Index(i, adm.sp, "lower")
    j_lo = Index(j, adm.sp, "lower")
    Dj_N = SpatialCovariantDeriv(N, j_lo, conn3)
    Di_Dj_N = SpatialCovariantDeriv(Dj_N, i_lo, conn3)

    term1 = ScalarMul(-1, Di_Dj_N)
    term2 = TensorProduct(N, R3_ij)
    term3 = TensorProduct(N, TensorProduct(K_tr, K_ij))
    term4 = ScalarMul(-2, TensorProduct(N, TensorProduct(K_ik, K_kj)))
    result = term1 + term2 + term3 + term4

    if include_shift_advection:
        N_vec = adm.shift("p", "upper")  # N^p (free에 사용 안 됨, advection의 X)
        K_for_lie = adm.extrinsic_curvature(i, j)
        result = result + LieDeriv(N_vec, K_for_lie)
    return result


# ─── 4D ↔ ADM split (slice_decompose) ──────────────────────


def slice_decompose(
    T: Tensor,
    sp_space: IndexSpace,
    *,
    name_suffix: bool = True,
    spatial_letters: str = "ijklmn",
) -> dict[tuple[str, ...], Tensor]:
    """4D Tensor T를 ADM (t/spatial) 컴포넌트 dict로 분해.

    각 slot에 대해 't'(시간)/'i'(공간) 둘 중 하나로 specialize. 결과는 슬롯 라벨
    튜플을 키로 한 dict, 값은 spatial-only 인덱스만 가진 Tensor.

    Parameters
    ----------
    T : Tensor
        분해할 4D 텐서. 모든 인덱스는 같은 4D space에 있어야 함 (혼합 X).
    sp_space : IndexSpace
        결과 컴포넌트의 spatial slot에 사용할 3D space.
    name_suffix : bool
        True 시 새 텐서 이름에 ``_<labels>`` suffix (예: ``T_ti``).
    spatial_letters : str
        spatial 슬롯에 차례대로 부여할 인덱스 이름 후보.

    Returns
    -------
    dict
        키: ``('t', 'i', ...)`` n-튜플; 값: ``Tensor``.

    Examples
    --------
    >>> g_lo = Tensor("g", [st.lower("μ"), st.lower("ν")], symmetric_pairs=[(0,1)])
    >>> comps = slice_decompose(g_lo, sp)
    >>> set(comps.keys())  # {('t','t'), ('t','i'), ('i','t'), ('i','j')}
    """
    from itertools import product as _product

    n = len(T.indices)
    pos_template = [idx.position for idx in T.indices]
    out: dict[tuple[str, ...], Tensor] = {}

    for combo in _product(("t", "s"), repeat=n):
        new_indices: list[Index] = []
        letter_iter = iter(spatial_letters)
        labels: list[str] = []
        for k, kind in enumerate(combo):
            if kind == "t":
                labels.append("t")
                continue
            try:
                letter = next(letter_iter)
            except StopIteration as e:
                raise ValueError(
                    f"Not enough spatial letters for rank-{n} decomposition; "
                    f"provide a longer spatial_letters string."
                ) from e
            labels.append(letter)
            new_indices.append(Index(letter, sp_space, pos_template[k]))

        key = tuple(labels)
        suffix = "".join(labels)
        new_name = f"{T.name}_{suffix}" if name_suffix else T.name
        out[key] = Tensor(new_name, new_indices)

    return out


# ─── Gauss / Codazzi RHS ───────────────────────────────────


def gauss_rhs(
    adm: ADMSetup,
    *,
    i: str = "i", j: str = "j", k: str = "k", l: str = "l",
) -> TensorExpr:
    """Gauss 식 우변 ``R^{(3)}_{ijkl} + K_{ik} K_{jl} - K_{il} K_{jk}``.

    4D Riemann의 순 spatial 사영과 같다.
    """
    R3 = adm.riemann3(i, j, k, l)
    K_ik = adm.extrinsic_curvature(i, k)
    K_jl = adm.extrinsic_curvature(j, l)
    K_il = adm.extrinsic_curvature(i, l)
    K_jk = adm.extrinsic_curvature(j, k)
    return R3 + TensorProduct(K_ik, K_jl) - TensorProduct(K_il, K_jk)


def codazzi_rhs(
    adm: ADMSetup,
    conn3: Connection,
    *,
    j: str = "j", k: str = "k", l: str = "l",
) -> TensorExpr:
    """Codazzi 식 우변 ``D_l K_{jk} - D_k K_{jl}``.

    4D Riemann의 (3 spatial + 1 normal) 사영과 같다.
    """
    K_jk = adm.extrinsic_curvature(j, k)
    K_jl = adm.extrinsic_curvature(j, l)
    l_lo = Index(l, adm.sp, "lower")
    k_lo = Index(k, adm.sp, "lower")
    return SpatialCovariantDeriv(K_jk, l_lo, conn3) - SpatialCovariantDeriv(K_jl, k_lo, conn3)


# ─── 4D metric components (g_{μν}, g^{μν}) ─────────────────


_ONE_SCALAR = Tensor("1", [])  # 0-rank 'scalar' 표현용 wrapper


def metric_lower_components(adm: ADMSetup) -> dict[str, TensorExpr]:
    """``g_{μν}`` ADM 컴포넌트.

    Returns
    -------
    dict
        ``'tt'`` → -N^2 + N_k N^k    (scalar TensorExpr; free=[])
        ``'ti'`` → N_i               (1-index, free=[i])
        ``'ij'`` → h_{ij}            (2-index symmetric, free=[i,j])

    Note: 'ti'는 N_i (lower)만 반환 — g_{0i}는 N_i = h_{ij} N^j 로 lowered.
    """
    N = adm.lapse()
    Nk_lo = adm.shift("k", "lower")
    Nk_up = adm.shift("k", "upper")
    Ni = adm.shift("i", "lower")
    h_ij = adm.spatial_metric_lower("i", "j")

    g_tt = ScalarMul(-1, TensorProduct(N, N)) + TensorProduct(Nk_lo, Nk_up)
    g_ti = Ni
    g_ij = h_ij
    return {"tt": g_tt, "ti": g_ti, "ij": g_ij}


def metric_upper_components(adm: ADMSetup) -> dict[str, TensorExpr]:
    """``g^{μν}`` ADM 컴포넌트.

    Returns
    -------
    dict
        ``'tt'`` → -1/N^2                        (scalar)
        ``'ti'`` → N^i / N^2                     (1-index, free=[i])
        ``'ij'`` → h^{ij} - N^i N^j / N^2        (2-index, free=[i,j])

    스칼라 entry는 ``ScalarMul(SymPy expr, Tensor("1", []))`` wrapping.
    """
    N_sym = sp.Symbol(adm.lapse_name)
    inv_N2 = 1 / N_sym ** 2

    Ni_up = adm.shift("i", "upper")
    Nj_up = adm.shift("j", "upper")
    h_inv = adm.spatial_metric_upper("i", "j")

    g_inv_tt = ScalarMul(-inv_N2, _ONE_SCALAR)
    g_inv_ti = ScalarMul(inv_N2, Ni_up)
    g_inv_ij = h_inv + ScalarMul(-inv_N2, TensorProduct(Ni_up, Nj_up))
    return {"tt": g_inv_tt, "ti": g_inv_ti, "ij": g_inv_ij}
