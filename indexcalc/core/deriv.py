"""
미분 연산자: PartialDeriv, Connection, CovariantDeriv.

∂_μ T^ν_λ 같은 편미분과 ∇_μ T^ν_λ 같은 공변미분을 symbolic하게 표현한다.
Connection은 각 IndexSpace에 대한 연결(Christoffel, spin connection 등)을 정의하고,
CovariantDeriv는 텐서의 각 인덱스 공간에 맞는 connection을 자동으로 적용한다.
"""

from __future__ import annotations
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)


class PartialDeriv(TensorExpr):
    """편미분 연산자 ∂_μ를 텐서 표현식에 적용한 결과.

    ∂_μ T^ν_λ 는 free indices [_μ, ^ν, _λ]를 가진다.
    즉, 미분 인덱스 μ가 lower index로 추가된다.

    주의: ∂_μ T^ν_λ는 텐서가 아니다 (좌표변환 시 inhomogeneous term).
    하지만 covariant derivative의 빌딩 블록으로서 symbolic 표현이 필요하다.

    Parameters
    ----------
    expr : TensorExpr
        미분 대상 표현식.
    deriv_index : Index
        미분 인덱스. 반드시 lower position이어야 한다 (∂_μ).

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> V = Tensor("V", [st.upper("ν")])
    >>> dV = PartialDeriv(V, st.lower("μ"))
    >>> dV.free_indices
    [_μ, ^ν]
    """

    def __init__(self, expr: TensorExpr, deriv_index: Index):
        if deriv_index.position != "lower":
            raise ValueError(
                f"Partial derivative index must be lower (covariant), "
                f"got '{deriv_index.position}' for {deriv_index}"
            )
        self.expr = expr
        self.deriv_index = deriv_index

    @property
    def free_indices(self) -> list[Index]:
        return [self.deriv_index] + self.expr.free_indices

    def __repr__(self) -> str:
        return f"∂_{self.deriv_index.name}({self.expr})"


def partial(expr: TensorExpr, index: Index) -> PartialDeriv:
    """편미분 ∂_index(expr)을 생성한다.

    index가 upper로 주어지면 자동으로 lower로 변환한다.

    Parameters
    ----------
    expr : TensorExpr
        미분 대상.
    index : Index
        미분 인덱스.

    Returns
    -------
    PartialDeriv
    """
    if index.position == "upper":
        index = index.flip()
    return PartialDeriv(expr, index)


# ─── Leibniz rule (곱의 미분) ────────────────────────────────

def expand_partial(expr: TensorExpr) -> TensorExpr:
    """PartialDeriv를 Leibniz rule로 전개한다.

    ∂_μ (A * B) → (∂_μ A) * B + A * (∂_μ B)
    ∂_μ (c * A) → c * ∂_μ A
    ∂_μ (A + B) → ∂_μ A + ∂_μ B

    재귀적으로 적용하여 PartialDeriv가 Tensor(잎 노드)에만 남도록 한다.

    Parameters
    ----------
    expr : TensorExpr
        전개할 표현식.

    Returns
    -------
    TensorExpr
        Leibniz rule이 적용된 표현식.
    """
    if not isinstance(expr, PartialDeriv):
        # PartialDeriv가 아닌 노드는 하위를 재귀 탐색
        if isinstance(expr, TensorProduct):
            return TensorProduct(
                expand_partial(expr.left),
                expand_partial(expr.right),
            )
        if isinstance(expr, TensorSum):
            return TensorSum(
                expand_partial(expr.left),
                expand_partial(expr.right),
            )
        if isinstance(expr, ScalarMul):
            return ScalarMul(expr.scalar, expand_partial(expr.expr))
        return expr

    # PartialDeriv 노드: 내부 표현식 구조에 따라 전개
    d_idx = expr.deriv_index
    inner = expr.expr

    # ∂_μ (Tensor) → 그대로 (잎 노드에 도달)
    if isinstance(inner, Tensor):
        return expr

    # ∂_μ (A * B) → (∂_μ A) * B + A * (∂_μ B)
    if isinstance(inner, TensorProduct):
        dA_B = TensorProduct(PartialDeriv(inner.left, d_idx), inner.right)
        A_dB = TensorProduct(inner.left, PartialDeriv(inner.right, d_idx))
        result = TensorSum(dA_B, A_dB)
        return expand_partial(result)

    # ∂_μ (A + B) → ∂_μ A + ∂_μ B
    if isinstance(inner, TensorSum):
        result = TensorSum(
            PartialDeriv(inner.left, d_idx),
            PartialDeriv(inner.right, d_idx),
        )
        return expand_partial(result)

    # ∂_μ (c * A) → c * ∂_μ A
    if isinstance(inner, ScalarMul):
        result = ScalarMul(inner.scalar, PartialDeriv(inner.expr, d_idx))
        return expand_partial(result)

    # ∂_μ (∂_ν A) → ∂_μ ∂_ν A (중첩 미분 — 그대로 유지)
    if isinstance(inner, PartialDeriv):
        return PartialDeriv(expand_partial(inner), d_idx)

    return expr


# ─── Connection ──────────────────────────────────────────────

class Connection:
    """텐서 인덱스 공간에 대한 연결(connection).

    Connection은 공변미분에서 ∂ 외에 추가되는 보정 항을 정의한다.
    각 IndexSpace마다 다른 connection이 적용될 수 있다.

    Parameters
    ----------
    symbol : str
        Connection 기호 (e.g., "Γ", "ω", "A").
    space : IndexSpace
        이 connection이 작용하는 인덱스 공간.
    deriv_space : IndexSpace or None
        미분 인덱스의 공간. None이면 space와 동일.

    Examples
    --------
    >>> christoffel = Connection("Γ", spacetime)
    >>> spin_conn = Connection("ω", lorentz, deriv_space=spacetime)
    """

    def __init__(
        self,
        symbol: str,
        space: IndexSpace,
        deriv_space: IndexSpace | None = None,
    ):
        self.symbol = symbol
        self.space = space
        self.deriv_space = deriv_space or space

    def make_tensor(self, upper_name: str, lower1_name: str, lower2_name: str) -> Tensor:
        """Γ^{upper}_{lower1, lower2} 텐서를 생성한다.

        Parameters
        ----------
        upper_name : str
            Upper index 이름 (connection이 작용하는 space).
        lower1_name : str
            첫 번째 lower index 이름 (미분 인덱스, deriv_space).
        lower2_name : str
            두 번째 lower index 이름 (connection이 작용하는 space).

        Returns
        -------
        Tensor
            Γ^a_{bc} 형태의 텐서.
        """
        return Tensor(self.symbol, [
            Index(upper_name, self.space, "upper"),
            Index(lower1_name, self.deriv_space, "lower"),
            Index(lower2_name, self.space, "lower"),
        ])

    def __repr__(self) -> str:
        return f"Connection({self.symbol!r}, {self.space.name})"


class LeviCivitaConnection(Connection):
    """Levi-Civita connection (torsion-free, metric-compatible).

    Γ^μ_νλ = ½ g^{μρ}(∂_ν g_{ρλ} + ∂_λ g_{ρν} - ∂_ρ g_{νλ})

    Parameters
    ----------
    metric : Tensor
        Metric 텐서 (e.g., g_{μν}).
    inverse : Tensor
        Inverse metric 텐서 (e.g., g^{μν}).
    space : IndexSpace
        이 connection의 인덱스 공간.
    """

    def __init__(self, metric: Tensor, inverse: Tensor, space: IndexSpace):
        super().__init__("Γ", space)
        self.metric = metric
        self.inverse = inverse

    def definition(self) -> TensorExpr:
        """Christoffel symbol의 symbolic 정의를 반환한다.

        ½ g^{μρ}(∂_ν g_{ρλ} + ∂_λ g_{ρν} - ∂_ρ g_{νλ})

        인덱스 이름: μ(upper), ν(lower1), λ(lower2), ρ(dummy).
        이 공간의 indices 문자열에서 가져온다.

        Returns
        -------
        TensorExpr
            Christoffel symbol 정의 표현식.
        """
        chars = self.space.indices
        if len(chars) < 4:
            raise ValueError(
                f"Space '{self.space.name}' needs at least 4 index characters "
                f"for Christoffel definition, has {len(chars)}"
            )

        mu, nu, lam, rho = chars[0], chars[1], chars[2], chars[3]

        g_inv = Tensor(self.inverse.name, [
            Index(mu, self.space, "upper"),
            Index(rho, self.space, "upper"),
        ])

        g_rl = Tensor(self.metric.name, [
            Index(rho, self.space, "lower"),
            Index(lam, self.space, "lower"),
        ])
        g_rn = Tensor(self.metric.name, [
            Index(rho, self.space, "lower"),
            Index(nu, self.space, "lower"),
        ])
        g_nl = Tensor(self.metric.name, [
            Index(nu, self.space, "lower"),
            Index(lam, self.space, "lower"),
        ])

        nu_idx = Index(nu, self.space, "lower")
        lam_idx = Index(lam, self.space, "lower")
        rho_idx = Index(rho, self.space, "lower")

        bracket = (
            PartialDeriv(g_rl, nu_idx)
            + PartialDeriv(g_rn, lam_idx)
            - PartialDeriv(g_nl, rho_idx)
        )

        return ScalarMul(0.5, TensorProduct(g_inv, bracket))

    def __repr__(self) -> str:
        return f"LeviCivitaConnection({self.space.name}, metric={self.metric.name})"


# ─── CovariantDeriv ──────────────────────────────────────────

class CovariantDeriv(TensorExpr):
    """공변미분 ∇_μ T^..._{...}.

    각 인덱스의 공간에 맞는 connection을 자동으로 적용한다.

    Parameters
    ----------
    expr : TensorExpr
        미분 대상 표현식.
    deriv_index : Index
        미분 인덱스. 반드시 lower.
    connections : dict[str, Connection] or Connection
        IndexSpace.name → Connection 매핑.
        Connection 하나만 주면 해당 space에 대해서만 적용.

    Examples
    --------
    >>> nabla_V = CovariantDeriv(V, mu, {spacetime.name: christoffel})
    """

    def __init__(
        self,
        expr: TensorExpr,
        deriv_index: Index,
        connections: dict[str, Connection] | Connection,
    ):
        if deriv_index.position != "lower":
            raise ValueError(
                f"Covariant derivative index must be lower, "
                f"got '{deriv_index.position}'"
            )

        self.expr = expr
        self.deriv_index = deriv_index

        if isinstance(connections, Connection):
            self.connections = {connections.space.name: connections}
        else:
            self.connections = connections

    @property
    def free_indices(self) -> list[Index]:
        return [self.deriv_index] + self.expr.free_indices

    def __repr__(self) -> str:
        return f"∇_{self.deriv_index.name}({self.expr})"


def covariant(
    expr: TensorExpr,
    index: Index,
    connections: dict[str, Connection] | Connection,
) -> CovariantDeriv:
    """공변미분 ∇_index(expr)을 생성한다.

    index가 upper이면 자동으로 lower로 변환.

    Parameters
    ----------
    expr : TensorExpr
        미분 대상.
    index : Index
        미분 인덱스.
    connections : dict or Connection
        Connection 매핑.

    Returns
    -------
    CovariantDeriv
    """
    if index.position == "upper":
        index = index.flip()
    return CovariantDeriv(expr, index, connections)


# ─── CovariantDeriv 전개 ─────────────────────────────────────

def _collect_existing_names_deriv(expr: TensorExpr) -> set[str]:
    """표현식에서 사용 중인 모든 인덱스 이름을 수집한다."""
    names = set()

    def _walk(e):
        if isinstance(e, Tensor):
            for idx in e.indices:
                names.add(idx.name)
        elif isinstance(e, PartialDeriv):
            names.add(e.deriv_index.name)
            _walk(e.expr)
        elif isinstance(e, CovariantDeriv):
            names.add(e.deriv_index.name)
            _walk(e.expr)
        elif isinstance(e, TensorProduct):
            _walk(e.left)
            _walk(e.right)
        elif isinstance(e, TensorSum):
            _walk(e.left)
            _walk(e.right)
        elif isinstance(e, ScalarMul):
            _walk(e.expr)

    _walk(expr)
    return names


def _make_dummy(space: IndexSpace, existing: set[str]) -> str:
    """충돌 없는 dummy index 이름을 생성한다.

    Uses the first grapheme (base char + trailing combining marks) of
    ``space.indices`` so decorated index alphabets like "p̄q̄r̄s̄" resolve
    correctly instead of slicing off the combining mark.
    """
    if space.indices:
        base = space.indices[0]
        i = 1
        while i < len(space.indices) and 0x0300 <= ord(space.indices[i]) <= 0x036F:
            base += space.indices[i]
            i += 1
    else:
        base = "i"
    counter = 1
    while f"{base}_{counter}" in existing:
        counter += 1
    name = f"{base}_{counter}"
    existing.add(name)
    return name


def expand_covariant(expr: TensorExpr) -> TensorExpr:
    """CovariantDeriv를 ∂ + Γ 항으로 전개한다.

    ∇_μ T^ν_λ = ∂_μ T^ν_λ + Γ^ν_{μρ} T^ρ_λ - Γ^ρ_{μλ} T^ν_ρ

    규칙:
      - upper index ^a (space S): + Γ^a_{μ, dummy} T^{...dummy...}
      - lower index _a (space S): - Γ^{dummy}_{μ, a} T^{..._dummy...}
      - connection이 없는 space의 인덱스: 보정 없음

    Parameters
    ----------
    expr : TensorExpr
        전개할 표현식. CovariantDeriv 노드를 찾아서 전개.

    Returns
    -------
    TensorExpr
        ∂ + Γ 항으로 전개된 표현식.
    """
    if isinstance(expr, CovariantDeriv):
        return _expand_single_covariant(expr)

    if isinstance(expr, TensorProduct):
        return TensorProduct(
            expand_covariant(expr.left),
            expand_covariant(expr.right),
        )
    if isinstance(expr, TensorSum):
        return TensorSum(
            expand_covariant(expr.left),
            expand_covariant(expr.right),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(expr.scalar, expand_covariant(expr.expr))

    return expr


def _expand_single_covariant(cov: CovariantDeriv) -> TensorExpr:
    """단일 CovariantDeriv를 전개한다."""
    inner = cov.expr
    d_idx = cov.deriv_index
    connections = cov.connections

    # 내부가 Tensor가 아닌 복합 표현식이면, 먼저 Leibniz 적용
    # ∇_μ (A * B) = (∇_μ A) * B + A * (∇_μ B)
    # type(cov)로 재귀해야 SpatialCovariantDeriv 등 subclass가 유지됨
    cov_cls = type(cov)
    if isinstance(inner, TensorProduct):
        left_cov = cov_cls(inner.left, d_idx, connections)
        right_cov = cov_cls(inner.right, d_idx, connections)
        dA_B = TensorProduct(left_cov, inner.right)
        A_dB = TensorProduct(inner.left, right_cov)
        return expand_covariant(TensorSum(dA_B, A_dB))

    if isinstance(inner, TensorSum):
        return expand_covariant(TensorSum(
            cov_cls(inner.left, d_idx, connections),
            cov_cls(inner.right, d_idx, connections),
        ))

    if isinstance(inner, ScalarMul):
        return ScalarMul(
            inner.scalar,
            expand_covariant(cov_cls(inner.expr, d_idx, connections)),
        )

    # 내부가 Tensor(잎 노드): ∂ + Γ 항 생성
    if not isinstance(inner, Tensor):
        # PartialDeriv 등 다른 노드 — 일단 ∂만 적용
        return PartialDeriv(inner, d_idx)

    existing = _collect_existing_names_deriv(cov)
    result: TensorExpr = PartialDeriv(inner, d_idx)

    for slot, idx in enumerate(inner.indices):
        space_name = idx.space.name
        if space_name not in connections:
            continue

        conn = connections[space_name]
        dummy = _make_dummy(idx.space, existing)

        if idx.position == "upper":
            # + Γ^{idx.name}_{d_idx.name, dummy} * T^{...dummy_upper...}
            # Γ has dummy as lower(3rd slot), T needs dummy as upper to contract
            gamma = conn.make_tensor(idx.name, d_idx.name, dummy)

            new_indices = list(inner.indices)
            new_indices[slot] = Index(dummy, idx.space, "upper")
            t_replaced = Tensor(inner.name, new_indices)

            result = result + TensorProduct(gamma, t_replaced)

        else:  # lower
            # - Γ^{dummy}_{d_idx.name, idx.name} * T^{..._dummy_lower...}
            # Γ has dummy as upper(1st slot), T needs dummy as lower to contract
            gamma = conn.make_tensor(dummy, d_idx.name, idx.name)

            new_indices = list(inner.indices)
            new_indices[slot] = Index(dummy, idx.space, "lower")
            t_replaced = Tensor(inner.name, new_indices)

            result = result - TensorProduct(gamma, t_replaced)

    return result
