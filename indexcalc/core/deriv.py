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
    _resolve_einstein_pairs,
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
        """deriv_index와 inner free 인덱스를 모은 뒤 Einstein 자동 contraction.

        예: ∂_ρ Γ^ρ_νσ에서 deriv_index ρ↓가 inner의 ρ↑와 contract → free=[ν, σ].
        """
        return _resolve_einstein_pairs(
            [self.deriv_index] + self.expr.free_indices
        )

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

    def make_tensor(
        self, upper_name: str, lower1_name: str, lower2_name: str,
    ) -> Tensor:
        """Γ^a_{bc} Tensor — torsion-free symmetric (slots 1↔2)."""
        return Tensor(self.symbol, [
            Index(upper_name, self.space, "upper"),
            Index(lower1_name, self.deriv_space, "lower"),
            Index(lower2_name, self.space, "lower"),
        ], symmetric_pairs=[(1, 2)])

    def christoffel(
        self, upper_name: str, lower1_name: str, lower2_name: str,
    ) -> Tensor:
        """Convenience alias for ``make_tensor`` — explicit Γ tensor builder."""
        return self.make_tensor(upper_name, lower1_name, lower2_name)

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
        """deriv_index와 inner free 인덱스를 모은 뒤 Einstein 자동 contraction.

        예: ∇_ρ T^ρ_νσ에서 deriv_index ρ↓가 inner의 ρ↑와 contract → free=[ν, σ].
        """
        return _resolve_einstein_pairs(
            [self.deriv_index] + self.expr.free_indices
        )

    def __repr__(self) -> str:
        return f"∇_{self.deriv_index.name}({self.expr})"


# ─── G7: ∂ → ∇̄ + Γ̄·T forward conversion ───────────────────


def _next_dummy_for(base: str, taken: set[str]) -> str:
    """``base_1, base_2, …`` 중 ``taken``에 없는 첫 이름."""
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate not in taken:
            return candidate
        i += 1


def _replace_slot(T: Tensor, slot: int, new_idx: Index) -> Tensor:
    """T의 slot 위치 인덱스만 ``new_idx``로 바꾼 새 Tensor (속성 보존)."""
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


def _expand_partial_to_cov(
    T: Tensor, mu: Index, conn_map: dict[str, Connection],
) -> TensorExpr:
    """``∂_μ T = ∇̄_μ T - Σ_{upper s} Γ·T + Σ_{lower s} Γ·T`` explicit form.

    각 slot에 대해 connection이 있으면 보정 항을 누적. 없는 slot은 건너뜀
    (해당 인덱스 공간에 connection 미정의 → ∂ = ∇̄ 가정).
    """
    cov_cls = CovariantDeriv
    cov = cov_cls(T, mu, conn_map)

    correction_terms: list[TensorExpr] = []
    taken = {idx.name for idx in T.indices} | {mu.name}

    for slot, idx in enumerate(T.indices):
        space = idx.space
        if space.name not in conn_map:
            continue
        conn = conn_map[space.name]

        base = space.indices[0] if space.indices else "i"
        dummy = _next_dummy_for(base, taken)
        taken.add(dummy)

        if idx.position == "upper":
            # ∂_μ T^ρ = ∇̄_μ T^ρ - Γ^ρ_{μ α} T^α
            gamma = Tensor(conn.symbol, [
                Index(idx.name, space, "upper"),
                Index(mu.name, conn.deriv_space, "lower"),
                Index(dummy, space, "lower"),
            ])
            T_replaced = _replace_slot(T, slot, Index(dummy, space, "upper"))
            correction_terms.append(
                ScalarMul(-1, TensorProduct(gamma, T_replaced))
            )
        else:
            # ∂_μ T_ρ = ∇̄_μ T_ρ + Γ^α_{μ ρ} T_α
            gamma = Tensor(conn.symbol, [
                Index(dummy, space, "upper"),
                Index(mu.name, conn.deriv_space, "lower"),
                Index(idx.name, space, "lower"),
            ])
            T_replaced = _replace_slot(T, slot, Index(dummy, space, "lower"))
            correction_terms.append(TensorProduct(gamma, T_replaced))

    result: TensorExpr = cov
    for term in correction_terms:
        result = TensorSum(result, term)
    return result


def partial_to_covariant(
    expr: TensorExpr,
    conn_map: dict[str, Connection] | Connection,
    *,
    only_for: set[str] | None = None,
) -> TensorExpr:
    """``∂_μ T`` (T leaf Tensor) 패턴을 ``∇̄_μ T - Σ Γ̄·T``로 explicit 전개.

    .. math::
        \\partial_\\mu T^\\rho \\;=\\; \\nabla_\\mu T^\\rho \\;-\\; \\Gamma^\\rho{}_{\\mu\\alpha} T^\\alpha
        \\partial_\\mu T_\\rho \\;=\\; \\nabla_\\mu T_\\rho \\;+\\; \\Gamma^\\alpha{}_{\\mu\\rho} T_\\alpha

    각 slot마다 connection이 있으면 보정 항 1개씩.

    Parameters
    ----------
    expr : TensorExpr
    conn_map : dict[str, Connection] or Connection
        IndexSpace.name → connection. 단일 connection이면 자동으로 dict로 변환.
    only_for : set[str] or None
        지정 시 leaf Tensor.name이 이 집합 안에 있을 때만 변환. ``None``이면
        모든 leaf에 대해.

    Returns
    -------
    TensorExpr
        ``∂_μ T`` 패턴이 ``CovariantDeriv(T, μ) ± Γ·T`` 합으로 전환된 식.
        다른 노드(TensorProduct/Sum/ScalarMul/CovariantDeriv)는 재귀로 내려감.
    """
    if isinstance(conn_map, Connection):
        conn_map = {conn_map.space.name: conn_map}
    return _convert_partial(expr, conn_map, only_for)


def covariant_collapse(
    expr: TensorExpr,
    conn_map: dict[str, Connection] | Connection,
    *,
    only_for: set[str] | None = None,
    mreg=None,
) -> TensorExpr:
    """``∂_μ T + Σ Γ̄·T`` (보정 항) 묶음을 ``∇̄_μ T``로 collapse (G7 backward).

    ``partial_to_covariant``의 역 방향. TensorSum 안에서 다음 패턴을 찾아 묶는다:

    .. math::
        \\partial_\\mu T \\;+\\; \\sum_{\\text{upper s}} +\\Gamma\\cdot T
        \\;+\\; \\sum_{\\text{lower s}} -\\Gamma\\cdot T
        \\;=\\; \\nabla_\\mu T

    알고리즘:
        1. expr 트리에서 TensorSum 노드를 찾는다.
        2. summand들 중 ``PartialDeriv(T_leaf, μ_lo)`` 또는 ``±·PartialDeriv``
           (ScalarMul 외부 부호) 식별.
        3. ``simplify(expand_covariant(CovariantDeriv(T, μ, conn)), mreg)``로
           expected 보정 생성 (simplify가 Γ sym + dummy rename으로 self-trace
           cancellation을 정리).
        4. 외부 부호(scalar)에 따라 expected corrections에도 동일 scalar 곱.
        5. expected 보정 각 항을 sum 내 다른 summand와 dummy 이름 무관하게 매칭
           (``canonical_form_modulo_dummies`` + scalar 일치).
        6. 모두 매칭되면 그 group을 ``±·CovariantDeriv``로 교체.

    Parameters
    ----------
    expr : TensorExpr
    conn_map : dict[str, Connection] or Connection
    only_for : set[str] or None
        지정 시 이 leaf 텐서 이름들에 대해서만 collapse 시도.
    mreg : MetricRegistry or None
        제공 시 expected expansion에 ``simplify(..., mreg)``를 적용 — Γ symmetry
        등으로 self-cancelling 보정 항이 정리되어 매칭률 향상. 특히 self-traced
        ``T^μ_μν`` 같은 tensor에 ∇를 적용할 때 필수.

    Returns
    -------
    TensorExpr
        매칭 성공 항이 있으면 일부 ``∇̄`` 형태로 묶인 새 expr; 없으면 원본.
    """
    if isinstance(conn_map, Connection):
        conn_map = {conn_map.space.name: conn_map}
    # distribute_products로 ScalarMul(c, TensorSum) 를 평탄화 — 보정 항 매칭에 필수
    from indexcalc.core.simplify import distribute_products, pull_scalars
    expr = pull_scalars(distribute_products(expr))
    return _collapse_walk(expr, conn_map, only_for, mreg)


def _collapse_walk(expr, conn_map, only_for, mreg=None):
    if isinstance(expr, TensorSum):
        matched = _try_collapse_tensorsum(expr, conn_map, only_for, mreg)
        if matched is not None:
            return matched
        new_l = _collapse_walk(expr.left, conn_map, only_for, mreg)
        new_r = _collapse_walk(expr.right, conn_map, only_for, mreg)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, TensorProduct):
        new_l = _collapse_walk(expr.left, conn_map, only_for, mreg)
        new_r = _collapse_walk(expr.right, conn_map, only_for, mreg)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorProduct(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = _collapse_walk(expr.expr, conn_map, only_for, mreg)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    if isinstance(expr, PartialDeriv):
        new_inner = _collapse_walk(expr.expr, conn_map, only_for, mreg)
        if new_inner is not expr.expr:
            return PartialDeriv(new_inner, expr.deriv_index)
        return expr

    if isinstance(expr, CovariantDeriv):
        new_inner = _collapse_walk(expr.expr, conn_map, only_for, mreg)
        if new_inner is not expr.expr:
            return type(expr)(new_inner, expr.deriv_index, expr.connections)
        return expr

    return expr


def _try_collapse_tensorsum(expr, conn_map, only_for, mreg=None):
    """TensorSum 노드에서 ∂T + Γ corrections 패턴 매칭. 변경 시 새 expr, 아니면 None."""
    from indexcalc.core.simplify import (
        _flatten_sum, _split_scalar, canonical_form_modulo_dummies, simplify,
    )
    from indexcalc.core.variation import ZeroTensor

    summands = _flatten_sum(expr)
    consumed = [False] * len(summands)
    new_summands: list[TensorExpr] = []
    any_collapsed = False

    for i, s in enumerate(summands):
        if consumed[i]:
            continue
        partial_info = _identify_partial_leaf(s, only_for)
        if partial_info is None:
            new_summands.append(s)
            consumed[i] = True
            continue

        outer_scalar, T, mu = partial_info  # outer_scalar = ±1 or other

        # Build expected ∇̄_μ T explicit form, simplify로 self-cancelling 정리
        cov_node = CovariantDeriv(T, mu, conn_map)
        explicit = expand_covariant(cov_node)
        explicit = simplify(explicit, mreg)
        cov_terms = _flatten_sum(explicit)
        # 첫 항은 ∂T (또는 ScalarMul wrapping 후); 분리
        partial_idx = None
        for k, t in enumerate(cov_terms):
            sc, body = _split_scalar(t)
            if isinstance(body, PartialDeriv) and body.expr is T and sc == 1:
                partial_idx = k
                break
        if partial_idx is None:
            # ∂T가 simplified explicit에 없음 — collapse 불가
            new_summands.append(s)
            consumed[i] = True
            continue
        expected_corrections = [t for k, t in enumerate(cov_terms) if k != partial_idx]
        if not expected_corrections:
            new_summands.append(s)
            consumed[i] = True
            continue

        # outer_scalar 만큼 expected 보정에 곱.
        # (ex: original summand가 -∂T이면, -∇T = -∂T - corrections.
        #  매칭 대상 summand의 scalar는 -1 * expected의 scalar.)
        # 각 expected correction을 unconsumed summand와 매칭
        found_in: list[int] = []
        for ec in expected_corrections:
            ec_scalar, ec_body = _split_scalar(ec)
            target_scalar = outer_scalar * ec_scalar
            ec_canon = _safe_canon_modulo_dummies(ec_body)
            best_j = None
            for j in range(len(summands)):
                if consumed[j] or j == i or j in found_in:
                    continue
                ss = summands[j]
                ss_scalar, ss_body = _split_scalar(ss)
                if ss_scalar != target_scalar:
                    continue
                ss_canon = _safe_canon_modulo_dummies(ss_body)
                if ss_canon is None or ec_canon is None:
                    continue
                if ss_canon == ec_canon:
                    best_j = j
                    break
            if best_j is None:
                break
            found_in.append(best_j)

        if len(found_in) == len(expected_corrections):
            consumed[i] = True
            for j in found_in:
                consumed[j] = True
            replacement: TensorExpr = cov_node
            if outer_scalar != 1:
                replacement = ScalarMul(outer_scalar, cov_node)
            new_summands.append(replacement)
            any_collapsed = True
        else:
            new_summands.append(s)
            consumed[i] = True

    if not any_collapsed:
        return None

    if not new_summands:
        return ZeroTensor(expr.free_indices)
    if len(new_summands) == 1:
        return new_summands[0]
    result = new_summands[0]
    for s in new_summands[1:]:
        result = TensorSum(result, s)
    return result


def _identify_partial_leaf(s, only_for):
    """summand가 ``[ScalarMul(c, )] PartialDeriv(Tensor_leaf, μ_lo)``이면
    ``(c, T, μ_lo)`` 반환 (c는 기본 1 또는 scalar). 아니면 None."""
    scalar = 1
    body = s
    if isinstance(body, ScalarMul):
        scalar = body.scalar
        body = body.expr
    if isinstance(body, PartialDeriv) and isinstance(body.expr, Tensor):
        T = body.expr
        if only_for is None or T.name in only_for:
            return scalar, T, body.deriv_index
    return None


def _safe_canon_modulo_dummies(expr):
    """``canonical_form_modulo_dummies`` 호출 — 실패 시 None."""
    from indexcalc.core.simplify import canonical_form_modulo_dummies
    try:
        return canonical_form_modulo_dummies(expr)
    except Exception:
        return None


def _convert_partial(expr, conn_map, only_for):
    if isinstance(expr, PartialDeriv):
        new_inner = _convert_partial(expr.expr, conn_map, only_for)
        if isinstance(new_inner, Tensor):
            if only_for is None or new_inner.name in only_for:
                return _expand_partial_to_cov(new_inner, expr.deriv_index, conn_map)
        if new_inner is not expr.expr:
            return PartialDeriv(new_inner, expr.deriv_index)
        return expr

    if isinstance(expr, CovariantDeriv):
        new_inner = _convert_partial(expr.expr, conn_map, only_for)
        if new_inner is not expr.expr:
            return type(expr)(new_inner, expr.deriv_index, expr.connections)
        return expr

    if isinstance(expr, TensorProduct):
        l = _convert_partial(expr.left, conn_map, only_for)
        r = _convert_partial(expr.right, conn_map, only_for)
        if l is not expr.left or r is not expr.right:
            return TensorProduct(l, r)
        return expr

    if isinstance(expr, TensorSum):
        l = _convert_partial(expr.left, conn_map, only_for)
        r = _convert_partial(expr.right, conn_map, only_for)
        if l is not expr.left or r is not expr.right:
            return TensorSum(l, r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = _convert_partial(expr.expr, conn_map, only_for)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    return expr


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
            t_replaced = Tensor(
                inner.name, new_indices,
                antisymmetric_pairs=list(inner.antisymmetric_pairs),
                symmetric_pairs=list(inner.symmetric_pairs),
                traceless=list(inner.traceless),
                transverse=list(inner.transverse),
                reps=dict(inner.reps),
                statistics=inner.statistics,
            )

            result = result + TensorProduct(gamma, t_replaced)

        else:  # lower
            # - Γ^{dummy}_{d_idx.name, idx.name} * T^{..._dummy_lower...}
            # Γ has dummy as upper(1st slot), T needs dummy as lower to contract
            gamma = conn.make_tensor(dummy, d_idx.name, idx.name)

            new_indices = list(inner.indices)
            new_indices[slot] = Index(dummy, idx.space, "lower")
            t_replaced = Tensor(
                inner.name, new_indices,
                antisymmetric_pairs=list(inner.antisymmetric_pairs),
                symmetric_pairs=list(inner.symmetric_pairs),
                traceless=list(inner.traceless),
                transverse=list(inner.transverse),
                reps=dict(inner.reps),
                statistics=inner.statistics,
            )

            result = result - TensorProduct(gamma, t_replaced)

    return result
