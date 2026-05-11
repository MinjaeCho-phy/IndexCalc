"""Vielbein identity collapse + Backend 2 (vielbein/spin connection setup).

Vielbein (frame field, tetrad) ``e``는 frame index와 spacetime index를 잇는다:
    - lower: ``g_{μν} = e^a{}_μ η_{ab} e^b{}_ν``
    - upper: ``g^{μν} = e_a{}^μ η^{ab} e_b{}^ν``

이 모듈이 제공하는 것:
    - ``VielbeinRegistry`` + ``collapse_vielbein_identity``: e × η × e 패턴 인식·치환.
    - ``VielbeinSetup``: 이름·공간 묶음 + leaf 빌더 + spin connection.
    - ``SpinConnection``: frame space 위에 작용하는 connection (Connection subclass).
    - 항등식 빌더:
        - ``vielbein_compatibility_lhs(setup, christoffel)``: ∇_μ e^a_ν = 0의 LHS.
        - ``spin_connection_from_vielbein(setup, christoffel)``: ω_μ^{ab} = e^{aν} ∇_μ e^b_ν.

DFT/string-frame처럼 frame_space가 별도(2D, doubled)이거나 frame metric이
표준 Minkowski가 아니어도, register/setup만 하면 동일 메커니즘 동작.
"""

from __future__ import annotations

from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import Connection, CovariantDeriv
from indexcalc.core.simplify import collect_factors, _index_name_count


class VielbeinRegistry:
    """Vielbein identity 등록.

    각 entry는 ``(vielbein_name, frame_metric_name, spacetime_metric_name,
    frame_space, st_space)`` 튜플. 등록된 entry에 대해
    ``collapse_vielbein_identity``가 매칭 패턴을 찾는다.

    Examples
    --------
    >>> vbreg = VielbeinRegistry()
    >>> vbreg.register("e", "η", "g", frame, st)
    """

    def __init__(self):
        self._entries: list[tuple] = []

    def register(
        self,
        vielbein_name: str,
        frame_metric_name: str,
        spacetime_metric_name: str,
        frame_space: IndexSpace,
        st_space: IndexSpace,
    ) -> None:
        self._entries.append((
            vielbein_name, frame_metric_name, spacetime_metric_name,
            frame_space, st_space,
        ))

    def entries(self) -> list[tuple]:
        return list(self._entries)


def _product_of(factors: list[TensorExpr]) -> TensorExpr:
    """factor 리스트를 left-associated TensorProduct로 묶는다."""
    if not factors:
        raise ValueError("empty factor list")
    result = factors[0]
    for f in factors[1:]:
        result = TensorProduct(result, f)
    return result


def _classify_vielbein(
    t: Tensor, frame_sp: IndexSpace, st_sp: IndexSpace,
) -> tuple[Index, Index] | None:
    """Tensor t가 vielbein 형상 (frame index 1개 + st index 1개)이면 (frame_idx, st_idx) 반환."""
    if len(t.indices) != 2:
        return None
    frame_idx = next((i for i in t.indices if i.space == frame_sp), None)
    st_idx = next((i for i in t.indices if i.space == st_sp), None)
    if frame_idx is None or st_idx is None:
        return None
    return frame_idx, st_idx


def _is_frame_metric(
    t: Tensor, eta_name: str, frame_sp: IndexSpace,
) -> bool:
    return (
        t.name == eta_name
        and len(t.indices) == 2
        and all(i.space == frame_sp for i in t.indices)
    )


def collapse_vielbein_identity(
    expr: TensorExpr, vbreg: VielbeinRegistry,
) -> TensorExpr:
    """``e × η × e`` 세 factor 패턴을 spacetime metric으로 collapse.

    Walk: TensorSum, ScalarMul, CovariantDeriv 등은 재귀로 내려가고, 실제 패턴
    매칭은 TensorProduct 노드의 평탄화된 factor 리스트 위에서.

    Position semantics:
        - Lower: e^a{}_μ × η_{ab} × e^b{}_ν → g_{μν}.
          e의 frame index는 upper, η의 frame indices는 lower → contract.
        - Upper: e_a{}^μ × η^{ab} × e_b{}^ν → g^{μν}.
          e의 frame index는 lower, η는 upper → contract.

    Returns
    -------
    TensorExpr
        하나 이상의 패턴이 잡혀 collapse되면 새 expr; 없으면 원본 그대로.
    """
    from indexcalc.core.deriv import PartialDeriv, CovariantDeriv

    if isinstance(expr, TensorSum):
        new_l = collapse_vielbein_identity(expr.left, vbreg)
        new_r = collapse_vielbein_identity(expr.right, vbreg)
        if new_l is not expr.left or new_r is not expr.right:
            return TensorSum(new_l, new_r)
        return expr

    if isinstance(expr, ScalarMul):
        new_inner = collapse_vielbein_identity(expr.expr, vbreg)
        if new_inner is not expr.expr:
            return ScalarMul(expr.scalar, new_inner)
        return expr

    if isinstance(expr, PartialDeriv):
        new_inner = collapse_vielbein_identity(expr.expr, vbreg)
        if new_inner is not expr.expr:
            return PartialDeriv(new_inner, expr.deriv_index)
        return expr

    if isinstance(expr, CovariantDeriv):
        new_inner = collapse_vielbein_identity(expr.expr, vbreg)
        if new_inner is not expr.expr:
            return type(expr)(new_inner, expr.deriv_index, expr.connections)
        return expr

    if not isinstance(expr, TensorProduct):
        return expr

    # Flatten and try to find a triple
    factors = collect_factors(expr)
    name_counts = _index_name_count(factors)
    free_names = {idx.name for idx in expr.free_indices}

    for entry in vbreg.entries():
        e_name, eta_name, g_name, frame_sp, st_sp = entry

        e_indices_in_factors = [
            (k, f, _classify_vielbein(f, frame_sp, st_sp))
            for k, f in enumerate(factors)
            if isinstance(f, Tensor) and f.name == e_name
        ]
        e_indices_in_factors = [
            (k, f, c) for (k, f, c) in e_indices_in_factors if c is not None
        ]
        eta_indices_in_factors = [
            (k, f) for k, f in enumerate(factors)
            if isinstance(f, Tensor) and _is_frame_metric(f, eta_name, frame_sp)
        ]

        for ei1, e1, (e1_frame, e1_st) in e_indices_in_factors:
            for ei2, e2, (e2_frame, e2_st) in e_indices_in_factors:
                if ei1 == ei2:
                    continue
                if e1_frame.name == e2_frame.name:
                    continue  # 같은 dummy 이름이면 self-contraction; 별개 패턴

                # 두 e의 frame name이 dummy인지 (count == 2, 그리고 free 아님)
                if (
                    name_counts.get(e1_frame.name, 0) != 2
                    or name_counts.get(e2_frame.name, 0) != 2
                ):
                    continue
                if e1_frame.name in free_names or e2_frame.name in free_names:
                    continue

                for eti, eta in eta_indices_in_factors:
                    if eti in (ei1, ei2):
                        continue
                    eta_names = [i.name for i in eta.indices]
                    if e1_frame.name not in eta_names or e2_frame.name not in eta_names:
                        continue
                    eta_for_e1 = next(
                        i for i in eta.indices if i.name == e1_frame.name
                    )
                    eta_for_e2 = next(
                        i for i in eta.indices if i.name == e2_frame.name
                    )
                    # Contraction: positions opposite
                    if eta_for_e1.position == e1_frame.position:
                        continue
                    if eta_for_e2.position == e2_frame.position:
                        continue

                    # Build replacement g
                    # spacetime metric position = e의 spacetime index와 동일
                    # (lower 버전: e의 st_idx가 lower → g lower)
                    g = Tensor(
                        g_name,
                        [e1_st, e2_st],
                        symmetric_pairs=[(0, 1)],
                    )

                    new_factors = [
                        f for k, f in enumerate(factors)
                        if k not in (ei1, ei2, eti)
                    ]
                    new_factors.append(g)

                    if len(new_factors) == 1:
                        result = new_factors[0]
                    else:
                        result = _product_of(new_factors)
                    # 재귀: 추가 패턴 collapse
                    return collapse_vielbein_identity(result, vbreg)

    return expr


# ─── B2: SpinConnection + VielbeinSetup ─────────────────────


class SpinConnection(Connection):
    """Spin connection ω_μ^{a}{}_b on frame space.

    ``Connection``의 thin subclass — type 식별만 (필요 시 display/분기).
    ``make_tensor(a, μ, b)``는 ``ω^a{}_{μ b}`` (frame upper, st lower, frame
    lower) Tensor를 반환 (Connection.make_tensor 그대로).
    """

    def __repr__(self) -> str:
        return f"SpinConnection({self.symbol!r}, {self.space.name}, deriv_space={self.deriv_space.name})"


class VielbeinSetup:
    """Vielbein + frame metric + spacetime metric + spin connection 묶음.

    Parameters
    ----------
    st : IndexSpace
        Spacetime index space (μ, ν, ...).
    fr : IndexSpace
        Frame index space (a, b, ...).
    vielbein_name, frame_metric_name, spacetime_metric_name, spin_connection_name : str
        Builder가 만들 텐서·connection의 이름. 기본 'e', 'η', 'g', 'ω'.
    """

    def __init__(
        self,
        st: IndexSpace,
        fr: IndexSpace,
        *,
        vielbein_name: str = "e",
        frame_metric_name: str = "η",
        spacetime_metric_name: str = "g",
        spin_connection_name: str = "ω",
    ):
        self.st = st
        self.fr = fr
        self.vielbein_name = vielbein_name
        self.frame_metric_name = frame_metric_name
        self.spacetime_metric_name = spacetime_metric_name
        self.spin_connection_name = spin_connection_name

    # ─── Vielbein leaf builders ────────────────────────────

    def vielbein(self, a: str = "a", μ: str = "μ") -> Tensor:
        """``e^a{}_μ`` — frame upper, spacetime lower."""
        return Tensor(self.vielbein_name, [
            Index(a, self.fr, "upper"),
            Index(μ, self.st, "lower"),
        ])

    def vielbein_inverse(self, a: str = "a", μ: str = "μ") -> Tensor:
        """``e_a{}^μ`` — frame lower, spacetime upper (inverse vielbein)."""
        return Tensor(self.vielbein_name, [
            Index(a, self.fr, "lower"),
            Index(μ, self.st, "upper"),
        ])

    def vielbein_aμ_upper(self, a: str = "a", μ: str = "μ") -> Tensor:
        """``e^{aμ}`` — both upper (raised by η^ab and g^μν)."""
        return Tensor(self.vielbein_name, [
            Index(a, self.fr, "upper"),
            Index(μ, self.st, "upper"),
        ])

    def vielbein_aμ_lower(self, a: str = "a", μ: str = "μ") -> Tensor:
        """``e_{aμ}`` — both lower (lowered by η_ab and g_μν)."""
        return Tensor(self.vielbein_name, [
            Index(a, self.fr, "lower"),
            Index(μ, self.st, "lower"),
        ])

    # ─── Frame metric builders ─────────────────────────────

    def frame_metric_lower(self, a: str = "a", b: str = "b") -> Tensor:
        return Tensor(self.frame_metric_name, [
            Index(a, self.fr, "lower"),
            Index(b, self.fr, "lower"),
        ], symmetric_pairs=[(0, 1)])

    def frame_metric_upper(self, a: str = "a", b: str = "b") -> Tensor:
        return Tensor(self.frame_metric_name, [
            Index(a, self.fr, "upper"),
            Index(b, self.fr, "upper"),
        ], symmetric_pairs=[(0, 1)])

    # ─── Spacetime metric builders ─────────────────────────

    def spacetime_metric_lower(self, μ: str = "μ", ν: str = "ν") -> Tensor:
        return Tensor(self.spacetime_metric_name, [
            Index(μ, self.st, "lower"),
            Index(ν, self.st, "lower"),
        ], symmetric_pairs=[(0, 1)])

    def spacetime_metric_upper(self, μ: str = "μ", ν: str = "ν") -> Tensor:
        return Tensor(self.spacetime_metric_name, [
            Index(μ, self.st, "upper"),
            Index(ν, self.st, "upper"),
        ], symmetric_pairs=[(0, 1)])

    # ─── Spin connection ───────────────────────────────────

    def spin_connection(self) -> SpinConnection:
        """SpinConnection acting on frame space, deriv_space=spacetime."""
        return SpinConnection(self.spin_connection_name, self.fr, deriv_space=self.st)

    # ─── Conversion to flat registry ───────────────────────

    def to_registry(self) -> VielbeinRegistry:
        """``collapse_vielbein_identity``에 쓸 수 있는 ``VielbeinRegistry``."""
        r = VielbeinRegistry()
        r.register(
            self.vielbein_name, self.frame_metric_name,
            self.spacetime_metric_name, self.fr, self.st,
        )
        return r


# ─── B2: Identity builders ───────────────────────────────


def vielbein_compatibility_lhs(
    setup: VielbeinSetup,
    christoffel: Connection,
    *,
    μ: str = "μ", a: str = "a", ν: str = "ν",
) -> CovariantDeriv:
    """Vielbein compatibility 조건 ``∇_μ e^a{}_ν = 0`` 의 LHS (compact).

    ``CovariantDeriv`` 노드를 반환 — ``expand_covariant``로 explicit 형태:

    .. math::
        \\nabla_\\mu e^a{}_\\nu \\;=\\; \\partial_\\mu e^a{}_\\nu
            + \\omega_\\mu{}^a{}_b\\, e^b{}_\\nu
            - \\Gamma^\\rho{}_{\\mu\\nu}\\, e^a{}_\\rho

    Parameters
    ----------
    setup : VielbeinSetup
    christoffel : Connection
        Spacetime Christoffel connection.
    μ, a, ν : str
        Free index 이름.
    """
    e = setup.vielbein(a, ν)
    spin = setup.spin_connection()
    μ_lo = Index(μ, setup.st, "lower")
    return CovariantDeriv(e, μ_lo, {
        setup.st.name: christoffel,
        setup.fr.name: spin,
    })


def spin_connection_from_vielbein(
    setup: VielbeinSetup,
    christoffel: Connection,
    *,
    μ: str = "μ", a: str = "a", b: str = "b",
) -> TensorExpr:
    """Spin connection 정의식 ``ω_μ{}^{ab} = e^{aν} \\nabla_μ e^b{}_ν``.

    ``∇_μ`` 안쪽은 spacetime Christoffel만 사용 (∂_μ e^b_ν - Γ^ρ_{μν} e^b_ρ).

    Returns
    -------
    TensorExpr
        ``CovariantDeriv``를 포함한 compact form. ``expand_covariant`` 호출 시:
        ``e^{aν} ∂_μ e^b_ν - e^{aν} Γ^ρ_{μν} e^b_ρ``.
    """
    ν = "ν"
    if ν in (μ, a, b):
        ν = "σ"
    e_aν_up = setup.vielbein_aμ_upper(a, ν)
    e_b_lo = setup.vielbein(b, ν)
    μ_lo = Index(μ, setup.st, "lower")
    cov = CovariantDeriv(e_b_lo, μ_lo, {setup.st.name: christoffel})
    return TensorProduct(e_aν_up, cov)
