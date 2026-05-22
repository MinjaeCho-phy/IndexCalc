"""
Generator: Lie group의 infinitesimal 변환 작용을 표현한다.

Generator는 그룹 + rep별 작용 규칙(Tensor → TensorExpr 함수)으로 구성된다.
``apply_to(field)``는 field의 ``reps`` attribute를 lookup해서 적절한 작용을 적용한다.
field가 그룹의 rep tag를 갖지 않으면 그 그룹의 singlet으로 간주, ZeroTensor 반환.

본 모듈은 *generator만* 다룬다 — 트리 전체에 적용하는 substitution walk는
``core/substitution.py``를 참조하라.

Examples
--------
>>> from indexcalc.core.group import Group
>>> from indexcalc.core.tensor import Tensor
>>> from indexcalc.core.generator import Generator, u1_action
>>>
>>> u1 = Group("U(1)", abelian=True)
>>> u1.add_rep("+1", dim=1, charge=1.0)
>>> u1.add_rep("-1", dim=1, charge=-1.0)
>>>
>>> T_u1 = Generator("T_U(1)", u1)
>>> T_u1.declare_action("+1", u1_action(u1.get_rep("+1")))
>>> T_u1.declare_action("-1", u1_action(u1.get_rep("-1")))
>>>
>>> phi = Tensor("phi", [], reps={"U(1)": "+1"})
>>> T_u1.apply_to(phi)
1j * phi
"""

from __future__ import annotations
import itertools
from typing import Callable, Optional

from indexcalc.core.group import Group, Representation
from indexcalc.core.index import Index, IndexSpace
from indexcalc.core.tensor import Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul
from indexcalc.core.variation import ZeroTensor


# 모든 generator action factory가 공유하는 fresh dummy name counter.
# 같은 expression tree 안에서 여러 leaf에 action이 적용될 때 globally unique한
# 이름을 보장. ``canonical_form_modulo_dummies``의 ``_d{n}`` 와 충돌하지 않도록
# prefix 분리 (``_act{n}``).
_dummy_counter = itertools.count()


def _fresh_dummy_name(base: str = "_act") -> str:
    """Generator action factory들이 공유하는 globally-unique dummy 이름 발급."""
    return f"{base}{next(_dummy_counter)}"


# ─── Action signature ──────────────────────────────────────────
#   action(field: Tensor) -> TensorExpr
ActionFn = Callable[[Tensor], TensorExpr]


class Generator:
    """단일 generator (또는 generator family)의 그룹별·rep별 작용을 보관한다.

    Non-abelian의 경우 생성자 가족 $T^a$ 전체를 한 Generator 인스턴스가 표현한다 —
    adjoint 인덱스 $a$는 ``apply_to`` 결과에서 free index로 등장한다 (M2부터 의미 있음).

    Parameters
    ----------
    name : str
        generator 이름 (e.g., "T_U(1)", "T_SU(3)").
    group : Group
        이 generator가 속한 그룹.
    """

    def __init__(self, name: str, group: Group):
        self.name = name
        self.group = group
        self._actions: dict[str, ActionFn] = {}
        # Optional: action on PartialDeriv's free index ($\partial_\mu$). Lorentz
        # transformations rotate the deriv_index as a vector even though no
        # explicit Tensor leaf carries it. Other generators (gauge U(1)/SU(N))
        # leave it alone.
        self._deriv_index_action: Optional[Callable] = None

    def declare_action(self, rep_name: str, action: ActionFn) -> None:
        """rep ``rep_name``의 field에 대한 작용을 등록한다."""
        if not self.group.has_rep(rep_name):
            raise ValueError(
                f"Rep {rep_name!r} not in group {self.group.name!r}"
            )
        self._actions[rep_name] = action

    def has_action(self, rep_name: str) -> bool:
        return rep_name in self._actions

    def declare_deriv_index_action(self, action: Callable) -> None:
        """``PartialDeriv.deriv_index``를 회전시키는 작용을 등록한다.

        ``action(pd: PartialDeriv) -> TensorExpr`` 형태. Lorentz vector 회전이
        대표 use case. 미등록이면 ``apply_generator``는 deriv_index 회전 항을
        생성하지 않는다 (global gauge default).
        """
        self._deriv_index_action = action

    def apply_to_deriv_index(self, pd) -> Optional["TensorExpr"]:
        """등록된 deriv_index_action 호출. 없으면 ``None``."""
        if self._deriv_index_action is None:
            return None
        return self._deriv_index_action(pd)

    def apply_to(self, field: Tensor) -> TensorExpr:
        """단일 field에 generator를 적용한다.

        - field가 ``self.group``의 rep tag를 가지지 않으면 → singlet → ZeroTensor.
        - rep tag가 있으나 작용이 등록 안 되어 있으면 → ValueError.
        - 그 외엔 등록된 action 함수의 결과를 그대로 반환.
        """
        rep_name = field.reps.get(self.group.name)
        if rep_name is None:
            return ZeroTensor(field.free_indices)
        if rep_name not in self._actions:
            raise ValueError(
                f"Generator {self.name!r}: no action declared for "
                f"rep {rep_name!r} of group {self.group.name!r}"
            )
        return self._actions[rep_name](field)

    def __repr__(self) -> str:
        return (
            f"Generator({self.name!r}, group={self.group.name!r}, "
            f"reps_with_action={list(self._actions)})"
        )


# ─── Helper: U(1) action factory ────────────────────────────────


def u1_action(rep: Representation) -> ActionFn:
    """U(1) 작용 factory: $\\delta\\phi = i q \\phi$ ($q$ = rep.charge).

    Parameters
    ----------
    rep : Representation
        ``charge`` 가 정의된 abelian rep.
    """
    if rep.charge is None:
        raise ValueError(
            f"u1_action requires a rep with .charge set (rep={rep!r})"
        )
    q = rep.charge

    def action(field: Tensor) -> TensorExpr:
        # δφ = (i q) · φ. 파라미터 α는 stripping (전체적인 인자).
        return ScalarMul(1j * q, field)

    return action


# ─── Helper: register a Generator + standard U(1) actions ───────


# ─── Helper: SU(N) adjoint action factory ───────────────────────


def su_n_adj_action(
    adj_space: IndexSpace,
    parameter_name: str = "b",
    structure_const_name: str = "f",
) -> ActionFn:
    """SU(N) adjoint rep 작용 factory:

    .. math:: \\delta_b X^a = f^a{}_{bc} X^c

    convention 메모:
      - $f$의 첫 인덱스는 입력 field의 adj 인덱스 위치(upper/lower)를 따른다.
      - 두 번째 ($b$, parameter)와 세 번째 ($c$, dummy)는 lower로 고정.
      - $f$는 모든 인덱스 쌍에 antisymmetric (Cartan-Killing 정규화 $\\kappa = \\delta$ 가정).
      - dummy 'c'는 $f$에서 lower, field에서 upper로 contract.

    각 ``apply_to(field)`` 호출 시 dummy 이름은 충돌 회피용으로 fresh하게 발급.
    parameter 이름은 generator 인스턴스에 대해 고정 (모든 leaf 변환에 같은 free 인덱스).

    Parameters
    ----------
    adj_space : IndexSpace
        adjoint 인덱스 공간.
    parameter_name : str
        generator parameter index의 이름 (예: ``"b"``). free index로 결과에 등장.
    structure_const_name : str
        구조 상수 텐서의 이름 (default ``"f"``).
    """

    def action(field: Tensor) -> TensorExpr:
        adj_indices = [
            (i, idx) for i, idx in enumerate(field.indices)
            if idx.space == adj_space
        ]
        if len(adj_indices) != 1:
            raise ValueError(
                f"su_n_adj_action: field {field.name!r} expected to have exactly "
                f"one index in {adj_space.name!r}, got {len(adj_indices)}"
            )
        slot, adj_idx = adj_indices[0]

        # globally unique dummy (앞뒤 expression context에 등장하는 어떤 이름과도 충돌 X)
        dummy_name = _fresh_dummy_name()

        # f tensor: [field's adj position, param lower, dummy lower]. all-pair antisymmetric.
        f_first = Index(adj_idx.name, adj_space, adj_idx.position)
        f_param = Index(parameter_name, adj_space, "lower")
        f_dummy = Index(dummy_name, adj_space, "lower")
        f_tensor = Tensor(
            structure_const_name,
            [f_first, f_param, f_dummy],
            antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        )

        # field with adj index renamed adj_idx.name → dummy_name, position UPPER
        # (contracts with f's third index which is lower)
        new_indices = list(field.indices)
        new_indices[slot] = Index(dummy_name, adj_space, "upper")
        renamed_field = Tensor(
            field.name,
            new_indices,
            antisymmetric_pairs=list(field.antisymmetric_pairs),
            reps=field.reps,
            statistics=field.statistics,
        )

        return TensorProduct(f_tensor, renamed_field)

    return action


def su_n_fund_action(
    adj_space: IndexSpace,
    fund_space: IndexSpace,
    parameter_name: str = "a",
    rep_matrix_name: str = "T",
) -> ActionFn:
    """SU(N) fund / antifund rep 작용 factory.

    Convention:
        $\\delta_a \\phi^i = i (T^a)^i{}_j \\phi^j$ (fund — input fund-upper).
        $\\delta_a \\phi_i = -i \\phi_j (T^a)^j{}_i$ (antifund — input fund-lower).

    $T^a$는 rep matrix tensor (이름 ``rep_matrix_name``, 기본 ``"T"``):
        - adj parameter index (upper)
        - fund row index (matches input position)
        - fund col index (opposite to input — contracts with renamed field)

    각 호출 시 fresh dummy name 발급 (parameter는 generator 인스턴스에서 고정).

    Parameters
    ----------
    adj_space : IndexSpace
    fund_space : IndexSpace
    parameter_name : str
        adj parameter index 이름 (free).
    rep_matrix_name : str
        rep matrix tensor 이름.
    """

    def action(field: Tensor) -> TensorExpr:
        fund_indices = [
            (i, idx) for i, idx in enumerate(field.indices)
            if idx.space == fund_space
        ]
        if len(fund_indices) != 1:
            raise ValueError(
                f"su_n_fund_action: field {field.name!r} expected to have exactly "
                f"one index in {fund_space.name!r}, got {len(fund_indices)}"
            )
        slot, fund_idx = fund_indices[0]
        position = fund_idx.position

        # globally unique dummy
        dummy = _fresh_dummy_name()

        if position == "upper":
            # δ φ^i = i T^{a,i}_j φ^j
            T = Tensor(
                rep_matrix_name,
                [
                    Index(parameter_name, adj_space, "upper"),
                    Index(fund_idx.name, fund_space, "upper"),  # row matches input
                    Index(dummy, fund_space, "lower"),  # col contracts with renamed
                ],
            )
            new_indices = list(field.indices)
            new_indices[slot] = Index(dummy, fund_space, "upper")
            renamed = Tensor(
                field.name, new_indices,
                antisymmetric_pairs=list(field.antisymmetric_pairs),
                reps=dict(field.reps),
                statistics=field.statistics,
            )
            return ScalarMul(1j, TensorProduct(T, renamed))
        else:  # lower (antifund)
            # δ φ_i = -i φ_j T^{a,j}_i
            T = Tensor(
                rep_matrix_name,
                [
                    Index(parameter_name, adj_space, "upper"),
                    Index(dummy, fund_space, "upper"),  # row contracts with renamed
                    Index(fund_idx.name, fund_space, "lower"),  # col matches input
                ],
            )
            new_indices = list(field.indices)
            new_indices[slot] = Index(dummy, fund_space, "lower")
            renamed = Tensor(
                field.name, new_indices,
                antisymmetric_pairs=list(field.antisymmetric_pairs),
                reps=dict(field.reps),
                statistics=field.statistics,
            )
            return ScalarMul(-1j, TensorProduct(renamed, T))

    return action


# ─── Helper: Lorentz spinor generator action ─────────────────


def lorentz_spinor_action(
    frame_space: IndexSpace,
    spinor_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "Sigma",
) -> ActionFn:
    """Lorentz spinor rep 작용 ($SO(1, D-1)$, Dirac).

    Convention:
        $\\delta\\psi^\\alpha = -\\tfrac{i}{2}\\, \\Sigma^{ab}{}^\\alpha{}_\\beta\\, \\psi^\\beta$
        $\\delta\\bar\\psi_\\alpha = +\\tfrac{i}{2}\\, \\bar\\psi_\\beta\\, \\Sigma^{ab}{}^\\beta{}_\\alpha$

    Parameter indices ``(a, b)`` are antisymmetric (Lorentz generator pair).
    각 호출 시 fresh dummy spinor name 발급.

    Parameters
    ----------
    frame_space : IndexSpace
        Lorentz frame (vector) 인덱스 공간 — 보통 metric 갖는 Minkowski.
    spinor_space : IndexSpace
        Dirac spinor 공간 — 보통 metric 없음.
    parameter_names : tuple[str, str]
        antisym 쌍의 두 frame 이름 (default ``("a", "b")``).
    generator_name : str
        Σ tensor 이름 (default ``"Sigma"``).
    """
    a_name, b_name = parameter_names

    def action(field: Tensor) -> TensorExpr:
        spinor_indices = [
            (i, idx) for i, idx in enumerate(field.indices)
            if idx.space == spinor_space
        ]
        if len(spinor_indices) != 1:
            raise ValueError(
                f"lorentz_spinor_action: field {field.name!r} expected to have "
                f"exactly one index in {spinor_space.name!r}, got {len(spinor_indices)}"
            )
        slot, sp_idx = spinor_indices[0]
        position = sp_idx.position

        # globally unique dummy
        dummy = _fresh_dummy_name()

        if position == "upper":
            # δψ^α = -i/2 Σ^{ab,α}_β ψ^β
            Sigma = Tensor(
                generator_name,
                [
                    Index(a_name, frame_space, "upper"),
                    Index(b_name, frame_space, "upper"),
                    Index(sp_idx.name, spinor_space, "upper"),  # row matches input
                    Index(dummy, spinor_space, "lower"),  # col contracts with renamed
                ],
                antisymmetric_pairs=[(0, 1)],
            )
            new_indices = list(field.indices)
            new_indices[slot] = Index(dummy, spinor_space, "upper")
            renamed = Tensor(
                field.name, new_indices,
                antisymmetric_pairs=list(field.antisymmetric_pairs),
                reps=dict(field.reps),
                statistics=field.statistics,
            )
            return ScalarMul(-0.5j, TensorProduct(Sigma, renamed))
        else:  # lower (conj_spinor)
            # δψ̄_α = +i/2 ψ̄_β Σ^{ab,β}_α
            Sigma = Tensor(
                generator_name,
                [
                    Index(a_name, frame_space, "upper"),
                    Index(b_name, frame_space, "upper"),
                    Index(dummy, spinor_space, "upper"),  # row contracts with renamed
                    Index(sp_idx.name, spinor_space, "lower"),  # col matches input
                ],
                antisymmetric_pairs=[(0, 1)],
            )
            new_indices = list(field.indices)
            new_indices[slot] = Index(dummy, spinor_space, "lower")
            renamed = Tensor(
                field.name, new_indices,
                antisymmetric_pairs=list(field.antisymmetric_pairs),
                reps=dict(field.reps),
                statistics=field.statistics,
            )
            return ScalarMul(0.5j, TensorProduct(renamed, Sigma))

    return action


def lorentz_vector_action(
    frame_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "M_vec",
    cometric_antisym: bool = False,
) -> ActionFn:
    """Lorentz vector rep 작용 ($SO(1, D-1)$ on 4-vectors and rank-≥1 tensors).

    Convention (per slot):
        $\\delta V^\\mu = (M^{ab})^\\mu{}_\\nu V^\\nu$  (frame slot upper)
        $\\delta V_\\mu = -V_\\nu (M^{ab})^\\nu{}_\\mu$  (frame slot lower)

    Multi-frame-index 필드 ($T^{\\mu\\nu}$, $W^A_{\\mu\\nu}$ 등) 는 **Leibniz** —
    각 frame slot 마다 한 항씩 만들어 모두 합산한다. 비-frame 인덱스
    (예: $W$의 SU(2) adj 인덱스, spinor 등) 는 그대로 유지.

    $M^{ab}$는 vector rep matrix tensor — adj 인덱스 (a, b) antisym + frame
    (row, col). 구체적 components ($M^{ab}_{\\mu\\nu} = i(\\eta^{a\\mu}\\delta^b_\\nu - \\eta^{b\\mu}\\delta^a_\\nu)$)
    는 IR-level invariance 검증에 불필요 — rep 변환 구조만 표현.
    """
    a_name, b_name = parameter_names

    def _rotate_slot(field: Tensor, slot: int) -> TensorExpr:
        fr_idx = field.indices[slot]
        position = fr_idx.position
        dummy = _fresh_dummy_name()

        new_indices = list(field.indices)
        renamed_pos = "upper" if position == "upper" else "lower"
        new_indices[slot] = Index(dummy, frame_space, renamed_pos)
        renamed = Tensor(
            field.name, new_indices,
            antisymmetric_pairs=list(field.antisymmetric_pairs),
            symmetric_pairs=list(field.symmetric_pairs),
            traceless=list(field.traceless),
            transverse=list(field.transverse),
            reps=dict(field.reps),
            statistics=field.statistics,
        )

        # Vector row/col are slots 2,3 in both branches. For orthogonal
        # generators (so(N)) they are antisymmetric *once both are at the same
        # position* (after the δ-metric raise/lower) — deferred via
        # cometric_antisymmetric_pairs. Symplectic callers leave this off.
        cometric = [(2, 3)] if cometric_antisym else None
        if position == "upper":
            # δ V^μ = M^{ab,μ}_ν V^ν
            M = Tensor(
                generator_name,
                [
                    Index(a_name, frame_space, "upper"),
                    Index(b_name, frame_space, "upper"),
                    Index(fr_idx.name, frame_space, "upper"),
                    Index(dummy, frame_space, "lower"),
                ],
                antisymmetric_pairs=[(0, 1)],
                cometric_antisymmetric_pairs=cometric,
            )
            return TensorProduct(M, renamed)
        else:
            # δ V_μ = -V_ν M^{ab,ν}_μ
            M = Tensor(
                generator_name,
                [
                    Index(a_name, frame_space, "upper"),
                    Index(b_name, frame_space, "upper"),
                    Index(dummy, frame_space, "upper"),
                    Index(fr_idx.name, frame_space, "lower"),
                ],
                antisymmetric_pairs=[(0, 1)],
                cometric_antisymmetric_pairs=cometric,
            )
            return ScalarMul(-1.0, TensorProduct(renamed, M))

    def action(field: Tensor) -> TensorExpr:
        frame_slots = [
            i for i, idx in enumerate(field.indices)
            if idx.space == frame_space
        ]
        if len(frame_slots) == 0:
            raise ValueError(
                f"lorentz_vector_action: field {field.name!r} has no index in "
                f"{frame_space.name!r}; vector rep requires ≥1 frame slot"
            )

        terms = [_rotate_slot(field, slot) for slot in frame_slots]
        result = terms[0]
        for t in terms[1:]:
            result = TensorSum(result, t)
        return result

    return action


def lorentz_deriv_index_action(
    frame_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "M_vec",
):
    """Lorentz 변환의 ``PartialDeriv`` index에 대한 vector 회전 작용.

    Convention (lower position only — ``PartialDeriv.deriv_index`` 는 항상 lower):

        $\\delta\\partial_\\mu T = -\\partial_\\nu T \\cdot (M^{ab})^\\nu{}_\\mu$

    즉 ``δV_μ`` 와 동일 형태. Lorentz 파라미터 $(a, b)$ 는 결과의 free index로
    살아남는다.

    Returns
    -------
    Callable[[PartialDeriv], TensorExpr]
        ``Generator.declare_deriv_index_action`` 에 등록 가능한 함수.
    """
    a_name, b_name = parameter_names

    def action(pd) -> TensorExpr:
        from indexcalc.core.deriv import PartialDeriv

        d_idx = pd.deriv_index
        if d_idx.space != frame_space:
            return ZeroTensor([d_idx] + pd.expr.free_indices)
        dummy = _fresh_dummy_name()
        new_pd = PartialDeriv(pd.expr, Index(dummy, frame_space, "lower"))
        M = Tensor(
            generator_name,
            [
                Index(a_name, frame_space, "upper"),
                Index(b_name, frame_space, "upper"),
                Index(dummy, frame_space, "upper"),
                Index(d_idx.name, frame_space, "lower"),
            ],
            antisymmetric_pairs=[(0, 1)],
        )
        return ScalarMul(-1.0, TensorProduct(new_pd, M))

    return action


def make_lorentz_spinor_generator(
    group: Group,
    frame_space: IndexSpace,
    spinor_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "Sigma",
    name: Optional[str] = None,
) -> Generator:
    """Lorentz Group의 표준 generator: spinor / conj_spinor / vector / singlet 자동 등록.

    parameter_names ``(a, b)``는 frame 인덱스의 antisym pair.

    Examples
    --------
    >>> from indexcalc.core.index import IndexSpace
    >>> from indexcalc.core.group import Group
    >>> lorentz = Group("Lorentz", dim=6, abelian=False)
    >>> lorentz.add_rep("spinor", dim=4)
    >>> lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    >>> lorentz.add_rep("vector", dim=4)
    >>> lorentz.add_rep("singlet", dim=1)
    >>> st = IndexSpace("st", dim=4, indices="μνλρσ", metric="η")
    >>> sp = IndexSpace("dirac", dim=4, indices="αβγ")
    >>> g = make_lorentz_spinor_generator(lorentz, st, sp)
    >>> all(g.has_action(r) for r in ("spinor", "conj_spinor", "vector", "singlet"))
    True
    """
    gen = Generator(name or f"M_{group.name}", group)
    spinor_act = lorentz_spinor_action(
        frame_space, spinor_space, parameter_names, generator_name,
    )
    vector_act = lorentz_vector_action(
        frame_space, parameter_names,
    )
    if group.has_rep("spinor"):
        gen.declare_action("spinor", spinor_act)
    if group.has_rep("conj_spinor"):
        gen.declare_action("conj_spinor", spinor_act)
    # Chiral reps share the same Σ action — chirality is a separate quantum
    # number tracked at the field-tag level; under proper Lorentz the Σ
    # rotation acts identically on LH/RH spinors. {γ_5, Σ^{ab}} = 0 vanishes
    # since Σ ~ [γ^a, γ^b].
    for chiral_rep in ("L_spinor", "R_spinor"):
        if group.has_rep(chiral_rep):
            gen.declare_action(chiral_rep, spinor_act)
    for chiral_conj in ("conj_L_spinor", "conj_R_spinor"):
        if group.has_rep(chiral_conj):
            gen.declare_action(chiral_conj, spinor_act)
    if group.has_rep("vector"):
        gen.declare_action("vector", vector_act)
    if group.has_rep("singlet"):
        gen.declare_action("singlet", lambda f: ZeroTensor(f.free_indices))
    # PartialDeriv의 deriv_index 회전: vector rep 가용시 자동 등록
    if group.has_rep("vector"):
        gen.declare_deriv_index_action(
            lorentz_deriv_index_action(frame_space, parameter_names),
        )
    return gen


def make_su_n_generator(
    group: Group,
    adj_space: IndexSpace,
    parameter_name: str = "b",
    structure_const_name: str = "f",
    name: Optional[str] = None,
    fund_space: Optional[IndexSpace] = None,
    rep_matrix_name: str = "T",
) -> Generator:
    """SU(N) Group에 등록된 adj/singlet rep에 대한 표준 generator를 만든다.

    fund rep은 별도 representation matrix 텐서 처리가 필요하므로 v1에선 미등록 —
    fund field에 작용하면 ``ValueError`` 발생 (M3에서 spinor와 함께 추가 예정).

    Examples
    --------
    >>> from indexcalc.core.index import IndexSpace
    >>> from indexcalc.core.group import Group
    >>> sun = Group("SU(3)", dim=8, abelian=False)
    >>> sun.add_rep("adj", dim=8)
    >>> sun.add_rep("singlet", dim=1)
    >>> adj = IndexSpace("su3_adj", dim=8, indices="abcdefgh")
    >>> g = make_su_n_generator(sun, adj)
    >>> g.has_action("adj") and g.has_action("singlet")
    True
    """
    if group.abelian:
        raise ValueError(
            f"make_su_n_generator requires non-abelian group, got {group.name!r}"
        )
    gen = Generator(name or f"T_{group.name}", group)

    if group.has_rep("adj"):
        gen.declare_action(
            "adj",
            su_n_adj_action(
                adj_space, parameter_name=parameter_name,
                structure_const_name=structure_const_name,
            ),
        )
    if group.has_rep("singlet"):
        # singlet field에 대한 작용: 0 (ZeroTensor)
        gen.declare_action("singlet", lambda field: ZeroTensor(field.free_indices))

    # fund / antifund 등록 — fund_space가 주어진 경우만
    if fund_space is not None:
        fund_act = su_n_fund_action(
            adj_space, fund_space,
            parameter_name=parameter_name,
            rep_matrix_name=rep_matrix_name,
        )
        if group.has_rep("fund"):
            gen.declare_action("fund", fund_act)
        if group.has_rep("antifund"):
            gen.declare_action("antifund", fund_act)

    return gen


def make_o_n_generator(
    group: Group,
    vector_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "M",
    name: Optional[str] = None,
) -> Generator:
    """$O(N)$ 또는 $SO(N)$ Group의 vector rep + singlet generator.

    vector rep 작용은 ``lorentz_vector_action`` 그대로 재활용 ($M^{ab,c}{}_d$
    rep matrix 형식). $SO(N)$ Euclidean은 $SO(1, D-1)$의 Wick-rotated 친척이라
    IR 표현 일치.

    $SO(N)$ vs $O(N)$ 구분은 invariant tensor 단계에서 처리
    (ε_{i_1..i_N}는 SO(N) 한정). generator/algebra 수준에서 동일.

    **Backend gap (v2.5 도중 확인):** δ-bilinear $\\delta_{ij}V^iV^j$의 회전
    invariance를 oracle(apply_generator + simplify)만으로 확인하려면
    metric absorption + $M^{ab}{}_{ij}$의 vector-index antisymmetric 인코딩
    + post-absorption antisymmetric_pairs 추론, 세 가지의 coupled 변경 필요.
    v2.5 범위에서 단일 fix가 작음. ``lions.probe._structural_check``를
    fallback으로 유지해서 hybrid oracle 유지. v3+ 의 backend cleanup
    마일스톤에서 단일 path로 통일.

    Examples
    --------
    >>> so3 = Group("SO(3)", dim=3, abelian=False)
    >>> so3.add_rep("vector", dim=3)
    >>> so3.add_rep("singlet", dim=1)
    >>> vec = IndexSpace("so3_vec", dim=3, indices="ijklmn", metric="delta")
    >>> g = make_o_n_generator(so3, vec)
    >>> g.has_action("vector") and g.has_action("singlet")
    True
    """
    if group.abelian:
        raise ValueError(
            f"make_o_n_generator requires non-abelian group, got {group.name!r}"
        )
    gen = Generator(name or f"M_{group.name}", group)
    vec_act = lorentz_vector_action(
        vector_space, parameter_names, generator_name=generator_name,
        cometric_antisym=True,
    )
    if group.has_rep("vector"):
        gen.declare_action("vector", vec_act)
    if group.has_rep("singlet"):
        gen.declare_action(
            "singlet", lambda field: ZeroTensor(field.free_indices),
        )
    return gen


def make_sp_2n_generator(
    group: Group,
    vector_space: IndexSpace,
    parameter_names: tuple = ("a", "b"),
    generator_name: str = "M",
    name: Optional[str] = None,
) -> Generator:
    """$Sp(2N)$ Group의 fundamental rep + singlet generator.

    Symplectic fundamental rep은 $2N$-차원 공간 위에서 작용하며 IR-level
    변환 구조 $\\delta V^i = (M)^i{}_j V^j$ 는 ``lorentz_vector_action`` 의
    그것과 동일하다 — 그래서 ``make_o_n_generator`` 와 같은 패턴으로 재사용한다.

    $Sp(2N)$과 $O(2N)$의 차이는 *invariant tensor* 단계에서만 드러난다:
    $O(N)$은 대칭 $\\delta_{ij}$, $Sp(2N)$은 **반대칭** symplectic form
    $\\Omega_{ij} = -\\Omega_{ji}$ 를 보존한다 (``standard_sp_2n_invariants``).
    $Sp$ 대수 조건 ($M^T\\Omega + \\Omega M = 0$, 즉 $\\Omega M$ 대칭) 자체는
    $O(N)$의 δ-bilinear 와 동일하게 oracle(apply_generator+simplify)만으로는
    완결되지 않으며 ``lions.probe._structural_check`` fallback에 의존한다
    (``make_o_n_generator`` 의 backend gap 노트 참조). v3+ backend cleanup에서
    단일 path로 통일.

    Examples
    --------
    >>> sp4 = Group("Sp(4)", dim=10, abelian=False)
    >>> sp4.add_rep("vector", dim=4)
    >>> sp4.add_rep("singlet", dim=1)
    >>> vec = IndexSpace("sp4_vec", dim=4, indices="ijklmn", metric="omega")
    >>> g = make_sp_2n_generator(sp4, vec)
    >>> g.has_action("vector") and g.has_action("singlet")
    True
    """
    if group.abelian:
        raise ValueError(
            f"make_sp_2n_generator requires non-abelian group, got {group.name!r}"
        )
    gen = Generator(name or f"M_{group.name}", group)
    vec_act = lorentz_vector_action(
        vector_space, parameter_names, generator_name=generator_name,
    )
    if group.has_rep("vector"):
        gen.declare_action("vector", vec_act)
    if group.has_rep("singlet"):
        gen.declare_action(
            "singlet", lambda field: ZeroTensor(field.free_indices),
        )
    return gen


def make_u1_generator(group: Group, name: Optional[str] = None) -> Generator:
    """U(1) Group에 등록된 모든 charged rep에 대해 자동으로 action을 다는 헬퍼.

    Examples
    --------
    >>> u1 = Group("U(1)", abelian=True)
    >>> u1.add_rep("+1", dim=1, charge=1.0)
    >>> u1.add_rep("-1", dim=1, charge=-1.0)
    >>> g = make_u1_generator(u1)
    >>> g.has_action("+1") and g.has_action("-1")
    True
    """
    if not group.abelian:
        raise ValueError(
            f"make_u1_generator requires an abelian group, got {group.name!r}"
        )
    gen = Generator(name or f"T_{group.name}", group)
    for rep_name, rep in group.reps.items():
        gen.declare_action(rep_name, u1_action(rep))
    return gen
