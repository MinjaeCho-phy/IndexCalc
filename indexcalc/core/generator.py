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
from typing import Callable, Optional

from indexcalc.core.group import Group, Representation
from indexcalc.core.tensor import Tensor, TensorExpr, ScalarMul
from indexcalc.core.variation import ZeroTensor


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

    def declare_action(self, rep_name: str, action: ActionFn) -> None:
        """rep ``rep_name``의 field에 대한 작용을 등록한다."""
        if not self.group.has_rep(rep_name):
            raise ValueError(
                f"Rep {rep_name!r} not in group {self.group.name!r}"
            )
        self._actions[rep_name] = action

    def has_action(self, rep_name: str) -> bool:
        return rep_name in self._actions

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
