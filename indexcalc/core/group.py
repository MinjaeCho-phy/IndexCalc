"""
Group/Representation: Lie group과 그 representation의 메타데이터 관리.

이 모듈은 *수치 계산을 위한 group element*를 다루지 않는다.
용도는 **field에 rep tag를 달고**, generator와 invariant tensor가 어느 그룹의
어떤 rep에 작용하는지를 일관되게 식별하는 것이다.

Examples
--------
>>> u1 = Group("U(1)", abelian=True)
>>> u1.add_rep("+1", dim=1, charge=1.0)
>>> u1.add_rep("-1", dim=1, charge=-1.0)
>>> sun = Group("SU(3)", dim=8, abelian=False)
>>> sun.add_rep("fund", dim=3)
>>> sun.add_rep("antifund", dim=3, conjugate=True)
>>> sun.add_rep("adj", dim=8)
>>> sun.add_rep("singlet", dim=1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Representation:
    """단일 representation의 메타데이터.

    Parameters
    ----------
    name : str
        rep의 이름 (e.g., "fund", "adj", "singlet", "+1").
    group_name : str
        이 rep이 속한 그룹의 이름. 순환 참조 회피를 위해 이름만 저장.
    dim : int
        rep의 차원.
    conjugate : bool
        켤레 rep 여부. fund$^*$ = antifund 류.
    charge : float | None
        Abelian 그룹의 경우 generator의 고윳값 (e.g., U(1) hypercharge).
        Non-abelian이면 None.
    """

    name: str
    group_name: str
    dim: int
    conjugate: bool = False
    charge: Optional[float] = None


class Group:
    """Lie group의 메타데이터.

    이 클래스는 그룹 자체의 수치 표현(structure constants 등)이 아니라
    **rep과 generator의 등록 hub** 역할을 한다.

    Parameters
    ----------
    name : str
        그룹 이름 (e.g., "U(1)", "SU(3)", "SO(1,3)").
    dim : int | None
        Lie algebra의 차원 (= adjoint rep의 차원). U(1)이면 1.
    abelian : bool
        Abelian 여부. Abelian이면 generator가 단일이며 rep은 charge로 구분.
    compact : bool
        compact 여부. v1에서는 compact만 다룬다.

    Examples
    --------
    >>> u1 = Group("U(1)", dim=1, abelian=True)
    >>> u1.add_rep("+1", dim=1, charge=1.0)
    >>> u1.get_rep("+1").charge
    1.0
    """

    def __init__(
        self,
        name: str,
        dim: Optional[int] = None,
        abelian: bool = False,
        compact: bool = True,
    ):
        self.name = name
        self.dim = dim
        self.abelian = abelian
        self.compact = compact
        self._reps: dict[str, Representation] = {}

    def add_rep(
        self,
        name: str,
        dim: int,
        conjugate: bool = False,
        charge: Optional[float] = None,
    ) -> Representation:
        """rep을 등록한다. 같은 이름이 이미 있으면 ValueError."""
        if name in self._reps:
            raise ValueError(
                f"Rep {name!r} already registered in group {self.name!r}"
            )
        if self.abelian and charge is None:
            raise ValueError(
                f"Abelian group {self.name!r} requires 'charge' for rep {name!r}"
            )
        rep = Representation(
            name=name,
            group_name=self.name,
            dim=dim,
            conjugate=conjugate,
            charge=charge,
        )
        self._reps[name] = rep
        return rep

    def get_rep(self, name: str) -> Representation:
        """이름으로 rep을 조회한다."""
        if name not in self._reps:
            raise KeyError(
                f"Rep {name!r} not found in group {self.name!r}. "
                f"Available: {list(self._reps)}"
            )
        return self._reps[name]

    def has_rep(self, name: str) -> bool:
        return name in self._reps

    @property
    def reps(self) -> dict[str, Representation]:
        return dict(self._reps)

    def __repr__(self) -> str:
        kind = "abelian" if self.abelian else "non-abelian"
        return f"Group({self.name!r}, {kind}, dim={self.dim}, reps={list(self._reps)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Group):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class GroupRegistry:
    """전역(또는 컨텍스트별) Group 등록소.

    한 라그랑지안 시스템 안에서 일관된 그룹 이름 → Group 매핑을 제공한다.

    Examples
    --------
    >>> reg = GroupRegistry()
    >>> u1 = Group("U(1)", abelian=True)
    >>> u1.add_rep("+1", dim=1, charge=1.0)
    >>> reg.register(u1)
    >>> reg.get("U(1)") is u1
    True
    """

    def __init__(self):
        self._groups: dict[str, Group] = {}

    def register(self, group: Group) -> None:
        if group.name in self._groups:
            raise ValueError(f"Group {group.name!r} already registered")
        self._groups[group.name] = group

    def get(self, name: str) -> Group:
        if name not in self._groups:
            raise KeyError(
                f"Group {name!r} not registered. Available: {list(self._groups)}"
            )
        return self._groups[name]

    def has(self, name: str) -> bool:
        return name in self._groups

    @property
    def groups(self) -> dict[str, Group]:
        return dict(self._groups)
