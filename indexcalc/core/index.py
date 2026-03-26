"""
IndexSpace와 Index: 텐서 인덱스 시스템의 기초 단위.

IndexSpace는 인덱스가 속하는 공간을 정의한다 (spacetime, Lorentz, gauge 등).
Index는 특정 이름과 위치(upper/lower)를 가진 구체적인 인덱스 인스턴스이다.
"""

from __future__ import annotations
from dataclasses import dataclass


class IndexSpace:
    """인덱스가 살고 있는 공간을 정의한다.

    각 IndexSpace는 고유한 이름, 차원, 사용 가능한 인덱스 문자들,
    그리고 이 공간에서 raise/lower에 사용되는 metric 이름을 갖는다.

    Parameters
    ----------
    name : str
        공간의 이름 (e.g., "spacetime", "lorentz").
    dim : int
        공간의 차원.
    indices : str
        이 공간에서 사용할 수 있는 인덱스 문자들 (e.g., "μνλρσ").
        표기 convention을 정의하는 용도이며, 강제되지는 않는다.
    metric : str
        이 공간의 metric 텐서 이름 (e.g., "g", "η").

    Examples
    --------
    >>> spacetime = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> lorentz = IndexSpace("lorentz", dim=4, indices="abcde", metric="η")
    """

    def __init__(self, name: str, dim: int, indices: str = "", metric: str = ""):
        self.name = name
        self.dim = dim
        self.indices = indices
        self.metric = metric

    # --- Index 생성 헬퍼 ---

    def upper(self, name: str) -> Index:
        """이 공간에 속하는 upper(contravariant) index를 생성한다."""
        return Index(name, self, "upper")

    def lower(self, name: str) -> Index:
        """이 공간에 속하는 lower(covariant) index를 생성한다."""
        return Index(name, self, "lower")

    def __repr__(self) -> str:
        return f"IndexSpace({self.name!r}, dim={self.dim})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexSpace):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True)
class Index:
    """구체적인 인덱스 하나를 나타낸다.

    Index는 immutable(frozen)이다. 이름, 공간, 위치가 정해지면 변하지 않으며,
    위치를 바꾸려면 flip()으로 새 Index를 만든다.

    Parameters
    ----------
    name : str
        인덱스 문자 (e.g., "μ", "a", "M").
    space : IndexSpace
        이 인덱스가 속하는 공간.
    position : str
        "upper" (contravariant) 또는 "lower" (covariant).

    Examples
    --------
    >>> st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="g")
    >>> mu = st.upper("μ")
    >>> mu
    ^μ
    >>> mu.flip()
    _μ
    """

    name: str
    space: IndexSpace
    position: str  # "upper" or "lower"

    def __post_init__(self) -> None:
        if self.position not in ("upper", "lower"):
            raise ValueError(
                f"position must be 'upper' or 'lower', got {self.position!r}"
            )

    def flip(self) -> Index:
        """위치를 반전시킨 새 Index를 반환한다. upper ↔ lower."""
        new_pos = "lower" if self.position == "upper" else "upper"
        return Index(self.name, self.space, new_pos)

    def same_slot(self, other: Index) -> bool:
        """같은 이름, 같은 공간, 같은 위치인지 확인한다."""
        return (
            self.name == other.name
            and self.space == other.space
            and self.position == other.position
        )

    def contracts_with(self, other: Index) -> bool:
        """이 인덱스와 other가 contraction 쌍인지 확인한다.

        같은 이름, 같은 공간, 반대 위치(upper↔lower)이면 contraction된다.
        """
        return (
            self.name == other.name
            and self.space == other.space
            and self.position != other.position
        )

    def __repr__(self) -> str:
        symbol = "^" if self.position == "upper" else "_"
        return f"{symbol}{self.name}"
