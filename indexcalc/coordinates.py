"""
좌표계 정의 (Phase 6f).

Coordinates 클래스는 manifold 위의 좌표 chart를 정의한다.
preset("cartesian", "spherical", ...)을 사용하거나 직접 좌표 이름을 지정할 수 있다.

사용법
------
>>> # Custom 좌표
>>> coords = Coordinates(["t", "x", "y", "z"])
>>>
>>> # Preset — dim으로 공간 차원 지정
>>> coords = Coordinates.preset("spherical", dim=3)       # r, θ, φ
>>>
>>> # Preset — signature로 시간 차원 자동 추가
>>> coords = Coordinates.preset("spherical", signature=(3,1))  # t, r, θ, φ
"""

from __future__ import annotations
from dataclasses import dataclass


# ─── Preset 좌표 이름 ─────────────────────────────────────────

_PRESETS: dict[str, dict[int, list[str]]] = {
    "cartesian": {
        1: ["x"],
        2: ["x", "y"],
        3: ["x", "y", "z"],
        4: ["x", "y", "z", "w"],
    },
    "spherical": {
        1: ["r"],
        2: ["θ", "φ"],
        3: ["r", "θ", "φ"],
    },
    "cylindrical": {
        1: ["ρ"],
        2: ["ρ", "φ"],
        3: ["ρ", "φ", "z"],
    },
}

# 시간 좌표 이름 (signature에 - 가 있을 때 앞에 추가)
_TEMPORAL = "t"


@dataclass(frozen=True)
class Coordinates:
    """좌표계 정의.

    Attributes
    ----------
    names : tuple[str, ...]
        좌표 이름 (순서가 곧 인덱스 순서).
    system : str
        좌표계 유형. preset 사용 시 자동 설정, custom은 "generic".
    dim : int
        총 차원수 = len(names).
    """

    names: tuple[str, ...]
    system: str = "generic"

    def __init__(self, names: list[str] | tuple[str, ...], system: str = "generic"):
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "system", system)

    @property
    def dim(self) -> int:
        return len(self.names)

    @classmethod
    def preset(
        cls,
        system: str,
        dim: int | None = None,
        signature: tuple | None = None,
    ) -> Coordinates:
        """Preset 좌표계를 생성한다.

        Parameters
        ----------
        system : str
            "cartesian", "spherical", "cylindrical" 중 하나.
        dim : int or None
            공간 차원수. signature와 동시 지정 시 정합성 검사.
        signature : tuple or None
            (p, q) — p개의 +1, q개의 -1.
            또는 explicit sign tuple: (-1, 1, 1, 1).
            None이면 Riemannian (전부 +1).

        Returns
        -------
        Coordinates

        Examples
        --------
        >>> Coordinates.preset("spherical", dim=3)        # r, θ, φ
        >>> Coordinates.preset("spherical", signature=(3,1))  # t, r, θ, φ
        >>> Coordinates.preset("cartesian", dim=2)         # x, y
        """
        if system not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{system}'. "
                f"Available: {', '.join(_PRESETS.keys())}"
            )

        n_temporal = _count_temporal(signature)
        spatial_dim = _resolve_spatial_dim(system, dim, signature, n_temporal)

        if spatial_dim not in _PRESETS[system]:
            available = sorted(_PRESETS[system].keys())
            raise ValueError(
                f"Preset '{system}' does not support spatial dim={spatial_dim}. "
                f"Available: {available}"
            )

        spatial_names = list(_PRESETS[system][spatial_dim])
        temporal_names = [_TEMPORAL] * n_temporal
        all_names = temporal_names + spatial_names

        return cls(all_names, system=system)

    def __repr__(self) -> str:
        names_str = ", ".join(self.names)
        if self.system != "generic":
            return f"Coordinates({names_str} | {self.system})"
        return f"Coordinates({names_str})"

    def __len__(self) -> int:
        return self.dim

    def __getitem__(self, idx: int) -> str:
        return self.names[idx]


# ─── Signature 처리 ────────────────────────────────────────────

def parse_signature(
    signature: tuple | list | None,
    dim: int,
) -> tuple[int, ...]:
    """Signature를 부호 배열로 변환한다.

    Parameters
    ----------
    signature : tuple or None
        (p, q) 축약 또는 (-1, 1, 1, 1) 같은 explicit tuple.
        None → 전부 +1 (Riemannian).
    dim : int
        총 차원수.

    Returns
    -------
    tuple[int, ...]
        길이 dim의 부호 배열. 각 원소는 +1 또는 -1.

    Examples
    --------
    >>> parse_signature((3, 1), 4)
    (-1, 1, 1, 1)
    >>> parse_signature((-1, 1, 1, 1), 4)
    (-1, 1, 1, 1)
    >>> parse_signature(None, 3)
    (1, 1, 1)
    """
    if signature is None:
        return (1,) * dim

    sig = tuple(signature)

    # Explicit: 모든 원소가 ±1이고 길이가 dim이면 그대로
    if len(sig) == dim and all(s in (1, -1) for s in sig):
        return sig

    # 축약: (p, q) — p개의 +1, q개의 -1
    if len(sig) == 2:
        p, q = int(sig[0]), int(sig[1])
        if p + q != dim:
            raise ValueError(
                f"Signature ({p}, {q}) sums to {p + q}, but dim={dim}."
            )
        if p < 0 or q < 0:
            raise ValueError(f"Signature counts must be non-negative: ({p}, {q})")
        # Convention: -1이 앞에 (시간 좌표 먼저)
        return (-1,) * q + (1,) * p

    raise ValueError(
        f"Invalid signature {sig}. Expected (p, q) shorthand or "
        f"explicit sign tuple of length {dim}."
    )


def _count_temporal(signature: tuple | None) -> int:
    """Signature에서 시간 차원 (음의 고유값) 개수를 반환."""
    if signature is None:
        return 0

    sig = tuple(signature)

    # Explicit sign tuple: -1 개수
    if all(s in (1, -1) for s in sig):
        return sum(1 for s in sig if s == -1)

    # (p, q) 축약: q가 음의 고유값 수
    if len(sig) == 2:
        return int(sig[1])

    return 0


def _resolve_spatial_dim(
    system: str,
    dim: int | None,
    signature: tuple | None,
    n_temporal: int,
) -> int:
    """공간 차원수를 결정한다."""
    if dim is not None and signature is not None:
        # 둘 다 지정 → 정합성 검사
        total = _total_from_signature(signature)
        if total is not None and dim != total:
            raise ValueError(
                f"dim={dim} conflicts with signature {signature} "
                f"(total dim={total})."
            )
        return dim - n_temporal

    if signature is not None:
        total = _total_from_signature(signature)
        if total is not None:
            return total - n_temporal
        raise ValueError(f"Cannot determine dim from signature {signature}")

    if dim is not None:
        return dim - n_temporal

    raise ValueError("Either dim or signature must be specified for presets.")


def _total_from_signature(signature: tuple | None) -> int | None:
    """Signature에서 총 차원수를 추론."""
    if signature is None:
        return None
    sig = tuple(signature)
    if len(sig) == 2 and not all(s in (1, -1) for s in sig):
        return int(sig[0]) + int(sig[1])
    return len(sig)
