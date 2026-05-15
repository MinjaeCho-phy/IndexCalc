"""LIONS field registry — metadata layer above IndexCalc ``Tensor``.

IndexCalc core treats fields as ad-hoc ``Tensor(name, [...], reps={...})`` —
no mass dimension, no central registry. For dataset generation we need a
catalog the enumerator can iterate over.

A ``FieldSpec`` is metadata. The actual ``Tensor`` object is produced by
``FieldSpec.build(dummy_namer)`` which assigns fresh dummy indices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor


@dataclass(frozen=True)
class SlotSpec:
    """One slot of a field tensor.

    ``space`` is the IndexSpace the index lives in. ``position`` is
    ``"upper"`` or ``"lower"``. Slots are filled by ``FieldSpec.build`` with
    fresh dummy names from a caller-provided namer.
    """
    space: IndexSpace
    position: str

    def make_index(self, name: str) -> Index:
        return Index(name, self.space, self.position)


@dataclass(frozen=True)
class FieldSpec:
    """Catalog entry for one field.

    Parameters
    ----------
    name : str
        Tensor name used in IR (e.g., ``"H"`` for Higgs).
    slots : tuple[SlotSpec, ...]
        Index slots in order.
    reps : dict[str, str]
        Group-rep tags, same convention as ``Tensor.reps``.
    mass_dim : float
        Engineering mass dimension. Scalar=1, vector=1, fermion=3/2, etc.
    statistics : str
        ``"bosonic"`` or ``"fermionic"``.
    """
    name: str
    slots: tuple[SlotSpec, ...]
    reps: dict[str, str] = field(default_factory=dict)
    mass_dim: float = 1.0
    statistics: str = "bosonic"

    def build(self, namer: Callable[[], str]) -> Tensor:
        """Instantiate a ``Tensor`` with fresh dummy index names from ``namer``."""
        indices = [s.make_index(namer()) for s in self.slots]
        return Tensor(
            self.name,
            indices,
            reps=dict(self.reps),
            statistics=self.statistics,
        )


class FieldRegistry:
    """Ordered catalog of ``FieldSpec`` entries.

    The enumerator iterates ``registry.fields()`` for building-block alphabet.
    """

    def __init__(self):
        self._fields: dict[str, FieldSpec] = {}

    def add(self, spec: FieldSpec) -> FieldSpec:
        if spec.name in self._fields:
            raise ValueError(f"Field {spec.name!r} already registered")
        self._fields[spec.name] = spec
        return spec

    def get(self, name: str) -> FieldSpec:
        if name not in self._fields:
            raise KeyError(f"Field {name!r} not registered")
        return self._fields[name]

    def fields(self) -> list[FieldSpec]:
        return list(self._fields.values())

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __len__(self) -> int:
        return len(self._fields)
