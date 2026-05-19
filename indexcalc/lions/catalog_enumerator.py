"""LIONS v2.5 M2 — catalog-driven Lagrangian enumeration.

Helpers that turn a ``CatalogEntry`` into the inputs the existing
``enumerate_scalar_invariants`` expects: a ``FieldRegistry`` of
anonymized fields with the right reps and an invariant tensor alphabet
matching ``entry.invariants``.

Per family:
  abelian (U(1))    — singlet + charged scalar fields. No tensor alphabet.
  unitary (U/SU)    — fund + antifund Hermitian forms; SU adds N-index ε.
  orthogonal (O/SO) — vector fields with δ (+ N-index ε for SO).
  lorentz           — 4D spacetime + Dirac/Weyl spinor reps; η, ε_{μνρσ}, γ^μ.
  poincare          — same as lorentz; translation handled at oracle time.

Field names are anonymized to ``F1, F2, ..., Fk`` here. Random renames
across training epochs happen later in the dataset adapter (M3).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.lions.catalog import CatalogEntry, build_groupspec
from indexcalc.lions.fields import (
    FieldSpec, FieldRegistry, SlotSpec, InvariantTensorSpec,
)
from indexcalc.lions.enumerate import (
    enumerate_scalar_invariants, EnumeratorCaps, EnumeratedSample,
)
from indexcalc.lions.probe import GroupSpec


# ─── Setup bundle ────────────────────────────────────────


@dataclass
class CatalogSetup:
    """Materialised inputs an entry needs for one enumeration pass."""
    entry: CatalogEntry
    spec: GroupSpec
    primary_space: IndexSpace      # vector/fund/frame depending on family
    registry: FieldRegistry
    invariants: list[InvariantTensorSpec]


# ─── Family-specific setup builders ──────────────────────


def _setup_abelian(entry: CatalogEntry, *, prefix: str, n_fields: int) -> CatalogSetup:
    """U(1): scalar fields with assorted charges. No invariant tensors."""
    spec = build_groupspec(entry, prefix=prefix)
    # Pick a no-index IndexSpace just so the enumerator signature is happy.
    # U(1) fields have no spatial indices.
    null = IndexSpace(f"{prefix}u1_dummy", dim=1, indices="·", metric="")
    reg = FieldRegistry()
    charges = ("+1", "-1", "0", "+1/2", "-1/2")
    for i in range(n_fields):
        ch = charges[i % len(charges)]
        reg.add(FieldSpec(
            name=f"F{i+1}",
            slots=(),               # scalar — no slots
            reps={"U(1)": ch},
            mass_dim=1.0, statistics="bosonic",
        ))
    return CatalogSetup(entry, spec, null, reg, invariants=[])


def _setup_orthogonal(entry: CatalogEntry, *, prefix: str, n_fields: int) -> CatalogSetup:
    """O(N)/SO(N): vector fields + δ; SO adds N-index ε.

    Uses the same IndexSpace the spec was built on so reps match.
    """
    spec = build_groupspec(entry, prefix=prefix)
    label = entry.label
    space = IndexSpace(
        f"{prefix}{label.lower().replace('(', '_').replace(')', '')}_vec",
        dim=entry.N, indices="ijklmnpqrstuv", metric="delta",
    )
    reg = FieldRegistry()
    for i in range(n_fields):
        reg.add(FieldSpec(
            name=f"F{i+1}",
            slots=(SlotSpec(space, "upper"),),
            reps={label: "vector"},
            mass_dim=1.0, statistics="bosonic",
        ))

    invariants: list[InvariantTensorSpec] = []
    if "delta" in entry.invariants:
        invariants.append(InvariantTensorSpec(
            name="delta",
            slots=(SlotSpec(space, "lower"), SlotSpec(space, "lower")),
            symmetric_pairs=((0, 1),),
        ))
    if "epsilon" in entry.invariants:
        slots = tuple(SlotSpec(space, "lower") for _ in range(entry.N))
        antisym_pairs = tuple(
            (a, b) for a in range(entry.N) for b in range(a + 1, entry.N)
        )
        invariants.append(InvariantTensorSpec(
            name="epsilon",
            slots=slots, antisymmetric_pairs=antisym_pairs,
        ))
    return CatalogSetup(entry, spec, space, reg, invariants)


def _setup_unitary(entry: CatalogEntry, *, prefix: str, n_fields: int) -> CatalogSetup:
    """U(N)/SU(N): fund + antifund Hermitian pairs.

    Each field has one upper-fund slot; ``δ^i_j`` contracts fund↔fund via
    the standard like-position matcher inside the enumerator. SU(N)'s
    ε_{i1..iN} is an N-slot lower-fund antisym invariant.
    """
    spec = build_groupspec(entry, prefix=prefix)
    label = entry.label
    fund = IndexSpace(
        f"{prefix}{label.lower().replace('(', '_').replace(')', '')}_fund",
        dim=entry.N, indices="ijklmnpqrstuv",
    )
    reg = FieldRegistry()
    # Half upper-fund, half antifund (lower-fund). Mixing both lets the
    # enumerator emit Hermitian bilinears F̄_i F^i.
    for i in range(n_fields):
        if i % 2 == 0:
            slots = (SlotSpec(fund, "upper"),)
            rep = "fund"
        else:
            slots = (SlotSpec(fund, "lower"),)
            rep = "antifund"
        reg.add(FieldSpec(
            name=f"F{i+1}",
            slots=slots,
            reps={label: rep},
            mass_dim=1.0, statistics="bosonic",
        ))

    invariants: list[InvariantTensorSpec] = []
    if "epsilon" in entry.invariants:  # SU(N) only
        slots = tuple(SlotSpec(fund, "lower") for _ in range(entry.N))
        antisym_pairs = tuple(
            (a, b) for a in range(entry.N) for b in range(a + 1, entry.N)
        )
        invariants.append(InvariantTensorSpec(
            name="epsilon",
            slots=slots, antisymmetric_pairs=antisym_pairs,
        ))
    return CatalogSetup(entry, spec, fund, reg, invariants)


def _setup_lorentz_like(entry: CatalogEntry, *, prefix: str, n_fields: int) -> CatalogSetup:
    """Lorentz / Poincaré: D=4 spacetime + Dirac + Weyl. First round = vector
    fields only (spinor enumeration deferred — requires Fermi parity care
    + γ contractions; M2.2 follow-up)."""
    spec = build_groupspec(entry, prefix=prefix)
    label = entry.label
    st = IndexSpace(
        f"{prefix}{label.lower()}_st", dim=4,
        indices="μνρστλ", metric="eta",
    )
    reg = FieldRegistry()
    for i in range(n_fields):
        reg.add(FieldSpec(
            name=f"F{i+1}",
            slots=(SlotSpec(st, "upper"),),
            reps={label: "vector"},
            mass_dim=1.0, statistics="bosonic",
        ))
    invariants: list[InvariantTensorSpec] = []
    if "eta" in entry.invariants:
        invariants.append(InvariantTensorSpec(
            name="eta",
            slots=(SlotSpec(st, "lower"), SlotSpec(st, "lower")),
            symmetric_pairs=((0, 1),),
        ))
    if "epsilon" in entry.invariants:
        slots = tuple(SlotSpec(st, "lower") for _ in range(4))
        antisym_pairs = tuple(
            (a, b) for a in range(4) for b in range(a + 1, 4)
        )
        invariants.append(InvariantTensorSpec(
            name="epsilon",
            slots=slots, antisymmetric_pairs=antisym_pairs,
        ))
    return CatalogSetup(entry, spec, st, reg, invariants)


# ─── Public API ─────────────────────────────────────────


def setup_for_entry(
    entry: CatalogEntry, *, prefix: str = "", n_fields: int = 3,
) -> CatalogSetup:
    """Dispatch to the right family-specific setup."""
    family = entry.family
    if family == "abelian":
        return _setup_abelian(entry, prefix=prefix, n_fields=n_fields)
    if family == "unitary":
        return _setup_unitary(entry, prefix=prefix, n_fields=n_fields)
    if family == "orthogonal":
        return _setup_orthogonal(entry, prefix=prefix, n_fields=n_fields)
    if family in ("lorentz", "poincare"):
        return _setup_lorentz_like(entry, prefix=prefix, n_fields=n_fields)
    raise ValueError(f"unknown family {family!r}")


def enumerate_for_entry(
    entry: CatalogEntry,
    *,
    caps: Optional[EnumeratorCaps] = None,
    n_fields: int = 3,
    prefix: str = "",
) -> tuple[CatalogSetup, list[EnumeratedSample]]:
    """One-call helper: setup + enumerate for a single catalog entry."""
    setup = setup_for_entry(entry, prefix=prefix, n_fields=n_fields)
    caps = caps or EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
        max_invariants_per_kind=1, max_invariants_total=2,
    )
    samples = enumerate_scalar_invariants(
        setup.registry, spacetime=setup.primary_space, caps=caps,
        invariant_alphabet=setup.invariants,
    )
    return setup, samples
