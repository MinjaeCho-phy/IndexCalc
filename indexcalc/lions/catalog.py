"""LIONS v2.5 Lie group catalog — pattern matching prototype registry.

v2.5 narrative shift (`LIONS/notes/v2_5_redirect.md`): the model no longer
re-runs invariance proofs at inference time. Instead it scores a
Lagrangian against a fixed catalog of (group, N) prototypes and returns
a ranked top-K. This file defines that catalog.

First round = 19 entries:
- U(1)
- U(N), N=2..5      (4)
- SU(N), N=2..5     (4)
- O(N), N=2..5      (4)
- SO(N), N=2..5     (4)
- Lorentz (D=4)
- Poincaré (D=4)

Sp(2N), Majorana spinors, conformal, exceptional groups: deferred to
v2.6+ (see ``v2_5_redirect.md`` §6).

Each entry carries enough metadata for two consumers:
1. The dataset generator (M2) — pick supported reps and invariant
   tensors to build candidate Lagrangians from.
2. The probe oracle (already exists) — ``build_groupspec`` materialises
   an executable ``GroupSpec`` (IndexSpace + Group + Generator) so the
   labeler can call ``apply_generator + simplify`` on each candidate.

The catalog is intentionally name-centric. M3's GNN reads each entry's
``label`` as the ranking output token; ``family`` selects readout heads
that share weights across N variants of the same family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from indexcalc.core.index import IndexSpace
from indexcalc.core.group import Group


@dataclass(frozen=True)
class CatalogEntry:
    """One (group, N or D) prototype in the catalog."""
    group_name: str          # template "U(N)", concrete "Lorentz" / "Poincare"
    N: int                   # classical N, spacetime D, or 1 for U(1)
    family: str              # "abelian" | "unitary" | "orthogonal" | "lorentz" | "poincare"
    supported_reps: tuple    # rep labels that fields in this group can carry
    invariants: tuple        # invariant tensor names available for enumeration

    @property
    def label(self) -> str:
        """Concrete ranking token: 'U(3)', 'SO(4)', 'Lorentz', 'Poincare'."""
        if "N" in self.group_name:
            return self.group_name.replace("N", str(self.N))
        return self.group_name


# ─── Entry factories ─────────────────────────────────────


def _u1_entry() -> CatalogEntry:
    return CatalogEntry(
        group_name="U(1)", N=1, family="abelian",
        supported_reps=("singlet", "+1", "-1", "+1/2", "-1/2"),
        invariants=(),
    )


def _classical_entry(group_name: str, N: int) -> CatalogEntry:
    """U(N), SU(N), O(N), SO(N) for N>=2."""
    if group_name in ("U(N)", "SU(N)"):
        family = "unitary"
        reps = ("singlet", "fund", "antifund", "adj")
        # SU(N) has ε_{i1..iN} (N-index totally antisym). U(N) doesn't.
        invariants = ("epsilon",) if group_name == "SU(N)" else ()
    elif group_name in ("O(N)", "SO(N)"):
        family = "orthogonal"
        reps = ("singlet", "vector")
        # SO(N) has ε_{i1..iN}, O(N) doesn't (reflection flips its sign).
        invariants = ("delta", "epsilon") if group_name == "SO(N)" else ("delta",)
    else:
        raise ValueError(f"unknown classical group {group_name!r}")
    return CatalogEntry(group_name, N, family, reps, invariants)


def _lorentz_entry() -> CatalogEntry:
    return CatalogEntry(
        group_name="Lorentz", N=4, family="lorentz",
        supported_reps=("singlet", "vector", "spinor", "L_spinor", "R_spinor"),
        invariants=("eta", "epsilon", "gamma"),
    )


def _poincare_entry() -> CatalogEntry:
    """Same reps as Lorentz; translation gen handled via explicit-t check
    in the oracle, not via a Lie-algebra apply_generator path."""
    return CatalogEntry(
        group_name="Poincare", N=4, family="poincare",
        supported_reps=("singlet", "vector", "spinor", "L_spinor", "R_spinor"),
        invariants=("eta", "epsilon", "gamma"),
    )


# ─── The catalog ─────────────────────────────────────────


CATALOG: tuple[CatalogEntry, ...] = (
    _u1_entry(),
    *(_classical_entry("U(N)",  N) for N in (2, 3, 4, 5)),
    *(_classical_entry("SU(N)", N) for N in (2, 3, 4, 5)),
    *(_classical_entry("O(N)",  N) for N in (2, 3, 4, 5)),
    *(_classical_entry("SO(N)", N) for N in (2, 3, 4, 5)),
    _lorentz_entry(),
    _poincare_entry(),
)
assert len(CATALOG) == 19, f"expected 19 catalog entries, got {len(CATALOG)}"


def all_labels() -> list[str]:
    """List of every catalog label in insertion order."""
    return [e.label for e in CATALOG]


def get(label: str) -> CatalogEntry:
    """Look up an entry by its concrete label ('U(3)', 'SO(4)', 'Lorentz')."""
    for e in CATALOG:
        if e.label == label:
            return e
    raise KeyError(
        f"no catalog entry labelled {label!r}. "
        f"Known: {all_labels()}"
    )


# ─── build_groupspec — materialise an entry ──────────────


def build_groupspec(entry: CatalogEntry, *, prefix: str = ""):
    """Materialise ``entry`` into an executable ``GroupSpec``.

    Parameters
    ----------
    entry : CatalogEntry
    prefix : str
        Optional prefix for the IndexSpace names. Use this when building
        multiple specs in the same process to avoid IndexSpace name
        collisions (the registry de-dupes by name).
    """
    from indexcalc.lions.probe import GroupSpec, classical_group_spec

    if entry.family == "abelian":
        return classical_group_spec("U(1)", 1, None)

    if entry.family == "unitary":
        # SU(N) and U(N): fund space = N-dim, adj computed inside helper.
        name = entry.label  # "U(3)", "SU(4)"
        space = IndexSpace(
            f"{prefix}{name.lower().replace('(', '_').replace(')', '')}_fund",
            dim=entry.N, indices="ijklmnpqrstuv",
        )
        return classical_group_spec(name, entry.N, space)

    if entry.family == "orthogonal":
        name = entry.label  # "O(3)", "SO(4)"
        space = IndexSpace(
            f"{prefix}{name.lower().replace('(', '_').replace(')', '')}_vec",
            dim=entry.N, indices="ijklmnpqrstuv", metric="delta",
        )
        return classical_group_spec(name, entry.N, space)

    if entry.family == "lorentz":
        return _build_lorentz_spec(prefix=prefix)

    if entry.family == "poincare":
        # First round: reuse the Lorentz spec; translation invariance is
        # handled by a separate explicit-t check at oracle time, not via
        # apply_generator. We mark it by name so downstream tooling can
        # branch.
        spec = _build_lorentz_spec(prefix=prefix)
        return GroupSpec(
            name="Poincare", group=spec.group,
            generator=spec.generator, dim=spec.dim + 4,  # +4 for translations
        )

    raise ValueError(f"unknown family {entry.family!r} in {entry!r}")


def _build_lorentz_spec(*, prefix: str = ""):
    """Lorentz group spec — D=4 spacetime + 4-dim Dirac + 2-dim Weyl reps."""
    from indexcalc.core.generator import make_lorentz_spinor_generator
    from indexcalc.lions.probe import GroupSpec

    frame = IndexSpace(
        f"{prefix}lorentz_frame", dim=4,
        indices="abcdefgh", metric="eta",
    )
    spinor = IndexSpace(
        f"{prefix}lorentz_dirac", dim=4,
        indices="αβγδεζ",
    )
    g = Group("Lorentz", dim=6, abelian=False)
    g.add_rep("singlet", dim=1)
    g.add_rep("vector", dim=4)
    g.add_rep("spinor", dim=4)
    g.add_rep("conj_spinor", dim=4, conjugate=True)
    g.add_rep("L_spinor", dim=2)
    g.add_rep("R_spinor", dim=2)
    gen = make_lorentz_spinor_generator(g, frame, spinor)
    return GroupSpec(name="Lorentz", group=g, generator=gen, dim=6)
