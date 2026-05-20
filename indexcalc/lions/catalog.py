"""LIONS v2.5 Lie group catalog — pattern matching prototype registry.

v2.5 narrative shift (`LIONS/notes/v2_5_redirect.md`): the model no longer
re-runs invariance proofs at inference time. Instead it scores a
Lagrangian against a fixed catalog of (group, N) prototypes and returns
a ranked top-K. This file defines that catalog.

v3.3 round = 29 entries:
- U(1)
- U(N), N=2..5      (4)
- SU(N), N=2..5     (4)
- O(N), N=2..5      (4)
- SO(N), N=2..5     (4)
- Sp(2N), rank N=2..5 → Sp(4),Sp(6),Sp(8),Sp(10)  (4)  [v3.0]
- SO(d,2) conformal, d=2,3,4 → SO(2,2),SO(3,2),SO(4,2)  (3)  [v3.2]
- O(D,D) T-duality, D=2,3,4 → O(2,2),O(3,3),O(4,4)  (3)  [v3.3]
- Lorentz (D=4)
- Poincaré (D=4)

Sp(2N) added v3.0; conformal SO(d,2) added v3.2 and O(D,D) added v3.3 (both
embedding-space formalism — orthogonal vector rep on a higher-dim space with
an indefinite metric, like Lorentz=SO(1,3)). O(D,D) is the NS-NS T-duality
group on a D-torus and the prototype seed for the hidden-symmetry track.
Majorana (a reality condition, not a Lie group) and exceptional E_d:
deferred. See ``v3_catalog_expansion.md`` / ``v3_3_odd_catalog.md``.

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
        """Concrete ranking token: 'U(3)', 'SO(4)', 'Sp(6)', 'Lorentz'."""
        if "2N" in self.group_name:  # Sp(2N): N stores rank → dim 2N
            return self.group_name.replace("2N", str(2 * self.N))
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
    """U(N), SU(N), O(N), SO(N) for N>=2; Sp(2N) where N is the rank."""
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
    elif group_name == "Sp(2N)":
        # N = symplectic rank → vector rep dim 2N. Preserves antisymmetric Ω.
        family = "symplectic"
        reps = ("singlet", "vector")
        invariants = ("omega",)
    else:
        raise ValueError(f"unknown classical group {group_name!r}")
    return CatalogEntry(group_name, N, family, reps, invariants)


def _conformal_entry(d: int) -> CatalogEntry:
    """Conformal group SO(d,2), embedding-space formalism on (d+2)-dim.

    A genuine Lie group (unlike Majorana, a reality condition), handled like
    Lorentz=SO(1,3): orthogonal vector rep on the (d+2)-dim embedding space
    with an indefinite (d,2) metric. ``N`` stores the spacetime dim d; the
    embedding/vector dim is d+2. Distinguished from Euclidean SO(d+2) (δ
    metric) and Sp(d+2) (Ω) by the ``conf`` metric on its index space.
    """
    return CatalogEntry(
        group_name=f"SO({d},2)", N=d, family="conformal",
        supported_reps=("singlet", "vector"),
        invariants=("eta_conf", "epsilon"),
    )


def _odd_entry(D: int) -> CatalogEntry:
    """T-duality group O(D,D), embedding/doubled-space formalism on 2D-dim.

    A genuine Lie group handled like the conformal case: orthogonal vector
    rep on the 2D-dim doubled space with the indefinite split-signature
    ``dd`` metric. ``N`` stores D; the vector dim is 2D. Carries only the
    metric (no ε) — like O(N), and matching the full T-duality group O(D,D)
    (det=±1). Distinguished from Euclidean SO(2D) (δ), Sp(2D) (Ω), and
    conformal SO(2D−2,2) (η^conf) at equal dimension by the ``dd`` metric.
    """
    return CatalogEntry(
        group_name=f"O({D},{D})", N=D, family="split_orthogonal",
        supported_reps=("singlet", "vector"),
        invariants=("eta_dd",),
    )


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
    # Sp(2N) v3.0: rank N=2..5 → Sp(4), Sp(6), Sp(8), Sp(10).
    # N=1 (Sp(2)≅SU(2)) omitted to avoid degeneracy with the SU(2) entry.
    *(_classical_entry("Sp(2N)", N) for N in (2, 3, 4, 5)),
    # Conformal SO(d,2) v3.2: spacetime d=2,3,4 → embedding dim 4,5,6.
    *(_conformal_entry(d) for d in (2, 3, 4)),
    # O(D,D) T-duality v3.3: D=2,3,4 → doubled dim 4,6,8.
    *(_odd_entry(D) for D in (2, 3, 4)),
    _lorentz_entry(),
    _poincare_entry(),
)
assert len(CATALOG) == 29, f"expected 29 catalog entries, got {len(CATALOG)}"


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

    if entry.family == "symplectic":
        name = entry.label  # "Sp(4)", "Sp(6)"
        vector_dim = 2 * entry.N  # entry.N = rank
        space = IndexSpace(
            f"{prefix}{name.lower().replace('(', '_').replace(')', '')}_vec",
            dim=vector_dim, indices="ijklmnpqrstuv", metric="omega",
        )
        return classical_group_spec(name, vector_dim, space)

    if entry.family == "conformal":
        return _build_conformal_spec(entry, prefix=prefix)

    if entry.family == "split_orthogonal":
        return _build_odd_spec(entry, prefix=prefix)

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


def _build_conformal_spec(entry: CatalogEntry, *, prefix: str = ""):
    """Conformal SO(d,2) spec — orthogonal vector rep on the (d+2)-dim
    embedding space with the indefinite ``conf`` metric. Generator reuses
    ``make_o_n_generator`` (the rep action δV^A = M^A_B V^B is signature-
    independent; the metric only enters invariant contractions)."""
    from indexcalc.core.generator import make_o_n_generator
    from indexcalc.lions.probe import GroupSpec

    name = entry.label  # "SO(4,2)"
    emb_dim = entry.N + 2
    space = IndexSpace(
        f"{prefix}{name.lower().replace('(', '_').replace(')', '').replace(',', '_')}_vec",
        dim=emb_dim, indices="ABCDEFGHIJKL", metric="conf",
    )
    dim = emb_dim * (emb_dim - 1) // 2  # SO(d,2): (d+2)(d+1)/2
    g = Group(name, dim=dim, abelian=False)
    g.add_rep("vector", dim=emb_dim)
    g.add_rep("singlet", dim=1)
    gen = make_o_n_generator(g, space)
    return GroupSpec(name=name, group=g, generator=gen, dim=dim)


def _build_odd_spec(entry: CatalogEntry, *, prefix: str = ""):
    """O(D,D) spec — orthogonal vector rep on the 2D-dim doubled space with
    the indefinite split-signature ``dd`` metric. Generator reuses
    ``make_o_n_generator`` (δV^M = M^M_N V^N is signature-independent; the
    metric only enters invariant contractions) — exactly as conformal."""
    from indexcalc.core.generator import make_o_n_generator
    from indexcalc.lions.probe import GroupSpec

    name = entry.label  # "O(4,4)"
    doubled_dim = 2 * entry.N
    space = IndexSpace(
        f"{prefix}{name.lower().replace('(', '_').replace(')', '').replace(',', '_')}_vec",
        dim=doubled_dim, indices="ABCDEFGHIJKL", metric="dd",
    )
    dim = doubled_dim * (doubled_dim - 1) // 2  # so(D,D): 2D(2D-1)/2
    g = Group(name, dim=dim, abelian=False)
    g.add_rep("vector", dim=doubled_dim)
    g.add_rep("singlet", dim=1)
    gen = make_o_n_generator(g, space)
    return GroupSpec(name=name, group=g, generator=gen, dim=dim)


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
