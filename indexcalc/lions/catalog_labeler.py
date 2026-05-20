"""LIONS v2.5 M2.2 — structural multi-positive labeling.

Given an enumerated Lagrangian and the catalog entry it was generated
from, decide which *other* catalog entries this Lagrangian also "looks
like". The decision is structural: we walk the expression for invariant
tensors (δ, ε, η, γ), record their IndexSpace properties (dim, metric),
and match against each catalog entry's declared ``invariants``.

Rationale (`notes/v2_5_redirect.md` §6 Q3): full apply_generator on
foreign groups returns trivial ✓ for any L whose fields use index
spaces the foreign generator doesn't know about — label explosion.
Structural matching catches the cases the user actually cares about
(L = φ² → every group; L = δ_ij F^i F^j → O/SO of the same N; L = ε
ABC → SO of that N; L = ψ̄γψ → Lorentz/Poincaré) without that explosion.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv

from indexcalc.lions.catalog import CATALOG, CatalogEntry


# Tensor name → which (group_family, requires_metric) it implies.
# - "delta" + metric="delta"  → orthogonal
# - "epsilon" + metric="delta" → orthogonal (SO only)
# - "epsilon" + metric=""      → unitary (SU only — fund space)
# - "omega" + metric=""        → symplectic (Sp — antisym form, metric-less space)
# - "eta"   + metric="eta"   → lorentz/poincare
# - "eta_conf" + metric="conf" → conformal SO(d,2) (symmetric, indefinite)
# - "eta_dd" + metric="dd"   → O(D,D) T-duality (symmetric, split signature)
# - "gamma" + metric="eta"   → lorentz/poincare (Dirac γ^μ)
INVARIANT_TENSORS = ("delta", "epsilon", "omega", "eta", "eta_conf", "eta_dd", "gamma")


# ─── Tensor signature collection ────────────────────────


def collect_tensor_signature(expr: TensorExpr) -> set[tuple[str, int, str]]:
    """Walk ``expr`` and return a set of (tensor_name, index_dim, metric).

    Only *invariant* tensors are recorded (δ/ε/η/γ — names listed in
    ``INVARIANT_TENSORS``). Field tensors are ignored — they're labeled
    via reps separately.
    """
    sig: set[tuple[str, int, str]] = set()

    def visit(e):
        if isinstance(e, Tensor):
            if e.name in INVARIANT_TENSORS and e.indices:
                sp = e.indices[0].space
                # Enumerator quirk: when given an orthogonal IndexSpace
                # (metric="delta") as its `spacetime` arg, the like-position
                # matcher auto-inserts a tensor named "eta" that's really
                # a δ trace on that same space. Normalize back so the
                # tensor signature reflects the actual invariant being used.
                name = e.name
                if name == "eta" and sp.metric == "delta":
                    name = "delta"
                # Same quirk on a conformal (metric="conf") space → η^conf.
                elif name == "eta" and sp.metric == "conf":
                    name = "eta_conf"
                # Same quirk on an O(D,D) (metric="dd") space → η^dd.
                elif name == "eta" and sp.metric == "dd":
                    name = "eta_dd"
                # M5.B.3: slot count matters for ε (SU(N)'s ε is N-slot,
                # SO(N)'s ε is N-slot, Lorentz ε_μνρσ is 4-slot). The OOD
                # eval surfaced cases where the enumerator emits a 2-slot
                # ε on a dim=3 IndexSpace — the structural label was wrong
                # before this field was tracked.
                slot_count = len(e.indices)
                sig.add((name, sp.dim, sp.metric, slot_count))
        elif isinstance(e, (TensorProduct, TensorSum)):
            visit(e.left); visit(e.right)
        elif isinstance(e, ScalarMul):
            visit(e.expr)
        elif isinstance(e, PartialDeriv):
            visit(e.expr)
        elif isinstance(e, TimeDeriv):
            visit(e.expr)
        elif isinstance(e, ScalarFunction):
            visit(e.arg)
        # Tensor leaves with names outside INVARIANT_TENSORS or ZeroTensor: ignore.

    visit(expr)
    return sig


# ─── Field rep signature (M5.RB) ─────────────────────────


def collect_field_rep_signature(expr: TensorExpr) -> dict[str, set[str]]:
    """Return ``{group_name: set(rep_label)}`` for every field in ``expr``.

    Walks the same tree as ``collect_tensor_signature`` but only records
    field-side tensor leaves (``reps`` non-empty). Invariant tensors
    (``reps == {}``) are skipped. Singlet entries are dropped — they're
    trivial for compatibility checks.

    Used by the labeler to distinguish "L = φ² is everything" (no reps
    declared) from "L = (φ†φ)²" (fund × antifund — only U/SU families).
    """
    sig: dict[str, set[str]] = defaultdict(set)

    def visit(e):
        if isinstance(e, Tensor):
            if e.reps:
                for group, rep in e.reps.items():
                    if rep != "singlet":
                        sig[group].add(rep)
        elif isinstance(e, (TensorProduct, TensorSum)):
            visit(e.left); visit(e.right)
        elif isinstance(e, ScalarMul):
            visit(e.expr)
        elif isinstance(e, PartialDeriv):
            visit(e.expr)
        elif isinstance(e, TimeDeriv):
            visit(e.expr)
        elif isinstance(e, ScalarFunction):
            visit(e.arg)
        # ZeroTensor, unknown tensor leaves: skip.

    visit(expr)
    return dict(sig)


def _entry_partner_label(entry: CatalogEntry) -> str | None:
    """Same-N family partner whose rep set the entry shares.

    O(N)↔SO(N), U(N)↔SU(N) (N≥2), Lorentz↔Poincaré. Returns ``None`` for
    U(1) (abelian, no partner).
    """
    if entry.family == "orthogonal":
        if entry.label.startswith("SO("):
            return f"O({entry.N})"
        if entry.label.startswith("O("):
            return f"SO({entry.N})"
    elif entry.family == "unitary":
        if entry.label == "U(1)":
            return None
        if entry.label.startswith("SU("):
            return f"U({entry.N})"
        if entry.label.startswith("U("):
            return f"SU({entry.N})"
    elif entry.family == "lorentz":
        return "Poincare"
    elif entry.family == "poincare":
        return "Lorentz"
    return None


def _entry_compatible_with_rep_sig(
    entry: CatalogEntry, rep_sig: dict[str, set[str]],
) -> bool:
    """Closed-set rep matching, partner-aware.

    If ``rep_sig`` is non-empty, the entry (or its same-N partner) must
    appear in ``rep_sig`` and every declared rep must lie in
    ``entry.supported_reps``. Partner pairs share invariant tensors and
    rep sets (O(3)↔SO(3), SU(2)↔U(2), Lorentz↔Poincaré), so a label
    declared on either side qualifies both.
    """
    if not rep_sig:
        return True

    candidates = [entry.label]
    partner = _entry_partner_label(entry)
    if partner:
        candidates.append(partner)

    for cand in candidates:
        if cand in rep_sig:
            used = rep_sig[cand]
            return all(r in entry.supported_reps for r in used)
    return False


# ─── Per-entry compatibility ────────────────────────────


def _owned_space_signature(entry: CatalogEntry) -> tuple[int, str] | None:
    """The ``(dim, metric)`` of the index space ``entry``'s group acts on.

    Per-sector labeling (v3.1) judges an entry *only* by the invariant
    tensors living in this space; tensors in other spaces are singlets to
    this group and ignored. U(1) (abelian) transforms via charges, not an
    index space → ``None`` (no owned space; ``rep_sig`` alone gates it).

    Note the deliberate (dim, metric) collision between unitary fund and
    symplectic vector at equal dimension (e.g. SU(4) and Sp(4) both own
    ``(4, "")``): the two are separated downstream by the invariant-tensor
    *name* (ε ∈ SU only, Ω ∈ Sp only) and by ``rep_sig``.
    """
    fam = entry.family
    if fam == "abelian":
        return None
    if fam == "unitary":
        return (entry.N, "")            # fund space, metric-less
    if fam == "orthogonal":
        return (entry.N, "delta")
    if fam == "symplectic":
        return (2 * entry.N, "")        # 2N-dim vector, metric-less
    if fam == "conformal":
        return (entry.N + 2, "conf")    # SO(d,2): (d+2)-dim embedding space
    if fam == "split_orthogonal":
        return (2 * entry.N, "dd")      # O(D,D): 2D-dim doubled space
    if fam in ("lorentz", "poincare"):
        return (4, "eta")
    raise ValueError(f"unknown family {entry.family!r} in {entry!r}")


def _entry_compatible_with_sig(
    entry: CatalogEntry, sig: set[tuple[str, int, str, int]],
) -> bool:
    """Per-sector tensor compatibility (v3.1).

    Only tensors living in the entry's own index space (matched by
    ``(dim, metric)``) are judged: each must be one of the group's declared
    invariants, with the right slot count. Tensors in *other* spaces are
    singlets to this group and ignored — so a multi-sector Lagrangian
    (δ_ij A^iA^j + Ω_kl B^kB^l, or a single field ψ^{aα}_μ charged under
    several groups) is correctly compatible with each sector's group.
    Over-generation is held back by the per-group ``rep_sig`` check.
    """
    owned = _owned_space_signature(entry)
    if owned is None:
        return True  # abelian: no index space; rep_sig gates U(1)
    for (name, dim, metric, slot_count) in sig:
        if (dim, metric) != owned:
            continue  # foreign space — singlet to this group, ignore
        if name not in entry.invariants:
            return False
        # N-slot Levi-Civita / 2-slot symplectic slot-count checks.
        if entry.family in ("orthogonal", "unitary"):
            if name == "epsilon" and slot_count != entry.N:
                return False
        elif entry.family == "symplectic":
            if slot_count != 2:
                return False
        elif entry.family == "conformal":
            # ε on SO(d,2) is the (d+2)-slot Levi-Civita of the embedding space.
            if name == "epsilon" and slot_count != entry.N + 2:
                return False
        elif entry.family == "split_orthogonal":
            # η^dd is the symmetric 2-index O(D,D) metric (no ε on this space).
            if name == "eta_dd" and slot_count != 2:
                return False
        elif entry.family in ("lorentz", "poincare"):
            if name == "epsilon" and slot_count != 4:
                return False
    return True


# ─── Public labeling API ────────────────────────────────


def label_lagrangian(
    expr: TensorExpr,
    primary_entry: CatalogEntry,
    *,
    catalog: Iterable[CatalogEntry] = CATALOG,
) -> dict[str, bool]:
    """Return ``{entry.label: bool}`` for every catalog entry.

    Algorithm:
    - Primary entry: always True (enumerator-built).
    - Empty signature (no invariant tensors): all entries True
      (the user's "L = φ² is everything" case).
    - Non-empty signature: each entry True iff every recorded tensor
      is in ``entry.invariants`` AND its dim/metric matches the family
      conventions.
    """
    sig = collect_tensor_signature(expr)
    rep_sig = collect_field_rep_signature(expr)
    labels: dict[str, bool] = {}
    for entry in catalog:
        if entry.label == primary_entry.label:
            labels[entry.label] = True
            continue
        # Trivial-scalar shortcut: no invariant tensors AND no fields
        # declare a non-singlet rep → every group trivially matches.
        if not sig and not rep_sig:
            labels[entry.label] = True
            continue
        tensor_ok = _entry_compatible_with_sig(entry, sig) if sig else True
        rep_ok = _entry_compatible_with_rep_sig(entry, rep_sig)
        labels[entry.label] = tensor_ok and rep_ok
    return labels
