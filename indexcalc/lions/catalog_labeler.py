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
# - "eta"   + metric="eta"   → lorentz/poincare
# - "gamma" + metric="eta"   → lorentz/poincare (Dirac γ^μ)
INVARIANT_TENSORS = ("delta", "epsilon", "eta", "gamma")


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
                sig.add((name, sp.dim, sp.metric))
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


# ─── Per-entry compatibility ────────────────────────────


def _entry_compatible_with_sig(
    entry: CatalogEntry, sig: set[tuple[str, int, str]],
) -> bool:
    """Is every tensor in ``sig`` consistent with ``entry``?

    All-or-nothing: a single foreign tensor disqualifies the entry.
    """
    for (name, dim, metric) in sig:
        if name not in entry.invariants:
            return False
        # Family-specific dim / metric checks.
        if entry.family == "orthogonal":
            if metric != "delta" or dim != entry.N:
                return False
        elif entry.family == "unitary":
            # ε on fund space only; metric=="" for fund.
            # δ on fund isn't in entry.invariants for unitary, so name match
            # above already filters δ out for U/SU.
            if dim != entry.N:
                return False
            if name == "epsilon" and metric != "":
                return False
        elif entry.family in ("lorentz", "poincare"):
            if metric != "eta" or dim != 4:
                return False
        elif entry.family == "abelian":
            # U(1) has no tensor invariants → name not in entry.invariants
            # for any tensor, so we never get here when sig is non-empty.
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
    labels: dict[str, bool] = {}
    for entry in catalog:
        if entry.label == primary_entry.label:
            labels[entry.label] = True
            continue
        if not sig:
            labels[entry.label] = True
        else:
            labels[entry.label] = _entry_compatible_with_sig(entry, sig)
    return labels
