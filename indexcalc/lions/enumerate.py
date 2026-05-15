"""Forward generation — invariant monomial enumerator (B0 scope).

Given a FieldRegistry of scalar fields (no fermions, no vector fields),
enumerate gauge/Lorentz-invariant monomials up to caps on field counts
and derivative counts. Each candidate is fed through ``simplify`` to
drop trivially zero combinations (Bose × antisym etc.).

Algorithm (see notes/d2_enumerator_algorithm.md):

  1. Multiset enum: pick (count_f, partial_count_f) per field.
  2. Build factor list with fresh dummy index names.
  3. Per-IndexSpace perfect-matching of open indices, inserting
     ε/η when a pair has like positions.
  4. simplify() each candidate; drop ZeroTensor.
  5. canonical_form_modulo_dummies for de-duplication.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import Iterator
import itertools

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorExpr, TensorProduct
from indexcalc.core.deriv import PartialDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import (
    simplify,
    canonical_form_modulo_dummies,
)
from indexcalc.lions.fields import FieldSpec, FieldRegistry
from indexcalc.lions.builders import (
    make_eta,
    make_epsilon_su2,
    make_epsilon_su2_upper,
)


@dataclass(frozen=True)
class EnumeratorCaps:
    max_field_total: int = 4
    max_per_field: int = 4
    max_partials_total: int = 4
    max_partials_per_field: int = 2
    max_contractions_per_pattern: int = 1000


@dataclass
class EnumeratedSample:
    expr: TensorExpr
    mass_dim: float
    field_counts: dict[str, int]
    partial_count: int


# ─── Helpers ──────────────────────────────────────────────


_DUMMY_POOL = (
    "ijklmnpqrstuv"           # SU(2)-fund-ish
    "αβγδεζηθικλμνξ"          # spinor/Greek
    "abcdefghABCDEFGH"        # adjoint-ish
)


class _DummyNamer:
    """Per-multiset fresh-name generator. Names are unique within a
    single enumeration pass."""

    def __init__(self):
        self._iter = itertools.count()

    def __call__(self) -> str:
        n = next(self._iter)
        if n < len(_DUMMY_POOL):
            return _DUMMY_POOL[n]
        return f"d{n}"


def _multiset_choices(
    fields: list[FieldSpec], caps: EnumeratorCaps,
) -> Iterator[tuple[dict[str, int], dict[str, int]]]:
    """Yield (count_f, partial_count_f) dicts satisfying caps.

    Skips the empty multiset (no fields)."""
    names = [f.name for f in fields]
    # All count combinations
    count_ranges = [range(0, caps.max_per_field + 1) for _ in names]
    for counts in itertools.product(*count_ranges):
        total = sum(counts)
        if total == 0 or total > caps.max_field_total:
            continue
        # All partial-count combinations consistent with counts
        partial_ranges = [
            range(0, c * caps.max_partials_per_field + 1) for c in counts
        ]
        for partials in itertools.product(*partial_ranges):
            if sum(partials) > caps.max_partials_total:
                continue
            yield (
                dict(zip(names, counts)),
                dict(zip(names, partials)),
            )


def _distribute_partials(
    count: int, partials: int, per_cap: int,
) -> Iterator[tuple[int, ...]]:
    """Yield non-increasing partial-distributions across ``count`` instances
    summing to ``partials`` (each ≤ per_cap). Non-increasing → drop
    permutation-equivalent distributions."""
    if count == 0:
        if partials == 0:
            yield tuple()
        return
    # Generate all weakly-decreasing tuples
    def _rec(remaining: int, slots_left: int, max_val: int):
        if slots_left == 0:
            if remaining == 0:
                yield tuple()
            return
        upper = min(remaining, max_val, per_cap)
        # Lower bound: each slot ≥ 0, but to keep weakly decreasing the
        # smallest entry constraint comes from remaining/slots_left ceiling.
        # Allow free choice; non-increasing filter applied by max_val.
        for k in range(upper, -1, -1):
            # Ensure remaining-k can fit in (slots_left-1) slots with max k
            if (slots_left - 1) * k < remaining - k:
                continue
            for tail in _rec(remaining - k, slots_left - 1, k):
                yield (k,) + tail

    yield from _rec(partials, count, per_cap)


def _build_factors(
    fields: list[FieldSpec],
    counts: dict[str, int],
    partials_per_instance: dict[str, list[int]],
    namer: _DummyNamer,
    spacetime: IndexSpace,
) -> tuple[list[TensorExpr], list[tuple[Index, int]]]:
    """Build the factor list for one multiset + partial-distribution.

    Returns (factors, open_indices) where open_indices is a list of
    (Index, factor_position) — the slot position is unused here, kept for
    debugging.
    """
    factors: list[TensorExpr] = []
    open_indices: list[tuple[Index, int]] = []

    for spec in fields:
        c = counts[spec.name]
        partials_list = partials_per_instance[spec.name]
        for inst_idx in range(c):
            tensor = spec.build(namer)
            n_partials = partials_list[inst_idx]
            current: TensorExpr = tensor
            # Record original tensor slots as open
            for idx in tensor.indices:
                open_indices.append((idx, len(factors)))
            # Apply partials, each adds a new lower spacetime index
            for _ in range(n_partials):
                deriv_idx = Index(namer(), spacetime, "lower")
                current = PartialDeriv(current, deriv_idx)
                open_indices.append((deriv_idx, len(factors)))
            factors.append(current)
    return factors, open_indices


def _group_open_by_space(
    open_indices: list[tuple[Index, int]],
) -> dict[IndexSpace, list[Index]]:
    """Group open indices by IndexSpace."""
    by_space: dict[IndexSpace, list[Index]] = {}
    for idx, _ in open_indices:
        by_space.setdefault(idx.space, []).append(idx)
    return by_space


def _perfect_matchings(items: list) -> Iterator[list[tuple]]:
    """All perfect matchings of an even-length list (as list of pairs)."""
    if len(items) == 0:
        yield []
        return
    if len(items) % 2 != 0:
        return  # no perfect matching
    first = items[0]
    for j in range(1, len(items)):
        partner = items[j]
        rest = items[1:j] + items[j + 1:]
        for tail in _perfect_matchings(rest):
            yield [(first, partner)] + tail


def _matching_to_factors_and_renames(
    matching: list[tuple[Index, Index]],
    space: IndexSpace,
    namer: _DummyNamer,
) -> tuple[list[Tensor], dict[str, str]]:
    """Convert a matching on indices of one space to:
       - extra invariant-tensor factors (ε, η) for like-position pairs.
       - a rename map for direct (upper, lower) Einstein contractions.

    Naming convention: if both indices are upper, insert ε^{ij}; if both
    lower, insert ε_{ij} (SU(2)) or η_{μν} (spacetime). For (upper, lower)
    we rename one to the other (Einstein contraction).
    """
    extra: list[Tensor] = []
    renames: dict[str, str] = {}

    for a, b in matching:
        if a.position != b.position:
            # Direct Einstein contraction — pick the lower-named one to win
            # so the contraction is unambiguous.
            # We rename one of them to the other's name. Pick deterministically.
            up = a if a.position == "upper" else b
            lo = b if a.position == "upper" else a
            # Rename lo.name → up.name (so the lower one carries upper's dummy)
            renames[lo.name] = up.name
            continue

        # Like-position pair: the invariant tensor must carry the OPPOSITE
        # position so Einstein convention contracts it against both hosts.
        # (a, b both lower) → insert tensor with both UPPER, same names.
        # (a, b both upper) → insert tensor with both LOWER, same names.
        inv_position = "upper" if a.position == "lower" else "lower"

        if a.position == "lower":
            if space.metric:
                tensor = Tensor(
                    "eta",
                    [Index(a.name, space, inv_position),
                     Index(b.name, space, inv_position)],
                    symmetric_pairs=[(0, 1)], reps={},
                )
            else:
                tensor = make_epsilon_su2_upper(space, a.name, b.name)
            extra.append(tensor)
        else:
            if space.metric:
                tensor = Tensor(
                    "eta",
                    [Index(a.name, space, inv_position),
                     Index(b.name, space, inv_position)],
                    symmetric_pairs=[(0, 1)], reps={},
                )
            else:
                tensor = make_epsilon_su2(space, a.name, b.name)
            extra.append(tensor)

    return extra, renames


def _apply_renames(expr: TensorExpr, renames: dict[str, str]) -> TensorExpr:
    """Apply name substitution to all indices in the tree.

    Defers to indexcalc.core.simplify.rename_index for the heavy work.
    """
    if not renames:
        return expr
    from indexcalc.core.simplify import rename_index
    return rename_index(expr, renames)


def _product(factors: list[TensorExpr]) -> TensorExpr:
    if not factors:
        raise ValueError("empty factor list")
    out: TensorExpr = factors[0]
    for f in factors[1:]:
        out = TensorProduct(out, f)
    return out


# ─── Main enumerator ──────────────────────────────────────


def enumerate_scalar_invariants(
    registry: FieldRegistry,
    *,
    spacetime: IndexSpace,
    caps: EnumeratorCaps = EnumeratorCaps(),
) -> list[EnumeratedSample]:
    """Enumerate gauge-invariant scalar monomials over the registered
    fields, deduplicated by canonical form and stripped of zeros.

    Currently scoped for B0: scalar fields only, no fermion/vector. The
    matching loop walks every IndexSpace; un-paired open indices fail the
    perfect-matching check and that contraction is dropped.
    """
    fields = registry.fields()
    seen: dict[tuple, EnumeratedSample] = {}

    for counts, total_partials in _multiset_choices(fields, caps):
        # For each field, enumerate distributions of partials across instances.
        per_instance_distros = {
            spec.name: list(_distribute_partials(
                counts[spec.name],
                total_partials[spec.name],
                caps.max_partials_per_field,
            ))
            for spec in fields
        }
        # If any field has no valid distribution, skip this multiset.
        if any(len(d) == 0 for d in per_instance_distros.values()):
            continue

        # Cartesian product of per-field distributions
        for distros in itertools.product(*per_instance_distros.values()):
            partials_per_instance = {
                spec.name: list(distros[i])
                for i, spec in enumerate(fields)
            }
            namer = _DummyNamer()
            factors, open_idxs = _build_factors(
                fields, counts, partials_per_instance, namer, spacetime,
            )
            by_space = _group_open_by_space(open_idxs)

            # Generate per-space matchings; Cartesian product across spaces
            per_space_matchings = []
            ok = True
            for space, idxs in by_space.items():
                if len(idxs) % 2 != 0:
                    ok = False
                    break
                matchings = list(_perfect_matchings(idxs))
                per_space_matchings.append((space, matchings))
            if not ok:
                continue

            # Bail if combinatorial explosion
            n_matches = 1
            for _, ms in per_space_matchings:
                n_matches *= max(1, len(ms))
            if n_matches > caps.max_contractions_per_pattern:
                continue

            for matching_choice in itertools.product(
                *[ms for _, ms in per_space_matchings]
            ):
                all_extras: list[Tensor] = []
                all_renames: dict[str, str] = {}
                for (space, _), matching in zip(
                    per_space_matchings, matching_choice
                ):
                    extras, renames = _matching_to_factors_and_renames(
                        matching, space, namer,
                    )
                    all_extras.extend(extras)
                    # Merge renames — collisions imply two indices being
                    # forced into different targets, which means this
                    # matching is ill-formed; skip.
                    collide = False
                    for k, v in renames.items():
                        if k in all_renames and all_renames[k] != v:
                            collide = True
                            break
                        all_renames[k] = v
                    if collide:
                        break
                else:
                    full_factors = factors + all_extras
                    expr = _product(full_factors)
                    expr = _apply_renames(expr, all_renames)
                    # Drop zeros via simplifier
                    simpl = simplify(expr)
                    if isinstance(simpl, ZeroTensor):
                        continue
                    key = canonical_form_modulo_dummies(simpl)
                    if key in seen:
                        continue
                    # Mass dim metadata (downstream use)
                    mdim = sum(
                        counts[s.name] * s.mass_dim for s in fields
                    ) + sum(total_partials.values())
                    seen[key] = EnumeratedSample(
                        expr=simpl,
                        mass_dim=float(mdim),
                        field_counts=dict(counts),
                        partial_count=sum(total_partials.values()),
                    )

    return list(seen.values())
