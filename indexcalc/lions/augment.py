"""LIONS dataset augmentation — negative + positive sample synthesis.

D8b (negative): take an invariant ``LabeledSample`` and mutate the rep
tag of one (field, group) pair. Re-run the oracle to assign a new
label dict. Provenance is set to ``"negative"``. Resulting labels may
be partial-False (still invariant under some groups but not others) or
all-False — both are useful for ML training as discriminative signal.

Symmetry-breaking by additive sums and field-redefinition style
negatives are deferred (D8b would explode otherwise; v1 ships
wrong-rep only).

D8c (positive augmentation) lives in this module too: ``permute_dummy_indices``,
``swap_factor_order``, ``scale_by`` for ``LabeledSample``. Each preserves
``labels`` since the oracle is invariant under those mutations.
"""

from __future__ import annotations
from dataclasses import dataclass

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import rename_index
from indexcalc.core.generator import Generator

from indexcalc.lions.dataset import LabeledSample, label_expression


# ─── Tree walker: mutate Tensor.reps ────────────────────


def mutate_field_reps(
    expr: TensorExpr,
    field_name: str,
    group_name: str,
    new_rep: str,
) -> TensorExpr:
    """Walk ``expr`` and, on every ``Tensor`` whose ``name == field_name``
    that carries a key ``group_name`` in ``reps``, swap to ``new_rep``.

    All other Tensor metadata (indices, antisym/sym/traceless/transverse,
    statistics) is preserved bit-for-bit. Non-matching Tensors and
    container nodes are reconstructed unchanged.
    """
    if isinstance(expr, Tensor):
        if expr.name != field_name or group_name not in expr.reps:
            return expr
        new_reps = dict(expr.reps)
        new_reps[group_name] = new_rep
        return Tensor(
            expr.name, list(expr.indices),
            antisymmetric_pairs=list(expr.antisymmetric_pairs),
            symmetric_pairs=list(expr.symmetric_pairs),
            traceless=list(expr.traceless),
            transverse=list(expr.transverse),
            reps=new_reps,
            statistics=expr.statistics,
        )
    if isinstance(expr, TensorProduct):
        return TensorProduct(
            mutate_field_reps(expr.left, field_name, group_name, new_rep),
            mutate_field_reps(expr.right, field_name, group_name, new_rep),
        )
    if isinstance(expr, TensorSum):
        return TensorSum(
            mutate_field_reps(expr.left, field_name, group_name, new_rep),
            mutate_field_reps(expr.right, field_name, group_name, new_rep),
        )
    if isinstance(expr, ScalarMul):
        return ScalarMul(
            expr.scalar,
            mutate_field_reps(expr.expr, field_name, group_name, new_rep),
        )
    if isinstance(expr, PartialDeriv):
        return PartialDeriv(
            mutate_field_reps(expr.expr, field_name, group_name, new_rep),
            expr.deriv_index,
        )
    if isinstance(expr, ZeroTensor):
        return expr
    return expr


# ─── Negative synthesis ────────────────────────────────


@dataclass(frozen=True)
class RepMutation:
    """One wrong-rep mutation spec.

    Apply via ``apply_rep_mutation`` — that re-labels via the oracle.
    """
    field_name: str
    group_name: str
    new_rep: str


def apply_rep_mutation(
    sample: LabeledSample,
    mutation: RepMutation,
    generators: dict[str, Generator],
) -> LabeledSample:
    """Apply one ``RepMutation`` and return a ``LabeledSample`` with
    freshly oracle-derived labels and ``provenance="negative"``.
    """
    new_expr = mutate_field_reps(
        sample.expr, mutation.field_name, mutation.group_name, mutation.new_rep,
    )
    new_labels = label_expression(new_expr, generators)
    return LabeledSample(
        expr=new_expr,
        labels=new_labels,
        mass_dim=sample.mass_dim,
        field_counts=dict(sample.field_counts),
        partial_count=sample.partial_count,
        invariant_counts=dict(sample.invariant_counts),
        provenance="negative",
    )


def enumerate_wrong_rep_negatives(
    samples: list[LabeledSample],
    generators: dict[str, Generator],
    mutations: list[RepMutation],
    *,
    require_label_change: bool = True,
) -> list[LabeledSample]:
    """For each (sample, mutation) pair build a negative ``LabeledSample``.

    Parameters
    ----------
    require_label_change
        If True (default), drop mutations that don't actually change the
        label dict — those are uninformative duplicates. If False, keep
        all results (useful for debugging or measuring oracle stability).
    """
    out: list[LabeledSample] = []
    for s in samples:
        for m in mutations:
            new_s = apply_rep_mutation(s, m, generators)
            if require_label_change and new_s.labels == s.labels:
                continue
            out.append(new_s)
    return out


# ─── Positive augmentation (D8c building blocks) ────────


def permute_dummy_indices(
    sample: LabeledSample, mapping: dict[str, str],
) -> LabeledSample:
    """Rename dummy indices via ``mapping``. Labels and provenance
    semantically unchanged — but provenance updated to ``"augmented"``
    to track that the IR was modified.
    """
    new_expr = rename_index(sample.expr, mapping)
    return LabeledSample(
        expr=new_expr,
        labels=dict(sample.labels),
        mass_dim=sample.mass_dim,
        field_counts=dict(sample.field_counts),
        partial_count=sample.partial_count,
        invariant_counts=dict(sample.invariant_counts),
        provenance="augmented",
    )


def swap_top_product(sample: LabeledSample) -> LabeledSample:
    """If ``sample.expr`` is a ``TensorProduct(L, R)``, swap to
    ``TensorProduct(R, L)``. For bosonic fields this preserves the IR's
    semantic value. Idempotent on non-product expressions (returns the
    same sample, provenance updated)."""
    expr = sample.expr
    if isinstance(expr, TensorProduct):
        new_expr = TensorProduct(expr.right, expr.left)
    else:
        new_expr = expr
    return LabeledSample(
        expr=new_expr,
        labels=dict(sample.labels),
        mass_dim=sample.mass_dim,
        field_counts=dict(sample.field_counts),
        partial_count=sample.partial_count,
        invariant_counts=dict(sample.invariant_counts),
        provenance="augmented",
    )


def scale_by(sample: LabeledSample, c) -> LabeledSample:
    """Wrap ``sample.expr`` in ``ScalarMul(c, ...)``. Labels preserved
    (linearity of generator action — a nonzero scalar doesn't change
    whether δ(L) = 0). Provenance → "augmented".
    """
    if c == 0:
        # ScalarMul(0, X) collapses to ZeroTensor — not useful.
        raise ValueError("scale_by(0) collapses the sample to ZeroTensor")
    new_expr = ScalarMul(c, sample.expr)
    return LabeledSample(
        expr=new_expr,
        labels=dict(sample.labels),
        mass_dim=sample.mass_dim,
        field_counts=dict(sample.field_counts),
        partial_count=sample.partial_count,
        invariant_counts=dict(sample.invariant_counts),
        provenance="augmented",
    )


# ─── Orchestrator: expand_dataset ──────────────────────


_DEFAULT_SCALES = (-1.0, 0.5, 2.0)


def augment_sample(
    sample: LabeledSample,
    *,
    include_swap: bool = True,
    scales: tuple = _DEFAULT_SCALES,
) -> list[LabeledSample]:
    """Generate label-preserving variants of one sample.

    Returns the original first, then (optional) a top-product swap,
    then a ScalarMul wrap for each scalar in ``scales``. All variants
    share the same label dict as the input by construction.
    """
    out: list[LabeledSample] = [sample]
    if include_swap and isinstance(sample.expr, TensorProduct):
        out.append(swap_top_product(sample))
    for c in scales:
        if c == 0:
            continue
        out.append(scale_by(sample, c))
    return out


def expand_dataset(
    samples: list[LabeledSample],
    *,
    include_swap: bool = True,
    scales: tuple = _DEFAULT_SCALES,
) -> list[LabeledSample]:
    """Expand each sample via ``augment_sample``; concatenate."""
    return [v for s in samples for v in augment_sample(
        s, include_swap=include_swap, scales=scales,
    )]
