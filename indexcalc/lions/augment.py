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


# ─── N3: dangling-term hard negatives ──────────────────


from indexcalc.core.index import Index


def _collect_index_names(expr: TensorExpr) -> set[str]:
    """Return every index name occurring in ``expr`` (free + dummy)."""
    names: set[str] = set()
    if isinstance(expr, Tensor):
        for idx in expr.indices:
            names.add(idx.name)
        return names
    if isinstance(expr, ZeroTensor):
        for idx in expr.free_indices:
            names.add(idx.name)
        return names
    if isinstance(expr, (TensorProduct, TensorSum)):
        return _collect_index_names(expr.left) | _collect_index_names(expr.right)
    if isinstance(expr, ScalarMul):
        return _collect_index_names(expr.expr)
    if isinstance(expr, PartialDeriv):
        return _collect_index_names(expr.expr) | {expr.deriv_index.name}
    raise TypeError(f"_collect_index_names: unsupported {type(expr).__name__}")


def _disambiguate_indices(
    p_expr: TensorExpr, q_expr: TensorExpr, *, q_suffix: str = "_q",
) -> TensorExpr:
    """Rename every index name in ``q_expr`` that collides with ``p_expr``.

    Strategy: append ``q_suffix`` to colliding names; if still clash, add
    a numeric counter. Returns a renamed q_expr where every index name
    is disjoint from p_expr's index set.
    """
    p_names = _collect_index_names(p_expr)
    q_names = _collect_index_names(q_expr)
    mapping: dict[str, str] = {}
    for n in q_names:
        if n not in p_names:
            continue
        candidate = n + q_suffix
        i = 0
        while candidate in p_names or candidate in q_names or candidate in mapping.values():
            i += 1
            candidate = f"{n}{q_suffix}{i}"
        mapping[n] = candidate
    if not mapping:
        return q_expr
    return rename_index(q_expr, mapping)


def add_n3_dangling_term(
    positive: LabeledSample,
    broken_term: LabeledSample,
    generators: dict[str, Generator],
    *,
    rng=None,
) -> LabeledSample:
    """Form ``TensorSum(positive.expr, broken_term.expr)`` after dummy
    disambiguation, then re-label via the oracle.

    The resulting sample has *same node-feature distribution* as a longer
    enumeration sample but its invariance status depends on whether each
    term is independently invariant. This is a topology-level hard
    negative: rep tags alone cannot decide it.
    """
    q_disambig = _disambiguate_indices(positive.expr, broken_term.expr)
    new_expr = TensorSum(positive.expr, q_disambig)
    new_labels = label_expression(new_expr, generators)
    # Combined metadata (sum of counts, max of mass dim — neither is
    # exact, but these fields are advisory for ML features).
    combined_field_counts: dict[str, int] = {}
    for d in (positive.field_counts, broken_term.field_counts):
        for k, v in d.items():
            combined_field_counts[k] = combined_field_counts.get(k, 0) + v
    return LabeledSample(
        expr=new_expr,
        labels=new_labels,
        mass_dim=max(positive.mass_dim, broken_term.mass_dim),
        field_counts=combined_field_counts,
        partial_count=positive.partial_count + broken_term.partial_count,
        invariant_counts={},
        provenance="hard_negative_n3",
    )


# ─── I4: deeper multi-term hard negatives ──────────────


def _combine_metadata(
    expr: TensorExpr, sources: list[LabeledSample],
    generators: dict[str, Generator], provenance: str,
) -> LabeledSample:
    """Re-label ``expr`` via the oracle, sum/max source metadata."""
    new_labels = label_expression(expr, generators)
    combined_counts: dict[str, int] = {}
    total_partials = 0
    max_mass_dim = 0.0
    for s in sources:
        for k, v in s.field_counts.items():
            combined_counts[k] = combined_counts.get(k, 0) + v
        total_partials += s.partial_count
        max_mass_dim = max(max_mass_dim, s.mass_dim)
    return LabeledSample(
        expr=expr, labels=new_labels,
        mass_dim=max_mass_dim,
        field_counts=combined_counts,
        partial_count=total_partials,
        invariant_counts={},
        provenance=provenance,
    )


def add_n3_positive_pair(
    a: LabeledSample, b: LabeledSample,
    generators: dict[str, Generator],
) -> LabeledSample:
    """``TensorSum(a, b)`` with disambiguated dummies. Both terms should
    be invariant ⇒ the sum is invariant. Provenance ``n3_positive``."""
    b_disambig = _disambiguate_indices(a.expr, b.expr)
    expr = TensorSum(a.expr, b_disambig)
    return _combine_metadata(expr, [a, b], generators, "n3_positive")


def add_n3_double_broken_pair(
    a: LabeledSample, b: LabeledSample,
    generators: dict[str, Generator],
) -> LabeledSample:
    """``TensorSum(a, b)`` of two broken samples. Sum is broken under
    every group that broke in either term (no magic cancellation).
    Provenance ``n3_double_broken``."""
    b_disambig = _disambiguate_indices(a.expr, b.expr)
    expr = TensorSum(a.expr, b_disambig)
    return _combine_metadata(expr, [a, b], generators, "n3_double_broken")


def add_n4_nested(
    a: LabeledSample, b: LabeledSample, c: LabeledSample,
    generators: dict[str, Generator],
) -> LabeledSample:
    """``TensorSum(a, TensorSum(b, c))``. Dummies in b are disambiguated
    against a; dummies in c are disambiguated against the (a, b)
    combined expression. Provenance ``n4_nested``."""
    b_disambig = _disambiguate_indices(a.expr, b.expr, q_suffix="_b")
    ab = TensorSum(a.expr, b_disambig)
    c_disambig = _disambiguate_indices(ab, c.expr, q_suffix="_c")
    expr = TensorSum(a.expr, TensorSum(b_disambig, c_disambig))
    return _combine_metadata(expr, [a, b, c], generators, "n4_nested")


def enumerate_n3_positives(
    positive_pool: list[LabeledSample],
    generators: dict[str, Generator],
    *, n_per_seed: int = 2, rng=None,
) -> list[LabeledSample]:
    """Pair each seed in ``positive_pool`` with ``n_per_seed`` other
    distinct positives to build ``TensorSum(inv, inv)`` samples."""
    import random as _random
    if rng is None:
        rng = _random.Random(0)
    out: list[LabeledSample] = []
    if len(positive_pool) < 2:
        return out
    for i, a in enumerate(positive_pool):
        others = [p for j, p in enumerate(positive_pool) if j != i]
        picks = rng.sample(others, k=min(n_per_seed, len(others)))
        for b in picks:
            try:
                out.append(add_n3_positive_pair(a, b, generators))
            except Exception:
                continue
    return out


def enumerate_n3_double_broken(
    broken_pool: list[LabeledSample],
    generators: dict[str, Generator],
    *, n_per_seed: int = 2, rng=None,
) -> list[LabeledSample]:
    """Pair broken samples to build ``TensorSum(broken, broken)``."""
    import random as _random
    if rng is None:
        rng = _random.Random(0)
    out: list[LabeledSample] = []
    if len(broken_pool) < 2:
        return out
    for i, a in enumerate(broken_pool):
        others = [p for j, p in enumerate(broken_pool) if j != i]
        picks = rng.sample(others, k=min(n_per_seed, len(others)))
        for b in picks:
            try:
                out.append(add_n3_double_broken_pair(a, b, generators))
            except Exception:
                continue
    return out


def enumerate_n4_nested(
    positive_pool: list[LabeledSample],
    broken_pool: list[LabeledSample],
    generators: dict[str, Generator],
    *, n_per_seed: int = 2, rng=None,
) -> list[LabeledSample]:
    """Build ``TensorSum(inv, TensorSum(inv, broken))`` — a 3-term sum
    where exactly one term breaks. Sum is broken; I2's min readout has
    to propagate the broken-term signal across two layers of TensorSum."""
    import random as _random
    if rng is None:
        rng = _random.Random(0)
    out: list[LabeledSample] = []
    if not broken_pool or len(positive_pool) < 2:
        return out
    for i, a in enumerate(positive_pool):
        others = [p for j, p in enumerate(positive_pool) if j != i]
        b_picks = rng.sample(others, k=min(n_per_seed, len(others)))
        for b in b_picks:
            c = rng.choice(broken_pool)
            try:
                out.append(add_n4_nested(a, b, c, generators))
            except Exception:
                continue
    return out


def enumerate_n3_negatives(
    positive_pool: list[LabeledSample],
    broken_pool: list[LabeledSample],
    generators: dict[str, Generator],
    *,
    n_per_seed: int = 2,
    rng=None,
    require_label_change_from_pos: bool = True,
) -> list[LabeledSample]:
    """For each positive in ``positive_pool``, sample ``n_per_seed``
    broken terms from ``broken_pool`` and emit one N3 negative each.

    Parameters
    ----------
    positive_pool
        Seeds. Typically fully-invariant samples — their labels remain
        unchanged after adding a broken term IF the broken term's labels
        are all True (which we filter against), so we require the broken
        pool to be NON-fully-invariant.
    broken_pool
        Broken samples. ``label_samples`` output where at least one
        group label is False.
    n_per_seed
        Number of broken terms attached per positive seed.
    require_label_change_from_pos
        If True, drop combinations whose label dict is identical to the
        seed positive's (i.e. the broken term turned out to also break
        in a way that the simplifier happens to cancel). Default True.
    """
    import random as _random
    if rng is None:
        rng = _random.Random(0)

    out: list[LabeledSample] = []
    if not broken_pool:
        return out
    for pos in positive_pool:
        picks = rng.sample(
            broken_pool, k=min(n_per_seed, len(broken_pool)),
        )
        for q in picks:
            try:
                neg = add_n3_dangling_term(pos, q, generators)
            except Exception:
                continue
            if (require_label_change_from_pos
                    and neg.labels == pos.labels):
                continue
            out.append(neg)
    return out
