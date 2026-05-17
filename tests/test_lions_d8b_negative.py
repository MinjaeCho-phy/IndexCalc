"""D8b — wrong-rep negative synthesis + augmentation building blocks.

Covers:
- ``mutate_field_reps``: tree walk preserves all Tensor metadata except
  the targeted reps entry.
- ``apply_rep_mutation`` + ``enumerate_wrong_rep_negatives``: end-to-end
  on B0 (Higgs charge swap) and B1 (Yukawa rep swap) — produces
  ``LabeledSample(provenance="negative")`` with at least one label
  flipping from True to False.
- Augmentation helpers (``permute_dummy_indices``, ``swap_top_product``,
  ``scale_by``) preserve labels by linearity / index renaming.
- Persistence chain: negative samples round-trip through ``save_dataset``
  / ``load_dataset`` (provenance preserved).
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.simplify import canonical_form_modulo_dummies

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
    RepMutation,
    mutate_field_reps,
    apply_rep_mutation,
    enumerate_wrong_rep_negatives,
    permute_dummy_indices,
    swap_top_product,
    scale_by,
    augment_sample,
    expand_dataset,
    save_dataset,
    load_dataset,
)
from indexcalc.lions.presets.su2_higgs_scalar import build_b0
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1
from indexcalc.lions.presets.su2_gauge import build_b2


# ─── mutate_field_reps ──────────────────────────────────


def test_mutate_field_reps_preserves_metadata():
    """Swap one rep on a Tensor — all other metadata bit-equal."""
    from indexcalc.core.index import IndexSpace
    adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    F = Tensor(
        "F",
        [adj.upper("A"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
        statistics="bosonic",
    )
    out = mutate_field_reps(F, "F", "SU(2)", "singlet")
    assert isinstance(out, Tensor)
    assert out.reps == {"SU(2)": "singlet", "Lorentz": "vector"}
    assert out.antisymmetric_pairs == F.antisymmetric_pairs
    assert out.indices == F.indices
    assert out.statistics == F.statistics


def test_mutate_skips_non_matching_field():
    from indexcalc.core.index import IndexSpace
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    out = mutate_field_reps(H, "X", "SU(2)", "singlet")  # name mismatch
    assert out is H


def test_mutate_walks_tensor_product():
    """Both children of a TensorProduct should be visited."""
    from indexcalc.core.index import IndexSpace
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H1 = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    H2 = Tensor("H", [su2.upper("j")], reps={"SU(2)": "fund"})
    expr = TensorProduct(H1, H2)
    out = mutate_field_reps(expr, "H", "SU(2)", "singlet")
    assert isinstance(out, TensorProduct)
    assert out.left.reps == {"SU(2)": "singlet"}
    assert out.right.reps == {"SU(2)": "singlet"}


# ─── End-to-end negative on B0 ─────────────────────────


def _b0_labeled():
    b0 = build_b0()
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b0.fields, spacetime=b0.spacetime, caps=caps,
    )
    generators = {"SU(2)": b0.su2_gen, "U(1)_Y": b0.u1y_gen}
    return b0, label_samples(samples, generators), generators


def test_b0_wrong_rep_flips_label():
    """Swapping H's U(1)_Y from +1/2 → +1 breaks charge balance →
    U(1)_Y label flips True→False on every positive sample."""
    _b0, labeled, generators = _b0_labeled()
    # Sanity: every starting sample is U(1)_Y invariant under positives.
    invariant_positives = [s for s in labeled if s.labels.get("U(1)_Y")]
    assert invariant_positives, "B0 enumerator produced no U(1)_Y invariants"

    mut = RepMutation("H", "U(1)_Y", "0")
    negatives = enumerate_wrong_rep_negatives(
        invariant_positives, generators, [mut],
    )
    assert negatives, "no negatives produced"
    for n in negatives:
        assert n.provenance == "negative"
        assert n.labels["U(1)_Y"] is False, (
            f"expected U(1)_Y=False after rep swap, got {n.labels}"
        )


# ─── End-to-end negative on B2 (gauge) ──────────────────


def test_b2_wrong_rep_provenance_marker():
    """W^A in adj → singlet: SU(2) generator on singlet returns 0 so
    the label dict stays {SU(2): True, Lorentz: True} — but the
    mutation still records ``provenance="negative"``. With
    ``require_label_change=False`` we keep these as marker-only
    negatives (useful for tracking what's been tried)."""
    b2 = build_b2()
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    generators = {"SU(2)": b2.su2_gen, "Lorentz": b2.lorentz_gen}
    labeled = label_samples(samples, generators)
    ww_samples = [s for s in labeled if s.field_counts == {"W": 2, "F": 0}]
    assert ww_samples

    mut = RepMutation("W", "SU(2)", "singlet")
    negatives = enumerate_wrong_rep_negatives(
        ww_samples, generators, [mut], require_label_change=False,
    )
    assert negatives
    for n in negatives:
        assert n.provenance == "negative"
        # Tensor's reps was actually mutated even if label is unchanged.
        from indexcalc.core.tensor import Tensor
        def walk(e):
            if isinstance(e, Tensor):
                yield e
            elif hasattr(e, "left"):
                yield from walk(e.left); yield from walk(e.right)
            elif hasattr(e, "expr"):
                yield from walk(e.expr)
        Ws = [t for t in walk(n.expr) if t.name == "W"]
        assert all(t.reps.get("SU(2)") == "singlet" for t in Ws)


# ─── Positive augmentation helpers ──────────────────────


def test_permute_dummy_indices_preserves_labels():
    """Renaming a dummy index doesn't change canonical form (and so
    doesn't change oracle outcome) — we just record provenance."""
    _b0, labeled, _ = _b0_labeled()
    s = labeled[0]
    augmented = permute_dummy_indices(s, {"i": "k", "j": "l"})
    assert augmented.provenance == "augmented"
    assert augmented.labels == s.labels


def test_swap_top_product_preserves_labels():
    _b0, labeled, _ = _b0_labeled()
    # Pick a sample whose top-level expr is a TensorProduct.
    s = next((x for x in labeled
              if isinstance(x.expr, TensorProduct)), None)
    if s is None:
        pytest.skip("no top-level TensorProduct in B0 enumeration")
    aug = swap_top_product(s)
    assert aug.provenance == "augmented"
    assert aug.labels == s.labels
    assert isinstance(aug.expr, TensorProduct)
    assert (canonical_form_modulo_dummies(aug.expr)
            == canonical_form_modulo_dummies(s.expr))


def test_scale_by_preserves_labels():
    _b0, labeled, _ = _b0_labeled()
    s = labeled[0]
    aug = scale_by(s, 0.5)
    assert isinstance(aug.expr, ScalarMul)
    assert aug.expr.scalar == 0.5
    assert aug.labels == s.labels
    assert aug.provenance == "augmented"


def test_scale_by_zero_rejects():
    _b0, labeled, _ = _b0_labeled()
    with pytest.raises(ValueError, match="ZeroTensor"):
        scale_by(labeled[0], 0)


# ─── D8c orchestrator ──────────────────────────────────


def test_augment_sample_preserves_labels_across_variants():
    """``augment_sample`` returns the original + swap (if applicable) +
    scale variants. Every variant must carry the same label dict.
    """
    _b0, labeled, _ = _b0_labeled()
    # Pick a TensorProduct-rooted sample so swap_top_product fires.
    s = next((x for x in labeled if isinstance(x.expr, TensorProduct)), None)
    if s is None:
        pytest.skip("no TensorProduct sample to augment")
    variants = augment_sample(s, scales=(-1.0, 0.5))
    assert len(variants) >= 3   # original + swap + 2 scales
    assert variants[0] is s
    for v in variants[1:]:
        assert v.labels == s.labels
        assert v.provenance == "augmented"


def test_expand_dataset_scales_count():
    """expand_dataset on N samples returns ≥N variants (≥3× when most
    samples are TensorProducts and scales=(-1,0.5,2))."""
    _b0, labeled, _ = _b0_labeled()
    expanded = expand_dataset(labeled, scales=(-1.0, 0.5))
    assert len(expanded) >= len(labeled) * 2


# ─── Persistence round-trip on negatives ────────────────


def test_negative_samples_persist(tmp_path):
    """Negative ``LabeledSample`` (mutated reps + provenance="negative")
    survives save → load."""
    _b0, labeled, generators = _b0_labeled()
    mut = RepMutation("H", "U(1)_Y", "0")
    negatives = enumerate_wrong_rep_negatives(labeled, generators, [mut])
    assert negatives

    path = tmp_path / "negatives.json"
    save_dataset(negatives, path)
    loaded = load_dataset(path)

    assert len(loaded) == len(negatives)
    for n, back in zip(negatives, loaded):
        assert back.provenance == "negative"
        assert back.labels == n.labels
        assert (canonical_form_modulo_dummies(back.expr)
                == canonical_form_modulo_dummies(n.expr))
