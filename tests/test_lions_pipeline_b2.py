"""D6 — B2 end-to-end pipeline test (gauge sector).

Enumerate gauge-sector invariants from FieldSpec(W^A_μ) and
FieldSpec(F^A_{μν}, antisym(μν)). Verify:

- W·W ( = W^A_μ W^B_ν η_{AB} η^{μν} ) is enumerated.
- F·F ( = F^A_{μν} F^B_{ρσ} η_{AB} η^{μρ} η^{νσ} ) is enumerated.

M9.6 landed: ``absorb_einstein_metric`` normalizes η_{AB} W^A W^B
into raise/lower form so the antisym × symmetric = 0 pattern fires.
M9.7 landed: enumerator dedupes sign-equivalent F·F forms via
``canonical_form_with_sign`` so labelling runs once on the canonically
sign-normalized body. Result: B2 produces exactly two samples
(W·W and F·F), both SU(2)+Lorentz invariant.

- F = ∂W + WW synthesis is deferred to D7+ (notes/d6_gauge_field.md §1).
- W^A_μ alone (free indices) is never produced — perfect-matching closes
  every open slot; v1 enumerator therefore does not test single-W
  non-invariance.
"""

from __future__ import annotations
import pytest

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
)
from indexcalc.lions.presets.su2_gauge import build_b2


@pytest.fixture
def b2():
    return build_b2()


@pytest.fixture
def generators(b2):
    return {"SU(2)": b2.su2_gen, "Lorentz": b2.lorentz_gen}


def test_b2_enumerator_yields_gauge_invariants(b2):
    """A W·W (only-W) sample and an F·F (only-F) sample both appear in
    the enumeration. Identification is via field counts because M9.6
    metric absorption rewrites the enumerator output form away from a
    fixed hand-built template.
    """
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    counts = [s.field_counts for s in samples]

    assert {"W": 2, "F": 0} in counts, "W·W missing from enumeration"
    assert {"W": 0, "F": 2} in counts, "F·F missing from enumeration"


def test_b2_labeler_recovers_gauge_invariants(b2, generators):
    """Post-M9.6+M9.7: B2 has exactly two samples (W·W, F·F) and every
    one is SU(2)+Lorentz invariant. No false negatives in the dataset."""
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    labeled = label_samples(samples, generators)

    field_counts = [s.field_counts for s in labeled]
    assert {"W": 2, "F": 0} in field_counts, "W·W missing"
    assert {"W": 0, "F": 2} in field_counts, "F·F missing"
    # M9.7 dedupes sign-equivalent F·F variants → exactly one F·F sample.
    ff_count = sum(1 for fc in field_counts if fc == {"W": 0, "F": 2})
    assert ff_count == 1, f"expected 1 F·F after M9.7 dedupe, got {ff_count}"

    for samp in labeled:
        assert samp.labels["SU(2)"], (
            f"{samp.field_counts}: SU(2) False — false-negative regression"
        )
        assert samp.labels["Lorentz"], (
            f"{samp.field_counts}: Lorentz False — false-negative regression"
        )
