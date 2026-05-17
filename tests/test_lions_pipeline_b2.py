"""D6 — B2 end-to-end pipeline test (gauge sector).

Enumerate gauge-sector invariants from FieldSpec(W^A_μ) and
FieldSpec(F^A_{μν}, antisym(μν)). Verify:

- W·W ( = W^A_μ W^B_ν η_{AB} η^{μν} ) is enumerated.
- F·F ( = F^A_{μν} F^B_{ρσ} η_{AB} η^{μρ} η^{νσ} ) is enumerated.

M9.6 landed: ``absorb_einstein_metric`` simplifier rule normalizes
η_{AB} W^A W^B forms by raise/lower, so the antisym × symmetric = 0
pattern is now recognised. W·W and the canonical F·F enumerator form
are SU(2)+Lorentz invariant. A separate enumerator dedupe issue means
F·F appears as two samples (cross-contraction sign-equivalent forms)
and one of them still labels SU(2)=False — that's a sign-aware
``canonical_form_modulo_dummies`` gap (M9.7 candidate).

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
    """W·W and the canonical F·F enumerator form are recovered as
    SU(2)+Lorentz invariant after M9.6 metric absorption.

    F·F appears in two enumerator forms (μ-ρ-ν-σ vs μ-σ-ν-ρ contraction
    orders). The second is sign-equivalent via F's antisym(μν), but
    enumerator dedupe doesn't realise that yet (M9.7 candidate). It's
    fine for at least one F·F form to be fully invariant — that's
    enough for downstream labelling to pick up the (+1) instance.
    """
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    labeled = label_samples(samples, generators)

    ww_invariant_count = 0
    ff_invariant_count = 0
    for samp in labeled:
        all_invariant = samp.labels["SU(2)"] and samp.labels["Lorentz"]
        if samp.field_counts == {"W": 2, "F": 0} and all_invariant:
            ww_invariant_count += 1
        if samp.field_counts == {"W": 0, "F": 2} and all_invariant:
            ff_invariant_count += 1

    assert ww_invariant_count >= 1, (
        "W·W not labelled SU(2)+Lorentz invariant"
    )
    assert ff_invariant_count >= 1, (
        "no F·F form labelled SU(2)+Lorentz invariant"
    )
