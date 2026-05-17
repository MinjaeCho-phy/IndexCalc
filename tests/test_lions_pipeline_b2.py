"""D6 — B2 end-to-end pipeline test (gauge sector).

Enumerate gauge-sector invariants from FieldSpec(W^A_μ) and
FieldSpec(F^A_{μν}, antisym(μν)). Verify:

- W·W ( = W^A_μ W^B_ν η_{AB} η^{μν} ) is enumerated.
- F·F ( = F^A_{μν} F^B_{ρσ} η_{AB} η^{μρ} η^{νσ} ) is enumerated.

Backend gap (M9.6, notes/d6_gauge_field.md §2.5):
  Current simplifier cannot absorb η_{AB} W^A W^B into W·W (raise/lower
  normalization). The SU(2) and Lorentz invariance labels of W·W and F·F
  are therefore marked False by the labeler even though both are
  mathematically invariant. After M9.6 metric-absorption is added these
  labels flip to True; this test asserts the *current* state and serves
  as a marker — flip the asserts when M9.6 lands.

- F = ∂W + WW synthesis is deferred to D7+ (notes/d6_gauge_field.md §1).
- W^A_μ alone (free indices) is never produced — perfect-matching closes
  every open slot; v1 enumerator therefore does not test single-W
  non-invariance.
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct
from indexcalc.core.simplify import canonical_form_modulo_dummies

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


def _key_WW(b2):
    """W^A_μ W^B_ν η_{AB} η^{μν} — both W in FieldSpec position (adj upper,
    st lower); like-position matchings insert η to close.
    """
    W1 = Tensor(
        "W",
        [b2.su2_adj.upper("A"), b2.spacetime.lower("μ")],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    W2 = Tensor(
        "W",
        [b2.su2_adj.upper("B"), b2.spacetime.lower("ν")],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    eta_adj = Tensor(
        "eta",
        [b2.su2_adj.lower("A"), b2.su2_adj.lower("B")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    eta_st = Tensor(
        "eta",
        [b2.spacetime.upper("μ"), b2.spacetime.upper("ν")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    expr = TensorProduct(
        TensorProduct(W1, W2),
        TensorProduct(eta_adj, eta_st),
    )
    return canonical_form_modulo_dummies(expr)


def _key_FF(b2):
    """F^A_{μν} F^B_{ρσ} η_{AB} η^{μρ} η^{νσ} — same convention as W·W;
    F slot order (adj, μ, ν) reused for both instances.
    """
    F1 = Tensor(
        "F",
        [b2.su2_adj.upper("A"),
         b2.spacetime.lower("μ"), b2.spacetime.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    F2 = Tensor(
        "F",
        [b2.su2_adj.upper("B"),
         b2.spacetime.lower("ρ"), b2.spacetime.lower("σ")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
    )
    eta_adj = Tensor(
        "eta",
        [b2.su2_adj.lower("A"), b2.su2_adj.lower("B")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    eta_st1 = Tensor(
        "eta",
        [b2.spacetime.upper("μ"), b2.spacetime.upper("ρ")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    eta_st2 = Tensor(
        "eta",
        [b2.spacetime.upper("ν"), b2.spacetime.upper("σ")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    expr = TensorProduct(
        TensorProduct(F1, F2),
        TensorProduct(TensorProduct(eta_adj, eta_st1), eta_st2),
    )
    return canonical_form_modulo_dummies(expr)


def test_b2_enumerator_yields_gauge_invariants(b2):
    """W·W and F·F appear in the enumeration output (no derivatives needed)."""
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    keys = {canonical_form_modulo_dummies(s.expr) for s in samples}

    assert _key_WW(b2) in keys, "W·W missing from B2 enumeration"
    assert _key_FF(b2) in keys, "F·F missing from B2 enumeration"


def test_b2_labeler_current_simplifier_gap(b2, generators):
    """**M9.6 marker**: with the current simplifier the SU(2) and Lorentz
    labels of W·W and F·F come out False (false-negative). Both are
    mathematically invariant; the failure mode is that the simplifier
    cannot absorb η_{AB} into raise/lower on the host tensor. Flip these
    asserts to True after M9.6 metric-absorption lands.
    """
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    labeled = label_samples(samples, generators)

    targets = {_key_WW(b2): "W·W", _key_FF(b2): "F·F"}
    seen = set()
    for samp in labeled:
        key = canonical_form_modulo_dummies(samp.expr)
        if key in targets:
            name = targets[key]
            # Pre-M9.6 behaviour: both labels are False (gap-documented).
            assert not samp.labels["SU(2)"], (
                f"{name}: SU(2) label True — has M9.6 landed? "
                f"Flip this assert."
            )
            assert not samp.labels["Lorentz"], (
                f"{name}: Lorentz label True — has M9.6 landed? "
                f"Flip this assert."
            )
            seen.add(name)
    assert seen == {"W·W", "F·F"}, f"missed targets: {set(targets.values()) - seen}"
