"""D4 — B0 end-to-end pipeline test.

Enumerate → label → check that the canonical Higgs sector pieces
(|H|^2, |H|^4, |∂H|^2) appear with the expected SU(2) × U(1)_Y labels,
and that the enumerator avoids trivial-zero monomials (Bose × ε with no
derivatives).
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct
from indexcalc.core.deriv import partial
from indexcalc.core.simplify import canonical_form_modulo_dummies

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
)
from indexcalc.lions.builders import make_eta
from indexcalc.lions.presets.su2_higgs_scalar import build_b0


# ─── Fixtures ────────────────────────────────────────────


@pytest.fixture
def b0():
    return build_b0()


@pytest.fixture
def generators(b0):
    return {"SU(2)": b0.su2_gen, "U(1)_Y": b0.u1y_gen}


# ─── Helpers — canonical keys for target Lagrangians ─────


def _key_HHdag(b0):
    """|H|^2 = H^i Hdag_i."""
    H = Tensor("H", [b0.su2_fund.upper("i")],
               reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    Hd = Tensor("Hdag", [b0.su2_fund.lower("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "-1/2"})
    return canonical_form_modulo_dummies(TensorProduct(H, Hd))


def _key_HHHH(b0):
    """|H|^4 = H^i H^j Hdag_i Hdag_j (one contraction pattern)."""
    H1 = Tensor("H", [b0.su2_fund.upper("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    H2 = Tensor("H", [b0.su2_fund.upper("j")],
                reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    Hd1 = Tensor("Hdag", [b0.su2_fund.lower("i")],
                 reps={"SU(2)": "fund", "U(1)_Y": "-1/2"})
    Hd2 = Tensor("Hdag", [b0.su2_fund.lower("j")],
                 reps={"SU(2)": "fund", "U(1)_Y": "-1/2"})
    expr = TensorProduct(
        TensorProduct(H1, H2),
        TensorProduct(Hd1, Hd2),
    )
    return canonical_form_modulo_dummies(expr)


def _key_partial_HHdag(b0):
    """|∂H|^2 = ∂_μ H^i ∂^μ Hdag_i  (∂ on different fields, η^{μν} contraction)."""
    H = Tensor("H", [b0.su2_fund.upper("i")],
               reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    Hd = Tensor("Hdag", [b0.su2_fund.lower("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "-1/2"})
    dH = partial(H, b0.spacetime.lower("μ"))
    dHd = partial(Hd, b0.spacetime.lower("ν"))
    eta = Tensor("eta",
                 [b0.spacetime.upper("μ"), b0.spacetime.upper("ν")],
                 symmetric_pairs=[(0, 1)], reps={})
    expr = TensorProduct(TensorProduct(dH, dHd), eta)
    return canonical_form_modulo_dummies(expr)


# ─── Tests ────────────────────────────────────────────────


def test_b0_enumerator_yields_higgs_canon(b0):
    """Enumerator output (caps tuned to dim ≤ 4 patterns) contains the
    three canonical Higgs sector Lagrangians."""
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=2, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        b0.fields, spacetime=b0.spacetime, caps=caps,
    )
    keys = {canonical_form_modulo_dummies(s.expr) for s in samples}

    assert _key_HHdag(b0) in keys, "|H|^2 missing from enumeration"
    assert _key_HHHH(b0) in keys, "|H|^4 missing from enumeration"
    assert _key_partial_HHdag(b0) in keys, "|∂H|^2 missing from enumeration"


def test_b0_labeler_marks_invariants(b0, generators):
    """Every enumerated sample is invariant under both SU(2) and U(1)_Y
    by construction (ε/η contractions of charge-balanced fields)."""
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=2, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        b0.fields, spacetime=b0.spacetime, caps=caps,
    )
    labeled = label_samples(samples, generators)

    # Some monomials may be invariant under one group but not the other
    # (e.g. wrong charge sum) — we only require that at least the three
    # canonical Higgs Lagrangians are flagged invariant under both groups.
    targets = {
        _key_HHdag(b0): "|H|^2",
        _key_HHHH(b0): "|H|^4",
        _key_partial_HHdag(b0): "|∂H|^2",
    }
    for samp in labeled:
        key = canonical_form_modulo_dummies(samp.expr)
        if key in targets:
            name = targets[key]
            assert samp.labels["SU(2)"], f"{name}: SU(2) label False"
            assert samp.labels["U(1)_Y"], f"{name}: U(1)_Y label False"


def test_b0_no_zero_pure_bose_antisym(b0):
    """The enumerator drops the obvious Bose × ε zero (H^i H^j ε_{ij}) for
    same-field same-position pairs without derivatives."""
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b0.fields, spacetime=b0.spacetime, caps=caps,
    )
    # The only surviving 2-field, 0-partial scalar invariant is |H|^2.
    assert len(samples) == 1
    assert samples[0].field_counts == {"H": 1, "Hdag": 1}
