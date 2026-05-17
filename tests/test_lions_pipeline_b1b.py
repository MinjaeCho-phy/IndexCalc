"""D5b — chiral kinetic enumeration with γ^μ in invariant alphabet.

B1 + γ^μ + 1 partial on a fermion field → enumerator recovers the
chiral kinetic terms:

    $i \\bar e_R \\gamma^\\mu \\partial_\\mu e_R$  (R chiral, SU(2) singlet)
    $i \\bar L \\gamma^\\mu \\partial_\\mu L \\cdot \\epsilon$  (L doublet, ε contraction)

Both are triply-invariant under SU(2) × U(1)_Y × Lorentz after the M8.1
ε-invariance partner-unwrap extension (which closes the SU(2)
cancellation across γ·∂ bridges).
"""

from __future__ import annotations
import pytest

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
)
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1


@pytest.fixture
def b1():
    return build_b1()


@pytest.fixture
def labeled(b1):
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=1, max_partials_per_field=1,
        max_invariants_per_kind=1, max_invariants_total=1,
        max_contractions_per_pattern=2000,
    )
    samples = enumerate_scalar_invariants(
        b1.fields, spacetime=b1.spacetime, caps=caps,
        forbid_like_position_spaces={b1.dirac},
        invariant_alphabet=b1.invariant_alphabet,
    )
    return label_samples(
        samples,
        {"SU(2)": b1.su2_gen, "U(1)_Y": b1.u1y_gen, "Lorentz": b1.lorentz_gen},
    )


def _has_field_sig(s, sig):
    return all(s.field_counts.get(k, 0) == v for k, v in sig.items())


def test_b1b_eR_kinetic_triply_invariant(labeled):
    """At least one (eR=1, eRbar=1, γ=1, ∂=1) sample is invariant under
    all three groups — the right-handed-lepton chiral kinetic term."""
    sig = {"eR": 1, "eRbar": 1, "L": 0, "Lbar": 0, "H": 0, "Hdag": 0}
    cands = [s for s in labeled
             if _has_field_sig(s, sig)
             and s.partial_count == 1
             and s.invariant_counts.get("gamma", 0) == 1]
    assert cands, "no (eR, eRbar, γ, ∂) sample enumerated"
    triply = [s for s in cands
              if s.labels["SU(2)"] and s.labels["U(1)_Y"]
              and s.labels["Lorentz"]]
    assert triply, (
        f"no triply-invariant eR-kinetic sample; labels: "
        f"{[s.labels for s in cands]}"
    )


def test_b1b_L_kinetic_triply_invariant(labeled):
    """L-kinetic (Lbar, L, γ, ∂) — at least one ε-bridged pattern is
    triply invariant after M8.1 partner-unwrap extension closes the
    SU(2) cancellation across γ·∂ bridges."""
    sig = {"L": 1, "Lbar": 1, "eR": 0, "eRbar": 0, "H": 0, "Hdag": 0}
    cands = [s for s in labeled
             if _has_field_sig(s, sig)
             and s.partial_count == 1
             and s.invariant_counts.get("gamma", 0) == 1]
    assert cands, "no (L, Lbar, γ, ∂) sample enumerated"
    triply = [s for s in cands
              if s.labels["SU(2)"] and s.labels["U(1)_Y"]
              and s.labels["Lorentz"]]
    assert triply, (
        "L-kinetic should now be triply invariant under M8.1; "
        f"labels: {[s.labels for s in cands]}"
    )


def test_b1b_b1_b0_still_intact(b1):
    """Adding γ to the alphabet must not lose B0/B1 invariants without
    partials. Enumerate at B1 caps (no γ, no ∂) and verify the Yukawa
    signature still appears."""
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b1.fields, spacetime=b1.spacetime, caps=caps,
        forbid_like_position_spaces={b1.dirac},
        # No invariant_alphabet passed — guards against accidental coupling.
    )
    sig = {"H": 1, "Hdag": 0, "L": 0, "Lbar": 1, "eR": 1, "eRbar": 0}
    matched = [s for s in samples if _has_field_sig(s, sig)]
    assert matched, "Yukawa signature lost from no-alphabet B1 enumeration"
