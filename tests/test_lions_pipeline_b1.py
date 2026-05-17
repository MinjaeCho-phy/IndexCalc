"""D5a — B1 end-to-end pipeline test.

B1 = B0 (SU(2) Higgs scalar) + chiral leptons (L, Lbar, eR) for the
Yukawa sector. This test verifies:

1. The Yukawa monomial $\\bar L^i H^j \\epsilon_{ij} e_R$ appears in the
   enumerator output with SU(2) / U(1)_Y / Lorentz all flagged invariant.
2. B0 pieces (|H|^2, |H|^4) still appear (regression).
3. Fermi parity filter — no single-fermion or triple-fermion monomial
   gets produced.
4. Forbidden dirac like-position policy — no spurious ε_{αβ} insertion.

γ^μ-bearing terms (chiral kinetic) are D5b, intentionally absent here:
the caps disable partial-derivative bridges on fermion fields.
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
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1


# ─── Fixtures ────────────────────────────────────────────


@pytest.fixture
def b1():
    return build_b1()


@pytest.fixture
def generators(b1):
    return {
        "SU(2)": b1.su2_gen,
        "U(1)_Y": b1.u1y_gen,
        "Lorentz": b1.lorentz_gen,
    }


# ─── Reference Lagrangian canonical keys ────────────────


def _key_HHdag(b1):
    H = Tensor("H", [b1.su2_fund.upper("i")],
               reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"})
    Hd = Tensor("Hdag", [b1.su2_fund.lower("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "singlet"})
    return canonical_form_modulo_dummies(TensorProduct(H, Hd))


def _key_HHHH(b1):
    H1 = Tensor("H", [b1.su2_fund.upper("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"})
    H2 = Tensor("H", [b1.su2_fund.upper("j")],
                reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"})
    Hd1 = Tensor("Hdag", [b1.su2_fund.lower("i")],
                 reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "singlet"})
    Hd2 = Tensor("Hdag", [b1.su2_fund.lower("j")],
                 reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "singlet"})
    expr = TensorProduct(TensorProduct(H1, H2), TensorProduct(Hd1, Hd2))
    return canonical_form_modulo_dummies(expr)


def _yukawa_signature():
    """Field-count signature of the chiral Yukawa: one H, one Lbar, one eR.

    We match by signature (not canonical key) because
    ``canonical_form_modulo_dummies`` does not normalize antisymmetric slot
    orderings of ε, so two structurally equivalent Yukawa terms can have
    distinct keys that differ only by a (-1) factor we don't care about
    at the "invariant existence" level.
    """
    return {"H": 1, "Hdag": 0, "L": 0, "Lbar": 1, "eR": 1}


# ─── Tests ────────────────────────────────────────────────


def _b1_samples(b1):
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    return enumerate_scalar_invariants(
        b1.fields,
        spacetime=b1.spacetime,
        caps=caps,
        forbid_like_position_spaces={b1.dirac},
    )


def test_b1_enumerator_yields_yukawa(b1):
    """Chiral Yukawa monomial Lbar H ε eR is enumerated under the B1 caps."""
    samples = _b1_samples(b1)
    sig = _yukawa_signature()
    matched = [s for s in samples
               if {k: s.field_counts.get(k, 0) for k in sig} == sig]
    assert matched, (
        "no enumerated sample matches the (H=1, Lbar=1, eR=1) Yukawa signature"
    )


def test_b1_b0_regression(b1):
    """|H|^2 and |H|^4 still enumerate alongside the new fermion fields."""
    samples = _b1_samples(b1)
    keys = {canonical_form_modulo_dummies(s.expr) for s in samples}
    assert _key_HHdag(b1) in keys
    assert _key_HHHH(b1) in keys


def test_b1_labeler_marks_yukawa_triply_invariant(b1, generators):
    """At least one (H=1, Lbar=1, eR=1) monomial must be invariant under
    SU(2), U(1)_Y, AND Lorentz simultaneously — that's the chiral Yukawa.

    Distinct contraction patterns at the same field signature may have
    different invariance labels (e.g. wrong ε-position breaks SU(2));
    existence of one triply-invariant pattern is what we assert.
    """
    samples = _b1_samples(b1)
    labeled = label_samples(samples, generators)
    sig = _yukawa_signature()
    cands = [s for s in labeled
             if {k: s.field_counts.get(k, 0) for k in sig} == sig]
    assert cands, "no Yukawa-signature samples in labeled set"
    triply = [s for s in cands
              if s.labels["SU(2)"] and s.labels["U(1)_Y"]
              and s.labels["Lorentz"]]
    assert triply, (
        "no Yukawa-signature sample passes SU(2) ∧ U(1)_Y ∧ Lorentz "
        f"invariance (labels: {[s.labels for s in cands]})"
    )


def test_b1_fermi_parity_filter(b1):
    """No enumerated monomial has an odd total fermion count.

    Without the filter, single-fermion patterns like ``L H`` (which can't
    contract its dirac slot against anything) would survive only via the
    perfect-matching odd-count drop — but multi-fermion *odd* counts
    (e.g. L L eR with 3 fermions) would still pass index-matching given
    the right field shapes. The parity filter rules them out upstream.
    """
    samples = _b1_samples(b1)
    fermion_names = {"L", "Lbar", "eR", "eRbar"}
    for s in samples:
        n_fermion = sum(
            s.field_counts.get(n, 0) for n in fermion_names
        )
        assert n_fermion % 2 == 0, (
            f"odd-fermion monomial slipped through: {s.field_counts}"
        )


def test_b1_no_spurious_epsilon_in_dirac(b1):
    """``forbid_like_position_spaces={dirac}`` blocks ε_{αβ} insertion on
    the dirac space — every dirac contraction in the output must be a
    direct Einstein (upper, lower) rename, not a like-position pair
    needing an invariant tensor that does not exist in our alphabet."""
    samples = _b1_samples(b1)
    # Heuristic — check that no produced expression contains an
    # ``epsilon`` Tensor whose slots live in the dirac space.
    from indexcalc.core.tensor import TensorExpr

    def walk(node):
        yield node
        for attr in ("left", "right", "factor", "operand", "expr"):
            child = getattr(node, attr, None)
            if isinstance(child, TensorExpr):
                yield from walk(child)
        for child in getattr(node, "args", ()):
            if isinstance(child, TensorExpr):
                yield from walk(child)

    for s in samples:
        for node in walk(s.expr):
            if isinstance(node, Tensor) and node.name == "epsilon":
                for idx in node.indices:
                    assert idx.space != b1.dirac, (
                        f"spurious ε on dirac in {s.expr!r}"
                    )
