"""LIONS M9.7 acceptance — sign-aware antisym normalization for
enumerator dedupe.

``normalize_antisym_signs`` sorts each ``Tensor.antisymmetric_pairs``
slot pair by index name and accumulates the parity into an outer
``ScalarMul`` sign. ``canonical_form_with_sign`` is the (sign, key)
pair built from the normalized form — used by the LIONS enumerator
to collapse sign-equivalent duplicates (e.g. F·F's μν-cross sample
that the bare canonical_form_modulo_dummies treats as distinct).

Scope discipline: ``normalize_antisym_signs`` is **not** in the
``simplify`` fixed-point. Putting it there breaks the existing
position-collapsed cancellation paths (M7-C four-term Lorentz fold).
M9.7 is a dedupe helper only.

- A: F^A_{νμ} normalizes to -F^A_{μν}.
- B: TensorProduct of two antisym Ts accumulates signs multiplicatively.
- C: canonical_form_with_sign — sign-equivalent forms get same key,
     opposite sign.
- D: B2 enumerator dedupe — F·F appears once (not twice) post-M9.7.
- E: simplify is unchanged — M7-C (W^A_{μν} W_A^{μν} Lorentz) still ZeroTensor.
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.group import Group
from indexcalc.core.generator import make_lorentz_spinor_generator
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import (
    simplify, normalize_antisym_signs, canonical_form_with_sign,
)
from indexcalc.core.variation import ZeroTensor


@pytest.fixture
def st():
    return IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")


@pytest.fixture
def adj():
    return IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")


# ─── A: single antisym Tensor sign flip ─────────────────


def test_a_antisym_swap_yields_minus(st, adj):
    """F^A_{νμ} → ScalarMul(-1, F^A_{μν}) (μ < ν lexicographically)."""
    F_swapped = Tensor(
        "F",
        [adj.upper("A"), st.lower("ν"), st.lower("μ")],
        antisymmetric_pairs=[(1, 2)],
        reps={},
    )
    out = normalize_antisym_signs(F_swapped)
    assert isinstance(out, ScalarMul)
    assert out.scalar == -1
    inner = out.expr
    assert isinstance(inner, Tensor)
    # Sorted slot 1, 2 names: μ, ν (μ < ν)
    assert inner.indices[1].name == "μ"
    assert inner.indices[2].name == "ν"


# ─── B: multiplicative sign accumulation across factors ──


def test_b_two_swaps_yield_plus(st, adj):
    """Two swapped antisym tensors compose to +1 (parity 2)."""
    F1 = Tensor(
        "F", [adj.upper("A"), st.lower("ν"), st.lower("μ")],
        antisymmetric_pairs=[(1, 2)], reps={},
    )
    F2 = Tensor(
        "F", [adj.upper("B"), st.lower("σ"), st.lower("ρ")],
        antisymmetric_pairs=[(1, 2)], reps={},
    )
    product = TensorProduct(F1, F2)
    out = normalize_antisym_signs(product)
    # Two -1 signs → +1, no ScalarMul wrap
    assert not isinstance(out, ScalarMul), f"got {out!r}"


# ─── C: canonical_form_with_sign maps equivalents together ──


def test_c_canonical_form_with_sign_matches_equivalents(st, adj):
    """F^A_{μν} F_A^{μν} and F^A_{νμ} F_A^{μν} share the same canonical
    key, with opposite signs (the second is -1 × the first)."""
    F1a = Tensor("F",
                 [adj.upper("A"), st.lower("μ"), st.lower("ν")],
                 antisymmetric_pairs=[(1, 2)], reps={})
    F1b = Tensor("F",
                 [adj.lower("A"), st.upper("μ"), st.upper("ν")],
                 antisymmetric_pairs=[(1, 2)], reps={})
    e1 = TensorProduct(F1a, F1b)

    F2a = Tensor("F",
                 [adj.upper("A"), st.lower("ν"), st.lower("μ")],
                 antisymmetric_pairs=[(1, 2)], reps={})
    e2 = TensorProduct(F2a, F1b)

    s1, k1 = canonical_form_with_sign(e1)
    s2, k2 = canonical_form_with_sign(e2)
    assert k1 == k2, "sign-equivalent forms should share canonical key"
    assert s1 == -s2, f"signs should differ; got {s1} vs {s2}"


# ─── D: enumerator dedupes F·F to a single sample ────────


def test_d_b2_enumerator_dedupes_ff(st, adj):
    """LIONS B2 preset enumeration: F·F should appear once after M9.7
    (was two before — μ-ρ-ν-σ vs μ-σ-ν-ρ contraction orders are
    sign-equivalent)."""
    from indexcalc.lions.presets.su2_gauge import build_b2
    from indexcalc.lions import EnumeratorCaps, enumerate_scalar_invariants
    b2 = build_b2()
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=2,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        b2.fields, spacetime=b2.spacetime, caps=caps,
    )
    ff_count = sum(1 for s in samples if s.field_counts == {"W": 0, "F": 2})
    assert ff_count == 1, f"expected 1 F·F sample, got {ff_count}"


# ─── E: simplify unchanged — M7-C still folds to ZeroTensor ──


def test_e_m7c_lorentz_invariance_unchanged(st, adj):
    """Regression guard: M9.7 must not affect the existing M7-C
    cancellation (W^A_{μν} W_A^{μν} Lorentz invariance via 4-term fold
    that relies on position-collapsed canonical_form, not antisym sign
    sorting)."""
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")
    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)
    gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )

    W_low = Tensor("W",
                   [adj.upper("A"), st.lower("μ"), st.lower("ν")],
                   antisymmetric_pairs=[(1, 2)],
                   reps={"Lorentz": "vector"})
    W_up = Tensor("W",
                  [adj.lower("A"), st.upper("μ"), st.upper("ν")],
                  antisymmetric_pairs=[(1, 2)],
                  reps={"Lorentz": "vector"})
    L = ScalarMul(-0.25, TensorProduct(W_low, W_up))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"M7-C regression: got {final!r}"
