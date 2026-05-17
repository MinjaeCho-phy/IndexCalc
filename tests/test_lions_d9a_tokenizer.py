"""D9a — IR ↔ prefix-token round-trip tests.

The decoder side of the Task 3 (inverse generation) pipeline emits
prefix tokens. Round-trip safety = the generated tokens, once parsed,
can be re-fed through the IndexCalc oracle to *verify* the candidate.
That's the whole point of writing tokens that map back to IR.

Tests:
- atomic units (Tensor with full metadata, ScalarMul with each scalar
  type, PartialDeriv, ZeroTensor).
- B0 / B1 / B2 enumerator outputs round-trip.
- SM-lite dataset (~670 sample) round-trips: every parsed expr has the
  same ``canonical_form_modulo_dummies`` key as the source.
- Oracle determinism after parse: re-running ``apply_generator+simplify``
  on the parsed IR gives the same label dict as the source LabeledSample.
- Vocab build covers every token in the dataset.
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import partial
from indexcalc.core.simplify import canonical_form_modulo_dummies, simplify
from indexcalc.core.substitution import apply_generator
from indexcalc.core.variation import ZeroTensor

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
    expr_to_tokens,
    tokens_to_expr,
    build_vocab,
    STRUCTURE_TOKENS,
)
from indexcalc.lions.serializer import collect_spaces
from indexcalc.lions.presets.su2_higgs_scalar import build_b0
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1
from indexcalc.lions.presets.su2_gauge import build_b2
from indexcalc.lions.presets.sm_lite import build_sm_lite


# ─── Atomic units ──────────────────────────────────────


def test_tensor_round_trip_with_metadata():
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    F = Tensor(
        "F",
        [adj.upper("A"), st.lower("μ"), st.lower("ν")],
        antisymmetric_pairs=[(1, 2)],
        reps={"SU(2)": "adj", "Lorentz": "vector"},
        statistics="bosonic",
    )
    tokens = expr_to_tokens(F)
    back = tokens_to_expr(tokens, {"spacetime": st, "su2_adj": adj})
    assert canonical_form_modulo_dummies(F) == canonical_form_modulo_dummies(back)
    assert back.antisymmetric_pairs == F.antisymmetric_pairs
    assert back.reps == F.reps


def test_scalar_mul_each_scalar_type():
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    T = Tensor("T", [st.upper("μ")], reps={})
    for scalar in (1, -1, 0.5, -0.25, 1j, complex(0.5, -0.5)):
        expr = ScalarMul(scalar, T)
        tokens = expr_to_tokens(expr)
        back = tokens_to_expr(tokens, {"spacetime": st})
        assert isinstance(back, ScalarMul)
        assert back.scalar == scalar, (
            f"scalar mismatch {scalar!r} → {back.scalar!r}"
        )


def test_partial_deriv_round_trip():
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    dH = partial(H, st.lower("μ"))
    tokens = expr_to_tokens(dH)
    back = tokens_to_expr(tokens, {"spacetime": st, "su2_fund": su2})
    assert (canonical_form_modulo_dummies(dH)
            == canonical_form_modulo_dummies(back))


def test_zero_tensor_round_trip():
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    z = ZeroTensor([st.upper("μ")])
    tokens = expr_to_tokens(z)
    back = tokens_to_expr(tokens, {"spacetime": st})
    assert isinstance(back, ZeroTensor)
    assert len(back.free_indices) == 1


# ─── Preset round-trip ─────────────────────────────────


@pytest.fixture(params=[
    ("b0", build_b0, lambda b: {"SU(2)": b.su2_gen, "U(1)_Y": b.u1y_gen}),
    ("b1", build_b1, lambda b: {"SU(2)": b.su2_gen, "U(1)_Y": b.u1y_gen,
                                "Lorentz": b.lorentz_gen}),
    ("b2", build_b2, lambda b: {"SU(2)": b.su2_gen, "Lorentz": b.lorentz_gen}),
])
def preset_setup(request):
    name, build, gens = request.param
    setup = build()
    return name, setup, gens(setup)


def _enum(setup, generators, **caps_kw):
    defaults = dict(
        max_field_total=3, max_per_field=2,
        max_partials_total=1, max_partials_per_field=1,
    )
    defaults.update(caps_kw)
    samples = enumerate_scalar_invariants(
        setup.fields, spacetime=setup.spacetime,
        caps=EnumeratorCaps(**defaults),
        invariant_alphabet=getattr(setup, "invariant_alphabet", None),
        forbid_like_position_spaces=(
            {setup.dirac} if hasattr(setup, "dirac") else None
        ),
    )
    return label_samples(samples, generators)


def test_preset_round_trip(preset_setup):
    name, setup, generators = preset_setup
    labeled = _enum(setup, generators)
    assert labeled, f"{name}: nothing enumerated"
    for s in labeled:
        spaces = collect_spaces(s.expr)
        tokens = expr_to_tokens(s.expr)
        back = tokens_to_expr(tokens, spaces)
        assert (canonical_form_modulo_dummies(s.expr)
                == canonical_form_modulo_dummies(back)), (
            f"{name}: round-trip canonical mismatch"
        )


def test_preset_oracle_determinism(preset_setup):
    """After tokenize → parse, re-running the oracle gives the same labels."""
    name, setup, generators = preset_setup
    labeled = _enum(setup, generators)
    for s in labeled:
        spaces = collect_spaces(s.expr)
        back = tokens_to_expr(expr_to_tokens(s.expr), spaces)
        for g_name, gen in generators.items():
            delta = apply_generator(back, gen)
            final = simplify(delta)
            recomputed = isinstance(final, ZeroTensor)
            assert recomputed == s.labels[g_name], (
                f"{name}: oracle disagreement under {g_name} on "
                f"{s.field_counts}"
            )


# ─── SM-lite at scale ───────────────────────────────────


@pytest.fixture(scope="module")
def sm_dataset():
    sm = build_sm_lite()
    caps = EnumeratorCaps(
        max_field_total=4, max_per_field=2,
        max_partials_total=1, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        sm.fields, spacetime=sm.spacetime, caps=caps,
        invariant_alphabet=sm.invariant_alphabet,
        forbid_like_position_spaces={sm.dirac},
    )
    generators = {
        "SU(2)": sm.su2_gen, "U(1)_Y": sm.u1y_gen, "Lorentz": sm.lorentz_gen,
    }
    return sm, label_samples(samples, generators)


def test_sm_lite_round_trip_at_scale(sm_dataset):
    _sm, labeled = sm_dataset
    for s in labeled:
        spaces = collect_spaces(s.expr)
        back = tokens_to_expr(expr_to_tokens(s.expr), spaces)
        assert (canonical_form_modulo_dummies(s.expr)
                == canonical_form_modulo_dummies(back))


def test_sm_lite_vocab_covers_dataset(sm_dataset):
    _sm, labeled = sm_dataset
    vocab = build_vocab(labeled)
    # Every structural token must be present.
    for st_tok in STRUCTURE_TOKENS:
        assert st_tok in vocab
    # Every dataset token must be present.
    for s in labeled:
        for tok in expr_to_tokens(s.expr):
            assert tok in vocab, f"token {tok!r} missing from vocab"
    # Vocab ids are dense (no gaps).
    ids = sorted(vocab.values())
    assert ids == list(range(len(vocab)))
