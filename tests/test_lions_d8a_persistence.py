"""D8a — dataset persistence round-trip tests.

Save → load preserves:
- the IR structure (canonical_form_modulo_dummies key match),
- the labels dict,
- oracle determinism (apply_generator+simplify on the loaded expr
  gives the same ZeroTensor outcome).

Covers Tensor / TensorProduct / TensorSum / ScalarMul / PartialDeriv /
ZeroTensor, plus all slot metadata (antisym/sym/traceless/transverse,
reps, statistics), via the three LIONS preset pipelines (B0, B1, B2).
"""

from __future__ import annotations
import pytest

from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import partial
from indexcalc.core.simplify import canonical_form_modulo_dummies
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify
from indexcalc.core.variation import ZeroTensor

from indexcalc.lions import (
    EnumeratorCaps,
    enumerate_scalar_invariants,
    label_samples,
    save_dataset,
    load_dataset,
    expr_to_dict,
    expr_from_dict,
    sample_to_dict,
    sample_from_dict,
)
from indexcalc.lions.serializer import collect_spaces, SCHEMA_VERSION
from indexcalc.lions.presets.su2_higgs_scalar import build_b0
from indexcalc.lions.presets.su2_higgs_yukawa import build_b1
from indexcalc.lions.presets.su2_gauge import build_b2


# ─── Building-block round-trips ──────────────────────────


def test_tensor_with_all_metadata_round_trip():
    """Tensor with antisym + sym + reps + statistics survives JSON."""
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
    d = expr_to_dict(F)
    spaces = {"spacetime": st, "su2_adj": adj}
    F_loaded = expr_from_dict(d, spaces)

    assert canonical_form_modulo_dummies(F) == canonical_form_modulo_dummies(F_loaded)
    assert F_loaded.antisymmetric_pairs == F.antisymmetric_pairs
    assert F_loaded.reps == F.reps
    assert F_loaded.statistics == F.statistics


def test_scalar_mul_complex_round_trip():
    """ScalarMul(1j, ...) survives via {re,im} encoding."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    T = Tensor("T", [st.upper("μ")], reps={})
    expr = ScalarMul(1j, T)
    d = expr_to_dict(expr)
    loaded = expr_from_dict(d, {"spacetime": st})
    assert isinstance(loaded, ScalarMul)
    assert loaded.scalar == 1j


def test_partial_deriv_round_trip():
    """PartialDeriv with deriv_index round-trips."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    T = Tensor("φ", [], reps={})
    dT = partial(T, st.lower("μ"))
    d = expr_to_dict(dT)
    loaded = expr_from_dict(d, {"spacetime": st})
    assert canonical_form_modulo_dummies(dT) == canonical_form_modulo_dummies(loaded)


# ─── End-to-end via presets ──────────────────────────────


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


def _enumerate_labeled(setup, generators):
    caps = EnumeratorCaps(
        max_field_total=3, max_per_field=2,
        max_partials_total=2, max_partials_per_field=1,
    )
    samples = enumerate_scalar_invariants(
        setup.fields, spacetime=setup.spacetime, caps=caps,
        invariant_alphabet=getattr(setup, "invariant_alphabet", None),
        forbid_like_position_spaces=(
            {setup.dirac} if hasattr(setup, "dirac") else None
        ),
    )
    return label_samples(samples, generators)


def test_preset_dataset_round_trip(tmp_path, preset_setup):
    """save → load preserves canonical form + labels for every preset."""
    name, setup, generators = preset_setup
    labeled = _enumerate_labeled(setup, generators)
    assert labeled, f"{name}: enumerate produced nothing"

    path = tmp_path / f"{name}_dataset.json"
    save_dataset(labeled, path)
    loaded = load_dataset(path)

    assert len(loaded) == len(labeled)
    for orig, back in zip(labeled, loaded):
        # IR structural equality (modulo dummies).
        assert (canonical_form_modulo_dummies(orig.expr)
                == canonical_form_modulo_dummies(back.expr))
        # Label dict equality.
        assert orig.labels == back.labels
        # Counts preserved.
        assert orig.field_counts == back.field_counts
        assert orig.mass_dim == back.mass_dim
        assert orig.partial_count == back.partial_count
        assert orig.provenance == back.provenance


def test_oracle_determinism_after_reload(tmp_path, preset_setup):
    """After load, re-running the oracle gives the same labels.

    Guards against (a) reps/statistics drift on Tensor reconstruction,
    (b) generator dependence on the *original* Tensor identity (we want
    purely value-based action).
    """
    name, setup, generators = preset_setup
    labeled = _enumerate_labeled(setup, generators)

    path = tmp_path / f"{name}_oracle.json"
    save_dataset(labeled, path)
    loaded = load_dataset(path)

    for back in loaded:
        for g_name, gen in generators.items():
            delta = apply_generator(back.expr, gen)
            final = simplify(delta)
            recomputed = isinstance(final, ZeroTensor)
            assert recomputed == back.labels[g_name], (
                f"{name}: oracle disagreement on {back.field_counts} "
                f"under {g_name}: stored={back.labels[g_name]}, "
                f"recomputed={recomputed}"
            )


# ─── Schema versioning ──────────────────────────────────


def test_load_rejects_bad_schema_version(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 999, "spaces": {}, "samples": []}))
    with pytest.raises(ValueError, match="schema version"):
        load_dataset(path)


def test_collect_spaces_picks_up_partial_deriv():
    """Index inside PartialDeriv.deriv_index also gets into the header."""
    from indexcalc.core.index import IndexSpace
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    T = Tensor("φ", [], reps={})
    expr = partial(T, st.lower("μ"))
    spaces = collect_spaces(expr)
    assert "spacetime" in spaces
