"""v3.1 R2: multi-index fields charged under several groups at once.

``setup_multi_index`` / ``enumerate_multi_index`` build fields ψ carrying a
slot per sector (ψ^{iμ} ∈ SU(3) fund × Lorentz vector, etc.). With the
per-sector labeler (R1) every sector's group comes out positive — and a
single term may carry invariant tensors from *two* spaces (δ_ik Ω_jl F^{ij}
F^{kl}), exercising the per-sector tensor logic directly.
"""

from __future__ import annotations
import pytest

from indexcalc.lions.catalog import get
from indexcalc.lions.catalog_enumerator import (
    setup_multi_index, enumerate_multi_index,
)
from indexcalc.lions.catalog_labeler import (
    label_lagrangian, collect_tensor_signature, collect_field_rep_signature,
)


def _positives(expr, primary):
    labels = label_lagrangian(expr, primary)
    return {k for k, v in labels.items() if v}


def test_setup_multi_index_builds_dual_charged_fields():
    entries = [get("Lorentz"), get("SU(3)")]
    setup = setup_multi_index(entries, prefix="t_")
    f = setup.registry.fields()[0]
    # Each field carries one slot per sector and reps for both groups.
    assert len(f.slots) == 2
    assert set(f.reps.keys()) == {"Lorentz", "SU(3)"}


@pytest.mark.parametrize("pair,expected", [
    (["Lorentz", "SU(3)"], {"Lorentz", "Poincare", "SU(3)", "U(3)"}),
    (["Lorentz", "SO(3)"], {"Lorentz", "Poincare", "SO(3)", "O(3)"}),
    (["SO(3)", "Sp(4)"], {"SO(3)", "O(3)", "Sp(4)"}),
])
def test_multi_index_samples_are_multipositive_per_sector(pair, expected):
    entries = [get(p) for p in pair]
    setup, samples = enumerate_multi_index(entries, prefix="t_")
    assert len(samples) > 0
    # Every sample's positives must contain exactly the union of both
    # sectors' groups (no foreign group, no missing sector).
    for s in samples[:25]:
        pos = _positives(s.expr, entries[0])
        assert expected <= pos, f"{s.expr}: {pos} missing {expected - pos}"
        # No wrong-N leakage from the same families.
        assert "SO(4)" not in pos
        assert "Sp(6)" not in pos


def test_so_sp_dual_tensor_term_exists_and_labels_both():
    """A single multi-index field F^{ij} (i∈SO(3), j∈Sp(4)) admits a term
    carrying δ (SO) and Ω (Sp) at once — both sectors must label positive."""
    entries = [get("SO(3)"), get("Sp(4)")]
    setup, samples = enumerate_multi_index(entries, prefix="t_")
    dual = [
        s for s in samples
        if {("delta", 3, "delta", 2), ("omega", 4, "", 2)}
        <= collect_tensor_signature(s.expr)
    ]
    assert dual, "no δ+Ω dual-tensor multi-index term enumerated"
    pos = _positives(dual[0].expr, entries[0])
    assert {"SO(3)", "O(3)", "Sp(4)"} <= pos, pos
