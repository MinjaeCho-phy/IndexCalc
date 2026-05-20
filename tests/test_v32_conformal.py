"""v3.2: conformal SO(d,2) catalog entry (embedding-space formalism).

SO(d,2) is treated as an orthogonal group on the (d+2)-dim embedding space
with an indefinite ``conf`` metric — like Lorentz=SO(1,3). Its symmetric
invariant η^conf is distinguished from Euclidean SO(d+2)'s δ and Sp(d+2)'s Ω
purely by the metric name, so a dim-6 η^conf bilinear labels SO(4,2) only —
not SO(6) or Sp(6).
"""

from __future__ import annotations
import pytest

from indexcalc.core.invariant_tensors import standard_conformal_invariants
from indexcalc.lions.catalog import CATALOG, get, build_groupspec
from indexcalc.lions.catalog_enumerator import enumerate_for_entry
from indexcalc.lions.catalog_labeler import (
    label_lagrangian, collect_tensor_signature, _owned_space_signature,
)


# ─── invariants ──────────────────────────────────────────


def test_conformal_invariants_symmetric_metric_plus_epsilon():
    inv = standard_conformal_invariants(4)  # SO(4,2), embedding dim 6
    names = {t.name for t in inv}
    assert names == {"eta_conf", "eta_conf_mixed", "epsilon"}
    eta = next(t for t in inv if t.name == "eta_conf")
    assert eta.symmetry == "symmetric"
    assert eta.group_name == "SO(4,2)"
    eps = next(t for t in inv if t.name == "epsilon")
    assert eps.index_pattern == ("vector_lower",) * 6  # (d+2)-slot


# ─── catalog ─────────────────────────────────────────────


def test_catalog_has_three_conformal_entries():
    conf = [e for e in CATALOG if e.family == "conformal"]
    assert [e.label for e in conf] == ["SO(2,2)", "SO(3,2)", "SO(4,2)"]


@pytest.mark.parametrize("label,emb_dim,dim", [
    ("SO(2,2)", 4, 6), ("SO(3,2)", 5, 10), ("SO(4,2)", 6, 15),
])
def test_conformal_groupspec_dims(label, emb_dim, dim):
    spec = build_groupspec(get(label), prefix="t_")
    assert spec.group.get_rep("vector").dim == emb_dim
    assert spec.dim == dim  # (d+2)(d+1)/2


def test_owned_space_is_conf_metric():
    assert _owned_space_signature(get("SO(4,2)")) == (6, "conf")
    assert _owned_space_signature(get("SO(3,2)")) == (5, "conf")


# ─── enumeration + labeling ──────────────────────────────


def test_conformal_enumerates_eta_conf_bilinears():
    setup, samples = enumerate_for_entry(get("SO(4,2)"), prefix="e_")
    assert len(samples) > 0
    sigs = [collect_tensor_signature(s.expr) for s in samples]
    assert any(("eta_conf", 6, "conf", 2) in sg for sg in sigs)


def test_eta_conf_dim6_labels_so42_only_not_so6_or_sp6():
    """η^conf on the dim-6 conf space → SO(4,2); the dim-6 neighbours
    SO(6) (δ) and Sp(6) (Ω) must NOT match."""
    setup, samples = enumerate_for_entry(get("SO(4,2)"), prefix="g_")
    bilinear = next(
        s for s in samples
        if ("eta_conf", 6, "conf", 2) in collect_tensor_signature(s.expr)
    )
    labels = label_lagrangian(bilinear.expr, get("SO(4,2)"))
    pos = {k for k, v in labels.items() if v}
    assert pos == {"SO(4,2)"}, pos
    assert not labels["SO(5)"]   # if present
    assert not labels["Sp(6)"]
    assert not labels["Lorentz"]
