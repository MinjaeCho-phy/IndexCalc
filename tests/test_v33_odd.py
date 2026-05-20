"""v3.3: O(D,D) T-duality catalog entry (doubled/embedding-space formalism).

O(D,D) is treated as an orthogonal group on the 2D-dim doubled space with an
indefinite split-signature ``dd`` metric — like the conformal SO(d,2) case.
Its symmetric invariant η^dd is distinguished from Euclidean δ, conformal
η^conf, and symplectic Ω purely by the metric name, so a dim-8 η^dd bilinear
labels O(4,4) only — not Sp(8); and a dim-4 η^dd bilinear labels O(2,2) only,
not SO(4)/Sp(4)/SO(2,2). Unlike SO(d,2), O(D,D) carries no Levi-Civita (it is
the full T-duality group O, not SO), mirroring O(N).
"""

from __future__ import annotations
import pytest

from indexcalc.core.invariant_tensors import standard_o_dd_invariants
from indexcalc.lions.catalog import CATALOG, get, build_groupspec
from indexcalc.lions.catalog_enumerator import enumerate_for_entry
from indexcalc.lions.catalog_labeler import (
    label_lagrangian, collect_tensor_signature, _owned_space_signature,
)
from indexcalc.lions.ml.features_v25 import (
    NODE_NAME, PRIMARY_METRIC, node_feature_ids_v25,
)


# ─── invariants ──────────────────────────────────────────


def test_o_dd_invariants_symmetric_metric_no_epsilon():
    inv = standard_o_dd_invariants(4)  # O(4,4), doubled dim 8
    names = {t.name for t in inv}
    assert names == {"eta_dd", "eta_dd_mixed"}  # no ε (O, not SO)
    eta = next(t for t in inv if t.name == "eta_dd")
    assert eta.symmetry == "symmetric"
    assert eta.group_name == "O(4,4)"


# ─── catalog ─────────────────────────────────────────────


def test_catalog_has_three_odd_entries():
    odd = [e for e in CATALOG if e.family == "split_orthogonal"]
    assert [e.label for e in odd] == ["O(2,2)", "O(3,3)", "O(4,4)"]
    assert len(CATALOG) == 29


@pytest.mark.parametrize("label,doubled_dim,dim", [
    ("O(2,2)", 4, 6), ("O(3,3)", 6, 15), ("O(4,4)", 8, 28),
])
def test_odd_groupspec_dims(label, doubled_dim, dim):
    spec = build_groupspec(get(label), prefix="t_")
    assert spec.group.get_rep("vector").dim == doubled_dim
    assert spec.dim == dim  # 2D(2D-1)/2


def test_owned_space_is_dd_metric():
    assert _owned_space_signature(get("O(4,4)")) == (8, "dd")
    assert _owned_space_signature(get("O(2,2)")) == (4, "dd")


# ─── enumeration + labeling ──────────────────────────────


def test_odd_enumerates_eta_dd_bilinears():
    setup, samples = enumerate_for_entry(get("O(4,4)"), prefix="e_")
    assert len(samples) > 0
    sigs = [collect_tensor_signature(s.expr) for s in samples]
    assert any(("eta_dd", 8, "dd", 2) in sg for sg in sigs)


def test_eta_dd_dim8_labels_o44_only_not_sp8():
    """η^dd on the dim-8 doubled space → O(4,4); the dim-8 neighbour Sp(8)
    (Ω, metric-less) must NOT match."""
    setup, samples = enumerate_for_entry(get("O(4,4)"), prefix="g_")
    bilinear = next(
        s for s in samples
        if ("eta_dd", 8, "dd", 2) in collect_tensor_signature(s.expr)
    )
    labels = label_lagrangian(bilinear.expr, get("O(4,4)"))
    pos = {k for k, v in labels.items() if v}
    assert pos == {"O(4,4)"}, pos
    assert not labels["Sp(8)"]
    assert not labels["Lorentz"]


def test_sp8_omega_does_not_label_o44():
    """Reverse: a Sp(8) Ω-bilinear (metric-less, antisymmetric) must NOT
    match O(4,4) (symmetric dd metric) at the shared dim 8."""
    setup, samples = enumerate_for_entry(get("Sp(8)"), prefix="s_")
    om = next(
        s for s in samples
        if ("omega", 8, "", 2) in collect_tensor_signature(s.expr)
    )
    labels = label_lagrangian(om.expr, get("Sp(8)"))
    assert labels["Sp(8)"]
    assert not labels["O(4,4)"]


def test_eta_dd_dim4_labels_o22_only_not_so4_sp4_so22():
    """The dim-4 four-way collision: η^dd → O(2,2) only; SO(4) (δ),
    Sp(4) (Ω), SO(2,2) (η^conf) all share dim 4 but differ by metric."""
    setup, samples = enumerate_for_entry(get("O(2,2)"), prefix="h_")
    bilinear = next(
        s for s in samples
        if ("eta_dd", 4, "dd", 2) in collect_tensor_signature(s.expr)
    )
    labels = label_lagrangian(bilinear.expr, get("O(2,2)"))
    pos = {k for k, v in labels.items() if v}
    assert pos == {"O(2,2)"}, pos
    for neighbour in ("SO(4)", "Sp(4)", "SO(2,2)", "O(4)"):
        assert not labels[neighbour]


# ─── ML features ─────────────────────────────────────────


def test_eta_dd_node_feature_encoding():
    assert NODE_NAME["eta_dd"] == 16
    assert PRIMARY_METRIC["dd"] == 5
    feat = node_feature_ids_v25(
        "invariant", "eta_dd", 2, "bosonic",
        primary_dim=8, primary_metric="dd",
    )
    # layout: [kind, name, rank, statistics, stats_hint, antisym_hint,
    #          primary_dim, primary_metric_id]
    assert feat[1] == 16        # name → eta_dd
    assert feat[6] == 8         # primary_dim (raw)
    assert feat[7] == 5         # primary_metric → dd
