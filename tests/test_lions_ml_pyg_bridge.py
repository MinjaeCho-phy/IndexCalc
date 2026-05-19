"""ML-D10 PyG bridge — EncodedGraph → torch_geometric.data.Data.

Coverage:
- |H|² topology: 2 nodes, 2 edges (forward + reverse), correct edge_type.
- Labels: y vector follows GROUP_ORDER, y_mask = 1 for declared groups.
- ZeroTensor / empty graph: graceful (encoded_list_to_pyg filters None).
- SM-lite scale: every encoded sample produces a valid Data object,
  Batch.from_data_list collates without error.
"""

from __future__ import annotations
import pytest

# Skip the entire module cleanly if torch / PyG are missing.
torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct
from indexcalc.lions import graph_encode, EncodedGraph
from indexcalc.lions.ml.features import GROUP_ORDER
from indexcalc.lions.ml.pyg_bridge import (
    encoded_to_pyg_data, encoded_list_to_pyg,
)


def _h_norm_graph() -> EncodedGraph:
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    g = graph_encode(TensorProduct(H, Hd))
    g.labels = {"SU(2)": True, "U(1)_Y": True, "Lorentz": True}
    return g


def test_h_norm_squared_data_shape():
    """|H|² → 2 nodes, 2 directed edges (forward + reverse)."""
    g = _h_norm_graph()
    d = encoded_to_pyg_data(g)
    assert d.num_nodes == 2
    # 4 base (kind, name, rank, statistics) + len(GROUP_ORDER)=5 rep ids
    assert d.x.shape == (2, 9)
    assert d.x.dtype == torch.long
    assert d.edge_index.shape == (2, 2), "1 contraction edge → 2 directed"
    assert d.edge_type.shape == (2,)
    assert d.edge_attr.shape == (2, 4)


def test_label_vector_alignment():
    """y vector follows GROUP_ORDER; y_mask = 1 where declared."""
    g = _h_norm_graph()
    d = encoded_to_pyg_data(g)
    assert d.y.shape == (1, len(GROUP_ORDER))
    assert d.y_mask.shape == (1, len(GROUP_ORDER))
    # _h_norm_graph declares all v1 SM-lite groups True; v2 NR groups
    # (O(3)/SO(3)) are absent → y_mask = 0 for those slots.
    su2_idx = GROUP_ORDER.index("SU(2)")
    u1y_idx = GROUP_ORDER.index("U(1)_Y")
    lorz_idx = GROUP_ORDER.index("Lorentz")
    declared = {su2_idx, u1y_idx, lorz_idx}
    y = d.y[0].tolist()
    ym = d.y_mask[0].tolist()
    for i in range(len(GROUP_ORDER)):
        if i in declared:
            assert y[i] == 1.0 and ym[i] == 1.0
        else:
            assert ym[i] == 0.0


def test_missing_group_masked():
    """Group not in g.labels → y=0, y_mask=0 (loss-masked)."""
    g = _h_norm_graph()
    g.labels = {"SU(2)": True}  # only SU(2) declared
    d = encoded_to_pyg_data(g)
    su2_idx = GROUP_ORDER.index("SU(2)")
    y = d.y[0].tolist()
    ym = d.y_mask[0].tolist()
    for i in range(len(GROUP_ORDER)):
        if i == su2_idx:
            assert y[i] == 1.0 and ym[i] == 1.0
        else:
            assert ym[i] == 0.0


def test_scalar_carries_real_imag():
    g = _h_norm_graph()
    g.scalar = -1.5 + 0.5j
    d = encoded_to_pyg_data(g)
    assert d.scalar_re.item() == pytest.approx(-1.5)
    assert d.scalar_im.item() == pytest.approx(0.5)


def test_node_features_distinguish_field_vs_invariant():
    """A γ^μ invariant tensor should map to kind=invariant."""
    from indexcalc.lions.builders import make_gamma
    st = IndexSpace("spacetime", dim=4, indices="μν", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγ")
    psi = Tensor("psi", [dirac.upper("β")], reps={"Lorentz": "spinor"})
    gamma = make_gamma(st, dirac, mu="μ", row="α", col="β")
    expr = TensorProduct(gamma, psi)
    g = graph_encode(expr)
    assert g is not None
    kinds = sorted(n.kind for n in g.nodes)
    assert "invariant" in kinds and "field" in kinds


def test_list_filter_skips_none():
    """encoded_list_to_pyg drops None (ZeroTensor) entries."""
    g = _h_norm_graph()
    pyg_list = encoded_list_to_pyg([g, None, g, None])
    assert len(pyg_list) == 2


def test_batchable_with_pyg():
    """PyG Batch.from_data_list collates without error."""
    g = _h_norm_graph()
    batch = Batch.from_data_list([encoded_to_pyg_data(g) for _ in range(4)])
    assert batch.num_graphs == 4
    assert batch.x.shape[0] == 8  # 2 nodes × 4 graphs
    assert batch.y.shape == (4, len(GROUP_ORDER))


def test_charge_numeric_feature():
    """I1: U(1)_Y rep tags map to numeric scalar (Data.x_float)."""
    from indexcalc.lions.ml.features import node_charge_features

    # Direct vocab check
    assert node_charge_features({"U(1)_Y": "+1/2"}) == [0.5]
    assert node_charge_features({"U(1)_Y": "-1/2"}) == [-0.5]
    assert node_charge_features({"U(1)_Y": "+1"})   == [1.0]
    assert node_charge_features({"U(1)_Y": "-1"})   == [-1.0]
    assert node_charge_features({"U(1)_Y": "0"})    == [0.0]
    assert node_charge_features({})                 == [0.0]  # <none>

    # End-to-end through bridge: H (+1/2) · Hdag (-1/2) → [0.5, -0.5]
    g = _h_norm_graph()
    # H/Hdag are SU(2)-only in the test stub; redo with U(1)_Y reps.
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")],
               reps={"SU(2)": "fund", "U(1)_Y": "+1/2"})
    Hd = Tensor("Hdag", [su2.lower("i")],
                reps={"SU(2)": "fund", "U(1)_Y": "-1/2"})
    g = graph_encode(TensorProduct(H, Hd))
    g.labels = {"SU(2)": True, "U(1)_Y": True, "Lorentz": True}
    d = encoded_to_pyg_data(g)
    # x_float shape [2, 1] with [+0.5, -0.5]
    assert d.x_float.shape == (2, 1)
    charges = sorted(d.x_float.flatten().tolist())
    assert charges == [-0.5, 0.5]


def test_term_id_default_single_term():
    """Flat (non-Sum) graph → data.term_id all zeros, data.num_terms==1."""
    g = _h_norm_graph()
    d = encoded_to_pyg_data(g)
    assert d.term_id.shape == (2,)
    assert d.term_id.dtype == torch.long
    assert d.term_id.tolist() == [0, 0]
    assert d.num_terms.tolist() == [1]


def test_term_id_two_terms_through_bridge():
    """TensorSum of two |H|² scalars (disjoint dummies) → term_id
    [0,0,1,1], num_terms=2."""
    from indexcalc.core.tensor import TensorSum
    su2 = IndexSpace("su2_fund", dim=2, indices="ijkl")
    H1 = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd1 = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    H2 = Tensor("H", [su2.upper("k")], reps={"SU(2)": "fund"})
    Hd2 = Tensor("Hdag", [su2.lower("k")], reps={"SU(2)": "fund"})
    expr = TensorSum(TensorProduct(H1, Hd1), TensorProduct(H2, Hd2))
    g = graph_encode(expr)
    g.labels = {"SU(2)": True, "U(1)_Y": True, "Lorentz": True}
    d = encoded_to_pyg_data(g)
    assert d.term_id.tolist() == [0, 0, 1, 1]
    assert d.num_terms.tolist() == [2]


def test_term_id_batches_correctly():
    """PyG Batch concats per-node term_id along node dim; num_terms
    becomes [B] (one entry per graph)."""
    from indexcalc.core.tensor import TensorSum
    su2 = IndexSpace("su2_fund", dim=2, indices="ijkl")
    H1 = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd1 = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    H2 = Tensor("H", [su2.upper("k")], reps={"SU(2)": "fund"})
    Hd2 = Tensor("Hdag", [su2.lower("k")], reps={"SU(2)": "fund"})
    # One single-term graph (2 nodes) + one two-term graph (4 nodes).
    flat = graph_encode(TensorProduct(H1, Hd1))
    flat.labels = {"SU(2)": True}
    summed = graph_encode(TensorSum(TensorProduct(H1, Hd1),
                                    TensorProduct(H2, Hd2)))
    summed.labels = {"SU(2)": False}
    batch = Batch.from_data_list([
        encoded_to_pyg_data(flat),
        encoded_to_pyg_data(summed),
    ])
    assert batch.term_id.tolist() == [0, 0, 0, 0, 1, 1]
    assert batch.num_terms.tolist() == [1, 2]


def test_sm_lite_small_batch_smoke():
    """SM-lite enumeration at tight caps encodes + batches end-to-end."""
    from indexcalc.lions import (
        EnumeratorCaps, enumerate_scalar_invariants, label_samples,
        encode_dataset,
    )
    from indexcalc.lions.presets.sm_lite import build_sm_lite

    sm = build_sm_lite()
    caps = EnumeratorCaps(
        max_field_total=2, max_per_field=1,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        sm.fields, spacetime=sm.spacetime, caps=caps,
        invariant_alphabet=[],
        forbid_like_position_spaces={sm.dirac},
    )
    generators = {
        "SU(2)": sm.su2_gen,
        "U(1)_Y": sm.u1y_gen,
        "Lorentz": sm.lorentz_gen,
    }
    labeled = label_samples(samples, generators)
    encoded = encode_dataset(labeled)
    pyg_list = encoded_list_to_pyg(encoded)
    assert len(pyg_list) > 0, "tight caps produced zero encodable samples"
    batch = Batch.from_data_list(pyg_list[:8])
    assert batch.num_graphs == min(8, len(pyg_list))
