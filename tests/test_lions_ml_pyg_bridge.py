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
    assert d.x.shape == (2, 7)
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
    assert d.y.tolist() == [[1.0, 1.0, 1.0]]
    assert d.y_mask.tolist() == [[1.0, 1.0, 1.0]]


def test_missing_group_masked():
    """Group not in g.labels → y=0, y_mask=0 (loss-masked)."""
    g = _h_norm_graph()
    g.labels = {"SU(2)": True}  # only SU(2) declared
    d = encoded_to_pyg_data(g)
    # GROUP_ORDER = ("SU(2)", "U(1)_Y", "Lorentz")
    assert d.y.tolist() == [[1.0, 0.0, 0.0]]
    assert d.y_mask.tolist() == [[1.0, 0.0, 0.0]]


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
