"""ML-D10 model wiring tests — RGCNClassifier forward + tiny overfit.

These tests need torch/PyG. Skip cleanly if missing.
"""

from __future__ import annotations
import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch
from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct
from indexcalc.lions import graph_encode
from indexcalc.lions.ml.features import GROUP_ORDER, num_relations
from indexcalc.lions.ml.pyg_bridge import encoded_to_pyg_data
from indexcalc.lions.ml.models import RGCNClassifier, GTClassifier
from indexcalc.lions.ml.train import auc_roc


def _two_class_synthetic(n_per_class: int = 8):
    """Two trivially-separable graphs:

    - class A: H · Hdag (SU(2) labels all True).
    - class B: H · H   (SU(2) False for SU(2) head).
    """
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H_up = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    H2 = Tensor("H", [su2.upper("j")], reps={"SU(2)": "fund"})

    pos = TensorProduct(H_up, Hd)
    neg = TensorProduct(H_up, H2)
    pos_g = graph_encode(pos); pos_g.labels = {g: True for g in GROUP_ORDER}
    neg_g = graph_encode(neg); neg_g.labels = {g: False for g in GROUP_ORDER}

    data_list = (
        [encoded_to_pyg_data(pos_g) for _ in range(n_per_class)]
        + [encoded_to_pyg_data(neg_g) for _ in range(n_per_class)]
    )
    return data_list


def test_rgcn_forward_shape():
    """Forward pass on a small batch returns [batch, num_groups] logits."""
    data_list = _two_class_synthetic(n_per_class=2)
    batch = Batch.from_data_list(data_list)
    model = RGCNClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, dropout=0.0,
    )
    out = model(batch)
    assert out.shape == (4, len(GROUP_ORDER))
    assert out.dtype == torch.float32


def test_rgcn_overfits_two_class():
    """Trivially separable two-class problem reaches AUC=1 within 30 epochs."""
    torch.manual_seed(0)
    data_list = _two_class_synthetic(n_per_class=8)
    loader = [Batch.from_data_list(data_list)]  # one batch, full data
    model = RGCNClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, dropout=0.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(30):
        for batch in loader:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()

    with torch.no_grad():
        out = model(Batch.from_data_list(data_list))
        probs = torch.sigmoid(out)
        ys = torch.cat([d.y for d in data_list], dim=0)
        # SU(2) head should be perfectly ranked (positives>negatives).
        auc = auc_roc(ys[:, 0], probs[:, 0])
        assert auc == 1.0, f"SU(2) AUC = {auc}, expected 1.0"


def _h_norm_pyg(term_label_su2: bool = True):
    """Single-term |H|² PyG data with SU(2) label set explicitly."""
    su2 = IndexSpace("su2_fund", dim=2, indices="ij")
    H = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    g = graph_encode(TensorProduct(H, Hd))
    g.labels = {grp: term_label_su2 for grp in GROUP_ORDER}
    return encoded_to_pyg_data(g)


def test_per_term_min_aggregation_matches_global_for_single_term():
    """For a 1-term graph the I2 per-term path must agree (exactly) with
    the legacy global_mean_pool readout — otherwise we'd silently change
    the meaning of every non-Sum sample."""
    from torch_geometric.nn import global_mean_pool
    torch.manual_seed(0)
    d = _h_norm_pyg()
    batch = Batch.from_data_list([d, d, d])
    model = RGCNClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        out_per_term = model(batch)
        # Manually replicate the legacy path: global_mean_pool over batch.
        x_float = getattr(batch, "x_float", None)
        h = model.encode_nodes(batch.x, x_float)
        for conv in model.convs:
            h = torch.nn.functional.relu(conv(h, batch.edge_index, batch.edge_type))
        legacy = model.head(global_mean_pool(h, batch.batch))
    assert torch.allclose(out_per_term, legacy, atol=1e-6)


def test_per_term_min_picks_broken_term():
    """Manually construct logits across two terms and verify that the
    AND-via-min readout returns the smaller (broken) term's logit."""
    from indexcalc.core.tensor import TensorSum

    # Build a two-term graph: two scalar |H|² terms with disjoint dummy
    # names. The min-readout target is structural (each summand has its
    # own per-term logit); the actual labels don't matter for this test.
    su2 = IndexSpace("su2_fund", dim=2, indices="ijkl")
    H1 = Tensor("H", [su2.upper("i")], reps={"SU(2)": "fund"})
    Hd1 = Tensor("Hdag", [su2.lower("i")], reps={"SU(2)": "fund"})
    H2 = Tensor("H", [su2.upper("k")], reps={"SU(2)": "fund"})
    Hd2 = Tensor("Hdag", [su2.lower("k")], reps={"SU(2)": "fund"})
    inv = graph_encode(TensorProduct(H1, Hd1))
    inv.labels = {grp: True for grp in GROUP_ORDER}
    brk = graph_encode(TensorSum(TensorProduct(H1, Hd1),
                                 TensorProduct(H2, Hd2)))
    brk.labels = {grp: False for grp in GROUP_ORDER}

    torch.manual_seed(0)
    model = RGCNClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, dropout=0.0,
    )
    model.eval()
    batch = Batch.from_data_list([
        encoded_to_pyg_data(inv),
        encoded_to_pyg_data(brk),
    ])
    with torch.no_grad():
        out = model(batch)
        # Manually pull per-term logits for the second sample and check
        # that out[1] == min over its two per-term logits.
        x_float = getattr(batch, "x_float", None)
        h = model.encode_nodes(batch.x, x_float)
        for conv in model.convs:
            h = torch.nn.functional.relu(conv(h, batch.edge_index, batch.edge_type))
        # composite IDs: graph 0 has 1 term (id 0), graph 1 has 2 terms
        # (ids 1, 2).
        from torch_geometric.nn import global_mean_pool
        offsets = torch.tensor([0, 1])
        composite = offsets[batch.batch] + batch.term_id
        pooled = global_mean_pool(h, composite)
        per_term = model.head(pooled)  # [3, G]
        expected = torch.minimum(per_term[1], per_term[2])
    assert torch.allclose(out[1], expected, atol=1e-6)
    # And graph 0 is single-term → out[0] == per_term[0].
    assert torch.allclose(out[0], per_term[0], atol=1e-6)


def test_gt_forward_shape():
    """GT forward returns the same [batch, num_groups] shape as RGCN."""
    data_list = _two_class_synthetic(n_per_class=2)
    batch = Batch.from_data_list(data_list)
    model = GTClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, num_heads=4, dropout=0.0,
    )
    out = model(batch)
    assert out.shape == (4, len(GROUP_ORDER))
    assert out.dtype == torch.float32


def test_gt_overfits_two_class():
    """GT can perfectly separate the two trivial classes in 30 epochs."""
    torch.manual_seed(0)
    data_list = _two_class_synthetic(n_per_class=8)
    loader = [Batch.from_data_list(data_list)]
    model = GTClassifier(
        hidden_dim=32, num_relations=num_relations(),
        num_layers=2, num_heads=4, dropout=0.0,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(30):
        for batch in loader:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
    with torch.no_grad():
        out = model(Batch.from_data_list(data_list))
        probs = torch.sigmoid(out)
        ys = torch.cat([d.y for d in data_list], dim=0)
        auc = auc_roc(ys[:, 0], probs[:, 0])
        assert auc == 1.0, f"GT SU(2) AUC={auc}, expected 1.0"


def test_auc_implementation():
    """Mann–Whitney AUC sanity: perfect=1, worst=0, random≈0.5."""
    assert auc_roc(
        torch.tensor([0., 0., 1., 1.]),
        torch.tensor([0.1, 0.2, 0.8, 0.9]),
    ) == 1.0
    assert auc_roc(
        torch.tensor([0., 1.]),
        torch.tensor([1., 0.]),
    ) == 0.0
    # All-positive or all-negative is undefined → nan.
    import math
    auc = auc_roc(
        torch.tensor([1., 1., 1.]),
        torch.tensor([0.1, 0.5, 0.9]),
    )
    assert math.isnan(auc)
