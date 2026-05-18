"""``EncodedGraph`` → ``torch_geometric.data.Data`` bridge.

This is the only place where the dependency-free LIONS graph meets
PyTorch. Everything upstream (``indexcalc.lions.graph``) stays free of
torch — backend acceptance tests do not need it installed.

Conventions (from ``LIONS/notes/ml_training_v1.md`` §5):
- One ``Data`` object per ``EncodedGraph``. Multi-graph batching is
  PyG's standard ``Batch.from_data_list``.
- Edges are emitted in BOTH directions (D9b stores canonical one only).
- ``y`` is a fixed-length float vector over ``features.GROUP_ORDER``;
  ``y_mask`` is 1 where the sample has that group declared, 0 otherwise.
- ``edge_type`` is the packed int from ``features.edge_feature_ids``;
  ``edge_attr`` is the unpacked 4-int component list.
"""

from __future__ import annotations
from typing import Optional, Sequence

from indexcalc.lions.ml import _require_torch
from indexcalc.lions.ml.features import (
    GROUP_ORDER, node_feature_ids, edge_feature_ids,
)
from indexcalc.lions.graph import EncodedGraph


def encoded_to_pyg_data(
    g: EncodedGraph,
    label_order: Sequence[str] = GROUP_ORDER,
):
    """Convert an ``EncodedGraph`` into a PyG ``Data`` object.

    Parameters
    ----------
    g
        Output of ``indexcalc.lions.graph.graph_encode`` (or
        ``encode_sample``). Must be non-None — callers handle ZeroTensor.
    label_order
        Group names in the order ``y[i]`` corresponds to. Defaults to
        ``features.GROUP_ORDER`` = ("SU(2)", "U(1)_Y", "Lorentz").
    """
    _require_torch()
    import torch
    from torch_geometric.data import Data

    # ── Node features ───────────────────────────────────
    if not g.nodes:
        # Defensive: a fully-empty graph would be a degenerate sample.
        # Caller should usually have filtered ZeroTensor → None upstream.
        x = torch.zeros((0, 7), dtype=torch.long)
    else:
        x = torch.tensor(
            [node_feature_ids(
                n.kind, n.name, n.rank, n.statistics, n.reps,
            ) for n in g.nodes],
            dtype=torch.long,
        )

    # ── Edges: emit both directions ─────────────────────
    if g.edges:
        src_list, dst_list, type_list, attr_list = [], [], [], []
        for e in g.edges:
            et, attr = edge_feature_ids(e.kind, e.space, e.src_pos, e.dst_pos)
            # forward
            src_list.append(e.src); dst_list.append(e.dst)
            type_list.append(et); attr_list.append(attr)
            # reverse (mirror src/dst positions)
            et_rev, attr_rev = edge_feature_ids(
                e.kind, e.space, e.dst_pos, e.src_pos,
            )
            src_list.append(e.dst); dst_list.append(e.src)
            type_list.append(et_rev); attr_list.append(attr_rev)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.long)

    # ── Labels ──────────────────────────────────────────
    y_vals = []
    y_mask = []
    for group in label_order:
        if group in g.labels:
            y_vals.append(1.0 if g.labels[group] else 0.0)
            y_mask.append(1.0)
        else:
            y_vals.append(0.0)
            y_mask.append(0.0)
    y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(0)
    y_mask_t = torch.tensor(y_mask, dtype=torch.float).unsqueeze(0)

    # ── Graph-level scalar ──────────────────────────────
    scalar = complex(g.scalar)
    scalar_re = torch.tensor([scalar.real], dtype=torch.float)
    scalar_im = torch.tensor([scalar.imag], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        y=y,
        y_mask=y_mask_t,
        scalar_re=scalar_re,
        scalar_im=scalar_im,
        num_nodes=len(g.nodes),
    )


def encoded_list_to_pyg(
    encoded: Sequence[Optional[EncodedGraph]],
    label_order: Sequence[str] = GROUP_ORDER,
) -> list:
    """Vectorized convenience: drop None (ZeroTensor) entries."""
    return [
        encoded_to_pyg_data(g, label_order)
        for g in encoded if g is not None
    ]
