"""LIONS ML models.

v1: ``RGCNClassifier`` — multi-relation GCN baseline for Task 1
(group-invariance multi-label binary classification).

Per ``LIONS/notes/ml_training_v1.md`` §6.1:
- Node features = embedding lookup over 5 categorical fields
  (kind, name, statistics, rep_SU(2), rep_U(1)_Y, rep_Lorentz)
  + scalar rank (normalized).
- Edge dispatch = RGCNConv on packed ``edge_type`` (G7 default).
- Graph readout = global_mean_pool.
- Output = 3-head logit vector (per ``features.GROUP_ORDER``).
"""

from __future__ import annotations
from indexcalc.lions.ml import _require_torch

_require_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool

from indexcalc.lions.ml.features import (
    GROUP_ORDER, NODE_KIND, NODE_NAME, STATISTICS, REP_VOCAB,
    num_charge_features,
)


def _vsize(table: dict) -> int:
    """Vocabulary size = max ID + 1 (IDs are not contiguous; use max)."""
    return max(table.values()) + 1


class RGCNClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_relations: int = 1600,   # cf. features.num_relations()
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Vocab sizes from features.py.
        v_kind   = _vsize(NODE_KIND)
        v_name   = _vsize(NODE_NAME)
        v_stat   = _vsize(STATISTICS)
        v_su2    = _vsize(REP_VOCAB["SU(2)"])
        v_u1y    = _vsize(REP_VOCAB["U(1)_Y"])
        v_lorz   = _vsize(REP_VOCAB["Lorentz"])

        emb_dim = hidden_dim // 4  # 16 per embedding when hidden_dim=64
        self.emb_kind = nn.Embedding(v_kind, emb_dim)
        self.emb_name = nn.Embedding(v_name, emb_dim)
        self.emb_stat = nn.Embedding(v_stat, emb_dim // 2)
        self.emb_su2  = nn.Embedding(v_su2,  emb_dim // 2)
        self.emb_u1y  = nn.Embedding(v_u1y,  emb_dim // 2)
        self.emb_lorz = nn.Embedding(v_lorz, emb_dim // 2)

        # I1: numeric charge features (U(1)_Y for now). Concatenated
        # alongside the categorical embeddings so the GNN can compute
        # charge sums via message passing.
        self.n_charge = num_charge_features()
        node_in_dim = (emb_dim * 2          # kind, name
                       + (emb_dim // 2) * 4 # stat, su2, u1y, lorentz
                       + 1                  # rank scalar
                       + self.n_charge)     # u1y charge etc.
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations,
                     num_bases=32)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(GROUP_ORDER)),
        )

    def encode_nodes(
        self, x: torch.Tensor, x_float: torch.Tensor = None,
    ) -> torch.Tensor:
        # x: [N, 7] long — see features.node_feature_ids order:
        # [kind, name, rank, stat, su2, u1y, lorentz]
        # x_float: [N, n_charge] float (I1 charge features). May be None
        # for back-compat with checkpoints that predate I1.
        kind = self.emb_kind(x[:, 0])
        name = self.emb_name(x[:, 1])
        rank = x[:, 2].float().unsqueeze(-1) / 4.0    # /4 rough normalize
        stat = self.emb_stat(x[:, 3])
        su2  = self.emb_su2 (x[:, 4])
        u1y  = self.emb_u1y (x[:, 5])
        lorz = self.emb_lorz(x[:, 6])
        if x_float is None:
            x_float = torch.zeros(
                x.shape[0], self.n_charge, device=x.device, dtype=torch.float,
            )
        h = torch.cat(
            [kind, name, rank, stat, su2, u1y, lorz, x_float], dim=-1,
        )
        return self.node_proj(h)

    def forward(self, data):
        x_float = getattr(data, "x_float", None)
        h = self.encode_nodes(data.x, x_float)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_type)
            h = F.relu(h)
            h = self.dropout(h)
        g = global_mean_pool(h, data.batch)
        return self.head(g)   # [batch, num_groups]
