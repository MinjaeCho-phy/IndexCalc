"""LIONS ML models.

v1: ``RGCNClassifier`` — multi-relation GCN baseline for Task 1
(group-invariance multi-label binary classification).

Per ``LIONS/notes/ml_training_v1.md`` §6.1:
- Node features = embedding lookup over 5 categorical fields
  (kind, name, statistics, rep_SU(2), rep_U(1)_Y, rep_Lorentz)
  + scalar rank (normalized).
- Edge dispatch = RGCNConv on packed ``edge_type`` (G7 default).
- Graph readout = global_mean_pool, **per TensorSum term** (I2).
  Per-term logits are aggregated with ``min`` across terms — sum is
  invariant ⇔ every term is invariant ⇒ logit-space AND ≈ min.
- Output = 3-head logit vector (per ``features.GROUP_ORDER``).
"""

from __future__ import annotations
from indexcalc.lions.ml import _require_torch

_require_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, TransformerConv, global_mean_pool

from indexcalc.lions.ml.features import (
    GROUP_ORDER, NODE_KIND, NODE_NAME, STATISTICS, REP_VOCAB,
    num_charge_features,
)


def _vsize(table: dict) -> int:
    """Vocabulary size = max ID + 1 (IDs are not contiguous; use max)."""
    return max(table.values()) + 1


def _per_term_min_readout(h, data, head):
    """Shared I2 readout. Pool per (batch, term), apply head per term,
    aggregate with min over terms (AND-semantics in logit space).

    Falls back to ``head(global_mean_pool(h, data.batch))`` when the
    sample carries no term metadata (legacy Data construction)."""
    term_id = getattr(data, "term_id", None)
    num_terms = getattr(data, "num_terms", None)
    if term_id is None or num_terms is None:
        g = global_mean_pool(h, data.batch)
        return head(g)
    B = int(num_terms.shape[0])
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=h.device),
        num_terms.cumsum(0)[:-1],
    ])
    composite = offsets[data.batch] + term_id
    pooled = global_mean_pool(h, composite)
    logits_per_term = head(pooled)
    G = logits_per_term.shape[-1]
    batch_for_term = torch.repeat_interleave(
        torch.arange(B, device=h.device), num_terms,
    )
    out = logits_per_term.new_full((B, G), float("inf"))
    idx = batch_for_term.unsqueeze(-1).expand(-1, G)
    out.scatter_reduce_(
        0, idx, logits_per_term, reduce="amin", include_self=False,
    )
    return out


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
        return _per_term_min_readout(h, data, self.head)


class GTClassifier(nn.Module):
    """Graph Transformer baseline — drop-in alternative to ``RGCNClassifier``.

    Uses ``torch_geometric.nn.TransformerConv``: multi-head attention over
    1-hop neighbours, with edge features driving attention scores. To stay
    fair against the R-GCN relation dispatch, ``edge_type`` is embedded
    into a ``hidden_dim``-wide vector and passed as ``edge_attr``; the
    attention layer learns one relation-aware projection per head.

    Node feature encoding and the per-term min readout are identical to
    ``RGCNClassifier`` so the architecture difference is *only* in the
    propagation layer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_relations: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        # --- Same node embedding tables as RGCN ---
        v_kind   = _vsize(NODE_KIND)
        v_name   = _vsize(NODE_NAME)
        v_stat   = _vsize(STATISTICS)
        v_su2    = _vsize(REP_VOCAB["SU(2)"])
        v_u1y    = _vsize(REP_VOCAB["U(1)_Y"])
        v_lorz   = _vsize(REP_VOCAB["Lorentz"])
        emb_dim = hidden_dim // 4
        self.emb_kind = nn.Embedding(v_kind, emb_dim)
        self.emb_name = nn.Embedding(v_name, emb_dim)
        self.emb_stat = nn.Embedding(v_stat, emb_dim // 2)
        self.emb_su2  = nn.Embedding(v_su2,  emb_dim // 2)
        self.emb_u1y  = nn.Embedding(v_u1y,  emb_dim // 2)
        self.emb_lorz = nn.Embedding(v_lorz, emb_dim // 2)
        self.n_charge = num_charge_features()
        node_in_dim = (emb_dim * 2
                       + (emb_dim // 2) * 4
                       + 1
                       + self.n_charge)
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        # --- Edge-type embedding fed as edge_attr ---
        self.edge_emb = nn.Embedding(num_relations, hidden_dim)

        # --- Transformer conv stack ---
        per_head = hidden_dim // num_heads
        self.convs = nn.ModuleList([
            TransformerConv(
                hidden_dim, per_head, heads=num_heads,
                concat=True, edge_dim=hidden_dim, dropout=dropout,
            )
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
        kind = self.emb_kind(x[:, 0])
        name = self.emb_name(x[:, 1])
        rank = x[:, 2].float().unsqueeze(-1) / 4.0
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
        e = self.edge_emb(data.edge_type)
        for conv in self.convs:
            h = conv(h, data.edge_index, edge_attr=e)
            h = F.relu(h)
            h = self.dropout(h)
        return _per_term_min_readout(h, data, self.head)
