"""LIONS v2.5 — GroupPrototypeClassifier.

Architecture per the v2.5 redirect (Output choice B → graph embed × N
prototype embeds → dot-product score, ranked top-K). Node features come
from ``features_v25.node_feature_ids_v25`` (kind, name, rank, statistics,
stats_hint, antisym_hint).

Pipeline:
  x (6-int per node) → embed lookups + concat → node_proj → hidden
  RGCN ×L → global_mean_pool → graph embedding g ∈ R^hidden
  g · p_k for each prototype k ∈ [0, 19) → logits (B, 19)
"""

from __future__ import annotations
from indexcalc.lions.ml import _require_torch

_require_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool

from indexcalc.lions.ml.features_v25 import (
    NODE_KIND, NODE_NAME, STATISTICS, PROP_STATISTICS, PROP_ANTISYM,
    PRIMARY_METRIC, PRIMARY_DIM_VOCAB, LABEL_ORDER, num_relations,
)


def _vsize(table: dict) -> int:
    return max(table.values()) + 1


class GroupPrototypeClassifier(nn.Module):
    """v2.5 catalog classifier — graph embed dotted against prototype table.

    Parameters
    ----------
    hidden_dim : int
        Both node embedding output and prototype dimension.
    num_relations : int
        RGCNConv relation count; default ``features_v25.num_relations()``.
    num_layers : int
        Number of stacked RGCNConv layers.
    n_prototypes : int
        Catalog size. Defaults to len(LABEL_ORDER) = 19.
    temperature : float
        Score divisor τ. Lower τ → sharper softmax.
    dropout : float
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_relations: int = None,
        num_layers: int = 3,
        n_prototypes: int = None,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_relations is None:
            from indexcalc.lions.ml.features_v25 import num_relations as _nr
            num_relations = _nr()
        if n_prototypes is None:
            n_prototypes = len(LABEL_ORDER)

        v_kind = _vsize(NODE_KIND)
        v_name = _vsize(NODE_NAME)
        v_stat = _vsize(STATISTICS)
        v_psh = _vsize(PROP_STATISTICS)
        v_pah = _vsize(PROP_ANTISYM)
        v_metric = _vsize(PRIMARY_METRIC)
        v_pdim = _vsize(PRIMARY_DIM_VOCAB)

        emb = hidden_dim // 4
        half = max(1, emb // 2)
        self.emb_kind = nn.Embedding(v_kind, emb)
        self.emb_name = nn.Embedding(v_name, emb)
        self.emb_stat = nn.Embedding(v_stat, half)
        self.emb_psh = nn.Embedding(v_psh, half)
        self.emb_pah = nn.Embedding(v_pah, half)
        self.emb_metric = nn.Embedding(v_metric, half)   # M4: index-space metric
        self.emb_pdim = nn.Embedding(v_pdim, half)        # v3.1: discrete dim

        # Lookup table raw IndexSpace dim → PRIMARY_DIM_VOCAB index. Buffer so
        # it rides along with .to(device) / state_dict.
        max_dim = max(PRIMARY_DIM_VOCAB)
        lut = torch.zeros(max_dim + 1, dtype=torch.long)
        for d, i in PRIMARY_DIM_VOCAB.items():
            lut[d] = i
        self.register_buffer("_dim_lut", lut)
        self._max_dim = max_dim

        # rank scalar (+1) + 7 half-embeddings (stat, psh, pah, metric, pdim,
        # and v3.3 secondary metric + secondary pdim for multi-index fields).
        node_in_dim = emb * 2 + half * 7 + 1
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)

        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim,
                     num_relations=num_relations, num_bases=32)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Prototype table — learned vector per catalog entry.
        self.prototypes = nn.Embedding(n_prototypes, hidden_dim)
        nn.init.normal_(self.prototypes.weight, std=0.1)

        self.temperature = temperature

    # ── encoding ────────────────────────────────────────

    def encode_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 10] long
        # [kind, name, rank, statistics, stats_hint, antisym_hint,
        #  primary_dim, primary_metric, secondary_dim, secondary_metric]
        kind = self.emb_kind(x[:, 0])
        name = self.emb_name(x[:, 1])
        rank = x[:, 2].float().unsqueeze(-1) / 4.0
        stat = self.emb_stat(x[:, 3])
        psh = self.emb_psh(x[:, 4])
        pah = self.emb_pah(x[:, 5])
        # v3.1: discrete dim embedding (raw dim → vocab idx via LUT). Resolves
        # close high dims (Sp(8) vs Sp(10)) the old /5 scalar conflated.
        dim_idx = self._dim_lut[x[:, 6].long().clamp(0, self._max_dim)]
        primary_dim = self.emb_pdim(dim_idx)
        metric = self.emb_metric(x[:, 7])
        # v3.3: second distinct index space of a multi-index field (ψ^{iα}
        # charged under two groups). Reuses the dim/metric tables — a sector's
        # identity is its (dim, metric), slot-independent; (0,"unknown") when
        # the node is single-sector. Without this the model only saw the first
        # sector and confused a dim-3 SU(3) index with a dim-8 Sp one.
        dim_idx2 = self._dim_lut[x[:, 8].long().clamp(0, self._max_dim)]
        secondary_dim = self.emb_pdim(dim_idx2)
        metric2 = self.emb_metric(x[:, 9])
        h = torch.cat([kind, name, rank, stat, psh, pah,
                       primary_dim, metric, secondary_dim, metric2], dim=-1)
        return self.node_proj(h)

    # ── forward ────────────────────────────────────────

    def forward(self, data) -> torch.Tensor:
        """Return raw logits [B, n_prototypes]. Sigmoid applied by the loss."""
        h = self.encode_nodes(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_type)
            h = F.relu(h)
            h = self.dropout(h)
        g = global_mean_pool(h, data.batch)         # [B, hidden]
        p = self.prototypes.weight                  # [n_proto, hidden]
        return (g @ p.t()) / self.temperature       # [B, n_proto]

    @torch.no_grad()
    def top_k(self, data, k: int = 5):
        """Return top-K (label, score) for a single batch entry."""
        scores = torch.sigmoid(self.forward(data))[0]
        vals, idx = scores.topk(min(k, scores.shape[0]))
        return [(LABEL_ORDER[i.item()], v.item()) for i, v in zip(idx, vals)]
