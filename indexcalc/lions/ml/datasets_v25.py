"""LIONS v2.5 — PyG dataset for the catalog-driven JSON schema.

Reads ``data/v2-5-catalog/{train,val,test}.json`` and serves PyG ``Data``
objects with:
- Label vector over the full catalog (catalog order, all bits valid → no
  mask needed).
- Node features = ``node_feature_ids_v25`` (10 ints per node).
- Optional random rename augment: at ``__getitem__`` time, F1..F8 token
  IDs are remapped through a fresh permutation so the model can't latch
  onto a fixed slot↔name correspondence.
- Edge features = same packing as v1.x (the graph encoder is unchanged).

This module is intentionally independent of v1.x ``datasets.py``.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Sequence

from indexcalc.lions.ml import _require_torch
from indexcalc.lions.ml.features_v25 import (
    LABEL_ORDER, FIELD_TOKEN_IDS,
    node_feature_ids_v25, kind_from_name,
)
from indexcalc.lions.ml.features import edge_feature_ids
from indexcalc.lions.serializer import expr_from_dict, space_from_dict
from indexcalc.lions.graph import graph_encode, EncodedGraph
from indexcalc.lions.augment import shuffle_order

_require_torch()
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# ─── Row → encoded graph (no torch yet) ──────────────────


def _row_to_graph(row: dict, spaces: dict):
    """Decode one JSON row into (expr, graph, field_property_hints, labels)."""
    expr = expr_from_dict(row["expr"], spaces)
    g = graph_encode(expr)
    if g is None:
        raise ValueError(f"row encoded to None: provenance={row.get('provenance')}")

    # Field property hints — only the named fields in the row's dump
    # have hints; everything else is "unknown". Translate dataset
    # statistics to the PROP_STATISTICS / PROP_ANTISYM vocab.
    raw_props: dict = row.get("field_properties", {}) or {}
    hints: dict[str, dict] = {}
    for fname, p in raw_props.items():
        hints[fname] = {
            "stats_hint": p.get("statistics", "unknown"),
            "antisym_hint": (
                "has_antisym" if p.get("antisymmetric_pairs") else "none"
            ),
        }
    return expr, g, hints, row["labels"]


# ─── Encode one graph → PyG Data ─────────────────────────


def _encode_to_pyg(g: EncodedGraph, hints: dict, labels: dict,
                   field_token_remap: dict[int, int] | None = None) -> Data:
    """Bundle the graph + 19-bit labels into a PyG ``Data``."""
    node_x = []
    for n in g.nodes:
        # Property hints attach by field name; invariants/operators get "unknown".
        h = hints.get(n.name, {})
        # M4 + v3.3: pull the node's *distinct* index spaces (dim, metric).
        # ``index_spaces`` is per-slot, so dedupe + sort canonically — this
        # makes the (primary, secondary) assignment order-invariant (a field
        # ψ^{iα} encodes the same whether the i or α slot comes first). The
        # secondary slot exposes a multi-index field's second sector, which
        # the M4 "first slot only" path discarded (v3.3 Sp false-positive).
        spaces = sorted(set(n.index_spaces))
        primary_dim, primary_metric = spaces[0] if spaces else (0, "unknown")
        secondary_dim, secondary_metric = spaces[1] if len(spaces) > 1 else (0, "unknown")
        # M5.AN: decide kind from the node's name rather than its reps
        # so user input with empty/dummy reps still classifies as a field.
        # ``n.kind`` from graph_encode stays the trusted source for
        # operators (PartialDeriv/TimeDeriv/ScalarFunction) — those
        # branches set kind directly, never via reps.
        if n.kind == "operator":
            kind = "operator"
        else:
            kind = kind_from_name(n.name, fallback="field")
        feats = node_feature_ids_v25(
            kind, n.name, n.rank, n.statistics,
            stats_hint=h.get("stats_hint", "unknown"),
            antisym_hint=h.get("antisym_hint", "unknown"),
            primary_dim=primary_dim, primary_metric=primary_metric,
            secondary_dim=secondary_dim, secondary_metric=secondary_metric,
            func_name=getattr(n, "func_name", ""),
        )
        if field_token_remap is not None:
            # Remap the name slot (index 1) through the rename perm.
            feats[1] = field_token_remap.get(feats[1], feats[1])
        node_x.append(feats)

    x = torch.tensor(node_x, dtype=torch.long)

    # I2 per-term ids (preserve from EncodedGraph if set).
    if g.node_term_ids:
        term_id = torch.tensor(g.node_term_ids, dtype=torch.long)
        num_terms_val = int(g.num_terms)
    else:
        term_id = torch.zeros((len(g.nodes),), dtype=torch.long)
        num_terms_val = 1
    num_terms = torch.tensor([num_terms_val], dtype=torch.long)

    if g.edges:
        src_list, dst_list, type_list, attr_list = [], [], [], []
        for e in g.edges:
            et, attr = edge_feature_ids(e.kind, e.space, e.src_pos, e.dst_pos)
            src_list.append(e.src); dst_list.append(e.dst)
            type_list.append(et); attr_list.append(attr)
            et_rev, attr_rev = edge_feature_ids(e.kind, e.space, e.dst_pos, e.src_pos)
            src_list.append(e.dst); dst_list.append(e.src)
            type_list.append(et_rev); attr_list.append(attr_rev)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(type_list, dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.long)

    y_vals = [1.0 if labels.get(g_label, False) else 0.0 for g_label in LABEL_ORDER]
    y = torch.tensor(y_vals, dtype=torch.float).unsqueeze(0)

    scalar = complex(g.scalar)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        y=y,
        scalar_re=torch.tensor([scalar.real], dtype=torch.float),
        scalar_im=torch.tensor([scalar.imag], dtype=torch.float),
        num_nodes=len(g.nodes),
        term_id=term_id,
        num_terms=num_terms,
    )


# ─── Dataset class ───────────────────────────────────────


class LionsV25Dataset(Dataset):
    """v2.5 catalog JSON → PyG Data, optionally with rename augment."""

    def __init__(self, json_path: Path | str, *,
                 random_rename: bool = False,
                 order_shuffle: bool = False,
                 rename_seed: int | None = None):
        self.json_path = Path(json_path)
        self.random_rename = random_rename
        self.order_shuffle = order_shuffle
        self._rng = random.Random(rename_seed)

        raw = json.loads(self.json_path.read_text())
        spaces = {nm: space_from_dict(nm, d) for nm, d in raw["spaces"].items()}

        # Pre-decode (expr, graph, hints, labels). Encoding to PyG Data happens
        # in __getitem__ so we can refresh the rename perm; the expr is kept so
        # ``order_shuffle`` can re-encode a fresh term/factor permutation per
        # epoch (v3.1) — semantically a no-op, augments against order bias.
        self._graphs: list[tuple[object, EncodedGraph, dict, dict]] = []
        for row in raw["rows"]:
            try:
                expr, g, hints, labels = _row_to_graph(row, spaces)
            except (NotImplementedError, ValueError):
                continue
            if len(g.nodes) == 0:
                continue
            self._graphs.append((expr, g, hints, labels))

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        expr, g, hints, labels = self._graphs[idx]
        if self.order_shuffle:
            g = graph_encode(shuffle_order(expr, self._rng))
        remap = None
        if self.random_rename:
            shuffled = list(FIELD_TOKEN_IDS)
            self._rng.shuffle(shuffled)
            remap = dict(zip(FIELD_TOKEN_IDS, shuffled))
        return _encode_to_pyg(g, hints, labels, field_token_remap=remap)


def collate_v25(data_list):
    return Batch.from_data_list(list(data_list))
