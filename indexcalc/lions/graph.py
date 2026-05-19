"""LIONS IR → graph encoder (D9b).

Encoder side of the Task 1·2 ML pipeline (recognition / property
classification). Output is dependency-free: ``EncodedGraph`` is a plain
dataclass of plain dataclasses, ready for a ~10-line bridge to PyG /
DGL / a custom Transformer.

Design decisions are frozen in ``LIONS/notes/graph_encoding_spec.md``
(F1-F8). Highlights:
- Position(up/down) is an edge attribute (F1).
- TensorProduct grouping is dropped — graph is set-of-factors (F2).
- TensorSum becomes multiple graphs per call site; v1 ``graph_encode``
  walks a single Sum into one graph and emits both terms' tensors as
  separate nodes (F3). Acceptable since enumerator output is monomial.
- No named-term super-nodes (F4) — let the model discover.
- PartialDeriv is its own operator node with ``acts_on`` edges (F5).
- ScalarMul collapses into a graph-level ``scalar`` field (F7).
- ZeroTensor → ``graph_encode`` returns ``None`` (F8).
- CovariantDeriv NotImplementedError — same v1 deferral as D8a/D9a.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.adm import TimeDeriv

from indexcalc.lions.dataset import LabeledSample


# ─── Dataclasses ─────────────────────────────────────────


@dataclass
class GraphNode:
    kind: str           # "field" | "invariant" | "operator"
    name: str
    rank: int
    reps: dict[str, str]
    statistics: str


@dataclass
class GraphEdge:
    src: int
    dst: int
    kind: str           # "contraction" | "acts_on"
    space: str          # "" for acts_on
    src_pos: str        # "upper" | "lower" | "" for acts_on
    dst_pos: str


@dataclass
class EncodedGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    scalar: complex = 1 + 0j
    # Optional carried metadata from LabeledSample (set by encode_sample).
    labels: dict[str, bool] = field(default_factory=dict)
    field_counts: dict[str, int] = field(default_factory=dict)
    mass_dim: float = 0.0
    # I2: term partition. For a flat (non-Sum) expression every node has
    # term_id=0 and num_terms=1. A TensorSum allocates a fresh term_id
    # for the right branch (recursively), so for ``TensorSum(A, B)`` the
    # nodes of A get id 0 and those of B get id 1. Empty list ⇒ legacy
    # construction; pyg_bridge falls back to single-term semantics.
    node_term_ids: list[int] = field(default_factory=list)
    num_terms: int = 1


# ─── Encoder ─────────────────────────────────────────────


def graph_encode(expr: TensorExpr) -> Optional[EncodedGraph]:
    """Walk ``expr`` and produce an ``EncodedGraph``.

    Returns ``None`` for a top-level ``ZeroTensor`` (F8). Raises
    ``NotImplementedError`` for ``CovariantDeriv``.
    """
    if isinstance(expr, ZeroTensor):
        return None

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_term_ids: list[int] = []
    # index name -> list of (node_id, position, space_name)
    index_occ: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    scalar: complex = 1
    # next_term_id[0] = next id to hand out for a fresh TensorSum branch.
    # Root walk uses term 0; the right branch of each TensorSum allocates
    # next_term_id[0] and bumps the counter.
    next_term_id: list[int] = [1]

    def add_node(node: GraphNode, term_id: int) -> int:
        nodes.append(node)
        node_term_ids.append(term_id)
        return len(nodes) - 1

    def add_tensor(t: Tensor, term_id: int) -> int:
        kind = "field" if t.reps else "invariant"
        nid = add_node(GraphNode(
            kind=kind, name=t.name, rank=len(t.indices),
            reps=dict(t.reps), statistics=t.statistics,
        ), term_id)
        for idx in t.indices:
            index_occ[idx.name].append((nid, idx.position, idx.space.name))
        return nid

    def walk(e: TensorExpr, term_id: int) -> list[int]:
        """Return the list of node IDs emitted by this subtree (used by
        PartialDeriv to wire ``acts_on`` edges to its operand tensors).

        ``term_id`` is the TensorSum-branch id assigned to every node
        emitted in this subtree (unless a nested TensorSum overrides it
        for its right branch).
        """
        nonlocal scalar
        if isinstance(e, Tensor):
            return [add_tensor(e, term_id)]
        if isinstance(e, ZeroTensor):
            return []
        if isinstance(e, TensorProduct):
            return walk(e.left, term_id) + walk(e.right, term_id)
        if isinstance(e, TensorSum):
            # I2: split the right branch into a fresh term id so the ML
            # readout can pool each summand independently.
            left_ids = walk(e.left, term_id)
            new_term = next_term_id[0]
            next_term_id[0] += 1
            right_ids = walk(e.right, new_term)
            return left_ids + right_ids
        if isinstance(e, ScalarMul):
            scalar = scalar * e.scalar
            return walk(e.expr, term_id)
        if isinstance(e, PartialDeriv):
            op_id = add_node(GraphNode(
                kind="operator", name="partial", rank=1,
                reps={}, statistics="bosonic",
            ), term_id)
            index_occ[e.deriv_index.name].append(
                (op_id, e.deriv_index.position, e.deriv_index.space.name),
            )
            inner_ids = walk(e.expr, term_id)
            for tid in inner_ids:
                src, dst = (op_id, tid) if op_id < tid else (tid, op_id)
                edges.append(GraphEdge(
                    src=src, dst=dst, kind="acts_on",
                    space="", src_pos="", dst_pos="",
                ))
            return [op_id]
        if isinstance(e, TimeDeriv):
            op_id = add_node(GraphNode(
                kind="operator", name="TimeDeriv", rank=0,
                reps={}, statistics="bosonic",
            ), term_id)
            inner_ids = walk(e.expr, term_id)
            for tid in inner_ids:
                src, dst = (op_id, tid) if op_id < tid else (tid, op_id)
                edges.append(GraphEdge(
                    src=src, dst=dst, kind="acts_on",
                    space="", src_pos="", dst_pos="",
                ))
            return [op_id]
        if isinstance(e, ScalarFunction):
            op_id = add_node(GraphNode(
                kind="operator", name="ScalarFunction", rank=0,
                reps={}, statistics="bosonic",
            ), term_id)
            inner_ids = walk(e.arg, term_id)
            for tid in inner_ids:
                src, dst = (op_id, tid) if op_id < tid else (tid, op_id)
                edges.append(GraphEdge(
                    src=src, dst=dst, kind="acts_on",
                    space="", src_pos="", dst_pos="",
                ))
            return [op_id]
        if isinstance(e, CovariantDeriv):
            raise NotImplementedError(
                "CovariantDeriv graph encoding deferred (D9b v1 scope)."
            )
        raise TypeError(f"unsupported expr type {type(e).__name__}")

    walk(expr, 0)

    # Now turn index_occ into contraction edges.
    for name, occs in index_occ.items():
        if len(occs) == 1:
            # Free index — record nothing. (Enumerator output is scalar
            # so this is rare; PartialDeriv outside an invariant context
            # could leave one.)
            continue
        if len(occs) != 2:
            raise ValueError(
                f"index {name!r} appears {len(occs)} times in expr — "
                f"invalid IR for graph encoding"
            )
        (n1, p1, s1), (n2, p2, s2) = occs
        if s1 != s2:
            raise ValueError(
                f"index {name!r} contracts across different spaces "
                f"({s1} vs {s2})"
            )
        src, dst = (n1, n2) if n1 < n2 else (n2, n1)
        src_pos, dst_pos = (p1, p2) if n1 < n2 else (p2, p1)
        edges.append(GraphEdge(
            src=src, dst=dst, kind="contraction",
            space=s1, src_pos=src_pos, dst_pos=dst_pos,
        ))

    return EncodedGraph(
        nodes=nodes, edges=edges, scalar=scalar,
        node_term_ids=node_term_ids, num_terms=next_term_id[0],
    )


def encode_sample(sample: LabeledSample) -> Optional[EncodedGraph]:
    """Encode a ``LabeledSample`` and attach its labels/metadata to the
    resulting graph for downstream ML feature extraction."""
    g = graph_encode(sample.expr)
    if g is None:
        return None
    g.labels = dict(sample.labels)
    g.field_counts = dict(sample.field_counts)
    g.mass_dim = sample.mass_dim
    return g


def encode_dataset(samples: list[LabeledSample]) -> list[EncodedGraph]:
    """Encode each sample; drop ``None`` (ZeroTensor) results."""
    out: list[EncodedGraph] = []
    for s in samples:
        g = encode_sample(s)
        if g is not None:
            out.append(g)
    return out
