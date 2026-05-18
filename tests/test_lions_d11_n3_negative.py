"""D11 — N3 dangling-term hard negatives.

Coverage:
- Index disambiguation prevents triple-occurrence errors after TensorSum.
- TensorSum(P, Q) is broken whenever Q is broken under at least one
  group (linearity of generator action: δ(P+Q) = δP + δQ = 0 + δQ).
- Generated negatives carry provenance="hard_negative_n3" and pass
  graph_encode (TensorSum supported as one combined graph).
- Round-trip through serializer.
"""

from __future__ import annotations
import pytest

from indexcalc.lions import (
    EnumeratorCaps, enumerate_scalar_invariants, label_samples,
    enumerate_n3_negatives, add_n3_dangling_term,
    graph_encode, save_dataset, load_dataset,
)
from indexcalc.lions.presets.sm_lite import build_sm_lite


@pytest.fixture(scope="module")
def sm_pools():
    sm = build_sm_lite()
    caps = EnumeratorCaps(
        max_field_total=3, max_per_field=1,
        max_partials_total=0, max_partials_per_field=0,
    )
    samples = enumerate_scalar_invariants(
        sm.fields, spacetime=sm.spacetime, caps=caps,
        invariant_alphabet=sm.invariant_alphabet,
        forbid_like_position_spaces={sm.dirac},
    )
    gens = {"SU(2)": sm.su2_gen,
            "U(1)_Y": sm.u1y_gen,
            "Lorentz": sm.lorentz_gen}
    labeled = label_samples(samples, gens)
    pos = [s for s in labeled if all(s.labels.values())]
    broken = [s for s in labeled if not all(s.labels.values())]
    return sm, gens, pos, broken


def test_n3_generation_smoke(sm_pools):
    """N3 enumerator returns at least one negative for a typical pool."""
    _sm, gens, pos, broken = sm_pools
    import random
    n3 = enumerate_n3_negatives(pos, broken, gens, n_per_seed=2,
                                rng=random.Random(0))
    assert n3, "no N3 negatives generated"
    for n in n3:
        assert n.provenance == "hard_negative_n3"


def test_n3_label_broken_in_at_least_one_group(sm_pools):
    """Every N3 result should fail invariance under at least one group."""
    _sm, gens, pos, broken = sm_pools
    import random
    n3 = enumerate_n3_negatives(pos, broken, gens, n_per_seed=2,
                                rng=random.Random(0))
    for n in n3:
        assert not all(n.labels.values()), (
            f"N3 sample is fully invariant — should be broken; "
            f"labels={n.labels}"
        )


def test_n3_graph_encode(sm_pools):
    """graph_encode handles TensorSum without raising. The number of
    nodes equals the sum of node counts from the two terms (TensorSum
    walks both terms into one graph per F3)."""
    _sm, gens, pos, broken = sm_pools
    n = add_n3_dangling_term(pos[0], broken[0], gens)
    g_p = graph_encode(pos[0].expr)
    g_q = graph_encode(broken[0].expr)
    g_n = graph_encode(n.expr)
    assert g_n is not None
    assert len(g_n.nodes) == len(g_p.nodes) + len(g_q.nodes)
    # All within-term edges preserved; no spurious cross-term edges.
    assert len(g_n.edges) == len(g_p.edges) + len(g_q.edges)


def test_n3_index_disambiguation_no_collision(sm_pools):
    """If both P and Q use the same dummy name 'i', TensorSum(P, Q)
    must not trip graph_encode's triple-occurrence guard."""
    _sm, gens, pos, broken = sm_pools
    # Pick samples that likely share index names — first ones (enumerator
    # uses similar dummy naming).
    n = add_n3_dangling_term(pos[0], broken[0], gens)
    # Should not raise.
    g = graph_encode(n.expr)
    assert g is not None


def test_n3_serialization_round_trip(sm_pools, tmp_path):
    _sm, gens, pos, broken = sm_pools
    import random
    n3 = enumerate_n3_negatives(pos, broken, gens, n_per_seed=1,
                                rng=random.Random(1))
    path = tmp_path / "n3.json"
    save_dataset(n3[:5], path)
    loaded = load_dataset(path)
    assert len(loaded) == min(5, len(n3))
    for orig, back in zip(n3[:5], loaded):
        assert orig.labels == back.labels
        assert orig.provenance == back.provenance
