"""D14 / I4 — deeper multi-term hard negatives.

Coverage for the three TensorSum variants added in ``augment.py``:
- ``add_n3_positive_pair``: ``TensorSum(inv, inv)`` ⇒ sum is invariant.
- ``add_n3_double_broken_pair``: ``TensorSum(broken, broken)`` ⇒ sum is
  broken under any group broken in either term.
- ``add_n4_nested``: ``TensorSum(A, TensorSum(B, C))`` with one broken
  C ⇒ nested 3-term sum, sum is broken.

We also smoke-test the enumerator wrappers and the graph encoder's
``num_terms``/``node_term_ids`` for nested cases.
"""

from __future__ import annotations
import random
import pytest

from indexcalc.lions import (
    EnumeratorCaps, enumerate_scalar_invariants, label_samples,
    add_n3_positive_pair, add_n3_double_broken_pair, add_n4_nested,
    enumerate_n3_positives, enumerate_n3_double_broken, enumerate_n4_nested,
    graph_encode,
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


def test_n3_positive_sum_is_invariant(sm_pools):
    """TensorSum(inv, inv) ⇒ oracle re-label = all True."""
    _sm, gens, pos, _broken = sm_pools
    assert len(pos) >= 2, "need ≥2 positives for this fixture"
    s = add_n3_positive_pair(pos[0], pos[1], gens)
    assert s.provenance == "n3_positive"
    assert all(s.labels.values()), (
        f"sum of two invariants is not invariant: labels={s.labels}"
    )


def test_n3_double_broken_sum_is_broken(sm_pools):
    """TensorSum(broken, broken) ⇒ oracle re-label is non-all-True."""
    _sm, gens, _pos, broken = sm_pools
    assert len(broken) >= 2
    s = add_n3_double_broken_pair(broken[0], broken[1], gens)
    assert s.provenance == "n3_double_broken"
    assert not all(s.labels.values()), (
        f"sum of two broken is fully invariant: labels={s.labels}"
    )


def test_n4_nested_with_broken_c_is_broken(sm_pools):
    """TensorSum(inv, TensorSum(inv, broken)) ⇒ at least one head is False."""
    _sm, gens, pos, broken = sm_pools
    assert len(pos) >= 2 and broken
    s = add_n4_nested(pos[0], pos[1], broken[0], gens)
    assert s.provenance == "n4_nested"
    assert not all(s.labels.values())


def test_n4_nested_graph_has_three_terms(sm_pools):
    """Nested TensorSum produces num_terms == 3 and node_term_ids spans
    {0, 1, 2}. Confirms ``walk`` recurses through nested sums."""
    _sm, gens, pos, broken = sm_pools
    s = add_n4_nested(pos[0], pos[1], broken[0], gens)
    g = graph_encode(s.expr)
    assert g.num_terms == 3
    assert set(g.node_term_ids) == {0, 1, 2}


def test_enumerator_wrappers_smoke(sm_pools):
    """All three enumerators produce ≥1 sample on a typical SM-lite pool."""
    _sm, gens, pos, broken = sm_pools
    n3p = enumerate_n3_positives(pos, gens, n_per_seed=1,
                                 rng=random.Random(0))
    n3db = enumerate_n3_double_broken(broken, gens, n_per_seed=1,
                                      rng=random.Random(0))
    n4 = enumerate_n4_nested(pos, broken, gens, n_per_seed=1,
                             rng=random.Random(0))
    assert n3p, "no n3_positive samples"
    assert n3db, "no n3_double_broken samples"
    assert n4, "no n4_nested samples"
    # Provenance correctness.
    assert all(s.provenance == "n3_positive" for s in n3p)
    assert all(s.provenance == "n3_double_broken" for s in n3db)
    assert all(s.provenance == "n4_nested" for s in n4)


def test_n3_positive_disambiguates_indices(sm_pools):
    """If the two positive seeds share dummy names, the resulting
    TensorSum should still encode cleanly (no triple-occurrence)."""
    _sm, gens, pos, _broken = sm_pools
    # Seeds 0 and 1 typically share dummy 'i' from the enumerator's
    # canonical naming.
    s = add_n3_positive_pair(pos[0], pos[1], gens)
    g = graph_encode(s.expr)
    assert g is not None
    assert g.num_terms == 2
