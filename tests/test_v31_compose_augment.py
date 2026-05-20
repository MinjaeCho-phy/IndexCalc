"""v3.1 R3/R4: multi-term compose + order-shuffle augment."""

from __future__ import annotations
import random
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum
from indexcalc.lions.catalog import get
from indexcalc.lions.catalog_enumerator import enumerate_for_entry
from indexcalc.lions.catalog_labeler import label_lagrangian
from indexcalc.lions.augment import compose_terms, shuffle_order


def _positives(expr):
    return {k for k, v in label_lagrangian(expr, get("U(1)")).items() if v}


# ─── R3: compose_terms ───────────────────────────────────


def test_compose_same_sector_keeps_sector_label():
    _, sp_samples = enumerate_for_entry(get("SO(3)"), prefix="a_")
    terms = [s.expr for s in sp_samples[:2]]
    composed = compose_terms(terms)
    assert isinstance(composed, TensorSum)
    pos = _positives(composed)
    assert {"SO(3)", "O(3)"} <= pos


def test_compose_cross_sector_is_per_sector_multipositive():
    """δ-term (SO) + Ω-term (Sp) → both SO and Sp positive (per-sector)."""
    _, so = enumerate_for_entry(get("SO(3)"), prefix="b_")
    _, sp = enumerate_for_entry(get("Sp(4)"), prefix="c_")
    composed = compose_terms([so[0].expr, sp[0].expr])
    pos = _positives(composed)
    assert {"SO(3)", "O(3)", "Sp(4)"} <= pos, pos
    assert "SO(4)" not in pos
    assert "Sp(6)" not in pos


def test_compose_disambiguates_indices():
    """Two terms reusing index name 'i' must not collide after compose."""
    _, so = enumerate_for_entry(get("SO(3)"), prefix="d_")
    composed = compose_terms([so[0].expr, so[0].expr])
    # If indices collided, the labeler's graph walk would see an index
    # appearing >2 times; labeling still succeeds → disjoint names.
    pos = _positives(composed)
    assert {"SO(3)", "O(3)"} <= pos


# ─── R4: shuffle_order ───────────────────────────────────


def test_shuffle_preserves_labels():
    _, so = enumerate_for_entry(get("SO(3)"), prefix="e_")
    _, sp = enumerate_for_entry(get("Sp(4)"), prefix="f_")
    composed = compose_terms([so[0].expr, sp[0].expr, so[1].expr])
    rng = random.Random(0)
    before = _positives(composed)
    for _ in range(5):
        shuffled = shuffle_order(composed, rng)
        assert _positives(shuffled) == before


def test_shuffle_can_change_structure():
    """At least one shuffle of a 3-term sum reorders the top-level terms."""
    _, so = enumerate_for_entry(get("SO(3)"), prefix="g_")
    terms = [so[i].expr for i in range(min(3, len(so)))]
    composed = compose_terms(terms)
    rng = random.Random(1)
    seen = {str(composed)}
    for _ in range(20):
        seen.add(str(shuffle_order(composed, rng)))
    assert len(seen) > 1, "shuffle never changed the serialized order"


def test_shuffle_product_factors_preserves_label():
    space = IndexSpace("t_so3", dim=3, indices="ij", metric="delta")
    F = Tensor("F", [space.upper("i")], reps={"SO(3)": "vector"})
    G = Tensor("G", [space.upper("j")], reps={"SO(3)": "vector"})
    delta = Tensor("delta", [space.lower("i"), space.lower("j")],
                   symmetric_pairs=[(0, 1)], reps={})
    expr = TensorProduct(TensorProduct(delta, F), G)
    rng = random.Random(2)
    before = _positives(expr)
    for _ in range(5):
        assert _positives(shuffle_order(expr, rng)) == before
