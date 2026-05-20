"""v3 S3: Sp(2N) enumeration + structural labeling.

검증:
1. ``enumerate_for_entry(Sp(4))`` 가 Ω_{ij}F^iG^j (두 다른 field) 만 생성하고
   vanishing single-field Ω F^iF^i 는 자동 drop (Bose × antisym → 0).
2. labeler: Ω-contraction on dim-2N → 그 Sp(2N) 만 positive.
3. δ-contraction (대칭) → Sp negative (O/SO positive). Ω → O/SO negative.
4. dim 분리: Ω on dim-4 → Sp(4) positive, Sp(6) negative.
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.lions.catalog import get
from indexcalc.lions.catalog_enumerator import enumerate_for_entry
from indexcalc.lions.catalog_labeler import (
    collect_tensor_signature, label_lagrangian,
)


def _sp_vec_space(rank: int, prefix: str = "t_"):
    """Metric-less 2·rank-dim space, mirroring the enumerator setup."""
    return IndexSpace(f"{prefix}sp_vec", dim=2 * rank, indices="ijklmn", metric="")


def _omega(space, i="i", j="j"):
    return Tensor(
        "omega", [space.lower(i), space.lower(j)],
        antisymmetric_pairs=[(0, 1)], reps={},
    )


def _sp_field(name: str, space, label: str):
    return Tensor(name, [space.upper(name[-1])], reps={label: "vector"})


# ─── Enumeration ─────────────────────────────────────────


def test_sp4_enumerates_only_two_distinct_field_omega_bilinears():
    setup, samples = enumerate_for_entry(get("Sp(4)"), prefix="e_")
    assert len(samples) > 0
    for s in samples:
        sig = collect_tensor_signature(s.expr)
        # Every Sp sample uses Ω on the 4-dim metric-less space, 2-slot.
        assert sig == {("omega", 4, "", 2)}, f"unexpected sig: {sig} for {s.expr}"


def test_sp4_drops_vanishing_single_field_bilinear():
    """No surviving sample is Ω contracting a field with itself (= 0)."""
    setup, samples = enumerate_for_entry(get("Sp(4)"), prefix="e2_")
    for s in samples:
        expr_str = str(s.expr)
        # crude but effective: a self-contraction would repeat one field name.
        import re
        fields = re.findall(r"F\d+", expr_str)
        assert len(set(fields)) == len(fields), (
            f"single-field Ω bilinear leaked (should vanish): {expr_str}"
        )


# ─── Labeling ────────────────────────────────────────────


def test_omega_n4_only_matches_sp4():
    space = _sp_vec_space(2)  # dim 4
    F = _sp_field("Fi", space, "Sp(4)")
    G = _sp_field("Gj", space, "Sp(4)")
    expr = _omega(space) * F * G
    labels = label_lagrangian(expr, primary_entry=get("Sp(4)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"Sp(4)"}, positives


def test_omega_n6_matches_sp6_not_sp4():
    space = _sp_vec_space(3)  # dim 6
    F = _sp_field("Fi", space, "Sp(6)")
    G = _sp_field("Gj", space, "Sp(6)")
    expr = _omega(space) * F * G
    labels = label_lagrangian(expr, primary_entry=get("Sp(6)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"Sp(6)"}, positives


def test_symmetric_delta_does_not_match_sp():
    """δ_{ij} (symmetric) → O/SO of that N, never Sp."""
    space = IndexSpace("t_o4_vec", dim=4, indices="ijkl", metric="delta")
    F = Tensor("Fi", [space.upper("i")], reps={"SO(4)": "vector"})
    G = Tensor("Gj", [space.upper("j")], reps={"SO(4)": "vector"})
    delta = Tensor("delta", [space.lower("i"), space.lower("j")],
                   symmetric_pairs=[(0, 1)], reps={})
    expr = delta * F * G
    labels = label_lagrangian(expr, primary_entry=get("SO(4)"))
    positives = {k for k, v in labels.items() if v}
    assert "Sp(4)" not in positives
    assert {"O(4)", "SO(4)"} <= positives


def test_omega_does_not_match_orthogonal():
    """Ω (antisymmetric, metric-less) → never O/SO (they need symmetric δ)."""
    space = _sp_vec_space(2)
    F = _sp_field("Fi", space, "Sp(4)")
    G = _sp_field("Gj", space, "Sp(4)")
    expr = _omega(space) * F * G
    labels = label_lagrangian(expr, primary_entry=get("Sp(4)"))
    for orth in ("O(4)", "SO(4)"):
        assert not labels[orth], f"{orth} should not match an Ω bilinear"
