"""LIONS v2.5 M2.2 — catalog_labeler tests.

Verifies the structural multi-positive labeling against four canonical
cases the user's redirect doc explicitly calls out.
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor
from indexcalc.lions.catalog import get
from indexcalc.lions.catalog_labeler import (
    collect_tensor_signature, label_lagrangian,
)


def _vec_space(N: int, prefix: str = "t_"):
    return IndexSpace(f"{prefix}so{N}_vec", dim=N,
                      indices="ijklmnp", metric="delta")


def _field(name: str, space, pos="upper", reps=None):
    idx = space.upper(name[-1]) if pos == "upper" else space.lower(name[-1])
    return Tensor(name, [idx], reps=reps or {"SO(3)": "vector"})


def _delta(space, i="i", j="j"):
    return Tensor(
        "delta", [space.lower(i), space.lower(j)],
        symmetric_pairs=[(0, 1)], reps={},
    )


def _epsilon_3(space, i="i", j="j", k="k"):
    return Tensor(
        "epsilon",
        [space.lower(i), space.lower(j), space.lower(k)],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        reps={},
    )


# ─── Tensor signature ────────────────────────────────────


def test_empty_signature_for_scalar():
    """Single field with no invariant tensor leaves → empty signature."""
    space = _vec_space(3)
    Phi = Tensor("Phi", [], reps={})
    assert collect_tensor_signature(Phi) == set()


def test_signature_picks_delta_dim_metric():
    space = _vec_space(3)
    F = _field("Fi", space)
    expr = _delta(space) * F * F
    sig = collect_tensor_signature(expr)
    # (name, dim, metric, slot_count) — δ is 2-slot.
    assert sig == {("delta", 3, "delta", 2)}


def test_signature_normalizes_eta_on_delta_space():
    """Spurious 'eta' tensor on a metric='delta' space → counted as δ."""
    space = _vec_space(3)
    spurious_eta = Tensor(
        "eta", [space.lower("i"), space.lower("j")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    sig = collect_tensor_signature(spurious_eta)
    assert sig == {("delta", 3, "delta", 2)}


def test_epsilon_slot_count_in_signature():
    """ε's slot count is tracked: SU(2) 2-slot vs SU(3) 3-slot are
    distinguishable even when they sit on the same IndexSpace dim."""
    space = _vec_space(3)  # dim=3 vec space (orthogonal metric)
    eps2 = Tensor(
        "epsilon",
        [space.lower("i"), space.lower("j")],
        antisymmetric_pairs=[(0, 1)], reps={},
    )
    eps3 = Tensor(
        "epsilon",
        [space.lower("i"), space.lower("j"), space.lower("k")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)], reps={},
    )
    assert collect_tensor_signature(eps2) == {("epsilon", 3, "delta", 2)}
    assert collect_tensor_signature(eps3) == {("epsilon", 3, "delta", 3)}


def test_two_slot_epsilon_on_dim3_matches_no_catalog_entry():
    """The OOD-eval quirk: ε_{ij} on a dim=3 IndexSpace has no catalog
    home — SO(3)/SU(3) want 3-slot ε, SO(2)/SU(2) want dim=2."""
    space = _vec_space(3)
    A = _field("Ai", space)
    B = _field("Bj", space)
    eps2 = Tensor(
        "epsilon",
        [space.lower("i"), space.lower("j")],
        antisymmetric_pairs=[(0, 1)], reps={},
    )
    expr = eps2 * A * B
    # Primary stays True (it's always set), but no other catalog entry
    # accepts the (epsilon, dim=3, slot=2) shape.
    labels = label_lagrangian(expr, primary_entry=get("SO(3)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"SO(3)"}, positives


# ─── Multi-positive labels ───────────────────────────────


def test_scalar_matches_every_entry():
    """User's 'L = φ²' case — bare singlet matches all 23 catalog entries
    (including Sp: an index-less scalar is a singlet under every group)."""
    space = _vec_space(3)
    Phi = Tensor("Phi", [], reps={})
    labels = label_lagrangian(Phi, primary_entry=get("U(1)"))
    assert all(labels.values()), f"non-trivial labels: {labels}"
    assert len(labels) == 23


def test_delta_n3_only_matches_o3_and_so3():
    space = _vec_space(3)
    F = _field("Fi", space)
    G = _field("Gj", space)
    expr = _delta(space) * F * G
    labels = label_lagrangian(expr, primary_entry=get("SO(3)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"O(3)", "SO(3)"}


def test_delta_n3_does_not_match_so4():
    space = _vec_space(3)
    F = _field("Fi", space)
    G = _field("Gj", space)
    expr = _delta(space) * F * G
    labels = label_lagrangian(expr, primary_entry=get("SO(3)"))
    assert labels["SO(4)"] is False
    assert labels["O(4)"] is False


def test_epsilon_n3_in_vec_space_only_matches_so3():
    """ε_{ijk} on orthogonal (metric=delta) → SO(3) only (O has no ε)."""
    space = _vec_space(3)
    A = _field("Ai", space)
    B = _field("Bj", space)
    C = _field("Ck", space)
    expr = _epsilon_3(space) * A * B * C
    labels = label_lagrangian(expr, primary_entry=get("SO(3)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"SO(3)"}, positives


def test_epsilon_n3_in_fund_space_only_matches_su3():
    """ε_{ijk} on unitary fund (metric='') → SU(3) only."""
    fund = IndexSpace("t_su3_fund", dim=3, indices="ijk")
    eps = Tensor(
        "epsilon",
        [fund.lower("i"), fund.lower("j"), fund.lower("k")],
        antisymmetric_pairs=[(0, 1), (0, 2), (1, 2)],
        reps={},
    )
    A = Tensor("A", [fund.upper("i")], reps={"SU(3)": "fund"})
    B = Tensor("B", [fund.upper("j")], reps={"SU(3)": "fund"})
    C = Tensor("C", [fund.upper("k")], reps={"SU(3)": "fund"})
    expr = eps * A * B * C
    labels = label_lagrangian(expr, primary_entry=get("SU(3)"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"SU(3)"}, positives


def test_field_rep_sig_filters_fund_pairs():
    """L = (φ†_i φ^i)² — fund × antifund Einstein contraction, no invariant
    tensor. Rep-sig narrows U/SU(2) only; SO/O are dropped because their
    supported_reps doesn't include 'fund'/'antifund'."""
    from indexcalc.core.tensor import TensorProduct
    fund = IndexSpace("t_su2_fund", dim=2, indices="ij")
    phi_u = Tensor("phi", [fund.upper("i")],
                   reps={"SU(2)": "fund", "U(2)": "fund"})
    phi_d = Tensor("phidag", [fund.lower("i")],
                   reps={"SU(2)": "antifund", "U(2)": "antifund"})
    expr = phi_d * phi_u  # |φ|²
    labels = label_lagrangian(expr, primary_entry=get("SU(2)"))
    positives = {k for k, v in labels.items() if v}
    # SU(2)/U(2) carry the fund pair; nothing else does.
    assert positives == {"SU(2)", "U(2)"}, positives


def test_lorentz_vector_field_filters_to_lorentz_family():
    """L = F^μ F_μ on a Lorentz spacetime field — Lorentz/Poincaré only
    (η is implicit via Einstein; expr's rep sig pins the family)."""
    st = IndexSpace("t_lst", dim=4, indices="μνρ", metric="eta")
    A_u = Tensor("F1", [st.upper("μ")], reps={"Lorentz": "vector"})
    A_d = Tensor("F1", [st.lower("μ")], reps={"Lorentz": "vector"})
    expr = A_u * A_d
    labels = label_lagrangian(expr, primary_entry=get("Lorentz"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"Lorentz", "Poincare"}, positives


def test_scalar_field_still_matches_everything():
    """Sanity: a pure scalar (no reps, no invariant tensor) still
    saturates every catalog entry — the user's L=φ² intent."""
    fund = IndexSpace("t_dummy", dim=1, indices="·")
    phi = Tensor("phi", [], reps={})
    expr = phi * Tensor("phi", [], reps={})
    labels = label_lagrangian(expr, primary_entry=get("U(1)"))
    assert all(labels.values()), f"non-trivial: {labels}"


def test_eta_only_matches_lorentz_and_poincare():
    st = IndexSpace("t_lorentz_st", dim=4, indices="μνρσ", metric="eta")
    eta = Tensor(
        "eta", [st.lower("μ"), st.lower("ν")],
        symmetric_pairs=[(0, 1)], reps={},
    )
    A = Tensor("A", [st.upper("μ")], reps={"Lorentz": "vector"})
    B = Tensor("B", [st.upper("ν")], reps={"Lorentz": "vector"})
    expr = eta * A * B
    labels = label_lagrangian(expr, primary_entry=get("Lorentz"))
    positives = {k for k, v in labels.items() if v}
    assert positives == {"Lorentz", "Poincare"}, positives
