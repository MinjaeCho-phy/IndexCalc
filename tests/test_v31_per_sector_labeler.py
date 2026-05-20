"""v3.1 R1: per-sector labeling.

A group judges only the invariant structure in *its own* index space;
tensors in other spaces are singlets to it and ignored. This lets a
multi-sector Lagrangian be multi-positive per sector:

- δ_ij A^iA^j (SO, dim-3) + Ω_kl B^kB^l (Sp(4), dim-4) → both SO(3) and Sp(4).
- a single field ψ^{iμ} charged under SU(3) (fund i) and Lorentz (vector μ)
  → both SU(3)/U(3) and Lorentz/Poincaré.

The old all-or-nothing labeler rejected both — a single foreign tensor
disqualified the entry.
"""

from __future__ import annotations
import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, TensorSum
from indexcalc.lions.catalog import get
from indexcalc.lions.catalog_labeler import (
    label_lagrangian, _entry_compatible_with_sig, _owned_space_signature,
)


# ─── owned-space signature ───────────────────────────────


def test_owned_space_signatures():
    assert _owned_space_signature(get("U(1)")) is None
    assert _owned_space_signature(get("SU(3)")) == (3, "")
    assert _owned_space_signature(get("U(3)")) == (3, "")
    assert _owned_space_signature(get("SO(3)")) == (3, "delta")
    assert _owned_space_signature(get("O(4)")) == (4, "delta")
    assert _owned_space_signature(get("Sp(4)")) == (4, "")
    assert _owned_space_signature(get("Sp(6)")) == (6, "")
    assert _owned_space_signature(get("Lorentz")) == (4, "eta")
    assert _owned_space_signature(get("Poincare")) == (4, "eta")


def test_foreign_space_tensor_is_ignored():
    """An Ω on (4,'') is foreign to SO(3) → SO(3) compatible (ignores it)."""
    sig = {("omega", 4, "", 2)}
    assert _entry_compatible_with_sig(get("SO(3)"), sig) is True
    # but Sp(4) owns (4,'') and Ω is its invariant → compatible too.
    assert _entry_compatible_with_sig(get("Sp(4)"), sig) is True
    # Sp(6) owns (6,'') → the (4,'') Ω is foreign → still compatible (ignored).
    assert _entry_compatible_with_sig(get("Sp(6)"), sig) is True


def test_owned_space_foreign_tensor_disqualifies():
    """A δ on SO(3)'s own (3,delta) space that isn't an SO invariant... but
    δ *is* an SO invariant. An Ω placed on (3,delta) would be foreign-named
    in SO(3)'s own space → disqualify."""
    assert _entry_compatible_with_sig(get("SO(3)"), {("delta", 3, "delta", 2)}) is True
    # omega is not an SO invariant; if it appeared in SO(3)'s own space it
    # would disqualify (constructed sig — exercises the name check).
    assert _entry_compatible_with_sig(get("SO(3)"), {("omega", 3, "delta", 2)}) is False


# ─── multi-sector Lagrangian (tensors in two spaces) ─────


def _so_term():
    sp = IndexSpace("t_so3", dim=3, indices="ijk", metric="delta")
    A = Tensor("A", [sp.upper("i")], reps={"SO(3)": "vector"})
    Ap = Tensor("Ap", [sp.upper("j")], reps={"SO(3)": "vector"})
    delta = Tensor("delta", [sp.lower("i"), sp.lower("j")],
                   symmetric_pairs=[(0, 1)], reps={})
    return TensorProduct(TensorProduct(delta, A), Ap)


def _sp_term():
    sp = IndexSpace("t_sp4", dim=4, indices="lmnp", metric="")
    B = Tensor("B", [sp.upper("l")], reps={"Sp(4)": "vector"})
    Bp = Tensor("Bp", [sp.upper("m")], reps={"Sp(4)": "vector"})
    omega = Tensor("omega", [sp.lower("l"), sp.lower("m")],
                   antisymmetric_pairs=[(0, 1)], reps={})
    return TensorProduct(TensorProduct(omega, B), Bp)


def test_multisector_delta_plus_omega_matches_both_so_and_sp():
    expr = TensorSum(_so_term(), _sp_term())
    labels = label_lagrangian(expr, primary_entry=get("SO(3)"))
    pos = {k for k, v in labels.items() if v}
    assert {"SO(3)", "O(3)", "Sp(4)"} <= pos, pos
    # wrong N within each family stays out (rep_sig gate).
    assert "SO(4)" not in pos
    assert "Sp(6)" not in pos


# ─── multi-index field ψ^{iμ} (SU(3) × Lorentz) ──────────


def test_multi_index_field_matches_su_and_lorentz():
    su = IndexSpace("t_su3", dim=3, indices="ijk", metric="")
    lor = IndexSpace("t_lst", dim=4, indices="μνρ", metric="eta")
    psi = Tensor("psi", [su.upper("i"), lor.upper("μ")],
                 reps={"SU(3)": "fund", "Lorentz": "vector"})
    psibar = Tensor("psibar", [su.lower("i"), lor.lower("μ")],
                    reps={"SU(3)": "antifund", "Lorentz": "vector"})
    expr = TensorProduct(psi, psibar)  # ψ^{iμ} ψ̄_{iμ}
    labels = label_lagrangian(expr, primary_entry=get("SU(3)"))
    pos = {k for k, v in labels.items() if v}
    # SU(3)/U(3) (fund pair) AND Lorentz/Poincaré (vector) — both sectors.
    assert {"SU(3)", "U(3)", "Lorentz", "Poincare"} <= pos, pos
    # not the wrong-N orthogonal/symplectic groups.
    assert "SO(3)" not in pos
    assert "Sp(4)" not in pos
