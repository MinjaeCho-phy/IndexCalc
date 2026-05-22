"""C0 — Noether symmetry verification via the total-time-derivative recognizer.

Covers the three behaviours verify_symmetry must distinguish:
  - time translation  δΦ=Φ̇   → δL = d/dt(L)   (total derivative, F=L)
  - rotation          δΦ=ωΦ   → δL = 0          (exact internal symmetry)
  - scaling           δΦ=Φ    → δL = 2L          (NOT a symmetry)
plus a direct dt_expand (Leibniz for ∂_t) check.
"""

from __future__ import annotations

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorSum, ScalarMul
from indexcalc.core.scalar_function import ScalarFunction
from indexcalc.core.group import Group
from indexcalc.core.generator import Generator, make_o_n_generator, u1_action
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import simplify
from indexcalc.adm import TimeDeriv
from indexcalc.core.noether import (
    dt_expand, is_total_time_derivative, verify_symmetry,
)


VEC = IndexSpace("q3", dim=3, indices="ijklmn", metric="delta")


def _group():
    """A single rotation group all the test generators share (so the field's
    rep tag matches every generator's group name)."""
    g = Group("G", dim=3, abelian=False)
    g.add_rep("vector", dim=3)
    g.add_rep("singlet", dim=1)
    return g


def _phi(i):
    return Tensor("Phi", [VEC.upper(i)], reps={"G": "vector"})


def _delta(a, b):
    return Tensor("delta", [VEC.lower(a), VEC.lower(b)], symmetric_pairs=[(0, 1)])


def _kinetic():
    """½ δ_ij Φ̇^i Φ̇^j (free particle)."""
    return ScalarMul(0.5, _delta("i", "j") * TimeDeriv(_phi("i")) * TimeDeriv(_phi("j")))


def _mass():
    """½ δ_ij Φ^i Φ^j."""
    return ScalarMul(0.5, _delta("i", "j") * _phi("i") * _phi("j"))


# ─── dt_expand: Leibniz for ∂_t, constants → 0 ──────────


def test_dt_expand_leibniz_and_constant_metric():
    # d/dt(δ_ij Φ^i Φ^j) = δ_ij(Φ̇^i Φ^j + Φ^i Φ̇^j); ∂_t(δ)=0.
    L = _delta("i", "j") * _phi("i") * _phi("j")
    expanded = dt_expand(TimeDeriv(L), {"Phi"})
    # Equals the hand-written Leibniz sum.
    hand = (_delta("i", "j") * TimeDeriv(_phi("i")) * _phi("j")
            + _delta("i", "j") * _phi("i") * TimeDeriv(_phi("j")))
    diff = TensorSum(expanded, ScalarMul(-1.0, hand))
    assert isinstance(simplify(diff), ZeroTensor)


# ─── time translation: δΦ = Φ̇  → total derivative ──────


def test_time_translation_is_total_derivative():
    gen = Generator("dt", _group())
    gen.declare_action("vector", lambda f: TimeDeriv(f))
    L = _kinetic()
    res = verify_symmetry(L, gen, field_names={"Phi"}, boundary_candidates=[L])
    assert res.is_symmetry
    assert not res.exact            # δL ≠ 0; it is d/dt(L)
    assert res.boundary_term is L


def test_time_translation_with_potential():
    """δΦ=Φ̇ on the harmonic oscillator L = ½Φ̇² + ½Φ² → δL = d/dt(L) too —
    verifier works with a potential term, not just the free particle."""
    gen = Generator("dt", _group())
    gen.declare_action("vector", lambda f: TimeDeriv(f))
    L = TensorSum(_kinetic(), _mass())
    res = verify_symmetry(L, gen, field_names={"Phi"}, boundary_candidates=[L])
    assert res.is_symmetry and not res.exact and res.boundary_term is L


def test_time_translation_needs_the_right_boundary_term():
    """Without a boundary candidate, the non-zero δL is not provable as a sym."""
    gen = Generator("dt", _group())
    gen.declare_action("vector", lambda f: TimeDeriv(f))
    res = verify_symmetry(_kinetic(), gen, field_names={"Phi"},
                          boundary_candidates=[])
    assert not res.is_symmetry      # δL is a total deriv, but F not supplied


# ─── U(1) phase: δφ = iφ  → exact zero ──────────────────


def test_u1_phase_is_exact_symmetry():
    """|φ|² = φ̄φ under δφ=iφ, δφ̄=−iφ̄ → δL = 0 (simplify handles this cleanly)."""
    u1 = Group("U(1)", abelian=True)
    u1.add_rep("+1", dim=1, charge=1.0)
    u1.add_rep("-1", dim=1, charge=-1.0)
    phi = Tensor("phi", [], reps={"U(1)": "+1"})
    phibar = Tensor("phibar", [], reps={"U(1)": "-1"})
    gen = Generator("Tu1", u1)
    gen.declare_action("+1", u1_action(u1.get_rep("+1")))
    gen.declare_action("-1", u1_action(u1.get_rep("-1")))
    res = verify_symmetry(phibar * phi, gen, field_names={"phi", "phibar"})
    assert res.is_symmetry
    assert res.exact                # δL = 0, no boundary term
    assert res.boundary_term is None


def test_son_rotation_hits_known_simplify_gap():
    """SO(N) rotation of a δ-bilinear: physically δL=0, but simplify cannot
    cancel it (the M-matrix vector-index antisymmetry is not encoded at IR
    level — documented gap D21, worked around by probe._structural_check).
    Recorded here so a future backend fix is validated against it. This gap is
    *orthogonal* to the C0 TimeDeriv-order fix."""
    gen = make_o_n_generator(_group(), VEC)
    res = verify_symmetry(_mass(), gen, field_names={"Phi"})
    assert not res.is_symmetry      # ← gap: should be True once SO(N) δ-bilinear lands


# ─── scaling: δΦ = Φ  → NOT a symmetry ──────────────────


def test_scaling_is_not_a_symmetry():
    gen = Generator("scale", _group())
    gen.declare_action("vector", lambda f: f)        # δΦ = Φ
    # δL = 2L for the kinetic term — not a total time derivative.
    res = verify_symmetry(_kinetic(), gen, field_names={"Phi"},
                          boundary_candidates=[_kinetic(), _mass()])
    assert not res.is_symmetry


# ─── C0b: Kepler with the 1/r ScalarFunction potential ──


def _r2():
    return _delta("k", "l") * _phi("k") * _phi("l")


def _kepler():
    """½ δ_ij Φ̇^i Φ̇^j + f(δ_kl Φ^k Φ^l)  — Kepler/Coulomb, f = inv_sqrt ≈ 1/r."""
    return TensorSum(_kinetic(), ScalarFunction("inv_sqrt", _r2()))


def test_dt_expand_scalarfunction_chain_rule():
    """∂_t f(I) = f'(I) ∂_t I — dt_expand must match apply_generator's chain
    rule so a non-polynomial potential (1/r) can appear in a boundary term."""
    expanded = dt_expand(TimeDeriv(ScalarFunction("inv_sqrt", _r2())), {"Phi"})
    hand = ScalarFunction("inv_sqrt_prime", _r2()) * dt_expand(TimeDeriv(_r2()), {"Phi"})
    assert isinstance(simplify(TensorSum(expanded, ScalarMul(-1.0, hand))), ZeroTensor)


def test_time_translation_verifies_full_kepler():
    """δΦ=Φ̇ on the full Kepler L (incl. the 1/r ScalarFunction) → δL = d/dt(L).
    Energy conservation verified off-shell — the potential is NOT a blocker:
    the ScalarFunction chain rule is consistent across variation and ∂_t."""
    gen = Generator("dt", _group())
    gen.declare_action("vector", lambda f: TimeDeriv(f))
    L = _kepler()
    res = verify_symmetry(L, gen, field_names={"Phi"}, boundary_candidates=[L])
    assert res.is_symmetry and not res.exact


def test_kepler_rotation_blocked_by_d21_gap():
    """Manifest SO(3) rotation of Kepler is physically a symmetry, but the
    pure-simplify verifier cannot confirm it: δ(δ-bilinear) leaves M-matrix
    terms simplify can't cancel (D21 gap). Since SO(4) ⊃ SO(3), the hidden
    SO(4) is gated on the same D21 lift — NOT on EOM (energy verifies above)
    nor representation (the 1/r potential is handled). Recorded so the D21 fix
    is validated against it."""
    gen = make_o_n_generator(_group(), VEC)
    res = verify_symmetry(_kepler(), gen, field_names={"Phi"})
    assert not res.is_symmetry      # ← unblocks once D21 SO(N) δ-bilinear lands


def test_is_total_time_derivative_zero_case():
    z = ZeroTensor([])
    ok, F = is_total_time_derivative(z, {"Phi"})
    assert ok and F is None


def test_simplify_distinguishes_timederiv_order():
    """C0 fix: simplify must NOT treat Φ̇ and Φ̈ as equal. Before the fix,
    canonical_form's factor key collapsed every TimeDeriv to ("TimeDeriv",),
    so Φ̇−Φ̈ falsely simplified to 0 — making the velocity-dependent verifier
    unsound (false positives). Locks in the canonical_form TimeDeriv branch."""
    dot = TimeDeriv(_phi("i"))
    ddot = TimeDeriv(TimeDeriv(_phi("i")))
    diff = TensorSum(dot, ScalarMul(-1.0, ddot))
    assert not isinstance(simplify(diff), ZeroTensor)
