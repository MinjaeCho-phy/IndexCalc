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
from indexcalc.core.generator import (
    Generator, make_o_n_generator, make_sp_2n_generator,
    lorentz_vector_action, u1_action, _fresh_dummy_name,
)
from indexcalc.core.substitution import apply_generator
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.simplify import simplify, collect_factors
from indexcalc.adm import TimeDeriv
from indexcalc.core.noether import (
    dt_expand, is_total_time_derivative, verify_symmetry, verify_symmetry_ft,
)
from indexcalc.core.deriv import PartialDeriv


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


def test_son_rotation_of_delta_bilinear_verifies():
    """SO(N) rotation of a δ-bilinear: δ(δ_ij Φ^i Φ^j) = 0, verified purely
    algebraically (D21 lift). The orthogonal generator carries its vector-index
    antisymmetry as ``cometric_antisymmetric_pairs``; once simplify promotes it
    (so(N) = antisymmetric matrices), antisym M × sym ΦΦ → 0 — no
    ``probe._structural_check`` workaround needed."""
    gen = make_o_n_generator(_group(), VEC)
    res = verify_symmetry(_mass(), gen, field_names={"Phi"})
    assert res.is_symmetry and res.exact      # δL = 0 exactly


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


def test_kepler_rotation_verifies_after_d21_lift():
    """Manifest SO(3) rotation of the *full* Kepler L (kinetic + 1/r potential)
    verifies algebraically after the D21 lift. δ(δ_ij Φ̇^i Φ̇^j) and
    δ(f(δ_kl Φ^k Φ^l)) both vanish term-by-term (antisym M × sym ΦΦ); the
    ScalarFunction chain rule carries the f'·0 of the potential. Since
    SO(4) ⊃ SO(3), the same machinery unblocks the hidden SO(4) rotation
    sector — this was the C0b gate (not EOM, not representation)."""
    gen = make_o_n_generator(_group(), VEC)
    res = verify_symmetry(_kepler(), gen, field_names={"Phi"})
    assert res.is_symmetry and res.exact      # δL = 0 exactly


# ─── D21: orthogonal-only scoping of the cometric antisymmetry ──


def _generator_M(out):
    Ms = [f for f in collect_factors(out)
          if isinstance(f, Tensor) and f.name.startswith("M")]
    assert len(Ms) == 1, f"expected one generator tensor, got {len(Ms)}"
    return Ms[0]


def test_orthogonal_generator_carries_cometric_antisym():
    """make_o_n_generator marks the vector slots as cometric-antisymmetric so
    simplify can promote them once they share a position (so(N) algebra)."""
    gen = make_o_n_generator(_group(), VEC)
    M = _generator_M(apply_generator(_phi("i"), gen))
    assert M.cometric_antisymmetric_pairs      # vector row/col slots


def test_symplectic_generator_has_no_cometric_antisym():
    """Sp(2N) preserves the *antisymmetric* Ω, so its generator is symmetric
    after lowering — it must NOT inherit the orthogonal antisymmetry, else
    Ω-bilinears would falsely cancel. Locks D21's orthogonal-only scoping."""
    sp = Group("Sp(4)", dim=10, abelian=False)
    sp.add_rep("vector", dim=4)
    sp.add_rep("singlet", dim=1)
    spvec = IndexSpace("sp4q", dim=4, indices="ijklmn", metric="omega")
    spgen = make_sp_2n_generator(sp, spvec)
    psi = Tensor("Psi", [spvec.upper("i")], reps={"Sp(4)": "vector"})
    M = _generator_M(apply_generator(psi, spgen))
    assert not M.cometric_antisymmetric_pairs


def test_nonorthogonal_action_does_not_cancel_delta_bilinear():
    """The δL→0 cancellation is gated on the orthogonal antisym mark, not on
    δ-bilinear structure alone: the same vector action *without* the mark
    leaves δL ≠ 0, so the verifier honestly reports 'not verified'."""
    gen = Generator("N", _group())
    gen.declare_action("vector", lorentz_vector_action(VEC, cometric_antisym=False))
    res = verify_symmetry(_mass(), gen, field_names={"Phi"})
    assert not res.is_symmetry


# ─── LRL: hidden SO(4) boost verified off-shell ─────────

_P = "p"   # LRL parameter direction (a vector index the Kepler L never uses)


def _delta_up(a, b):
    return Tensor("delta", [VEC.upper(a), VEC.upper(b)], symmetric_pairs=[(0, 1)])


def _x_dot_x():
    """δ_kl Φ^k Φ̇^l  (scalar x·ẋ), fresh dummies each call."""
    k, l = _fresh_dummy_name(), _fresh_dummy_name()
    return _delta(k, l) * _phi(k) * TimeDeriv(_phi(l))


def _lrl_action(field):
    """δΦ^i = 2 Φ̇^i Φ^p − Φ^i Φ̇^p − (x·ẋ) δ^{ip}."""
    i = field.indices[0].name
    t1 = ScalarMul(2.0, TimeDeriv(_phi(i)) * _phi(_P))
    t2 = ScalarMul(-1.0, field * TimeDeriv(_phi(_P)))
    t3 = ScalarMul(-1.0, _x_dot_x() * _delta_up(i, _P))
    return TensorSum(TensorSum(t1, t2), t3)


def _lrl_F():
    """F^p = (ẋ·ẋ) Φ^p − (x·ẋ) Φ̇^p + Φ^p / r."""
    k, l = _fresh_dummy_name(), _fresh_dummy_name()
    xdot2 = _delta(k, l) * TimeDeriv(_phi(k)) * TimeDeriv(_phi(l))
    F1 = xdot2 * _phi(_P)
    F2 = ScalarMul(-1.0, _x_dot_x() * TimeDeriv(_phi(_P)))
    F3 = _phi(_P) * ScalarFunction("inv_sqrt", _r2())
    return TensorSum(TensorSum(F1, F2), F3)


def test_kepler_lrl_boost_verifies_offshell():
    """Hidden SO(4): the velocity-dependent LRL boost
    δΦ^i = 2Φ̇^i Φ^p − Φ^i Φ̇^p − (x·ẋ)δ^{ip} is a Noether symmetry of the full
    Kepler L (incl. the 1/r ScalarFunction), with δL = d/dt F^p — verified
    *off-shell* (no EOM). Closes the LRL sector of Kepler's hidden SO(4): the
    C0b gate. Exercises the LRL backend lifts (free-index metric absorption,
    mixed-Kronecker elimination, inv_sqrt homogeneity)."""
    gen = Generator("LRL", _group())
    gen.declare_action("vector", _lrl_action)
    gen.declare_action("singlet", lambda f: ZeroTensor(f.free_indices))
    res = verify_symmetry(_kepler(), gen, field_names={"Phi"},
                          boundary_candidates=[_lrl_F()])
    assert res.is_symmetry and not res.exact      # δL = d/dt F (F ≠ 0)
    assert res.boundary_term is not None


def test_lrl_wrong_coefficient_is_not_a_symmetry():
    """Soundness: corrupt the LRL ansatz (coefficient 1 instead of 2 on the
    first term) — verify_symmetry must reject it (δL ≠ d/dt F^p)."""
    def bad_action(field):
        i = field.indices[0].name
        t1 = ScalarMul(1.0, TimeDeriv(_phi(i)) * _phi(_P))   # should be 2.0
        t2 = ScalarMul(-1.0, field * TimeDeriv(_phi(_P)))
        t3 = ScalarMul(-1.0, _x_dot_x() * _delta_up(i, _P))
        return TensorSum(TensorSum(t1, t2), t3)
    gen = Generator("LRLbad", _group())
    gen.declare_action("vector", bad_action)
    gen.declare_action("singlet", lambda f: ZeroTensor(f.free_indices))
    res = verify_symmetry(_kepler(), gen, field_names={"Phi"},
                          boundary_candidates=[_lrl_F()])
    assert not res.is_symmetry


def test_simplify_eliminates_mixed_kronecker():
    """δ^p{}_j Φ^j → Φ^p  (mixed-position Kronecker is the identity map)."""
    kron = Tensor("delta", [VEC.upper(_P), VEC.lower("j")])
    out = simplify(kron * _phi("j"))
    assert isinstance(out, Tensor) and out.name == "Phi"
    assert [idx.name for idx in out.free_indices] == [_P]


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


# ─── (C) Field-theory ∂_μ Noether: δL = ∂_μ J^μ (total divergence) ─────

_ST = IndexSpace("st", dim=4, indices="mnpqrsuvw", metric="eta")
_TR = Group("Tr", dim=4, abelian=False)
_TR.add_rep("scalar", dim=1)
_SPHI = Tensor("phi", [], reps={"Tr": "scalar"})


def _scalar_kinetic(d1, d2):
    """−½ η^{d1 d2} ∂_{d1}φ ∂_{d2}φ (massless free scalar)."""
    eta_inv = Tensor("eta", [_ST.upper(d1), _ST.upper(d2)], symmetric_pairs=[(0, 1)])
    return ScalarMul(-0.5, eta_inv * PartialDeriv(_SPHI, _ST.lower(d1))
                     * PartialDeriv(_SPHI, _ST.lower(d2)))


def _translation_action(field):
    """δφ = a^r ∂_r φ — spacetime translation (a constant, fresh dummy)."""
    r = _fresh_dummy_name()
    return Tensor("a", [_ST.upper(r)]) * PartialDeriv(field, _ST.lower(r))


def test_scalar_translation_is_total_divergence():
    """Free scalar spacetime translation δφ=a^ν∂_νφ → δL = ∂_μ(a^μ L), the
    field-theory analogue of time translation. Exercises the ∂_μ verifier
    (verify_symmetry_ft) and the two simplify lifts it needs: partial-derivative
    commutativity (∂_μ∂_ν=∂_ν∂_μ) and dummy canonicalization over the resulting
    symmetric second-derivative stack."""
    gen = Generator("transl", _TR)
    gen.declare_action("scalar", _translation_action)
    L = _scalar_kinetic("m", "n")
    J = Tensor("a", [_ST.upper("p")]) * _scalar_kinetic("u", "v")    # J^p = a^p L
    res = verify_symmetry_ft(L, gen, field_names={"phi"},
                             deriv_index=_ST.lower("p"), current_candidates=[J])
    assert res.is_symmetry and not res.exact
    assert res.boundary_term is not None


def test_scalar_translation_wrong_current_rejected():
    """Soundness: a wrong current (coefficient 2 instead of 1) must not verify."""
    gen = Generator("transl", _TR)
    gen.declare_action("scalar", _translation_action)
    L = _scalar_kinetic("m", "n")
    bad_J = ScalarMul(2.0, Tensor("a", [_ST.upper("p")]) * _scalar_kinetic("u", "v"))
    res = verify_symmetry_ft(L, gen, field_names={"phi"},
                             deriv_index=_ST.lower("p"), current_candidates=[bad_J])
    assert not res.is_symmetry


def test_partial_derivatives_commute_in_simplify():
    """∂_m∂_r(φ) − ∂_r∂_m(φ) = 0 (partial derivatives commute)."""
    a = PartialDeriv(PartialDeriv(_SPHI, _ST.lower("r")), _ST.lower("m"))
    b = PartialDeriv(PartialDeriv(_SPHI, _ST.lower("m")), _ST.lower("r"))
    assert isinstance(simplify(TensorSum(a, ScalarMul(-1.0, b))), ZeroTensor)
