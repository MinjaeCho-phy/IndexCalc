"""LIONS M6 acceptance — partial-derivative Lorentz hook + invariant-tensor commute.

검증 대상 (M6 commit ``27400f3`` 인프라):

1. ``commute_partial_through_constants`` simplifier:
   $\\partial_\\mu(\\Sigma \\cdot \\psi) \\to \\Sigma \\cdot \\partial_\\mu \\psi$ — invariant
   tensor (``reps={}``)는 spacetime 미분 밖으로 빠지고, dynamic field는 안에 머문다.

2. ``lorentz_deriv_index_action`` (``Generator._deriv_index_action`` hook):
   $\\delta\\partial_\\mu T = -\\partial_\\nu T \\cdot M^{ab,\\nu}{}_\\mu$.
   ``make_lorentz_spinor_generator``가 ``vector`` rep 등록 시 자동 활성화.

3. End-to-end: $V^\\mu \\partial_\\mu \\phi$ Lorentz invariance.
   $\\delta V^\\mu$ (M4 vector 회전) + $\\delta \\partial_\\mu \\phi$ (M6 deriv_index 회전)이
   같은 $M_{\\rm vec}$ body를 만들어 dummy renaming 후 cancel.

**제외 (Clifford simplifier 필요 — M6.5/M7):**

자유 Dirac 운동항 $i\\bar\\psi \\gamma^\\mu \\partial_\\mu \\psi$의 Lorentz invariance에는
$[\\Sigma^{ab}, \\gamma^c] = i(\\eta^{bc}\\gamma^a - \\eta^{ac}\\gamma^b)$ identity 적용
pass가 추가로 필요. 본 M6 인프라는 그 전제(γ를 ∂ 밖으로 commute, ∂의 vector 회전)
까지만 제공.
"""

import pytest

from indexcalc.core.index import IndexSpace, Index
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.deriv import PartialDeriv
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    make_lorentz_spinor_generator,
    make_su_n_generator,
)
from indexcalc.core.substitution import apply_generator
from indexcalc.core.simplify import simplify, commute_partial_through_constants


@pytest.fixture
def setup():
    """Lorentz Group with spinor + vector reps. ``vector`` rep 덕분에
    ``lorentz_deriv_index_action``이 자동 등록된다."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )
    return st, dirac, lorentz, gen


# ─── M6-A: commute_partial_through_constants ─────────────────────────────


def test_commute_partial_pulls_invariant_left(setup):
    """$\\partial_\\mu(\\Sigma \\cdot \\psi) \\to \\Sigma \\cdot \\partial_\\mu \\psi$.

    Σ는 ``reps={}`` (group invariant tensor) → constant 판정, ∂ 밖으로 빠짐.
    """
    st, dirac, *_ = setup
    psi = Tensor(
        "psi", [dirac.upper("α")],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )
    Sigma = Tensor(
        "Sigma",
        [Index("a", st, "upper"), Index("b", st, "upper"),
         dirac.upper("β"), dirac.lower("α")],
        reps={},  # invariant tensor convention
    )
    expr = PartialDeriv(TensorProduct(Sigma, psi), st.lower("μ"))

    out = commute_partial_through_constants(expr)
    assert isinstance(out, TensorProduct)
    assert out.left is Sigma
    assert isinstance(out.right, PartialDeriv)
    assert out.right.expr is psi
    assert out.right.deriv_index.name == "μ"


def test_commute_partial_pulls_invariant_right(setup):
    """$\\partial_\\mu(\\psi \\cdot \\gamma) \\to \\partial_\\mu \\psi \\cdot \\gamma$."""
    st, dirac, *_ = setup
    psi = Tensor(
        "psi", [dirac.upper("α")],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )
    gamma = Tensor(
        "gamma",
        [Index("c", st, "upper"), dirac.upper("β"), dirac.lower("α")],
        reps={},
    )
    expr = PartialDeriv(TensorProduct(psi, gamma), st.lower("μ"))

    out = commute_partial_through_constants(expr)
    assert isinstance(out, TensorProduct)
    assert isinstance(out.left, PartialDeriv)
    assert out.left.expr is psi
    assert out.right is gamma


def test_commute_partial_keeps_dynamic_pair(setup):
    """둘 다 dynamic이면 Leibniz 분배 안 함 — 그대로 유지."""
    st, dirac, *_ = setup
    psi = Tensor(
        "psi", [dirac.upper("α")],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )
    chi = Tensor(
        "chi", [dirac.upper("β")],
        reps={"Lorentz": "spinor"}, statistics="fermionic",
    )
    expr = PartialDeriv(TensorProduct(psi, chi), st.lower("μ"))

    out = commute_partial_through_constants(expr)
    assert out is expr  # 변경 없음


# ─── M6-B: lorentz_deriv_index_action 구조 sanity ───────────────────────


def test_lorentz_acts_on_partial_singlet(setup):
    """$\\delta(\\partial_\\mu \\phi) = -\\partial_\\nu \\phi \\cdot M^{ab,\\nu}{}_\\mu$ for singlet $\\phi$.

    φ singlet → δφ = 0 → inner_term은 ZeroTensor → drop. deriv_term만 살아남는다.
    """
    st, _dirac, _lorentz, gen = setup
    phi = Tensor("phi", [], reps={"Lorentz": "singlet"})
    pd = PartialDeriv(phi, st.lower("μ"))

    result = apply_generator(pd, gen)
    assert isinstance(result, ScalarMul)
    assert result.scalar == -1.0
    inner = result.expr
    assert isinstance(inner, TensorProduct)
    new_pd, M = inner.left, inner.right
    assert isinstance(new_pd, PartialDeriv)
    assert new_pd.expr is phi
    assert M.name == "M_vec"
    assert (0, 1) in M.antisymmetric_pairs
    # M의 col이 input deriv_index 이름과 일치 (lower)
    assert M.indices[3].name == "μ"
    assert M.indices[3].position == "lower"
    # M의 slot 2 (upper) ↔ new_pd의 deriv_index (lower) — dummy contract
    assert M.indices[2].position == "upper"
    assert new_pd.deriv_index.name == M.indices[2].name


# ─── M6-C: End-to-end Lorentz invariance — V^μ ∂_μ φ ────────────────────


def test_vector_dot_partial_scalar_invariant(setup):
    """$\\delta_{\\rm Lorentz}(V^\\mu \\partial_\\mu \\phi) = 0$.

    Leibniz 두 항:
        $(M^{ab,\\mu}{}_\\nu V^\\nu) \\partial_\\mu\\phi$ — vector 회전 (M4)
        $V^\\mu (-\\partial_\\nu\\phi \\cdot M^{ab,\\nu}{}_\\mu)$ — deriv_index 회전 (M6)

    같은 $M_{\\rm vec}$ contraction body, 부호만 반대 → ``canonical_form_modulo_dummies``
    + ``collect_scalar_terms``로 cancel.
    """
    st, _dirac, _lorentz, gen = setup
    Vu = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    phi = Tensor("phi", [], reps={"Lorentz": "singlet"})
    L = TensorProduct(Vu, PartialDeriv(phi, st.lower("μ")))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor), f"got {final!r}"


def test_vector_dot_partial_scalar_invariant_with_coefficient(setup):
    """ScalarMul wrapping ($\\tfrac12 V^\\mu \\partial_\\mu \\phi$ 등 운동항 prefactor)도 통과."""
    st, _dirac, _lorentz, gen = setup
    Vu = Tensor("V", [st.upper("μ")], reps={"Lorentz": "vector"})
    phi = Tensor("phi", [], reps={"Lorentz": "singlet"})
    L = ScalarMul(0.5, TensorProduct(Vu, PartialDeriv(phi, st.lower("μ"))))

    delta = apply_generator(L, gen)
    final = simplify(delta)
    assert isinstance(final, ZeroTensor)


# ─── M6-D: Negative — SU(N) generator는 deriv_index 안 건드림 ───────────


def test_su_n_does_not_rotate_deriv_index():
    """``make_su_n_generator``는 ``declare_deriv_index_action`` 호출 안 함.
    $\\partial_\\mu A^a$에 SU(N) 적용 시 deriv_index 회전 항 없음 — inner_term만 살아남고
    결과는 단일 ``PartialDeriv``."""
    st = IndexSpace("st_for_sun", dim=4, indices="μν", metric="η")
    adj = IndexSpace("su3_adj", dim=8, indices="abcd")

    sun = Group("SU(3)", dim=8, abelian=False)
    sun.add_rep("adj", dim=8)
    sun.add_rep("singlet", dim=1)

    g = make_su_n_generator(sun, adj)
    A = Tensor("A", [adj.upper("a")], reps={"SU(3)": "adj"})
    pd = PartialDeriv(A, st.lower("μ"))

    result = apply_generator(pd, g)
    # deriv_term None → 단일 PartialDeriv (deriv_index 그대로 'μ')
    assert isinstance(result, PartialDeriv)
    assert result.deriv_index.name == "μ"
    assert result.deriv_index.position == "lower"
