"""Generator 테스트 (LIONS M1 / E3) — U(1)만 (non-abelian은 M2)."""

import pytest

from indexcalc.core.group import Group
from indexcalc.core.tensor import Tensor, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.generator import Generator, u1_action, make_u1_generator


# ─── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def u1():
    g = Group("U(1)", dim=1, abelian=True)
    g.add_rep("+1", dim=1, charge=1.0)
    g.add_rep("-1", dim=1, charge=-1.0)
    g.add_rep("0", dim=1, charge=0.0)
    return g


@pytest.fixture
def u1_gen(u1):
    return make_u1_generator(u1)


# ─── declare_action / has_action ────────────────────────────


class TestDeclareAction:
    def test_declare_and_has(self, u1):
        gen = Generator("T", u1)
        gen.declare_action("+1", u1_action(u1.get_rep("+1")))
        assert gen.has_action("+1")
        assert not gen.has_action("-1")

    def test_unknown_rep_raises(self, u1):
        gen = Generator("T", u1)
        with pytest.raises(ValueError, match="not in group"):
            gen.declare_action("triplet", u1_action(u1.get_rep("+1")))


# ─── apply_to: U(1) ─────────────────────────────────────────


class TestApplyToU1:
    def test_charged_field(self, u1_gen):
        phi = Tensor("phi", [], reps={"U(1)": "+1"})
        result = u1_gen.apply_to(phi)
        assert isinstance(result, ScalarMul)
        assert result.scalar == 1j * 1.0
        assert result.expr is phi

    def test_negative_charge(self, u1_gen):
        phistar = Tensor("phistar", [], reps={"U(1)": "-1"})
        result = u1_gen.apply_to(phistar)
        assert isinstance(result, ScalarMul)
        assert result.scalar == 1j * (-1.0)

    def test_zero_charge(self, u1_gen):
        # charge 0 rep도 등록되어 있으면 action은 0·field = ScalarMul(0, field)
        # (singlet은 reps 자체를 비워야 ZeroTensor가 나옴)
        f = Tensor("f", [], reps={"U(1)": "0"})
        result = u1_gen.apply_to(f)
        assert isinstance(result, ScalarMul)
        assert result.scalar == 0j

    def test_singlet_field_returns_zero_tensor(self, u1_gen):
        # reps에 U(1) 키가 없으면 singlet 취급
        psi = Tensor("psi", [], reps={})
        result = u1_gen.apply_to(psi)
        assert isinstance(result, ZeroTensor)

    def test_field_with_other_group_rep_only(self, u1_gen):
        # U(1) tag 없음 → U(1) 입장에서 singlet
        chi = Tensor("chi", [], reps={"SU(3)": "fund"})
        result = u1_gen.apply_to(chi)
        assert isinstance(result, ZeroTensor)


# ─── Singleton Generator without registered action ──────────


def test_undeclared_action_raises(u1):
    gen = Generator("T_partial", u1)
    # +1 만 등록, -1은 미등록
    gen.declare_action("+1", u1_action(u1.get_rep("+1")))
    phistar = Tensor("phistar", [], reps={"U(1)": "-1"})
    with pytest.raises(ValueError, match="no action declared"):
        gen.apply_to(phistar)


# ─── make_u1_generator factory ──────────────────────────────


def test_make_u1_generator_covers_all_reps(u1):
    gen = make_u1_generator(u1)
    for rep_name in ("+1", "-1", "0"):
        assert gen.has_action(rep_name)


def test_make_u1_generator_non_abelian_raises():
    sun = Group("SU(3)", dim=8, abelian=False)
    sun.add_rep("fund", dim=3)
    with pytest.raises(ValueError, match="abelian"):
        make_u1_generator(sun)


# ─── u1_action factory edge cases ────────────────────────────


def test_u1_action_requires_charge():
    # 직접 charge=None인 rep을 만들어 시도 (Group.add_rep는 막지만 우회 테스트)
    from indexcalc.core.group import Representation

    rep_no_charge = Representation(name="x", group_name="G", dim=1, charge=None)
    with pytest.raises(ValueError, match="charge"):
        u1_action(rep_no_charge)
