"""Generator 테스트 (LIONS M1/M2 — U(1) + SU(N) adjoint)."""

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.group import Group
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul
from indexcalc.core.variation import ZeroTensor
from indexcalc.core.generator import (
    Generator, u1_action, make_u1_generator,
    su_n_adj_action, make_su_n_generator,
)


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


# ─── SU(N) adjoint action (M2) ──────────────────────────────


@pytest.fixture
def sun_setup():
    sun = Group("SU(3)", dim=8, abelian=False)
    sun.add_rep("adj", dim=8)
    sun.add_rep("singlet", dim=1)
    sun.add_rep("fund", dim=3)
    adj = IndexSpace("su3_adj", dim=8, indices="abcdefgh")
    return sun, adj


class TestSUNAdjAction:
    def test_action_on_adj_field_upper(self, sun_setup):
        sun, adj = sun_setup
        action = su_n_adj_action(adj, parameter_name="b")
        # X^a (adj index 'a' upper)
        X = Tensor("X", [adj.upper("a")], reps={"SU(3)": "adj"})
        result = action(X)

        # f^a_{bc} · X^c
        assert isinstance(result, TensorProduct)
        f, Xprime = result.left, result.right
        assert isinstance(f, Tensor) and f.name == "f"
        assert len(f.indices) == 3
        assert f.indices[0].name == "a" and f.indices[0].position == "upper"
        assert f.indices[1].name == "b" and f.indices[1].position == "lower"
        # third index: dummy lower in adj space
        assert f.indices[2].position == "lower"
        assert f.indices[2].space == adj
        # all-pair antisym
        assert set(f.antisymmetric_pairs) == {(0, 1), (0, 2), (1, 2)}

        # X' has dummy upper in adj
        assert Xprime.name == "X"
        assert Xprime.indices[0].position == "upper"
        assert Xprime.indices[0].name == f.indices[2].name  # contracts

    def test_action_on_adj_field_lower(self, sun_setup):
        sun, adj = sun_setup
        action = su_n_adj_action(adj, parameter_name="b")
        X = Tensor("X", [adj.lower("a")], reps={"SU(3)": "adj"})
        result = action(X)

        f, Xprime = result.left, result.right
        # f's first index follows field: lower
        assert f.indices[0].name == "a" and f.indices[0].position == "lower"
        # second is parameter lower
        assert f.indices[1].name == "b" and f.indices[1].position == "lower"
        # X' adj index renamed to dummy, position upper (contracts with f's third lower)
        assert Xprime.indices[0].position == "upper"
        assert Xprime.indices[0].name == f.indices[2].name

    def test_dummy_name_avoids_collision(self, sun_setup):
        sun, adj = sun_setup
        # adj 공간 인덱스 문자가 "abc..." 인 상황에서, field가 "a", "b" 둘 다 가져 충돌 가능
        action = su_n_adj_action(adj, parameter_name="b")
        # field has indices a (adj upper), b (some other space) → 'a' and 'b' both exist
        st = IndexSpace("spacetime", dim=4, indices="μνλ")
        X = Tensor(
            "X",
            [adj.upper("a"), st.lower("μ")],
            reps={"SU(3)": "adj"},
        )
        result = action(X)
        f = result.left
        # dummy must not be 'a' or 'b' or 'μ'
        used = {"a", "b", "μ"}
        assert f.indices[2].name not in used

    def test_extra_field_indices_preserved(self, sun_setup):
        sun, adj = sun_setup
        st = IndexSpace("spacetime", dim=4, indices="μνλ")
        action = su_n_adj_action(adj, parameter_name="b")
        F = Tensor(
            "F",
            [adj.upper("a"), st.lower("μ"), st.lower("ν")],
            antisymmetric_pairs=[(1, 2)],
            reps={"SU(3)": "adj"},
        )
        result = action(F)
        Fprime = result.right
        # μ, ν 보존
        assert Fprime.indices[1] == st.lower("μ")
        assert Fprime.indices[2] == st.lower("ν")
        # antisymmetric_pairs 보존
        assert (1, 2) in Fprime.antisymmetric_pairs

    def test_field_without_adj_index_raises(self, sun_setup):
        sun, adj = sun_setup
        action = su_n_adj_action(adj, parameter_name="b")
        # adj 인덱스 없는 field
        X = Tensor("X", [], reps={"SU(3)": "adj"})
        with pytest.raises(ValueError, match="exactly one"):
            action(X)


class TestMakeSUNGenerator:
    def test_registers_adj_and_singlet(self, sun_setup):
        sun, adj = sun_setup
        gen = make_su_n_generator(sun, adj)
        assert gen.has_action("adj")
        assert gen.has_action("singlet")
        # fund는 미등록
        assert not gen.has_action("fund")

    def test_singlet_returns_zero(self, sun_setup):
        sun, adj = sun_setup
        gen = make_su_n_generator(sun, adj)
        s = Tensor("s", [], reps={"SU(3)": "singlet"})
        result = gen.apply_to(s)
        assert isinstance(result, ZeroTensor)

    def test_adj_returns_product(self, sun_setup):
        sun, adj = sun_setup
        gen = make_su_n_generator(sun, adj, parameter_name="d")
        X = Tensor("X", [adj.upper("a")], reps={"SU(3)": "adj"})
        result = gen.apply_to(X)
        assert isinstance(result, TensorProduct)
        # parameter index 이름이 'd'로 들어갔는지
        assert result.left.indices[1].name == "d"

    def test_fund_field_raises(self, sun_setup):
        sun, adj = sun_setup
        gen = make_su_n_generator(sun, adj)
        psi = Tensor("psi", [], reps={"SU(3)": "fund"})
        with pytest.raises(ValueError, match="no action declared"):
            gen.apply_to(psi)

    def test_abelian_group_raises(self):
        u1 = Group("U(1)", abelian=True)
        u1.add_rep("+1", dim=1, charge=1.0)
        adj = IndexSpace("dummy", dim=1)
        with pytest.raises(ValueError, match="non-abelian"):
            make_su_n_generator(u1, adj)


def test_u1_action_requires_charge():
    # 직접 charge=None인 rep을 만들어 시도 (Group.add_rep는 막지만 우회 테스트)
    from indexcalc.core.group import Representation

    rep_no_charge = Representation(name="x", group_name="G", dim=1, charge=None)
    with pytest.raises(ValueError, match="charge"):
        u1_action(rep_no_charge)
