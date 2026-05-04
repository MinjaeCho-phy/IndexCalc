"""Group / Representation / GroupRegistry 테스트 (LIONS M1 / E1)."""

import pytest

from indexcalc.core.group import Group, Representation, GroupRegistry


# ─── Representation ─────────────────────────────────────────


def test_representation_immutable():
    r = Representation(name="fund", group_name="SU(3)", dim=3)
    with pytest.raises(Exception):
        r.dim = 5  # frozen dataclass


def test_representation_defaults():
    r = Representation(name="fund", group_name="SU(3)", dim=3)
    assert r.conjugate is False
    assert r.charge is None


# ─── Group: abelian (U(1)) ──────────────────────────────────


class TestAbelianGroup:
    def test_u1_register_charged_reps(self):
        u1 = Group("U(1)", dim=1, abelian=True)
        u1.add_rep("+1", dim=1, charge=1.0)
        u1.add_rep("-1", dim=1, charge=-1.0)
        u1.add_rep("0", dim=1, charge=0.0)

        assert u1.has_rep("+1")
        assert u1.get_rep("+1").charge == 1.0
        assert u1.get_rep("-1").charge == -1.0

    def test_abelian_requires_charge(self):
        u1 = Group("U(1)", abelian=True)
        with pytest.raises(ValueError, match="charge"):
            u1.add_rep("nope", dim=1)

    def test_duplicate_rep_raises(self):
        u1 = Group("U(1)", abelian=True)
        u1.add_rep("+1", dim=1, charge=1.0)
        with pytest.raises(ValueError, match="already registered"):
            u1.add_rep("+1", dim=1, charge=1.0)


# ─── Group: non-abelian (SU(N)) ─────────────────────────────


class TestNonAbelianGroup:
    def test_su3_standard_reps(self):
        sun = Group("SU(3)", dim=8, abelian=False)
        sun.add_rep("fund", dim=3)
        sun.add_rep("antifund", dim=3, conjugate=True)
        sun.add_rep("adj", dim=8)
        sun.add_rep("singlet", dim=1)

        assert sun.get_rep("fund").dim == 3
        assert sun.get_rep("antifund").conjugate is True
        assert sun.get_rep("adj").dim == 8
        assert sun.get_rep("singlet").dim == 1
        assert sun.dim == 8

    def test_unknown_rep_raises(self):
        sun = Group("SU(2)", dim=3, abelian=False)
        with pytest.raises(KeyError, match="not found"):
            sun.get_rep("triplet")

    def test_rep_carries_group_name(self):
        sun = Group("SU(3)", dim=8, abelian=False)
        rep = sun.add_rep("fund", dim=3)
        assert rep.group_name == "SU(3)"


# ─── GroupRegistry ──────────────────────────────────────────


class TestGroupRegistry:
    def test_register_and_get(self):
        reg = GroupRegistry()
        u1 = Group("U(1)", abelian=True)
        u1.add_rep("+1", dim=1, charge=1.0)
        reg.register(u1)

        assert reg.has("U(1)")
        assert reg.get("U(1)") is u1

    def test_double_register_raises(self):
        reg = GroupRegistry()
        u1 = Group("U(1)", abelian=True)
        reg.register(u1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(u1)

    def test_unknown_group_raises(self):
        reg = GroupRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("SU(5)")

    def test_groups_dict_is_copy(self):
        reg = GroupRegistry()
        u1 = Group("U(1)", abelian=True)
        reg.register(u1)
        snapshot = reg.groups
        snapshot.clear()  # 외부 dict 조작이 내부에 영향 X
        assert reg.has("U(1)")


# ─── Equality / hashing ─────────────────────────────────────


def test_group_equality_by_name():
    a = Group("SU(3)", abelian=False)
    b = Group("SU(3)", abelian=False)
    c = Group("SU(2)", abelian=False)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
