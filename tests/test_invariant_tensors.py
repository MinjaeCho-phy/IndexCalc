"""InvariantTensorRegistry 테스트 (LIONS M1 / E2)."""

import pytest

from indexcalc.core.invariant_tensors import (
    InvariantTensor,
    InvariantTensorRegistry,
    standard_su_n_invariants,
    standard_u_n_invariants,
    standard_lorentz_invariants,
)


# ─── InvariantTensor 메타데이터 ─────────────────────────────


def test_invariant_tensor_immutable():
    t = InvariantTensor("delta", "SU(3)", ("fund_upper", "fund_lower"))
    with pytest.raises(Exception):
        t.name = "f"  # frozen


# ─── Registry 기본 ──────────────────────────────────────────


class TestRegistryBasic:
    def test_declare_and_lookup(self):
        reg = InvariantTensorRegistry()
        reg.declare("delta", "SU(3)", ("fund_upper", "fund_lower"))
        assert reg.is_invariant("delta", "SU(3)")
        assert reg.get("delta", "SU(3)").index_pattern == (
            "fund_upper",
            "fund_lower",
        )

    def test_double_declare_raises(self):
        reg = InvariantTensorRegistry()
        reg.declare("f", "SU(2)", ("adj",) * 3, "totally_antisymmetric")
        with pytest.raises(ValueError, match="already declared"):
            reg.declare("f", "SU(2)", ("adj",) * 3, "totally_antisymmetric")

    def test_same_name_different_group_ok(self):
        reg = InvariantTensorRegistry()
        reg.declare("epsilon", "SU(2)", ("fund_lower",) * 2,
                    "totally_antisymmetric")
        reg.declare("epsilon", "SU(3)", ("fund_lower",) * 3,
                    "totally_antisymmetric")
        assert reg.is_invariant("epsilon", "SU(2)")
        assert reg.is_invariant("epsilon", "SU(3)")
        assert reg.get("epsilon", "SU(2)").index_pattern == ("fund_lower",) * 2
        assert reg.get("epsilon", "SU(3)").index_pattern == ("fund_lower",) * 3

    def test_unknown_lookup_raises(self):
        reg = InvariantTensorRegistry()
        with pytest.raises(KeyError):
            reg.get("delta", "SU(7)")

    def test_invalid_symmetry_rejected(self):
        reg = InvariantTensorRegistry()
        with pytest.raises(ValueError, match="unknown symmetry"):
            reg.declare("X", "G", ("a", "b"), symmetry="weird")

    def test_list_for_group(self):
        reg = InvariantTensorRegistry()
        reg.declare("delta", "SU(2)", ("fund_upper", "fund_lower"))
        reg.declare("f", "SU(2)", ("adj",) * 3, "totally_antisymmetric")
        reg.declare("eta", "Lorentz", ("frame_lower",) * 2, "symmetric")

        names = sorted(reg.list_for_group("SU(2)"))
        assert names == ["delta", "f"]
        assert reg.list_for_group("Lorentz") == ["eta"]
        assert reg.list_for_group("nonexistent") == []


# ─── 표준 헬퍼 ──────────────────────────────────────────────


class TestStandardHelpers:
    def test_su3_standard(self):
        items = standard_su_n_invariants(3)
        names = sorted(t.name for t in items)
        assert names == ["d", "delta", "epsilon", "f"]
        epsilon = next(t for t in items if t.name == "epsilon")
        assert len(epsilon.index_pattern) == 3
        assert epsilon.symmetry == "totally_antisymmetric"

    def test_su2_epsilon_rank2(self):
        items = standard_su_n_invariants(2)
        epsilon = next(t for t in items if t.name == "epsilon")
        assert len(epsilon.index_pattern) == 2

    def test_u_n_no_epsilon(self):
        items = standard_u_n_invariants(3)
        names = sorted(t.name for t in items)
        assert "epsilon" not in names
        assert "delta" in names

    def test_lorentz_standard(self):
        items = standard_lorentz_invariants()
        names = sorted(t.name for t in items)
        assert names == ["epsilon4", "eta"]


# ─── 표준 헬퍼와 레지스트리 통합 ────────────────────────────


def test_register_standard_su3_to_registry():
    reg = InvariantTensorRegistry()
    for inv in standard_su_n_invariants(3):
        reg.declare(inv.name, inv.group_name, inv.index_pattern, inv.symmetry)
    assert reg.is_invariant("delta", "SU(3)")
    assert reg.is_invariant("epsilon", "SU(3)")
    assert reg.get("epsilon", "SU(3)").symmetry == "totally_antisymmetric"
