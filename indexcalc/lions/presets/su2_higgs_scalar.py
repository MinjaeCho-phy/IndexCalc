"""B0 preset — SU(2)_L fundamental Higgs doublet + U(1)_Y, scalar-only.

The smallest non-trivial enumeration target: one bosonic doublet H and
its conjugate H†, two gauge groups, no fermion, no gauge field. The
expected invariant catalog includes the standard Higgs sector pieces:

    |H|^2,  |H|^4,  |∂H|^2

This preset returns the index spaces, group registry, generators, and a
LIONS ``FieldRegistry`` so the enumerator and labeler can be wired up
without rebuilding fixtures across tests.
"""

from __future__ import annotations
from dataclasses import dataclass

from indexcalc.core.index import IndexSpace
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    Generator, make_u1_generator, make_su_n_generator,
)
from indexcalc.lions.fields import FieldSpec, FieldRegistry, SlotSpec


@dataclass
class B0Setup:
    spacetime: IndexSpace
    su2_adj: IndexSpace
    su2_fund: IndexSpace
    su2: Group
    u1y: Group
    su2_gen: Generator
    u1y_gen: Generator
    fields: FieldRegistry


def build_b0() -> B0Setup:
    """Construct the B0 preset (idempotent: caller-owned fresh objects)."""
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ijklmnpq")

    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("fund", dim=2)
    su2.add_rep("singlet", dim=1)

    u1y = Group("U(1)_Y", dim=1, abelian=True)
    u1y.add_rep("+1/2", dim=1, charge=0.5)
    u1y.add_rep("-1/2", dim=1, charge=-0.5)
    u1y.add_rep("0", dim=1, charge=0.0)

    su2_gen = make_su_n_generator(
        su2, su2_adj, parameter_name="P", fund_space=su2_fund,
    )
    u1y_gen = make_u1_generator(u1y, name="T_U(1)_Y")

    reg = FieldRegistry()
    reg.add(FieldSpec(
        name="H",
        slots=(SlotSpec(su2_fund, "upper"),),
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2"},
        mass_dim=1.0,
        statistics="bosonic",
    ))
    reg.add(FieldSpec(
        name="Hdag",
        slots=(SlotSpec(su2_fund, "lower"),),
        reps={"SU(2)": "fund", "U(1)_Y": "-1/2"},
        mass_dim=1.0,
        statistics="bosonic",
    ))

    return B0Setup(
        spacetime=st, su2_adj=su2_adj, su2_fund=su2_fund,
        su2=su2, u1y=u1y, su2_gen=su2_gen, u1y_gen=u1y_gen,
        fields=reg,
    )
