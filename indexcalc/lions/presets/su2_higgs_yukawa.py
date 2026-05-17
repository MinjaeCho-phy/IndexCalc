"""B1 preset — B0 + chiral fermions for the lepton-Yukawa sector.

Extends ``su2_higgs_scalar.build_b0`` with three fermion fields:

- ``L^i_α``         SU(2) fund upper, U(1)_Y = -1/2, L_spinor (dirac upper)
- ``Lbar^i_α``      SU(2) fund upper (pseudoreal), U(1)_Y = +1/2,
                    conj_L_spinor (dirac lower)
- ``eR^α``          SU(2) singlet,  U(1)_Y = -1,    R_spinor (dirac upper)

Convention is identical to ``tests/test_m8_acceptance.py`` so the
enumerated $\\bar L^i H^j \\epsilon_{ij} e_R$ piece routes through the same
oracle that the M8 acceptance already verifies.
"""

from __future__ import annotations
from dataclasses import dataclass

from indexcalc.core.index import IndexSpace
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    Generator, make_u1_generator, make_su_n_generator,
    make_lorentz_spinor_generator,
)
from indexcalc.lions.fields import (
    FieldSpec, FieldRegistry, SlotSpec, InvariantTensorSpec,
)


@dataclass
class B1Setup:
    spacetime: IndexSpace
    su2_adj: IndexSpace
    su2_fund: IndexSpace
    dirac: IndexSpace
    su2: Group
    u1y: Group
    lorentz: Group
    su2_gen: Generator
    u1y_gen: Generator
    lorentz_gen: Generator
    fields: FieldRegistry
    invariant_alphabet: list  # list[InvariantTensorSpec]


def build_b1() -> B1Setup:
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    su2_fund = IndexSpace("su2_fund", dim=2, indices="ijklmnpq")
    dirac = IndexSpace("dirac", dim=4, indices="αβγδερστυφ")

    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("fund", dim=2)
    su2.add_rep("antifund", dim=2, conjugate=True)
    su2.add_rep("singlet", dim=1)

    u1y = Group("U(1)_Y", dim=1, abelian=True)
    u1y.add_rep("+1/2", dim=1, charge=0.5)
    u1y.add_rep("-1/2", dim=1, charge=-0.5)
    u1y.add_rep("+1", dim=1, charge=1.0)
    u1y.add_rep("-1", dim=1, charge=-1.0)
    u1y.add_rep("0", dim=1, charge=0.0)

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("L_spinor", dim=2)
    lorentz.add_rep("R_spinor", dim=2)
    lorentz.add_rep("conj_L_spinor", dim=2, conjugate=True)
    lorentz.add_rep("conj_R_spinor", dim=2, conjugate=True)
    lorentz.add_rep("spinor", dim=4)
    lorentz.add_rep("conj_spinor", dim=4, conjugate=True)
    # vector rep is required so make_lorentz_spinor_generator registers
    # deriv_index_action — without it ∂_μ does not transform under Lorentz
    # and kinetic-term cancellation fails.
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    su2_gen = make_su_n_generator(
        su2, su2_adj, parameter_name="P", fund_space=su2_fund,
    )
    u1y_gen = make_u1_generator(u1y, name="T_U(1)_Y")
    lorentz_gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )

    reg = FieldRegistry()
    reg.add(FieldSpec(
        name="H",
        slots=(SlotSpec(su2_fund, "upper"),),
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "singlet"},
        mass_dim=1.0,
        statistics="bosonic",
    ))
    reg.add(FieldSpec(
        name="Hdag",
        slots=(SlotSpec(su2_fund, "lower"),),
        reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "singlet"},
        mass_dim=1.0,
        statistics="bosonic",
    ))
    reg.add(FieldSpec(
        name="L",
        slots=(SlotSpec(su2_fund, "upper"), SlotSpec(dirac, "upper")),
        reps={"SU(2)": "fund", "U(1)_Y": "-1/2", "Lorentz": "L_spinor"},
        mass_dim=1.5,
        statistics="fermionic",
    ))
    reg.add(FieldSpec(
        name="Lbar",
        slots=(SlotSpec(su2_fund, "upper"), SlotSpec(dirac, "lower")),
        reps={"SU(2)": "fund", "U(1)_Y": "+1/2", "Lorentz": "conj_L_spinor"},
        mass_dim=1.5,
        statistics="fermionic",
    ))
    reg.add(FieldSpec(
        name="eR",
        slots=(SlotSpec(dirac, "upper"),),
        reps={"SU(2)": "singlet", "U(1)_Y": "-1", "Lorentz": "R_spinor"},
        mass_dim=1.5,
        statistics="fermionic",
    ))
    reg.add(FieldSpec(
        name="eRbar",
        slots=(SlotSpec(dirac, "lower"),),
        reps={"SU(2)": "singlet", "U(1)_Y": "+1", "Lorentz": "conj_R_spinor"},
        mass_dim=1.5,
        statistics="fermionic",
    ))

    invariant_alphabet = [
        InvariantTensorSpec(
            name="gamma",
            slots=(SlotSpec(st, "upper"),
                   SlotSpec(dirac, "upper"),
                   SlotSpec(dirac, "lower")),
            reps={},
        ),
    ]

    return B1Setup(
        spacetime=st, su2_adj=su2_adj, su2_fund=su2_fund, dirac=dirac,
        su2=su2, u1y=u1y, lorentz=lorentz,
        su2_gen=su2_gen, u1y_gen=u1y_gen, lorentz_gen=lorentz_gen,
        fields=reg, invariant_alphabet=invariant_alphabet,
    )
