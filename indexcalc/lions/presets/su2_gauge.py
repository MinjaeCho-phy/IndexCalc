"""B2 preset — SU(2) gauge field W^A_μ + field strength F^A_{μν} (gauge sector only).

D6 minimal cut: F^A_{μν} is registered as an **independent IR primitive** with
an antisymmetric (μν) pair, not synthesised from ∂_μW^A_ν − ∂_νW^A_μ. The
synthesis (and covariant-derivative lifting) is D7+ scope.

Targets recovered by the enumerator at dim ≤ 4:
    W·W   (= W^A_μ W_A^μ)        — SU(2)+Lorentz singlet, mass dim 2
    F·F   (= F^A_{μν} F_A^{μν})  — SU(2)+Lorentz singlet, mass dim 4

W^A_μ alone (free indices) is never produced by the enumerator (open indices
get matched); single-W invariance is therefore not a v1 acceptance target.
"""

from __future__ import annotations
from dataclasses import dataclass

from indexcalc.core.index import IndexSpace
from indexcalc.core.group import Group
from indexcalc.core.generator import (
    Generator, make_su_n_generator, make_lorentz_spinor_generator,
)
from indexcalc.lions.fields import FieldSpec, FieldRegistry, SlotSpec


@dataclass
class B2Setup:
    spacetime: IndexSpace
    su2_adj: IndexSpace
    dirac: IndexSpace                # placeholder, no fields use it
    su2: Group
    lorentz: Group
    su2_gen: Generator
    lorentz_gen: Generator
    fields: FieldRegistry


def build_b2() -> B2Setup:
    st = IndexSpace("spacetime", dim=4, indices="μνλρσ", metric="η")
    su2_adj = IndexSpace("su2_adj", dim=3, indices="ABCDE", metric="δ")
    # dirac is required by make_lorentz_spinor_generator's signature even
    # though we register no spinor reps; closure is never invoked.
    dirac = IndexSpace("dirac", dim=4, indices="αβγδε")

    su2 = Group("SU(2)", dim=3, abelian=False)
    su2.add_rep("adj", dim=3)
    su2.add_rep("singlet", dim=1)

    lorentz = Group("Lorentz", dim=6, abelian=False)
    lorentz.add_rep("vector", dim=4)
    lorentz.add_rep("singlet", dim=1)

    su2_gen = make_su_n_generator(
        su2, su2_adj, parameter_name="P",
    )
    lorentz_gen = make_lorentz_spinor_generator(
        lorentz, frame_space=st, spinor_space=dirac,
    )

    reg = FieldRegistry()
    reg.add(FieldSpec(
        name="W",
        slots=(SlotSpec(su2_adj, "upper"), SlotSpec(st, "lower")),
        reps={"SU(2)": "adj", "Lorentz": "vector"},
        mass_dim=1.0,
        statistics="bosonic",
    ))
    reg.add(FieldSpec(
        name="F",
        slots=(SlotSpec(su2_adj, "upper"),
               SlotSpec(st, "lower"),
               SlotSpec(st, "lower")),
        reps={"SU(2)": "adj", "Lorentz": "vector"},
        antisymmetric_pairs=((1, 2),),
        mass_dim=2.0,
        statistics="bosonic",
    ))

    return B2Setup(
        spacetime=st, su2_adj=su2_adj, dirac=dirac,
        su2=su2, lorentz=lorentz,
        su2_gen=su2_gen, lorentz_gen=lorentz_gen,
        fields=reg,
    )
