"""scripts/vielbein_demo.py — Backend 2 vielbein/spin connection demo.

VielbeinSetup으로 e^a_μ, η_{ab}, g_{μν}, ω_μ^a_b 셋업하고:
    - vielbein collapse (e η e → g)
    - vielbein compatibility identity (∇_μ e^a_ν = 0)
    - spin connection definition (ω_μ^{ab} = e^{aν} ∇_μ e^b_ν)
출력 형태와 expand 결과를 확인.
"""

from __future__ import annotations

from indexcalc import (
    IndexSpace, MetricRegistry, LeviCivitaConnection,
    TensorProduct, expand_covariant,
    VielbeinSetup, collapse_vielbein_identity,
    vielbein_compatibility_lhs, spin_connection_from_vielbein,
    to_latex,
)


def main() -> None:
    print("=" * 70)
    print("Backend 2: Vielbein / Spin connection demo")
    print("=" * 70)

    st = IndexSpace("st", dim=4, indices="μνρσλ", metric="g")
    fr = IndexSpace("fr", dim=4, indices="abcde", metric="η")
    setup = VielbeinSetup(st, fr)
    print(f"  Spacetime st  : dim={st.dim}, metric='{st.metric}'")
    print(f"  Frame fr      : dim={fr.dim}, metric='{fr.metric}'")

    # ─── Leaf builders ─────────────────────────────────────
    print()
    print("  Leaf builders:")
    print(f"    e^a{{}}_μ    : {to_latex(setup.vielbein())}")
    print(f"    e_a{{}}^μ    : {to_latex(setup.vielbein_inverse())}")
    print(f"    e^{{aμ}}     : {to_latex(setup.vielbein_aμ_upper())}")
    print(f"    η_ab        : {to_latex(setup.frame_metric_lower())}")
    print(f"    η^ab        : {to_latex(setup.frame_metric_upper())}")
    print(f"    g_μν        : {to_latex(setup.spacetime_metric_lower())}")

    # Connections
    g_lo = setup.spacetime_metric_lower()
    g_up = setup.spacetime_metric_upper()
    chr = LeviCivitaConnection(g_lo, g_up, st)
    spin = setup.spin_connection()
    print(f"    Christoffel : {chr}")
    print(f"    Spin conn.  : {spin}")

    # ─── Vielbein collapse identity ────────────────────────
    print()
    print("  e^a{}_μ η_{ab} e^b{}_ν → g_{μν}:")
    e1 = setup.vielbein("a", "μ")
    eta = setup.frame_metric_lower("a", "b")
    e2 = setup.vielbein("b", "ν")
    triple = TensorProduct(TensorProduct(e1, eta), e2)
    print(f"    raw      : {to_latex(triple)}")
    collapsed = collapse_vielbein_identity(triple, setup.to_registry())
    print(f"    collapsed: {to_latex(collapsed)}")

    # ─── Vielbein compatibility ────────────────────────────
    print()
    print("  Vielbein compatibility  ∇_μ e^a{}_ν = 0:")
    compat = vielbein_compatibility_lhs(setup, chr)
    print(f"    LHS (compact)  : {to_latex(compat)}")
    print(f"    LHS (expanded) : {to_latex(expand_covariant(compat))}")
    print(f"    free           : {[i.name for i in compat.free_indices]}")

    # ─── Spin connection definition ────────────────────────
    print()
    print("  Spin connection  ω_μ^{ab} = e^{aν} ∇_μ e^b_ν:")
    omg = spin_connection_from_vielbein(setup, chr)
    print(f"    compact   : {to_latex(omg)}")
    print(f"    expanded  : {to_latex(expand_covariant(omg))}")
    print(f"    free      : {[i.name for i in omg.free_indices]}")

    print()
    print("=" * 70)
    print("done.")


if __name__ == "__main__":
    main()
