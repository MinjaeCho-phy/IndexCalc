"""D16: ScalarFunction smoke + Kepler IR 표현.

NR 시간 미분은 기존 ``indexcalc.adm.TimeDeriv``를 재활용 (free_indices가
inner와 동일 — NR mechanics에도 그대로 적합).

검증 toy:
1. \\dot{Phi}^i 의 free_indices = [^i]
2. \\dot{Phi}^i \\dot{Phi}_i 의 free_indices = []
3. ScalarFunction("V", scalar)는 free_indices == []
4. ScalarFunction이 비스칼라 인수에 대해 ValueError
5. Kepler L = (1/2) M δ_{ij} \\dot{Phi}^i \\dot{Phi}^j + κ * inv_sqrt(δ_{kl} Phi^k Phi^l)
   가 IR 노드로 표현되며 free_indices == [] (스칼라 Lagrangian).
"""

from __future__ import annotations

import pytest

from indexcalc.core.index import IndexSpace
from indexcalc.core.tensor import Tensor, TensorProduct, ScalarMul, TensorSum
from indexcalc.adm import TimeDeriv
from indexcalc.core.scalar_function import ScalarFunction


@pytest.fixture
def so3():
    return IndexSpace("so3_vector", dim=3, indices="ijklmn", metric="delta")


def test_time_deriv_preserves_single_upper_index(so3):
    Phi = Tensor("Phi", [so3.upper("i")])
    dPhi = TimeDeriv(Phi)
    assert [idx.name for idx in dPhi.free_indices] == ["i"]
    assert [idx.position for idx in dPhi.free_indices] == ["upper"]


def test_time_deriv_scalar_after_einstein_pair(so3):
    Phi_up = Tensor("Phi", [so3.upper("i")])
    Phi_dn = Tensor("Phi", [so3.lower("i")])
    expr = TimeDeriv(Phi_up) * TimeDeriv(Phi_dn)
    # i 가 위·아래로 한 번씩 등장 → contraction
    assert expr.free_indices == []


def test_temporal_deriv_constructible(so3):
    Phi = Tensor("Phi", [so3.upper("i")])
    assert isinstance(TimeDeriv(Phi), TimeDeriv)


def test_scalar_function_scalar_arg_ok(so3):
    Phi_up = Tensor("Phi", [so3.upper("i")])
    Phi_dn = Tensor("Phi", [so3.lower("i")])
    inner = Phi_up * Phi_dn  # i contracted → scalar
    f = ScalarFunction("inv_sqrt", inner)
    assert f.free_indices == []
    assert f.name == "inv_sqrt"


def test_scalar_function_rejects_nonscalar_arg(so3):
    Phi = Tensor("Phi", [so3.upper("i")])
    with pytest.raises(ValueError, match="must be a scalar"):
        ScalarFunction("V", Phi)


def test_kepler_lagrangian_expressible(so3):
    """Kepler L 전체가 IR 트리로 표현되며 스칼라(자유 인덱스 없음)."""
    # 운동항: (1/2) M δ_{ij} \dot Phi^i \dot Phi^j
    Phi_i = Tensor("Phi", [so3.upper("i")])
    Phi_j = Tensor("Phi", [so3.upper("j")])
    delta_ij = Tensor("delta", [so3.lower("i"), so3.lower("j")],
                      symmetric_pairs=[(0, 1)])
    kinetic = ScalarMul(0.5, delta_ij * TimeDeriv(Phi_i)
                                       * TimeDeriv(Phi_j))
    # kinetic은 i, j가 양쪽에서 contract → 스칼라
    assert kinetic.free_indices == []

    # 퍼텐셜: κ / sqrt(δ_{kl} Phi^k Phi^l)
    Phi_k = Tensor("Phi", [so3.upper("k")])
    Phi_l = Tensor("Phi", [so3.upper("l")])
    delta_kl = Tensor("delta", [so3.lower("k"), so3.lower("l")],
                      symmetric_pairs=[(0, 1)])
    r_squared = delta_kl * Phi_k * Phi_l  # scalar
    assert r_squared.free_indices == []
    potential_func = ScalarFunction("inv_sqrt", r_squared)
    potential = ScalarMul(1.0, potential_func)  # κ는 일단 1로

    # 전체 Lagrangian
    L = TensorSum(kinetic, potential)
    assert L.free_indices == []
