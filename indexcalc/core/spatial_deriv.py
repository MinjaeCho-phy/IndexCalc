"""
Spatial covariant derivative D_i (ADM 3-slice 용).

일반 4D ∇_μ와 구조는 동일하지만:
  - 연산 기호: \\nabla 대신 D
  - 전개 시 쓰는 connection은 spatial (γ^i_{jk} = Christoffel of 3-metric Ω)

구현 전략: ``SpatialCovariantDeriv``를 ``CovariantDeriv``의 subclass로 둔다.
``expand_covariant``와 ``_distribute_nabla_once``는 이미 ``type(cov)``로 재귀해서
subclass 정체성이 유지되므로, 이 파일은 새 클래스 + 표시/파서 분기만 담당.

Variation(δD_i T)의 Palatini 보정도 기존 ``VariationRegistry``의
``declare_varying_connection`` 메커니즘을 그대로 쓴다 (예: γ를 varying으로 선언).
"""

from __future__ import annotations
from indexcalc.core.index import Index
from indexcalc.core.tensor import TensorExpr
from indexcalc.core.deriv import (
    CovariantDeriv, Connection, expand_covariant,
)


class SpatialCovariantDeriv(CovariantDeriv):
    """공간 covariant derivative D_i.

    ``CovariantDeriv``의 subclass. 내부 데이터 구조와 전개 규칙은 동일하고,
    단지 타입 식별을 통해 display/파싱에서 `D` 기호로 취급된다.

    Examples
    --------
    >>> sp = IndexSpace("spatial", dim=3, indices="ijklmn", metric="Ω")
    >>> gamma = LeviCivitaConnection(Ω, Ω_inv, sp)
    >>> V = Tensor("V", [sp.upper("i")])
    >>> D_j_V = SpatialCovariantDeriv(V, sp.lower("j"), gamma)
    """

    def __repr__(self) -> str:
        return f"D_{self.deriv_index.name}({self.expr})"


def spatial_covariant(
    expr: TensorExpr,
    index: Index,
    connections: dict[str, Connection] | Connection,
) -> SpatialCovariantDeriv:
    """D_index(expr) 생성. index가 upper면 자동으로 lower로 변환."""
    if index.position == "upper":
        index = index.flip()
    return SpatialCovariantDeriv(expr, index, connections)


def expand_spatial_covariant(expr: TensorExpr) -> TensorExpr:
    """D_i T를 ∂ + γ 항으로 전개. ``expand_covariant``와 동일 로직 재사용.

    Subclass가 유지되도록 ``CovariantDeriv``의 expand가 ``type(cov)``로 재귀하니
    이 wrapper는 편의성을 위한 별칭에 가깝다.
    """
    return expand_covariant(expr)
