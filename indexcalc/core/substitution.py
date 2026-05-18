"""
Substitution walk: TensorExpr 트리 전체에 generator 작용을 Leibniz로 적용.

이 모듈은 ``core/variation.py``의 ``expand_variation``과 같은 패턴이지만,
Leibniz 분배의 leaf 작용이 ``VariationRegistry``의 δ-prefixing 대신
``Generator.apply_to``로 결정된다.

주요 함수:
- ``apply_generator(expr, generator)``: 트리 walk + Leibniz + leaf에서 generator 적용.

ZeroTensor가 끼어드는 자리는 자동으로 정리된다 (``_simplify_zeros`` 재사용).
"""

from __future__ import annotations

from indexcalc.core.tensor import (
    Tensor, TensorExpr, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.deriv import PartialDeriv, CovariantDeriv
from indexcalc.core.variation import ZeroTensor, _simplify_zeros
from indexcalc.core.generator import Generator


def apply_generator(expr: TensorExpr, generator: Generator) -> TensorExpr:
    """Generator를 expr 트리 전체에 Leibniz로 적용한다.

    규칙:
        Tensor leaf:   generator.apply_to(leaf)
        TensorSum:     apply_generator(left) + apply_generator(right)
        TensorProduct: (δA)*B + A*(δB)
        ScalarMul:     c * apply_generator(expr)
        PartialDeriv:  ∂_μ(apply_generator(expr))   [global symmetry]
        CovariantDeriv: ∇_μ(apply_generator(expr))  [기하 connection은 background]
        ZeroTensor:    ZeroTensor (변하지 않음)

    주의: gauge generator의 경우 $\\partial_\\mu$와의 commute 성립이 보장된다
    (global, $x$-독립 변환). Local gauge transformation은 별도 generator로
    표현해야 하며 본 walk 규칙으로는 부족하다 — M2 이후 다룸.
    """
    if isinstance(expr, ZeroTensor):
        return expr

    if isinstance(expr, Tensor):
        return _simplify_zeros(generator.apply_to(expr))

    if isinstance(expr, TensorSum):
        dL = apply_generator(expr.left, generator)
        dR = apply_generator(expr.right, generator)
        # ZeroTensor 조기 처리 — TensorSum의 free index count check 회피.
        # 한 항이 singlet(0)인 경우 다른 항이 가진 parameter free index만 살아남음.
        if isinstance(dL, ZeroTensor) and isinstance(dR, ZeroTensor):
            return ZeroTensor(expr.free_indices)
        if isinstance(dL, ZeroTensor):
            return dR
        if isinstance(dR, ZeroTensor):
            return dL
        return _simplify_zeros(TensorSum(dL, dR))

    if isinstance(expr, TensorProduct):
        dA = apply_generator(expr.left, generator)
        dB = apply_generator(expr.right, generator)
        # Leibniz: 통계와 무관하게 (δA)·B + A·(δB).
        # Fermion grading 부호는 leaf REORDERING에서 발생하지 변분 분배에서 안 생김.
        # ZeroTensor 분기 — TensorSum이 free index 개수 불일치로 실패하기 전에 조기 처리.
        if isinstance(dA, ZeroTensor) and isinstance(dB, ZeroTensor):
            return ZeroTensor(expr.free_indices)
        if isinstance(dA, ZeroTensor):
            return _simplify_zeros(TensorProduct(expr.left, dB))
        if isinstance(dB, ZeroTensor):
            return _simplify_zeros(TensorProduct(dA, expr.right))
        return _simplify_zeros(
            TensorSum(
                TensorProduct(dA, expr.right),
                TensorProduct(expr.left, dB),
            )
        )

    if isinstance(expr, ScalarMul):
        return _simplify_zeros(
            ScalarMul(expr.scalar, apply_generator(expr.expr, generator))
        )

    if isinstance(expr, PartialDeriv):
        inner = apply_generator(expr.expr, generator)
        if isinstance(inner, ZeroTensor):
            inner_term: TensorExpr = ZeroTensor(
                [expr.deriv_index] + inner.free_indices
            )
        else:
            inner_term = PartialDeriv(inner, expr.deriv_index)

        # Vector-rep rotation of the deriv_index itself (Lorentz only).
        # Other generators leave deriv_index alone.
        deriv_term = generator.apply_to_deriv_index(expr)
        if deriv_term is None:
            return _simplify_zeros(inner_term)
        if isinstance(deriv_term, ZeroTensor):
            return _simplify_zeros(inner_term)
        if isinstance(inner_term, ZeroTensor):
            return _simplify_zeros(deriv_term)
        return _simplify_zeros(TensorSum(inner_term, deriv_term))

    if isinstance(expr, CovariantDeriv):
        inner = apply_generator(expr.expr, generator)
        if isinstance(inner, ZeroTensor):
            return ZeroTensor([expr.deriv_index] + inner.free_indices)
        return type(expr)(inner, expr.deriv_index, expr.connections)

    # TimeDeriv: 시간 미분은 글로벌 (non-time) symmetry와 commute.
    #   δ(\\dot T) = \\dot{δT}. inner이 invariant이면 결과도 invariant.
    # adm.TimeDeriv는 NR mechanics 용으로도 그대로 재활용.
    from indexcalc.adm import TimeDeriv
    if isinstance(expr, TimeDeriv):
        inner = apply_generator(expr.expr, generator)
        if isinstance(inner, ZeroTensor):
            return ZeroTensor(expr.free_indices)
        return TimeDeriv(inner)

    # ScalarFunction: δ(f(I)) = f'(I) δI. δI이 ZeroTensor이면 결과 ZeroTensor.
    # 아니면 f'(I) · δI 형태로 symbolic 유지 (invariant이 아님을 표현).
    from indexcalc.core.scalar_function import ScalarFunction
    if isinstance(expr, ScalarFunction):
        delta_arg = apply_generator(expr.arg, generator)
        if isinstance(delta_arg, ZeroTensor):
            return ZeroTensor([])
        return _simplify_zeros(TensorProduct(
            ScalarFunction(f"{expr.name}_prime", expr.arg),
            delta_arg,
        ))

    raise NotImplementedError(
        f"apply_generator not implemented for {type(expr).__name__}"
    )
