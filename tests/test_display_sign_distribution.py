"""Display: ScalarMul × TensorSum 부호 분배 회귀 테스트.

2026-05-11 발견된 버그:
    ScalarMul(c<0, TensorSum(A, B)) → "-A + B"  (X)
    ScalarMul(c<0, TensorSum(A, B)) → "-A - B"  (O)

Riemann tensor의 δR을 expand_variation으로 도출 시
δ(-A·B) = -δA·B - A·δB 가 잘못 -δA·B + A·δB 로 표시됐다.
"""

from indexcalc import (
    IndexSpace, Tensor, TensorProduct, TensorSum, ScalarMul,
)
from indexcalc.core.index import Index


def make_indices():
    st = IndexSpace("st", dim=4, indices="μνρσλ", metric="g")
    return st


def test_scalarmul_minus_one_distributes_over_sum():
    """ScalarMul(-1, A+B) → '-A - B' (each term carries the sign)."""
    st = make_indices()
    A = Tensor("A", [Index("μ", st, "upper")])
    B = Tensor("B", [Index("μ", st, "upper")])
    expr = ScalarMul(-1, TensorSum(A, B))
    out = expr.to_latex()
    assert "-A^{\\mu}" in out
    assert "- B^{\\mu}" in out
    assert "+ B" not in out, f"sign not distributed: {out}"


def test_scalarmul_minus_n_distributes_over_sum():
    """ScalarMul(c<-1, A+B) → '-c A - c B'."""
    st = make_indices()
    A = Tensor("A", [Index("μ", st, "upper")])
    B = Tensor("B", [Index("μ", st, "upper")])
    expr = ScalarMul(-3, TensorSum(A, B))
    out = expr.to_latex()
    # 정확한 출력 형식보다 '+ ' 가 안 나오는지가 핵심
    assert " + " not in out, f"sign not distributed: {out}"


def test_subtraction_of_grouped_sum():
    """X + (-(A + B)) → X - A - B (Riemann ΓΓ 항의 typical pattern)."""
    st = make_indices()
    X = Tensor("X", [Index("μ", st, "upper")])
    A = Tensor("A", [Index("μ", st, "upper")])
    B = Tensor("B", [Index("μ", st, "upper")])
    expr = TensorSum(X, ScalarMul(-1, TensorSum(A, B)))
    out = expr.to_latex()
    assert " + " not in out.replace("X^{\\mu} -", "")  # X 다음에는 - 나와야 함
    # 명시적: A, B 모두 - 부호로 분배
    assert out.count(" - ") == 2, f"expected 2 minuses, got: {out}"


def test_nested_negation_of_leibniz_pattern():
    """Riemann δ(-A·B) → δA·B 와 A·δB 모두 - 부호로 나옴."""
    from indexcalc import Variation, VariationRegistry, expand_variation
    from indexcalc.parse.display import to_latex
    st = make_indices()
    A = Tensor("A", [Index("μ", st, "upper")])
    B = Tensor("B", [Index("ν", st, "lower")])
    vreg = VariationRegistry()
    vreg.declare_varying("A")
    vreg.declare_varying("B")
    expr = ScalarMul(-1, TensorProduct(A, B))
    result = expand_variation(Variation(expr), vreg)
    out = to_latex(result)
    # δA, δB 둘 다 - 부호로 나와야 함 (어느 것도 +로 나오면 안 됨)
    assert "+ δA" not in out, f"δA should be negative: {out}"
    assert "+ A" not in out, f"A·δB should be negative: {out}"
    # leading "-" 또는 " - " 형태로 둘 다 음수 표시
    assert out.startswith("-") and " - " in out, f"both minus: {out}"
