"""LIONS adapter — forward generation pipeline + ML dataset builders.

This subpackage sits on top of IndexCalc core (which only provides the
verification oracle: ``apply_generator`` + ``simplify`` → ``ZeroTensor``).
LIONS adds the things the core deliberately does NOT have:

- A **field registry** with mass dimension metadata.
- **Invariant tensor builders** that turn metadata into concrete ``Tensor``
  objects matching the conventions used by core simplifiers.
- An **enumerator** for invariant monomials up to a mass-dimension cap.
- A **labeler** that wraps ``apply_generator`` + ``simplify`` to assign
  multi-group invariance labels to candidate Lagrangians.

Design note: ``notes/data_pipeline_design.md`` in the LIONS repo.
"""

from indexcalc.lions.fields import FieldSpec, FieldRegistry, SlotSpec
from indexcalc.lions.builders import (
    make_eta,
    make_kronecker,
    make_epsilon_su2,
    make_epsilon_su2_upper,
    make_partial,
)
from indexcalc.lions.enumerate import (
    EnumeratorCaps,
    EnumeratedSample,
    enumerate_scalar_invariants,
)
from indexcalc.lions.dataset import (
    LabeledSample,
    label_expression,
    label_samples,
)

__all__ = [
    "FieldSpec",
    "FieldRegistry",
    "SlotSpec",
    "make_eta",
    "make_kronecker",
    "make_epsilon_su2",
    "make_epsilon_su2_upper",
    "make_partial",
    "EnumeratorCaps",
    "EnumeratedSample",
    "enumerate_scalar_invariants",
    "LabeledSample",
    "label_expression",
    "label_samples",
]
