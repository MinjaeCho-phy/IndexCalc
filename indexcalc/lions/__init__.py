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

from indexcalc.lions.fields import (
    FieldSpec, FieldRegistry, SlotSpec, InvariantTensorSpec,
)
from indexcalc.lions.builders import (
    make_eta,
    make_kronecker,
    make_epsilon_su2,
    make_epsilon_su2_upper,
    make_partial,
    make_gamma,
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
from indexcalc.lions.serializer import (
    save_dataset,
    load_dataset,
    sample_to_dict,
    sample_from_dict,
    expr_to_dict,
    expr_from_dict,
)
from indexcalc.lions.augment import (
    RepMutation,
    mutate_field_reps,
    apply_rep_mutation,
    enumerate_wrong_rep_negatives,
    permute_dummy_indices,
    swap_top_product,
    scale_by,
)

__all__ = [
    "FieldSpec",
    "FieldRegistry",
    "SlotSpec",
    "InvariantTensorSpec",
    "make_eta",
    "make_kronecker",
    "make_epsilon_su2",
    "make_epsilon_su2_upper",
    "make_partial",
    "make_gamma",
    "EnumeratorCaps",
    "EnumeratedSample",
    "enumerate_scalar_invariants",
    "LabeledSample",
    "label_expression",
    "label_samples",
    "save_dataset",
    "load_dataset",
    "sample_to_dict",
    "sample_from_dict",
    "expr_to_dict",
    "expr_from_dict",
    "RepMutation",
    "mutate_field_reps",
    "apply_rep_mutation",
    "enumerate_wrong_rep_negatives",
    "permute_dummy_indices",
    "swap_top_product",
    "scale_by",
]
