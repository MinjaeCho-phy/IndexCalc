"""LIONS ML training pipeline (D10+).

Sits on top of ``indexcalc.lions`` (D1~D9b) and turns ``EncodedGraph`` /
``LabeledSample`` collections into trainable PyG datasets + models.

Optional dependency policy
--------------------------
This subpackage requires ``torch`` and ``torch_geometric``. Backend
acceptance tests (528+) must stay green without them, so this
``__init__`` does NOT eagerly import torch — concrete modules under
``indexcalc.lions.ml.*`` import torch at module scope and will fail with
a clear ``ImportError`` if torch is missing. Importing
``indexcalc.lions.ml`` itself is always safe.

Layout
------
- ``features``   — node/edge categorical → int ID encoders (vocab tables)
- ``pyg_bridge`` — ``EncodedGraph`` → ``torch_geometric.data.Data``
- ``datasets``   — ``LabeledSample`` list → PyG ``InMemoryDataset``
- ``models``     — ``RGCNClassifier`` (v1), ``GraphTransformerClassifier`` (v2)
- ``train``      — train loop (AdamW + BCEWithLogits + per-group AUC)
- ``eval``       — per-group metrics, calibration

Spec: ``LIONS/notes/ml_training_v1.md`` (G1~G6 frozen, G7~G12 open).
"""

from __future__ import annotations


def _require_torch():
    """Helper for modules that want a friendly ImportError surface."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "indexcalc.lions.ml requires torch and torch_geometric. "
            "Install with: pip install torch torch_geometric"
        ) from e


__all__ = ["_require_torch"]
