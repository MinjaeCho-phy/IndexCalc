"""PyG dataset wrapper around LIONS JSON splits.

Lazy: torch is imported only when this module is.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

from indexcalc.lions.ml import _require_torch
from indexcalc.lions.ml.features import GROUP_ORDER
from indexcalc.lions.ml.pyg_bridge import encoded_to_pyg_data

_require_torch()

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from indexcalc.lions import (
    load_dataset, graph_encode, encode_sample,
)


class LionsPyGDataset(Dataset):
    """Load a LIONS JSON dataset and serve PyG ``Data`` objects.

    Encoding ``EncodedGraph → Data`` happens eagerly at construction time
    (v1-toy fits in memory; ~1k samples). For v1 (10k) and v1.1 (100k+)
    we'll move to on-disk LMDB.
    """

    def __init__(
        self,
        json_path: Path,
        label_order: Sequence[str] = GROUP_ORDER,
    ):
        self.json_path = Path(json_path)
        samples = load_dataset(self.json_path)
        self._data: list = []
        for s in samples:
            try:
                g = encode_sample(s)
            except NotImplementedError:
                continue
            if g is None:
                continue
            d = encoded_to_pyg_data(g, label_order)
            if d.num_nodes == 0:
                continue
            self._data.append(d)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int):
        return self._data[i]


def collate_pyg(data_list):
    """Standard PyG batching helper."""
    return Batch.from_data_list(list(data_list))
