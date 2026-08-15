import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fields


#: Default is the KuaiRand-1K feather in the repo root. Set ``LFM_DATASET`` to
#: point at another dataset (e.g. ``dataset_27k.feather``) — the split dates in
#: :func:`Split` are identical for 1K and 27K (both span 20220408..20220508).
DATASET_PATH = Path(
    os.environ.get("LFM_DATASET",
                   str(Path(__file__).resolve().with_name("dataset.feather")))
)


class GpuBatches:
    """GPU-resident zero-copy batch iterator (opt-in fast path).

    The whole split is uploaded ONCE as a single ``int32 [N, F]`` tensor, so
    producing a batch is a pure device-side row gather: no pandas ``.iloc``
    per sample, no ``default_collate``, no per-batch host→device copy, no
    worker processes.

    Motivation: ``Dataset.__getitem__`` below does ``self.iloc[i].to_dict()``,
    i.e. **one pandas row lookup + dict build per sample**. At batch_size=10000
    that is 10k pandas scalar lookups per batch, which made training
    host-bound and pinned GPU utilization at 13–33%. The full KuaiRand-1K
    table is only 11.7M × 39 int32 ≈ 1.83 GB, so it fits on the device with
    room to spare.

    Batch semantics are identical to the DataLoader path: a dict mapping every
    column name to an ``int32`` device tensor of shape ``[B]``.

    Args:
        df:         a split DataFrame (columns == ``fields.all``)
        batch_size: rows per batch
        device:     CUDA device for both the table and the batches
        shuffle:    reshuffle row order every epoch
        seed:       optional seed for the shuffling generator
    """

    def __init__(self, df, batch_size, device, shuffle=False, seed=None):
        self.cols = list(df.columns)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device)
        # max value in the table is a date like 20220508, well inside int32
        self.data = torch.from_numpy(
            np.ascontiguousarray(df.to_numpy(dtype=np.int32))
        ).to(self.device)
        self.n = int(self.data.shape[0])
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, device=self.device,
                                   generator=self._gen)
        else:
            order = torch.arange(self.n, device=self.device)
        for chunk in order.split(self.batch_size):
            rows = self.data.index_select(0, chunk)
            yield {c: rows[:, j] for j, c in enumerate(self.cols)}


class Dataset(pd.DataFrame):
    def __getitem__(self, i):
        return self.iloc[i].to_dict()


def Split(scenario):
    df = pd.read_feather(DATASET_PATH)
    if scenario != "all":
        df = df.query(f"tab == {scenario}")
    return [
        df.query("            date < 20220503"),
        df.query("20220503 <= date < 20220506"),
        df.query("20220506 <= date           "),
    ]


if __name__ == "__main__":
    DATA_DIR = "/apdcephfs/private_xavieryu/database/KuaiRand-1K/data"
    df1 = pd.read_csv(f"{DATA_DIR}/log_standard_4_08_to_4_21_1k.csv")
    df2 = pd.read_csv(f"{DATA_DIR}/log_standard_4_22_to_5_08_1k.csv")
    df = pd.concat([df1, df2]).sort_values("time_ms")
    df = pd.merge(df, pd.read_csv(f"{DATA_DIR}/user_features_1k.csv"))
    df = pd.merge(df, pd.read_csv(f"{DATA_DIR}/video_features_basic_1k.csv"))
    for field in fields.user | fields.video:
        df[field] = df[field].factorize(use_na_sentinel=False)[0]
    df[fields.all].to_feather("dataset.feather")
