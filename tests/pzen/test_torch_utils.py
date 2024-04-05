from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import assert_type

from pzen import torch_utils


def test_iter_dataloader():

    class Batch(NamedTuple):
        x: torch.Tensor

    class MyDataset(Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, idx: int) -> Batch:
            return Batch(x=torch.tensor(idx))

    data_loader: DataLoader[Batch] = DataLoader(MyDataset())

    for batch in torch_utils.iter_dataloader(data_loader):
        assert_type(batch, Batch)
        assert isinstance(batch, Batch)
