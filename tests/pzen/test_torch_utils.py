from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import assert_type

from pzen import torch_utils
from pzen.torch_utils import assert_close


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


def test_inverse_sigmoid__basic():
    assert_close(torch_utils.inverse_sigmoid(torch.tensor(0.9)), +2.1972246170043945)
    assert_close(torch_utils.inverse_sigmoid(torch.tensor(0.5)), 0.0)
    assert_close(torch_utils.inverse_sigmoid(torch.tensor(0.1)), -2.1972246170043945)


def test_inverse_sigmoid__shape_preserving():
    x = torch.tensor(
        [
            [-1.0, 0.0, -7.0, 2.0],
            [3.0, 2.0, 4.0, -5.0],
        ]
    )
    sigmoid_x = torch.sigmoid(x)
    print(sigmoid_x)
    assert_close(torch_utils.inverse_sigmoid(sigmoid_x), x)
