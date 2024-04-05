from typing import Iterator, TypeVar

from torch.utils.data import DataLoader

T = TypeVar("T")


def iter_dataloader(data_loader: DataLoader[T]) -> Iterator[T]:
    """
    Work-around for: https://github.com/pytorch/pytorch/issues/119123
    """
    for batch in data_loader:
        yield batch
