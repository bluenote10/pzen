from typing import Iterator, TypeVar

import torch
from torch.testing._comparison import assert_close as assert_close_orig
from torch.utils.data import DataLoader

T = TypeVar("T")


def iter_dataloader(data_loader: DataLoader[T]) -> Iterator[T]:
    """
    Work-around for: https://github.com/pytorch/pytorch/issues/119123
    """
    for batch in data_loader:
        yield batch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # https://stackoverflow.com/a/77031136/1804173
    return torch.log(x) - torch.log(1 - x)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor | object,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected)
    print(f"\nActual:   {actual}")
    print(f"Expected: {expected}")
    assert_close_orig(actual, expected, rtol=rtol, atol=atol)
