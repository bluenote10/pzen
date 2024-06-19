from contextlib import contextmanager
from typing import Generator, Iterator, Sequence, TypeVar

import torch
from torch.testing._comparison import assert_close as assert_close_orig
from torch.utils.data import DataLoader

T = TypeVar("T")

# -----------------------------------------------------------------------------
# For general type safety
# -----------------------------------------------------------------------------


class TypeSafeModule(torch.nn.Module):
    # def __getattr__(self, name: str) -> torch.Tensor | torch.nn.Module:
    def __getattr__(self, name: str) -> object:
        # Type-safety work around for https://github.com/pytorch/pytorch/pull/104321
        return super().__getattr__(name)


# -----------------------------------------------------------------------------
# Type-safe data loading
# -----------------------------------------------------------------------------


def iter_dataloader(data_loader: DataLoader[T]) -> Iterator[T]:
    """
    Work-around for: https://github.com/pytorch/pytorch/issues/119123
    """
    for batch in data_loader:
        yield batch


# -----------------------------------------------------------------------------
# Misc utils
# -----------------------------------------------------------------------------


@contextmanager
def local_eval(module: torch.nn.Module) -> Generator[None, None, None]:
    """
    Allows to switch module to `eval` mode with automatic restoring of original state.
    """
    training_state_before = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training_state_before)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # https://stackoverflow.com/a/77031136/1804173
    return torch.log(x) - torch.log(1 - x)


# -----------------------------------------------------------------------------
# Testing utils
# -----------------------------------------------------------------------------


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor | Sequence[object] | float | int,
    rtol: float | None = None,
    atol: float | None = None,
    check_dtype: bool = True,
) -> None:
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected)
    print(f"\nActual:   {actual}")
    print(f"Expected: {expected}")
    assert_close_orig(actual, expected, rtol=rtol, atol=atol, check_dtype=check_dtype)
