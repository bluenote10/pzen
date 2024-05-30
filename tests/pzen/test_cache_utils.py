import numpy as np

from pzen.cache_utils import _CACHE_DIR, with_memory

_CALLS: list[tuple[float, str, np.ndarray]] = []


@with_memory
def f(a: float, b: str, c: np.ndarray) -> float:
    _CALLS.append((a, b, c))
    return (a + len(b)) * np.sum(c)


def test_cache_utils():
    f.clear()  # type: ignore[attr-defined] # MemorizedFunc magic to clear the cache.

    assert f(1.0, "foo", np.array([1, 2, 3])) == 24.0
    assert len(_CALLS) == 1
    assert f(1.0, "foo", np.array([1, 2, 3])) == 24.0
    assert len(_CALLS) == 1

    assert f(2.0, "foo", np.array([1, 2, 3])) == 30.0
    assert len(_CALLS) == 2
    assert f(2.0, "foo", np.array([1, 2, 3])) == 30.0
    assert len(_CALLS) == 2

    assert f(2.0, "fooo", np.array([1, 2, 3])) == 36.0
    assert len(_CALLS) == 3
    assert f(2.0, "fooo", np.array([1, 2, 3])) == 36.0
    assert len(_CALLS) == 3

    assert f(1.0, "foo", np.array([1, 2, 4])) == 28.0
    assert len(_CALLS) == 4
    assert f(1.0, "foo", np.array([1, 2, 4])) == 28.0
    assert len(_CALLS) == 4

    assert _CACHE_DIR.exists()
