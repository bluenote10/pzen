import datetime
import tempfile
from pathlib import Path
from typing import Any, Callable, TypeVar

from joblib import Memory
from typing_extensions import ParamSpec
from yachalk import chalk

_CACHE_DIR = Path(tempfile.gettempdir()) / "joblib_cache"
_MEMORY = Memory(
    _CACHE_DIR,
)


P = ParamSpec("P")
R = TypeVar("R")


def with_memory(f: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to apply joblib's Memory in a type-safe manner:
    https://joblib.readthedocs.io/en/latest/memory.html
    """

    def cache_validation_cb(metadata: dict[str, Any]) -> bool:
        time = datetime.datetime.fromtimestamp(metadata["time"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        duration = metadata["duration"]
        print(
            chalk.bold(
                f"Using cached result for `{f.__name__}` (computed at {time}, in {duration:.3f} s)"
            )
        )
        return True

    return _MEMORY.cache(
        verbose=0,
        cache_validation_callback=cache_validation_cb,
    )(
        f  # pyright: ignore
    )
