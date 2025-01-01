from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import assert_type

from pzen import mpl_utils


def test_subplots_wrapper__plot(tmp_path: Path):
    plot_path = tmp_path / "subdir" / "plot.png"
    with mpl_utils.plot(save_as=plot_path) as (fig, ax):
        assert_type(fig, Figure)
        assert isinstance(fig, Figure)
        assert_type(ax, Axes)
        assert isinstance(ax, Axes)

        ax.plot([1, 2, 3])

    assert plot_path.is_file()


def test_subplots_wrapper__plot_rows(tmp_path: Path):
    plot_path = tmp_path / "subdir" / "plot.png"
    with mpl_utils.plot_rows(3, save_as=plot_path) as (fig, axes):
        assert_type(fig, Figure)
        assert isinstance(fig, Figure)
        assert_type(axes, list[Axes])

        axes[0].plot([1, 2, 3])
        axes[1].plot([1, 2, 3])
        axes[2].plot([1, 2, 3])

    assert plot_path.is_file()


def test_subplots_wrapper__plot_cols(tmp_path: Path):
    plot_path = tmp_path / "subdir" / "plot.png"
    with mpl_utils.plot_cols(3, save_as=plot_path) as (fig, axes):
        assert_type(fig, Figure)
        assert isinstance(fig, Figure)
        assert_type(axes, list[Axes])

        axes[0].plot([1, 2, 3])
        axes[1].plot([1, 2, 3])
        axes[2].plot([1, 2, 3])

    assert plot_path.is_file()


def test_subplots_wrapper__plot_grid(tmp_path: Path):
    plot_path = tmp_path / "subdir" / "plot.png"
    with mpl_utils.plot_grid(2, 3, save_as=plot_path) as (fig, axes):
        assert_type(fig, Figure)
        assert isinstance(fig, Figure)
        assert_type(axes, list[list[Axes]])

        for i in range(2):
            for j in range(3):
                axes[i][j].plot([1, 2, 3])

    assert plot_path.is_file()
