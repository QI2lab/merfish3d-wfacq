import argparse
from functools import partial
from pathlib import Path
from typing import Literal

from pymmcore_gui import MicroManagerGUI, create_mmgui
from pymmcore_plus import CMMCorePlus
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from merfish3d_wfacq.gui import (
    _ensure_merfish_widget_action_registered,
    enhance_main_window,
)


def launch_merfish_app(
    *,
    mm_config: Path | str | None | Literal[False] = None,
    mmcore: CMMCorePlus | None = None,
    exec_app: bool = True,
) -> MicroManagerGUI:
    """Launch ``pymmcore-gui`` and attach the MERFISH extension dock.

    Parameters
    ----------
    mm_config : Path or str or None or Literal[False], optional
        Optional Micro-Manager configuration to load.
    mmcore : CMMCorePlus or None, optional
        Existing ``CMMCorePlus`` instance to reuse.
    exec_app : bool, optional
        Whether to enter the Qt event loop before returning.

    Returns
    -------
    MicroManagerGUI
        Created ``pymmcore-gui`` main window.
    """

    _ensure_merfish_widget_action_registered()
    window = create_mmgui(
        mm_config=mm_config,
        mmcore=mmcore,
        exec_app=False,
    )
    _schedule_merfish_attachment(window)
    if exec_app and (app := QApplication.instance()) is not None:
        app.exec()
    return window


def _schedule_merfish_attachment_step_two(window: MicroManagerGUI) -> None:
    """Attach the MERFISH dock on the second deferred event-loop turn.

    Parameters
    ----------
    window : MicroManagerGUI
        ``pymmcore-gui`` main window receiving the MERFISH dock.
    """

    QTimer.singleShot(0, partial(enhance_main_window, window))


def _schedule_merfish_attachment(window: MicroManagerGUI) -> None:
    """Attach the MERFISH dock after pyMM restores its saved layout.

    Parameters
    ----------
    window : MicroManagerGUI
        ``pymmcore-gui`` main window receiving the MERFISH dock.
    """

    # create_mmgui restores the saved dock state on the next event-loop turn.
    # Attach MERFISH one turn later or it will be immediately hidden again.
    QTimer.singleShot(0, partial(_schedule_merfish_attachment_step_two, window))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the MERFISH application entrypoint.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.
    """

    parser = argparse.ArgumentParser(
        prog="merfish3d-wfacq",
        description="Launch pymmcore-gui with the MERFISH fluidics dock attached.",
    )
    parser.add_argument(
        "--mm-config",
        dest="mm_config",
        default=None,
        help="Optional Micro-Manager config file to load at startup.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the MERFISH-enhanced ``pymmcore-gui`` application.

    Parameters
    ----------
    argv : list[str] or None, optional
        Optional command-line arguments to parse.

    Returns
    -------
    int
        Process exit code.
    """

    args = build_arg_parser().parse_args(argv)
    launch_merfish_app(mm_config=args.mm_config)
    return 0
