from pathlib import Path

import pandas as pd
import pytest
from pymmcore_gui import MicroManagerGUI
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QApplication, QMessageBox
from pymmcore_widgets.mda._core_mda import MDAWidget
from tests.merfish_builders import FITC_RHODAMINE_CY5_SEQUENCE, TWO_ROUND_EXP_ORDER
from useq import MDASequence

from merfish3d_wfacq.gui import (
    MERFISH_WIDGET_KEY,
    MerfishFluidicsWidget,
    enhance_main_window,
)
from merfish3d_wfacq.ui_state import (
    bit_mapping_preview,
    channel_specs_from_sequence_wavelength_rows,
    fluidics_round_options,
    wavelength_rows_for_sequence,
)
from merfish3d_wfacq.utils.data_io import read_fluidics_program


def _sequence() -> MDASequence:
    return FITC_RHODAMINE_CY5_SEQUENCE.model_copy(deep=True)


def _exp_order() -> pd.DataFrame:
    return TWO_ROUND_EXP_ORDER.copy(deep=True)


def _synthetic_run_rounds_program(_workspace_tmp_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [2, 2, 4, 4, 7, 7],
            "source": ["B02", "RUN", "B04", "RUN", "B07", "RUN"],
            "time": [0.1, 0.0, 0.1, 0.0, 0.1, 0.0],
            "pump": [10.0, 0.0, 11.0, 0.0, 12.0, 0.0],
        }
    )


def _initial_fluidics_program(_workspace_tmp_path: Path) -> pd.DataFrame:
    return read_fluidics_program(Path(__file__).parent / "initial.csv")
def _set_stage_explorer_overlap_percent(widget: object, overlap_percent: float) -> None:
    scan_menu = widget._stage_explorer._toolbar.scan_menu
    if hasattr(scan_menu, "_overlap_spin"):
        scan_menu._overlap_spin.setValue(float(overlap_percent))
    else:
        scan_menu.set_overlap_percent(float(overlap_percent))


def _click_next_message_box_button(button: QMessageBox.StandardButton) -> None:
    def _click() -> None:
        for candidate in QApplication.topLevelWidgets():
            if isinstance(candidate, QMessageBox):
                target = candidate.button(button)
                assert target is not None
                target.click()
                return

    QTimer.singleShot(0, _click)


def test_widget_request_refresh_handles_operator_confirmation(
    merfish_widget: object,
) -> None:
    widget = merfish_widget

    _click_next_message_box_button(QMessageBox.StandardButton.Yes)
    assert widget.request_refresh({"round": 2}) is True

    _click_next_message_box_button(QMessageBox.StandardButton.No)
    assert widget.request_refresh({"round": 3}) is False


def test_enhance_main_window_uses_pymmcore_gui_widget_registry(
    qtbot: object, offline_icons: None
) -> None:
    window = MicroManagerGUI()
    qtbot.addWidget(window)

    try:
        widget = enhance_main_window(window)

        assert widget is window.get_widget(MERFISH_WIDGET_KEY)
        assert window.get_dock_widget(MERFISH_WIDGET_KEY).widget() is widget
        assert window.get_dock_widget(MERFISH_WIDGET_KEY).toggleViewAction().isChecked()
    finally:
        window.close()
        QApplication.processEvents()


def test_tile_overlap_is_read_from_stage_explorer_widget(
    merfish_widget: object,
) -> None:
    widget = merfish_widget
    _set_stage_explorer_overlap_percent(widget, 35.0)

    assert "35.0%" in widget._tile_overlap_label.text()


def test_wavelength_rows_for_sequence_guess_demo_channels() -> None:
    assert wavelength_rows_for_sequence(_sequence()) == [
        ("FITC", "488", "520"),
        ("Rhodamine", "561", "590"),
        ("Cy5", "647", "680"),
    ]


def test_channel_specs_convert_gui_wavelengths_from_nm_to_um(
    merfish_widget: object,
) -> None:
    widget = merfish_widget
    widget._mda_widget.setValue(_sequence())
    widget._refresh_channel_table()

    channel_specs = channel_specs_from_sequence_wavelength_rows(
        widget._mda_widget.value(),
        [
            (
                widget._wavelength_table.item(row_index, 1).text(),
                widget._wavelength_table.item(row_index, 2).text(),
            )
            for row_index in range(widget._wavelength_table.rowCount())
        ],
    )

    assert [spec["excitation_um"] for spec in channel_specs] == pytest.approx(
        [0.488, 0.561, 0.647]
    )
    assert [spec["emission_um"] for spec in channel_specs] == pytest.approx(
        [0.520, 0.590, 0.680]
    )


def test_bit_mapping_preview_uses_mda_channel_order() -> None:
    headers, rows = bit_mapping_preview(_sequence(), _exp_order())

    assert headers == ["Round", "FITC", "Rhodamine", "Cy5"]
    assert rows == [[1, 0, 1, 2], [2, 0, 3, 4]]


def test_channel_tables_reflect_sequence_and_exp_order(
    merfish_widget: object,
) -> None:
    widget = merfish_widget
    widget._exp_order = _exp_order()
    widget._mda_widget.setValue(_sequence())
    widget._refresh_channel_table()

    assert widget._wavelength_table.rowCount() == 3
    assert widget._bit_mapping_table.rowCount() == 2
    assert widget._bit_mapping_table.columnCount() == 4
    assert widget._bit_mapping_table.item(0, 1).text() == "0"


def test_uniform_illumination_option_disables_profile_file_widgets(
    merfish_widget: object,
) -> None:
    widget = merfish_widget

    widget._mode_combo.setCurrentIndex(widget._mode_combo.findData("iterative"))
    widget._use_uniform_illumination.setChecked(True)

    assert widget._illumination_browse.isEnabled() is False
    assert widget._illumination_path_edit.text() == "Uniform / unknown (all ones)"


@pytest.mark.parametrize(
    ("program_loader", "expected_rounds"),
    [
        (_synthetic_run_rounds_program, [2, 4, 7]),
        (_initial_fluidics_program, []),
    ],
)
def test_single_round_options_follow_fluidics_run_rounds(
    merfish_widget: object,
    workspace_tmp_path: Path,
    program_loader: object,
    expected_rounds: list[int],
) -> None:
    widget = merfish_widget
    widget._fluidics_program = program_loader(workspace_tmp_path)
    widget._fluidics_path = None
    widget._refresh_single_round_options()
    widget._mode_combo.setCurrentIndex(widget._mode_combo.findData("single_round"))

    rounds = [
        widget._single_round_combo.itemData(index)
        for index in range(widget._single_round_combo.count())
    ]
    assert rounds == expected_rounds
    assert widget._single_round_combo.currentData() == (
        expected_rounds[0] if expected_rounds else None
    )
    assert fluidics_round_options(widget._fluidics_program) == expected_rounds


def test_iterative_uniform_config_enables_run_without_stage_explorer(
    qtbot: object,
    demo_core: object,
    offline_icons: None,
) -> None:
    mda_widget = MDAWidget(mmcore=demo_core)
    mda_widget.setValue(_sequence())
    widget = MerfishFluidicsWidget(
        mmcore=demo_core,
        mda_widget=mda_widget,
        stage_explorer=None,
    )
    qtbot.addWidget(mda_widget)
    qtbot.addWidget(widget)
    try:
        widget._mode_combo.setCurrentIndex(widget._mode_combo.findData("iterative"))
        widget._fluidics_program = pd.DataFrame(
            {
                "round": [1, 1, 2, 2],
                "source": ["B01", "RUN", "B02", "RUN"],
                "time": [0.1, 0.0, 0.1, 0.0],
                "pump": [10.0, 0.0, 10.0, 0.0],
            }
        )
        widget._exp_order = _exp_order()
        widget._codebook = pd.DataFrame(
            {
                "gene_id": ["gene"],
                "bit01": [1],
                "bit02": [0],
                "bit03": [1],
                "bit04": [0],
            }
        )
        widget._use_uniform_illumination.setChecked(True)
        widget._validate()
        assert widget._run_button.isEnabled() is True
        assert widget._status_label.text() == "MERFISH configuration is valid."
    finally:
        widget.close()
        mda_widget.close()




