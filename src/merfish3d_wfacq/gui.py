import json
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import pandas as pd
from pymmcore_gui import MicroManagerGUI, WidgetAction
from pymmcore_gui._qt.QtCore import Qt, Signal
from pymmcore_gui._qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pymmcore_gui.actions import WidgetActionInfo
from pymmcore_plus import CMMCorePlus
from useq import MDASequence

from merfish3d_wfacq.core_metadata import (
    CoreMetadataError,
    derive_core_metadata,
)
from merfish3d_wfacq.dispatch import prepare_merfish_dispatch
from merfish3d_wfacq.engine import MerfishMDAEngine
from merfish3d_wfacq.sequence import RunMode, tile_count_from_sequence
from merfish3d_wfacq.ui_state import (
    bit_mapping_preview,
    build_merfish_ui_state,
    fluidics_round_options,
    wavelength_rows_for_sequence,
)
from merfish3d_wfacq.utils.data_io import (
    read_codebook,
    read_exp_order,
    read_fluidics_program,
    read_illumination_profiles,
)
from merfish3d_wfacq.workflow import (
    normalize_merfish_ui_state,
)

if TYPE_CHECKING:
    import numpy as np


MERFISH_WIDGET_KEY = "merfish3d_wfacq.merfish_widget"


class _PyMMWidgetAdapter:
    """Small adapter around upstream pyMM widgets used by the MERFISH UI."""

    def __init__(
        self,
        *,
        mda_widget: QWidget,
        stage_explorer: QWidget | None = None,
    ) -> None:
        """Wrap the upstream MDA and Stage Explorer widgets.

        Parameters
        ----------
        mda_widget : QWidget
            Upstream MDA widget instance.
        stage_explorer : QWidget or None, optional
            Upstream Stage Explorer widget instance.
        """

        self.mda_widget = mda_widget
        self.stage_explorer = stage_explorer

    def current_sequence(self) -> MDASequence:
        """Return the current MDA sequence from the upstream widget.

        Returns
        -------
        MDASequence
            Current sequence held by the upstream MDA widget.
        """

        return self.mda_widget.value()

    def connect_sequence_changed(self, slot: Any) -> None:
        """Connect a slot to upstream MDA sequence changes.

        Parameters
        ----------
        slot : Any
            Qt-compatible slot to connect.
        """

        self.mda_widget.valueChanged.connect(slot)

    def ensure_ome_zarr_selected(self) -> None:
        """Force the upstream save widget into OME-Zarr mode."""

        save_info = self.mda_widget.save_info
        save_value = dict(save_info.value())
        save_value["should_save"] = True
        save_value["format"] = "ome-zarr"
        save_info.setValue(save_value)

    def prepare_mda(self) -> bool | str | Path | None:
        """Delegate to the upstream MDA preparation hook.

        Returns
        -------
        bool or str or Path or None
            Upstream prepare result from the MDA widget.
        """
        return self.mda_widget.prepare_mda()

    def connect_stage_overlap_changed(self, slot: Any) -> None:
        """Connect a slot to Stage Explorer overlap changes when available.

        Parameters
        ----------
        slot : Any
            Qt-compatible slot to connect.
        """

        if scan_menu := self._stage_explorer_scan_menu():
            scan_menu.valueChanged.connect(slot)

    def tile_overlap_fraction(self) -> float:
        """Return the Stage Explorer overlap as a unitless fraction.

        Returns
        -------
        float
            Tile overlap as a fraction between ``0`` and ``1``.
        """

        scan_menu = self._stage_explorer_scan_menu()
        if scan_menu is None or not hasattr(scan_menu, "value"):
            raise ValueError(
                "Stage Explorer scan settings are unavailable; cannot determine tile overlap."
            )
        overlap_percent, _mode = scan_menu.value()
        return float(overlap_percent) / 100.0

    def _stage_explorer_scan_menu(self) -> Any | None:
        """Return the private Stage Explorer scan menu used for overlap.

        Returns
        -------
        Any or None
            Stage Explorer scan menu object, if available.
        """

        toolbar = getattr(self.stage_explorer, "_toolbar", None)
        return getattr(toolbar, "scan_menu", None)


class MerfishFluidicsWidget(QWidget):
    """Dock widget that extends the built-in pyMM MDA workflow for MERFISH."""

    logMessage = Signal(str)
    statusMessage = Signal(str)
    refreshRequested = Signal(object)

    def __init__(
        self,
        *,
        mmcore: CMMCorePlus,
        mda_widget: QWidget,
        stage_explorer: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Create the MERFISH widget and bind it to the upstream pyMM UI.

        Parameters
        ----------
        mmcore : CMMCorePlus
            Active ``CMMCorePlus`` instance.
        mda_widget : QWidget
            Upstream MDA widget.
        stage_explorer : QWidget or None, optional
            Upstream Stage Explorer widget.
        parent : QWidget or None, optional
            Parent widget.
        """

        super().__init__(parent)
        self._mmc = mmcore
        self._mda_widget = mda_widget
        self._stage_explorer = stage_explorer
        self._upstream = _PyMMWidgetAdapter(
            mda_widget=mda_widget,
            stage_explorer=stage_explorer,
        )
        self._fluidics_program: pd.DataFrame | None = None
        self._exp_order: pd.DataFrame | None = None
        self._codebook: pd.DataFrame | None = None
        self._illumination_profiles: np.ndarray | None = None
        self._fluidics_path: Path | None = None
        self._exp_order_path: Path | None = None
        self._codebook_path: Path | None = None
        self._illumination_profiles_path: Path | None = None
        self._reference_tile_initialized = False
        self._core_metadata: dict[str, Any] | None = None
        self._core_metadata_error: str | None = None
        self._validated_ui_state_cache: dict[str, Any] | None = None
        self._validation_error: str | None = None

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Fluidics only", RunMode.FLUIDICS_ONLY.value)
        self._mode_combo.addItem("Iterative imaging", RunMode.ITERATIVE.value)
        self._mode_combo.addItem("Single round test", RunMode.SINGLE_ROUND.value)

        self._fluidics_path_edit = QLineEdit()
        self._fluidics_path_edit.setReadOnly(True)
        self._fluidics_browse = QPushButton("Load fluidics CSV")

        self._exp_order_path_edit = QLineEdit()
        self._exp_order_path_edit.setReadOnly(True)
        self._exp_order_browse = QPushButton("Load experiment order file")

        self._codebook_path_edit = QLineEdit()
        self._codebook_path_edit.setReadOnly(True)
        self._codebook_browse = QPushButton("Load codebook")

        self._illumination_path_edit = QLineEdit()
        self._illumination_path_edit.setReadOnly(True)
        self._illumination_browse = QPushButton("Load illumination profiles")
        self._use_uniform_illumination = QCheckBox(
            "Use uniform / unknown illumination"
        )

        self._single_round_combo = QComboBox()
        self._single_round_combo.setEnabled(False)

        self._reference_tile_spin = QSpinBox()
        self._reference_tile_spin.setRange(0, 0)

        self._microscope_type_combo = QComboBox()
        self._microscope_type_combo.addItems(["3D", "2D"])
        self._na_spin = self._make_double_spin(0.0, 2.0, 1.35, 3)
        self._ri_spin = self._make_double_spin(1.0, 2.0, 1.51, 3)
        self._tile_overlap_label = QLabel("")
        self._e_per_adu_spin = self._make_double_spin(0.0, 100.0, 0.51, 4)
        self._camera_offset_spin = self._make_double_spin(0.0, 100000.0, 0.0, 2)
        self._binning_spin = QSpinBox()
        self._binning_spin.setRange(1, 16)
        self._binning_spin.setValue(1)
        self._pixel_size_um_spin = self._make_double_spin(0.0, 100.0, 0.0, 4)
        self._camera_model_edit = QLineEdit()
        self._affine_zyx_px_edit = QLineEdit()
        self._affine_zyx_px_edit.setPlaceholderText(
            "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        )
        self._pixel_size_um_spin.setReadOnly(True)
        self._camera_model_edit.setReadOnly(True)
        self._e_per_adu_spin.setReadOnly(True)
        self._camera_offset_spin.setReadOnly(True)
        self._binning_spin.setReadOnly(True)
        self._affine_zyx_px_edit.setReadOnly(True)

        self._simulate_pump = QCheckBox("Simulate pump")
        self._simulate_pump.setChecked(True)
        self._simulate_valves = QCheckBox("Simulate valves")
        self._simulate_valves.setChecked(True)
        self._simulated_valve_count = QSpinBox()
        self._simulated_valve_count.setRange(4, 16)
        self._simulated_valve_count.setValue(4)

        self._pump_com_port = QLineEdit("COM3")
        self._valve_com_port = QLineEdit("COM4")
        self._enable_drift = QCheckBox("Enable drift correction")
        self._enable_drift.setChecked(True)

        self._program_table = QTableWidget(0, 0)
        self._program_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._program_table.horizontalHeader().setStretchLastSection(True)

        self._wavelength_table = QTableWidget(0, 3)
        self._wavelength_table.setHorizontalHeaderLabels(
            ["Channel", "Excitation (nm)", "Emission (nm)"]
        )
        self._wavelength_table.horizontalHeader().setStretchLastSection(True)
        self._bit_mapping_table = QTableWidget(0, 0)
        self._bit_mapping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._bit_mapping_table.horizontalHeader().setStretchLastSection(True)

        self._status_label = QLabel("Load MERFISH acquisition inputs to begin.")
        self._status_label.setWordWrap(True)
        self._wavelength_help_label = QLabel(
            "Channel wavelength metadata is independent of the experiment order file. Provide one "
            "excitation/emission pair for each active acquisition channel."
        )
        self._wavelength_help_label.setWordWrap(True)
        self._bit_mapping_help_label = QLabel(
            "Bit identity comes from the experiment order file. It must include one column "
            "per active channel, named to match the MDA channel configs; column order "
            "does not matter, and the unique all-zero channel column is treated as the fiducial."
        )
        self._bit_mapping_help_label.setWordWrap(True)
        self._illumination_help_label = QLabel(
            "Load measured illumination profiles for flatfield correction, or use the "
            "uniform / unknown option to generate all-ones profiles at runtime."
        )
        self._illumination_help_label.setWordWrap(True)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)

        self._run_button = QPushButton("Run acquisition")
        self._abort_button = QPushButton("Abort")
        self._abort_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()
        self._refresh_reference_tile_range()
        self._refresh_channel_table()
        self._refresh_core_metadata()
        self._refresh_tile_overlap_display()
        self._apply_mode_ui()
        self._validate()

    @staticmethod
    def _make_double_spin(
        minimum: float, maximum: float, value: float, decimals: int
    ) -> QDoubleSpinBox:
        """Create a configured ``QDoubleSpinBox``.

        Parameters
        ----------
        minimum : float
            Minimum value.
        maximum : float
            Maximum value.
        value : float
            Initial value.
        decimals : int
            Number of decimals to display.

        Returns
        -------
        QDoubleSpinBox
            Configured spin box.
        """

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(10 ** (-decimals + 1) if decimals > 1 else 0.1)
        return spin

    def _build_layout(self) -> None:
        """Build the widget layout and group boxes."""

        layout = QVBoxLayout(self)

        files_group = QGroupBox("Programs", self)
        files_layout = QGridLayout(files_group)
        files_layout.addWidget(QLabel("Run mode:"), 0, 0)
        files_layout.addWidget(self._mode_combo, 0, 1, 1, 2)
        files_layout.addWidget(QLabel("Fluidics program:"), 1, 0)
        files_layout.addWidget(self._fluidics_path_edit, 1, 1)
        files_layout.addWidget(self._fluidics_browse, 1, 2)
        files_layout.addWidget(QLabel("Experiment order file:"), 2, 0)
        files_layout.addWidget(self._exp_order_path_edit, 2, 1)
        files_layout.addWidget(self._exp_order_browse, 2, 2)
        files_layout.addWidget(QLabel("codebook:"), 3, 0)
        files_layout.addWidget(self._codebook_path_edit, 3, 1)
        files_layout.addWidget(self._codebook_browse, 3, 2)
        files_layout.addWidget(QLabel("illumination profiles:"), 4, 0)
        files_layout.addWidget(self._illumination_path_edit, 4, 1)
        files_layout.addWidget(self._illumination_browse, 4, 2)
        files_layout.addWidget(self._use_uniform_illumination, 5, 1, 1, 2)
        files_layout.addWidget(QLabel("Selected fluidics round:"), 6, 0)
        files_layout.addWidget(self._single_round_combo, 6, 1, 1, 2)

        datastore_group = QGroupBox("Datastore Metadata", self)
        datastore_layout = QFormLayout(datastore_group)
        datastore_layout.addRow("Microscope type", self._microscope_type_combo)
        datastore_layout.addRow("NA", self._na_spin)
        datastore_layout.addRow("RI", self._ri_spin)
        datastore_layout.addRow(
            "Tile overlap (Stage Explorer)", self._tile_overlap_label
        )
        datastore_layout.addRow("e-/ADU", self._e_per_adu_spin)
        datastore_layout.addRow("Camera offset (ADU)", self._camera_offset_spin)
        datastore_layout.addRow("Binning", self._binning_spin)
        datastore_layout.addRow("Pixel size XY (um)", self._pixel_size_um_spin)
        datastore_layout.addRow("Camera model", self._camera_model_edit)
        datastore_layout.addRow("Affine ZYX PX", self._affine_zyx_px_edit)

        hardware_group = QGroupBox("Fluidics Hardware", self)
        hardware_layout = QFormLayout(hardware_group)
        hardware_layout.addRow(self._simulate_pump)
        hardware_layout.addRow(self._simulate_valves)
        hardware_layout.addRow("Simulated valve count", self._simulated_valve_count)
        hardware_layout.addRow("Pump COM port", self._pump_com_port)
        hardware_layout.addRow("Valve COM port", self._valve_com_port)
        hardware_layout.addRow(self._enable_drift)
        hardware_layout.addRow("Reference tile", self._reference_tile_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self._run_button)
        buttons.addWidget(self._abort_button)

        layout.addWidget(files_group)
        layout.addWidget(datastore_group)
        layout.addWidget(hardware_group)
        layout.addWidget(self._illumination_help_label)
        layout.addWidget(QLabel("Channel wavelengths:"))
        layout.addWidget(self._wavelength_help_label)
        layout.addWidget(self._wavelength_table)
        layout.addWidget(QLabel("Round-to-channel bit mapping preview:"))
        layout.addWidget(self._bit_mapping_help_label)
        layout.addWidget(self._bit_mapping_table)
        layout.addWidget(QLabel("Loaded fluidics program preview:"))
        layout.addWidget(self._program_table, stretch=1)
        layout.addWidget(self._status_label)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Run log:"))
        layout.addWidget(self._log, stretch=1)

    def _connect_signals(self) -> None:
        """Connect Qt signals for widget state and MDA lifecycle events."""

        self._fluidics_browse.clicked.connect(self._load_fluidics_program)
        self._exp_order_browse.clicked.connect(self._load_exp_order)
        self._codebook_browse.clicked.connect(self._load_codebook)
        self._illumination_browse.clicked.connect(self._load_illumination_profiles)
        self._use_uniform_illumination.toggled.connect(
            self._on_uniform_illumination_toggled
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._connect_validation_signals(
            self._single_round_combo.currentIndexChanged,
            self._reference_tile_spin.valueChanged,
            self._simulate_pump.toggled,
            self._simulate_valves.toggled,
            self._simulated_valve_count.valueChanged,
            self._pump_com_port.textChanged,
            self._valve_com_port.textChanged,
            self._enable_drift.toggled,
            self._microscope_type_combo.currentIndexChanged,
            self._na_spin.valueChanged,
            self._ri_spin.valueChanged,
            self._e_per_adu_spin.valueChanged,
            self._camera_offset_spin.valueChanged,
            self._binning_spin.valueChanged,
        )
        self._wavelength_table.itemChanged.connect(
            self._on_wavelength_table_item_changed
        )
        self._run_button.clicked.connect(self._run_acquisition)
        self._abort_button.clicked.connect(self._mmc.mda.cancel)
        self.logMessage.connect(self._append_log)
        self.statusMessage.connect(self._set_status)
        self.refreshRequested.connect(self._handle_refresh_request)
        self._mmc.mda.events.sequenceStarted.connect(self._on_sequence_started)
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)
        self._connect_upstream_signals()

    def _connect_validation_signals(self, *signals: Any) -> None:
        """Connect many widget signals to the shared validation slot.

        Parameters
        ----------
        *signals : Any
            Bound Qt signals that should trigger validation.
        """

        for signal in signals:
            signal.connect(self._validate)

    def _connect_upstream_signals(self) -> None:
        """Connect upstream pyMM widget signals used by the MERFISH dock."""

        for slot in (
            self._refresh_reference_tile_range,
            self._refresh_channel_table,
            self._refresh_core_metadata,
            self._validate,
        ):
            self._upstream.connect_sequence_changed(slot)
        for slot in (self._refresh_tile_overlap_display, self._validate):
            self._upstream.connect_stage_overlap_changed(slot)

    def _current_mode(self) -> RunMode:
        """Return the currently selected MERFISH run mode.

        Returns
        -------
        RunMode
            Currently selected run mode.
        """

        return RunMode(self._mode_combo.currentData())

    def _current_sequence(self) -> MDASequence:
        """Return the current upstream MDA sequence.

        Returns
        -------
        MDASequence
            Sequence currently configured in the upstream MDA widget.
        """

        return self._upstream.current_sequence()

    def _load_fluidics_program(self) -> None:
        """Load and preview a fluidics CSV file."""

        path, _ = QFileDialog.getOpenFileName(
            self, "Select fluidics program", "", "CSV Files (*.csv *.tsv)"
        )
        if not path:
            return
        try:
            self._fluidics_program = read_fluidics_program(path)
            self._fluidics_path = Path(path)
            self._fluidics_path_edit.setText(path)
            self._populate_table(self._fluidics_program)
            self._refresh_single_round_options()
            self._append_log(f"Loaded fluidics program: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Fluidics program error", str(exc))
        self._validate()

    def _load_exp_order(self) -> None:
        """Load and preview an experiment order file."""

        path, _ = QFileDialog.getOpenFileName(
            self, "Select experiment order file", "", "Table Files (*.csv *.tsv)"
        )
        if not path:
            return
        try:
            self._exp_order = read_exp_order(path)
            self._exp_order_path = Path(path)
            self._exp_order_path_edit.setText(path)
            self._append_log(f"Loaded experiment order file: {path}")
            self._refresh_bit_mapping_table()
        except Exception as exc:
            QMessageBox.critical(self, "Experiment order file error", str(exc))
        self._validate()

    def _load_codebook(self) -> None:
        """Load a MERFISH codebook table."""

        path, _ = QFileDialog.getOpenFileName(
            self, "Select codebook", "", "Table Files (*.csv *.tsv)"
        )
        if not path:
            return
        try:
            self._codebook = read_codebook(path)
            self._codebook_path = Path(path)
            self._codebook_path_edit.setText(path)
            self._append_log(f"Loaded codebook: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Codebook error", str(exc))
        self._validate()

    def _load_illumination_profiles(self) -> None:
        """Load illumination profiles used for flatfield correction."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select illumination profiles",
            "",
            "Image Files (*.ome.tif *.ome.tiff *.tif *.tiff)",
        )
        if not path:
            return
        try:
            self._illumination_profiles = read_illumination_profiles(path)
            self._illumination_profiles_path = Path(path)
            if not self._use_uniform_illumination.isChecked():
                self._illumination_path_edit.setText(path)
            self._append_log(
                "Loaded illumination profiles: "
                f"{path} with shape {tuple(self._illumination_profiles.shape)}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Illumination-profile error", str(exc))
        self._validate()

    def _on_mode_changed(self) -> None:
        """Update widget state when the run mode changes."""

        self._apply_mode_ui()
        self._validate()

    def _on_uniform_illumination_toggled(self, checked: bool) -> None:
        """Toggle generated all-ones illumination profiles.

        Parameters
        ----------
        checked : bool
            Whether to use generated uniform illumination profiles.
        """

        if checked:
            self._illumination_path_edit.setText("Uniform / unknown (all ones)")
        elif self._illumination_profiles_path is not None:
            self._illumination_path_edit.setText(str(self._illumination_profiles_path))
        else:
            self._illumination_path_edit.clear()
        self._apply_mode_ui()
        self._validate()

    def _apply_mode_ui(self) -> None:
        """Enable and disable controls for the selected run mode."""

        mode = self._current_mode()
        is_fluidics_only = mode is RunMode.FLUIDICS_ONLY
        is_iterative = mode is RunMode.ITERATIVE
        is_single_round = mode is RunMode.SINGLE_ROUND

        self._run_button.setText(
            "Run fluidics"
            if is_fluidics_only
            else "Run single round imaging"
            if is_single_round
            else "Run acquisition"
        )

        imaging_widgets = [
            self._exp_order_path_edit,
            self._exp_order_browse,
            self._codebook_path_edit,
            self._codebook_browse,
            self._use_uniform_illumination,
            self._illumination_path_edit,
            self._illumination_browse,
            self._microscope_type_combo,
            self._na_spin,
            self._ri_spin,
            self._e_per_adu_spin,
            self._camera_offset_spin,
            self._binning_spin,
            self._pixel_size_um_spin,
            self._camera_model_edit,
            self._affine_zyx_px_edit,
            self._wavelength_table,
            self._bit_mapping_table,
        ]
        for widget in imaging_widgets:
            widget.setEnabled(not is_fluidics_only)

        if not is_fluidics_only:
            uniform_enabled = self._use_uniform_illumination.isChecked()
            self._illumination_path_edit.setEnabled(not uniform_enabled)
            self._illumination_browse.setEnabled(not uniform_enabled)

        fluidics_widgets = [
            self._simulate_pump,
            self._simulate_valves,
            self._simulated_valve_count,
            self._pump_com_port,
            self._valve_com_port,
            self._enable_drift,
            self._reference_tile_spin,
        ]
        for widget in fluidics_widgets:
            widget.setEnabled(not is_single_round)

        self._single_round_combo.setEnabled(is_single_round)
        if is_fluidics_only:
            self._single_round_combo.setEnabled(False)
        if is_iterative:
            self._single_round_combo.setEnabled(False)

    def _on_wavelength_table_item_changed(self, _item: QTableWidgetItem) -> None:
        """Revalidate after the wavelength table changes.

        Parameters
        ----------
        _item : QTableWidgetItem or None
            Changed table item.
        """

        self._validate()

    def _current_sequence_or_none(self) -> MDASequence | None:
        """Return the current upstream sequence when available.

        Returns
        -------
        MDASequence or None
            Current MDA sequence, or ``None`` when unavailable.
        """

        try:
            return self._current_sequence()
        except Exception:
            return None

    def _wavelength_table_values(self, sequence: MDASequence) -> list[tuple[str, str, str]]:
        """Return the current wavelength-table values keyed by channel name.

        Parameters
        ----------
        sequence : MDASequence
            Current upstream sequence.

        Returns
        -------
        list[tuple[str, str, str]]
            Rows of ``(channel, excitation_nm, emission_nm)``.
        """

        previous_values: dict[str, tuple[str, str]] = {}
        for row in range(self._wavelength_table.rowCount()):
            channel_item = self._wavelength_table.item(row, 0)
            exc_item = self._wavelength_table.item(row, 1)
            em_item = self._wavelength_table.item(row, 2)
            if channel_item is None:
                continue
            previous_values[channel_item.text()] = (
                exc_item.text() if exc_item is not None else "",
                em_item.text() if em_item is not None else "",
            )
        return wavelength_rows_for_sequence(sequence, previous_values)

    def _prepare_save_path(self, normalized_ui_state: dict[str, Any]) -> str | Path | None:
        """Prepare and return the upstream save path for imaging runs.

        Parameters
        ----------
        normalized_ui_state : dict[str, Any]
            Cached normalized widget state.

        Returns
        -------
        str or Path or None
            Save path selected by the upstream MDA widget, or ``None`` for fluidics-only runs.
        """

        if normalized_ui_state["mode"] is RunMode.FLUIDICS_ONLY:
            return None
        self._upstream.ensure_ome_zarr_selected()
        save_path = self._upstream.prepare_mda()
        if isinstance(save_path, bool):
            return None
        return save_path

    def _run_acquisition(self) -> None:
        """Validate widget state and dispatch the prepared MERFISH run."""

        normalized_ui_state, error = self._validated_ui_state()
        if error is not None:
            QMessageBox.warning(self, "MERFISH configuration", error)
            return

        try:
            save_path = self._prepare_save_path(normalized_ui_state)
            if (
                normalized_ui_state["mode"] is not RunMode.FLUIDICS_ONLY
                and save_path is None
            ):
                return
            sequence, output = prepare_merfish_dispatch(
                normalized_ui_state=normalized_ui_state,
                save_path=save_path,
                overwrite=True,
            )
            self._mmc.run_mda(sequence, output=output)
            self._append_log("Dispatched MERFISH acquisition to pymmcore-plus.")
        except Exception as exc:
            QMessageBox.critical(self, "Failed to run MERFISH acquisition", str(exc))
            self._append_log(f"ERROR: {exc}")

    def _refresh_core_metadata(self) -> None:
        """Refresh read-only datastore metadata derived from MMCore."""

        try:
            metadata = derive_core_metadata(self._mmc, self._current_sequence())
        except CoreMetadataError as exc:
            self._core_metadata = None
            self._core_metadata_error = str(exc)
            return
        except Exception as exc:
            self._core_metadata = None
            self._core_metadata_error = (
                f"Failed to derive required metadata from Micro-Manager core: {exc}"
            )
            return

        self._core_metadata = metadata
        self._core_metadata_error = None
        self._pixel_size_um_spin.blockSignals(True)
        self._camera_model_edit.blockSignals(True)
        self._e_per_adu_spin.blockSignals(True)
        self._camera_offset_spin.blockSignals(True)
        self._binning_spin.blockSignals(True)
        self._affine_zyx_px_edit.blockSignals(True)
        try:
            self._pixel_size_um_spin.setValue(float(metadata["pixel_size_um"]))
            self._camera_model_edit.setText(str(metadata["camera_model"]))
            self._e_per_adu_spin.setValue(float(metadata["e_per_adu"]))
            self._camera_offset_spin.setValue(float(metadata["camera_offset_adu"]))
            self._binning_spin.setValue(int(metadata["binning"]))
            self._affine_zyx_px_edit.setText(
                json.dumps(metadata["affine_zyx_px"], separators=(",", ":"))
            )
        finally:
            self._pixel_size_um_spin.blockSignals(False)
            self._camera_model_edit.blockSignals(False)
            self._e_per_adu_spin.blockSignals(False)
            self._camera_offset_spin.blockSignals(False)
            self._binning_spin.blockSignals(False)
            self._affine_zyx_px_edit.blockSignals(False)

    def _refresh_tile_overlap_display(self) -> None:
        """Refresh the displayed Stage Explorer overlap value."""

        try:
            overlap_fraction = self._upstream.tile_overlap_fraction()
        except ValueError:
            self._tile_overlap_label.setText("Unavailable")
            return

        overlap_percent = overlap_fraction * 100.0
        self._tile_overlap_label.setText(
            f"{overlap_percent:.1f}% ({overlap_fraction:.3f})"
        )

    def _refresh_reference_tile_range(self) -> None:
        """Update the valid reference-tile range from the current sequence."""

        try:
            sequence = self._current_sequence()
            count = tile_count_from_sequence(sequence)
        except Exception:
            count = 1
        self._reference_tile_spin.setRange(0, count - 1)
        midpoint = count // 2
        if self._reference_tile_spin.value() >= count:
            self._reference_tile_spin.setValue(midpoint)
        elif not self._reference_tile_initialized:
            self._reference_tile_spin.setValue(midpoint)
            self._reference_tile_initialized = True

    def _refresh_single_round_options(self) -> None:
        """Refresh the single-round selector from the loaded fluidics program."""

        current_round = self._selected_single_round()
        rounds = fluidics_round_options(self._fluidics_program)

        self._single_round_combo.blockSignals(True)
        self._single_round_combo.clear()
        for round_id in rounds:
            self._single_round_combo.addItem(str(round_id), round_id)
        if rounds:
            selected_round = (
                current_round if current_round in rounds else int(rounds[0])
            )
            self._single_round_combo.setCurrentIndex(rounds.index(selected_round))
        self._single_round_combo.blockSignals(False)

    def _selected_single_round(self) -> int | None:
        """Return the selected single-round test round, if applicable.

        Returns
        -------
        int or None
            Selected round label for single-round mode.
        """

        if self._current_mode() is not RunMode.SINGLE_ROUND:
            return None
        selected = self._single_round_combo.currentData()
        return int(selected) if selected is not None else None

    def _refresh_channel_table(self) -> None:
        """Refresh the wavelength table from the active MDA channels."""

        sequence = self._current_sequence_or_none()
        rows = [] if sequence is None else self._wavelength_table_values(sequence)

        self._wavelength_table.blockSignals(True)
        self._wavelength_table.setRowCount(len(rows))
        for row_index, (config_name, excitation_nm, emission_nm) in enumerate(rows):
            self._set_wavelength_table_item(row_index, 0, config_name, editable=False)
            self._set_wavelength_table_item(row_index, 1, excitation_nm, editable=True)
            self._set_wavelength_table_item(row_index, 2, emission_nm, editable=True)
        self._wavelength_table.blockSignals(False)
        self._refresh_bit_mapping_table()

    def _refresh_bit_mapping_table(self) -> None:
        """Refresh the round-to-bit preview table."""

        sequence = self._current_sequence_or_none()
        headers, rows = ([], []) if sequence is None else bit_mapping_preview(sequence, self._exp_order)

        if not headers:
            self._bit_mapping_table.clear()
            self._bit_mapping_table.setRowCount(0)
            self._bit_mapping_table.setColumnCount(0)
            return

        self._bit_mapping_table.clear()
        self._bit_mapping_table.setColumnCount(len(headers))
        self._bit_mapping_table.setHorizontalHeaderLabels(headers)
        self._bit_mapping_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._bit_mapping_table.setItem(row_index, column_index, item)

    def _set_wavelength_table_item(
        self, row: int, column: int, text: str, *, editable: bool
    ) -> None:
        """Populate one wavelength-table cell.

        Parameters
        ----------
        row : int
            Target row index.
        column : int
            Target column index.
        text : str
            Cell text.
        editable : bool
            Whether the cell should be editable.
        """

        item = QTableWidgetItem(str(text))
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._wavelength_table.setItem(row, column, item)

    def _populate_table(self, frame: pd.DataFrame) -> None:
        """Populate the fluidics preview table from a dataframe.

        Parameters
        ----------
        frame : pd.DataFrame
            Table to display in the preview widget.
        """

        self._program_table.clear()
        self._program_table.setRowCount(len(frame))
        self._program_table.setColumnCount(len(frame.columns))
        self._program_table.setHorizontalHeaderLabels(
            [str(column) for column in frame.columns]
        )
        for row_index, row in enumerate(frame.itertuples(index=False)):
            for column_index, value in enumerate(row):
                self._program_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(str(value)),
                )

    def _collect_ui_state(self) -> dict[str, Any]:
        """Collect normalized widget state for workflow helpers.

        Returns
        -------
        dict[str, Any]
            Normalized widget state used by workflow helpers.
        """

        mode = self._current_mode()
        sequence = self._current_sequence_or_none() if mode is not RunMode.FLUIDICS_ONLY else None
        if mode is not RunMode.FLUIDICS_ONLY and sequence is None:
            raise ValueError("MDA widget configuration is invalid.")

        try:
            tile_overlap = (
                self._upstream.tile_overlap_fraction()
                if mode is not RunMode.FLUIDICS_ONLY
                else None
            )
            tile_overlap_error = None
        except ValueError as exc:
            tile_overlap = None
            tile_overlap_error = str(exc)
        wavelength_rows: list[tuple[str, str]] | None = None
        if sequence is not None:
            wavelength_rows = []
            for row_index in range(len(sequence.channels)):
                excitation_item = self._wavelength_table.item(row_index, 1)
                emission_item = self._wavelength_table.item(row_index, 2)
                wavelength_rows.append(
                    (
                        excitation_item.text() if excitation_item is not None else "",
                        emission_item.text() if emission_item is not None else "",
                    )
                )

        return build_merfish_ui_state(
            mode=mode,
            sequence=sequence,
            tile_overlap=tile_overlap,
            tile_overlap_error=tile_overlap_error,
            wavelength_rows=wavelength_rows,
            selected_single_round=self._selected_single_round(),
            fluidics_program=self._fluidics_program,
            exp_order=self._exp_order,
            codebook=self._codebook,
            illumination_profiles=self._illumination_profiles,
            use_uniform_illumination=bool(
                self._use_uniform_illumination.isChecked()
            ),
            core_metadata=self._core_metadata,
            core_metadata_error=self._core_metadata_error,
            reference_tile=int(self._reference_tile_spin.value()),
            enable_drift_correction=bool(self._enable_drift.isChecked()),
            simulate_pump=bool(self._simulate_pump.isChecked()),
            simulate_valves=bool(self._simulate_valves.isChecked()),
            num_simulated_valves=int(self._simulated_valve_count.value()),
            pump_com_port=self._pump_com_port.text(),
            valve_com_port=self._valve_com_port.text(),
            microscope_type=self._microscope_type_combo.currentText(),
            numerical_aperture=float(self._na_spin.value()),
            refractive_index=float(self._ri_spin.value()),
            exp_order_path=self._exp_order_path,
            codebook_path=self._codebook_path,
            illumination_profiles_path=self._illumination_profiles_path,
        )

    def _validate(self) -> None:
        """Validate the current widget state and update run availability."""

        _normalized_ui_state, error = self._refresh_validated_ui_state()
        self._run_button.setEnabled(error is None and not self._mmc.mda.is_running())
        if error:
            self._set_status(error)
        else:
            self._set_status("MERFISH configuration is valid.")

    def _refresh_validated_ui_state(self) -> tuple[dict[str, Any] | None, str | None]:
        """Recompute and cache the validated widget state.

        Returns
        -------
        tuple[dict[str, Any] or None, str or None]
            Cached normalized widget state and the current validation error.
        """

        try:
            self._validated_ui_state_cache = normalize_merfish_ui_state(
                self._collect_ui_state()
            )
            self._validation_error = None
        except Exception as exc:
            self._validated_ui_state_cache = None
            self._validation_error = str(exc)
        return self._validated_ui_state_cache, self._validation_error

    def _validated_ui_state(self) -> tuple[dict[str, Any] | None, str | None]:
        """Return the cached validated widget state when available.

        Returns
        -------
        tuple[dict[str, Any] or None, str or None]
            Normalized widget state and the current validation error.
        """

        if self._validated_ui_state_cache is None and self._validation_error is None:
            return self._refresh_validated_ui_state()
        return self._validated_ui_state_cache, self._validation_error

    def _append_log(self, message: str) -> None:
        """Append one line to the widget log panel."""

        self._log.appendPlainText(str(message))

    def _set_status(self, message: str) -> None:
        """Update the widget status label."""

        self._status_label.setText(str(message))

    def request_refresh(self, payload: dict[str, Any]) -> bool:
        """Block until the operator responds to a REFRESH request.

        Parameters
        ----------
        payload : dict[str, Any]
            REFRESH payload for the current fluidics step.

        Returns
        -------
        bool
            ``True`` when the operator confirms the refresh step.
        """

        request = {"payload": payload, "approved": False, "event": Event()}
        self.refreshRequested.emit(request)
        request["event"].wait()
        return bool(request["approved"])

    def _handle_refresh_request(self, request: dict[str, Any]) -> None:
        """Show the REFRESH confirmation dialog for the operator."""

        payload = request["payload"]
        text = (
            f"Complete the REFRESH step for round {payload['round']} and click Yes "
            "to continue the acquisition."
        )
        answer = QMessageBox.question(self, "MERFISH fluidics refresh", text)
        request["approved"] = answer == QMessageBox.StandardButton.Yes
        request["event"].set()
    def _on_sequence_started(self, _sequence: object, _meta: dict[str, Any]) -> None:
        """Update button state when an acquisition starts."""

        self._run_button.setEnabled(False)
        self._abort_button.setEnabled(True)
        self._set_status("MERFISH acquisition running.")

    def _on_sequence_finished(self, _sequence: object) -> None:
        """Restore widget state when an acquisition finishes."""

        self._abort_button.setEnabled(False)
        self._validate()


def create_merfish_widget(parent: QWidget) -> MerfishFluidicsWidget:
    """Create the MERFISH widget and register its engine with the pyMM main window.

    Parameters
    ----------
    parent : QWidget
        ``pymmcore-gui`` main window.

    Returns
    -------
    MerfishFluidicsWidget
        Created MERFISH widget.
    """

    if TYPE_CHECKING:
        main_window = parent
    else:
        main_window = parent

    mda_widget = main_window.get_widget(WidgetAction.MDA_WIDGET)
    try:
        stage_explorer = main_window.get_widget(WidgetAction.STAGE_EXPLORER)
    except Exception:
        stage_explorer = None
    widget = MerfishFluidicsWidget(
        mmcore=main_window.mmcore,
        mda_widget=mda_widget,
        stage_explorer=stage_explorer,
        parent=main_window,
    )
    engine = MerfishMDAEngine(
        main_window.mmcore,
        log_callback=widget.logMessage.emit,
        status_callback=widget.statusMessage.emit,
        refresh_handler=widget.request_refresh,
    )
    main_window.mmcore.register_mda_engine(engine)
    return widget


def _ensure_merfish_widget_action_registered() -> str:
    """Register the MERFISH widget action with ``pymmcore-gui``.

    Returns
    -------
    str
        Widget-action key for the MERFISH dock.
    """

    try:
        WidgetActionInfo.for_key(MERFISH_WIDGET_KEY)
    except KeyError:
        WidgetActionInfo(
            key=MERFISH_WIDGET_KEY,
            text="MERFISH Fluidics",
            create_widget=create_merfish_widget,
        )
    return MERFISH_WIDGET_KEY


def enhance_main_window(main_window: MicroManagerGUI) -> MerfishFluidicsWidget:
    """Create and attach the MERFISH dock widget to a pyMM main window.

    Parameters
    ----------
    main_window : MicroManagerGUI
        ``pymmcore-gui`` main window receiving the MERFISH dock.

    Returns
    -------
    MerfishFluidicsWidget
        Attached MERFISH widget.
    """

    widget_key = _ensure_merfish_widget_action_registered()
    widget = main_window.get_widget(widget_key)
    merfish_dock = main_window.get_dock_widget(widget_key)

    try:
        mda_dock = main_window.get_dock_widget(WidgetAction.MDA_WIDGET)
    except Exception:
        mda_dock = None
    else:
        if merfish_dock.dockAreaWidget() is not mda_dock.dockAreaWidget():
            main_window.dock_manager.addDockWidgetTabToArea(
                merfish_dock, mda_dock.dockAreaWidget()
            )
        mda_dock.raise_()

    return widget


