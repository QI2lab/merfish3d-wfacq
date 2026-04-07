import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from useq import MDASequence

from merfish3d_wfacq.sequence import RunMode, ordered_channel_bits_for_rounds
from merfish3d_wfacq.utils.data_io import experiment_order_mapping, imaging_rounds


def fluidics_round_options(fluidics_program: pd.DataFrame | None) -> list[int]:
    """Return imaging-capable rounds for the single-round selector.

    Parameters
    ----------
    fluidics_program : pd.DataFrame or None
        Loaded fluidics program.

    Returns
    -------
    list[int]
        Rounds that contain a ``RUN`` command.
    """

    return imaging_rounds(fluidics_program)


def guess_wavelengths_nm(config_name: str) -> tuple[int, int] | None:
    """Guess excitation and emission wavelengths from a channel name.

    Parameters
    ----------
    config_name : str
        Channel configuration name from the MDA widget.

    Returns
    -------
    tuple[int, int] or None
        Guessed excitation and emission wavelengths in nanometers.
    """

    normalized = config_name.lower()
    named_mapping = {
        "dapi": (405, 450),
        "fitc": (488, 520),
        "blue": (488, 520),
        "rhodamine": (561, 590),
        "yellow": (561, 590),
        "cy5": (647, 680),
        "red": (647, 680),
    }
    for key, wavelengths in named_mapping.items():
        if key in normalized:
            return wavelengths
    mapping = {
        405: (405, 450),
        488: (488, 520),
        514: (514, 540),
        532: (532, 560),
        561: (561, 590),
        565: (565, 590),
        594: (594, 617),
        635: (635, 670),
        640: (640, 680),
        647: (647, 680),
        750: (750, 780),
        785: (785, 820),
    }
    for match in re.findall(r"\d{3}", normalized):
        wavelength = int(match)
        if wavelength in mapping:
            return mapping[wavelength]
    return None


def wavelength_rows_for_sequence(
    sequence: MDASequence,
    previous_values: dict[str, tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Build editable wavelength rows for the current MDA channels.

    Parameters
    ----------
    sequence : MDASequence
        Active MDA sequence from the upstream widget.
    previous_values : dict[str, tuple[str, str]] or None, optional
        Existing wavelength values keyed by channel config name.

    Returns
    -------
    list[tuple[str, str, str]]
        Rows of ``(channel, excitation_nm, emission_nm)``.
    """

    rows: list[tuple[str, str, str]] = []
    previous_values = previous_values or {}
    for row_index, channel in enumerate(sequence.channels):
        config_name = channel.config or f"channel_{row_index}"
        excitation_nm, emission_nm = previous_values.get(config_name, ("", ""))
        if not excitation_nm and not emission_nm:
            guess = guess_wavelengths_nm(config_name)
            if guess is not None:
                excitation_nm, emission_nm = str(guess[0]), str(guess[1])
        rows.append((config_name, str(excitation_nm), str(emission_nm)))
    return rows


def _preview_rows_for_channel_names(
    channel_names: list[str],
    exp_order: pd.DataFrame,
) -> list[list[int | str]]:
    """Return preview rows for the given channel names.

    Parameters
    ----------
    channel_names : list[str]
        Channel names in display order.
    exp_order : pd.DataFrame
        Normalized experiment-order table.

    Returns
    -------
    list[list[int | str]]
        Preview rows matching ``channel_names``.
    """

    round_labels = [int(round_id) for round_id in exp_order["round"].tolist()]
    ordered_mapping = experiment_order_mapping(exp_order)
    if all(
        all(channel_name in ordered_mapping[int(round_label)] for channel_name in channel_names)
        for round_label in round_labels
    ):
        return [
            [int(record["round_label"]), *[int(bit) for bit in record["channel_bits"]]]
            for record in ordered_channel_bits_for_rounds(
                channel_names,
                ordered_mapping,
                round_labels,
            )
        ]

    channel_columns = [column for column in exp_order.columns if column != "round"]
    return [
        [
            int(exp_row["round"]),
            *[
                int(exp_row[channel_name]) if channel_name in channel_columns else ""
                for channel_name in channel_names
            ],
        ]
        for _, exp_row in exp_order.iterrows()
    ]


def bit_mapping_preview(
    sequence: MDASequence,
    exp_order: pd.DataFrame | None,
) -> tuple[list[str], list[list[int | str]]]:
    """Build a per-round preview table for channel-to-bit mapping.

    Parameters
    ----------
    sequence : MDASequence
        Active MDA sequence from the upstream widget.
    exp_order : pd.DataFrame or None
        Loaded experiment-order table.

    Returns
    -------
    tuple[list[str], list[list[int | str]]]
        Preview headers and table rows.
    """

    channels = list(sequence.channels)
    if not channels or exp_order is None or exp_order.empty:
        return [], []

    headers = [
        "Round",
        *[
            str(channel.config or f"channel_{index}")
            for index, channel in enumerate(channels)
        ],
    ]
    channel_names = [
        str(channel.config or f"channel_{index}")
        for index, channel in enumerate(channels)
    ]
    return headers, _preview_rows_for_channel_names(channel_names, exp_order)


def channel_specs_from_wavelength_rows(
    rows: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Normalize wavelength-table rows into datastore channel specs.

    Parameters
    ----------
    rows : list[tuple[str, str, str]]
        Rows of ``(channel, excitation_nm, emission_nm)``.

    Returns
    -------
    list[dict[str, Any]]
        Normalized channel specs for datastore metadata.
    """

    specs: list[dict[str, Any]] = []
    for row_index, (config_name, excitation_text, emission_text) in enumerate(rows):
        try:
            excitation_nm = float(excitation_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid excitation wavelength for {config_name}: {excitation_text!r}"
            ) from exc
        try:
            emission_nm = float(emission_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid emission wavelength for {config_name}: {emission_text!r}"
            ) from exc
        specs.append(
            {
                "channel_index": int(row_index),
                "config_name": str(config_name),
                "excitation_um": excitation_nm / 1000.0,
                "emission_um": emission_nm / 1000.0,
            }
        )
    return specs


def channel_specs_from_sequence_wavelength_rows(
    sequence: MDASequence,
    rows: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Validate widget wavelength rows against the current sequence.

    Parameters
    ----------
    sequence : MDASequence
        Active MDA sequence from the upstream widget.
    rows : list[tuple[str, str]]
        User-entered excitation and emission values.

    Returns
    -------
    list[dict[str, Any]]
        Normalized channel specs for datastore metadata.
    """

    specs_input: list[tuple[str, str, str]] = []
    for row_index, channel in enumerate(sequence.channels):
        config_name = channel.config or f"channel_{row_index}"
        try:
            excitation_text, emission_text = rows[row_index]
        except IndexError as exc:
            raise ValueError(
                f"Missing wavelength metadata for {config_name}."
            ) from exc
        excitation_text = str(excitation_text).strip()
        emission_text = str(emission_text).strip()
        if not excitation_text:
            raise ValueError(f"Provide a excitation wavelength for {config_name}.")
        if not emission_text:
            raise ValueError(f"Provide a emission wavelength for {config_name}.")
        specs_input.append((str(config_name), excitation_text, emission_text))
    return channel_specs_from_wavelength_rows(specs_input)


def build_merfish_ui_state(
    *,
    mode: RunMode,
    sequence: MDASequence | None,
    tile_overlap: float | None,
    tile_overlap_error: str | None,
    wavelength_rows: list[tuple[str, str]] | None,
    selected_single_round: int | None,
    fluidics_program: pd.DataFrame | None,
    exp_order: pd.DataFrame | None,
    codebook: pd.DataFrame | None,
    illumination_profiles: np.ndarray | None,
    use_uniform_illumination: bool,
    core_metadata: dict[str, Any] | None,
    core_metadata_error: str | None,
    reference_tile: int,
    enable_drift_correction: bool,
    simulate_pump: bool,
    simulate_valves: bool,
    num_simulated_valves: int,
    pump_com_port: str,
    valve_com_port: str,
    microscope_type: str,
    numerical_aperture: float,
    refractive_index: float,
    exp_order_path: Path | None,
    codebook_path: Path | None,
    illumination_profiles_path: Path | None,
) -> dict[str, Any]:
    """Collect normalized MERFISH widget state for validation and dispatch.

    Parameters
    ----------
    mode : RunMode
        Selected MERFISH run mode.
    sequence : MDASequence or None
        Current MDA sequence.
    tile_overlap : float or None
        Stage Explorer overlap as a unitless fraction.
    tile_overlap_error : str or None
        Error emitted while reading overlap metadata.
    wavelength_rows : list[tuple[str, str]] or None
        User-entered excitation and emission values.
    selected_single_round : int or None
        Selected round for single-round mode.
    fluidics_program : pd.DataFrame or None
        Loaded fluidics program.
    exp_order : pd.DataFrame or None
        Loaded experiment-order table.
    codebook : pd.DataFrame or None
        Loaded codebook table.
    illumination_profiles : np.ndarray or None
        Loaded illumination profile stack.
    use_uniform_illumination : bool
        Whether the run should use generated all-ones illumination profiles.
    core_metadata : dict[str, Any] or None
        Core metadata derived from MMCore.
    core_metadata_error : str or None
        Error emitted while deriving core metadata.
    reference_tile : int
        Selected reference tile for drift correction.
    enable_drift_correction : bool
        Whether drift correction is enabled.
    simulate_pump : bool
        Whether the pump should be simulated.
    simulate_valves : bool
        Whether the valves should be simulated.
    num_simulated_valves : int
        Number of simulated valves to expose.
    pump_com_port : str
        Pump serial port.
    valve_com_port : str
        Valve serial port.
    microscope_type : str
        Microscope type label stored in datastore metadata.
    numerical_aperture : float
        Objective numerical aperture.
    refractive_index : float
        Sample refractive index.
    exp_order_path : Path or None
        Source path of the loaded experiment-order table.
    codebook_path : Path or None
        Source path of the loaded codebook.
    illumination_profiles_path : Path or None
        Source path of the loaded illumination profiles.

    Returns
    -------
    dict[str, Any]
        Normalized UI state used by validation and acquisition setup.
    """

    channel_specs: list[dict[str, Any]] | None = None
    if sequence is not None:
        channel_specs = channel_specs_from_sequence_wavelength_rows(
            sequence, wavelength_rows or []
        )

    return {
        "mode": mode,
        "sequence": sequence,
        "tile_overlap": tile_overlap,
        "tile_overlap_error": tile_overlap_error,
        "selected_single_round": selected_single_round,
        "channel_specs": channel_specs,
        "fluidics_program": fluidics_program,
        "exp_order": exp_order,
        "codebook": codebook,
        "illumination_profiles": illumination_profiles,
        "use_uniform_illumination": bool(use_uniform_illumination),
        "core_metadata": core_metadata,
        "core_metadata_error": core_metadata_error,
        "reference_tile": int(reference_tile),
        "enable_drift_correction": bool(enable_drift_correction),
        "simulate_pump": bool(simulate_pump),
        "simulate_valves": bool(simulate_valves),
        "num_simulated_valves": int(num_simulated_valves),
        "pump_com_port": str(pump_com_port),
        "valve_com_port": str(valve_com_port),
        "microscope_type": str(microscope_type),
        "numerical_aperture": float(numerical_aperture),
        "refractive_index": float(refractive_index),
        "exp_order_path": exp_order_path,
        "codebook_path": codebook_path,
        "illumination_profiles_path": illumination_profiles_path,
    }
