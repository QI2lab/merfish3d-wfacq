from pathlib import Path
from typing import Any, cast

from useq import Axis, MDAEvent, MDASequence

from merfish3d_wfacq.datastore import resolve_experiment_root
from merfish3d_wfacq.input_metadata import input_file_metadata
from merfish3d_wfacq.sequence import (
    ImageKind,
    RunMode,
    build_merfish_events,
    channel_index_key,
    tile_count_from_sequence,
)
from merfish3d_wfacq.utils.data_io import (
    fluidics_rounds,
    imaging_rounds,
    infer_fiducial_channel_name,
    validate_round_mappings,
)


def normalize_merfish_ui_state(ui_state: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate MERFISH widget state for acquisition setup.

    Parameters
    ----------
    ui_state : dict[str, Any]
        Normalized widget state built by ``build_merfish_ui_state``.

    Returns
    -------
    dict[str, Any]
        Validated run inputs used to build MERFISH metadata.
    """

    mode = ui_state["mode"]
    sequence = ui_state["sequence"]
    fluidics_program = ui_state["fluidics_program"]
    exp_order = ui_state["exp_order"]
    codebook = ui_state["codebook"]
    use_uniform_illumination = bool(ui_state["use_uniform_illumination"])
    illumination_profiles = ui_state["illumination_profiles"]
    core_metadata = ui_state["core_metadata"]
    core_metadata_error = ui_state["core_metadata_error"]
    tile_overlap = ui_state["tile_overlap"]
    selected_single_round = ui_state["selected_single_round"]
    channel_specs = ui_state["channel_specs"]
    reference_tile = int(ui_state["reference_tile"])
    enable_drift_correction = bool(ui_state["enable_drift_correction"])
    simulate_pump = bool(ui_state["simulate_pump"])
    simulate_valves = bool(ui_state["simulate_valves"])
    num_simulated_valves = int(ui_state["num_simulated_valves"])
    pump_com_port = str(ui_state["pump_com_port"])
    valve_com_port = str(ui_state["valve_com_port"])
    microscope_type = str(ui_state["microscope_type"])
    numerical_aperture = float(ui_state["numerical_aperture"])
    refractive_index = float(ui_state["refractive_index"])
    exp_order_path = ui_state["exp_order_path"]
    codebook_path = ui_state["codebook_path"]
    illumination_profiles_path = ui_state["illumination_profiles_path"]

    normalized_tile_overlap = 0.0 if tile_overlap is None else float(tile_overlap)

    if mode is RunMode.FLUIDICS_ONLY:
        if fluidics_program is None:
            raise ValueError("Load a fluidics program for fluidics-only runs.")
        return {
            "mode": mode,
            "sequence": MDASequence(),
            "rounds": fluidics_rounds(fluidics_program),
            "experiment_order": {},
            "reference_tile": reference_tile,
            "enable_drift_correction": enable_drift_correction,
            "simulate_pump": simulate_pump,
            "simulate_valves": simulate_valves,
            "num_simulated_valves": num_simulated_valves,
            "pump_com_port": pump_com_port,
            "valve_com_port": valve_com_port,
            "microscope_type": microscope_type,
            "numerical_aperture": numerical_aperture,
            "refractive_index": refractive_index,
            "channel_specs": [],
            "fiducial_channel_name": None,
            "fiducial_channel_index": 0,
            "tile_overlap": normalized_tile_overlap,
            "fluidics_program": fluidics_program,
            "exp_order": exp_order,
            "codebook": codebook,
            "illumination_profiles": illumination_profiles,
            "use_uniform_illumination": use_uniform_illumination,
            "core_metadata": core_metadata,
            "exp_order_path": exp_order_path,
            "codebook_path": codebook_path,
            "illumination_profiles_path": illumination_profiles_path,
        }

    if sequence is None:
        raise ValueError("MDA widget configuration is invalid.")
    if mode is RunMode.SINGLE_ROUND and fluidics_program is None:
        raise ValueError("Load a fluidics program for single-round imaging.")
    if (
        mode in {RunMode.ITERATIVE, RunMode.SINGLE_ROUND}
        and fluidics_program is not None
        and not imaging_rounds(fluidics_program)
    ):
        raise ValueError(
            "The loaded fluidics program does not contain any RUN commands for imaging."
        )
    if mode is RunMode.SINGLE_ROUND and selected_single_round is None:
        raise ValueError("Select a round from the loaded fluidics program.")

    channel_count = len(sequence.channels)
    if channel_count < 2:
        raise ValueError(
            "MERFISH imaging requires one fiducial channel and at least one readout "
            "channel."
        )

    if exp_order is None:
        raise ValueError("Load an experiment order file for imaging runs.")
    if codebook is None:
        raise ValueError("Load a codebook for imaging runs.")
    if illumination_profiles is None and not use_uniform_illumination:
        raise ValueError(
            "Load illumination profiles or enable the uniform / unknown illumination option for imaging runs."
        )
    if (
        not use_uniform_illumination
        and illumination_profiles is not None
        and illumination_profiles.shape[0] != channel_count
    ):
        raise ValueError(
            "Illumination-profile channel count must match the number of active MDA "
            f"channels. Expected {channel_count}, found {illumination_profiles.shape[0]}."
        )
    if mode is RunMode.ITERATIVE and fluidics_program is None:
        raise ValueError("Load a fluidics program for iterative imaging.")
    if core_metadata is None:
        raise ValueError(
            core_metadata_error
            or "Required metadata could not be derived from Micro-Manager core."
        )
    normalized_tile_overlap = 0.0 if tile_overlap is None else float(tile_overlap)
    if normalized_tile_overlap < 0.0 or normalized_tile_overlap > 1.0:
        raise ValueError(
            "Set the Stage Explorer overlap between 0% and 100% for MERFISH runs."
        )
    if channel_specs is None:
        raise ValueError("Channel wavelength metadata is incomplete.")

    expected_channel_names = [
        str(spec["config_name"])
        for spec in sorted(channel_specs, key=channel_index_key)
    ]
    experiment_order = validate_round_mappings(
        run_mode=mode.value,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        selected_single_round=selected_single_round,
        expected_channel_names=expected_channel_names,
    )
    fiducial_channel_name = infer_fiducial_channel_name(exp_order)
    channel_specs_by_name = {
        str(spec["config_name"]): dict(spec) for spec in channel_specs
    }
    fiducial_channel_index = int(
        channel_specs_by_name[fiducial_channel_name]["channel_index"]
    )
    resolved_channel_specs = []
    for spec in sorted(channel_specs, key=channel_index_key):
        resolved_spec = dict(spec)
        resolved_spec["role"] = (
            ImageKind.FIDUCIAL.value
            if str(resolved_spec["config_name"]) == fiducial_channel_name
            else ImageKind.READOUT.value
        )
        resolved_channel_specs.append(resolved_spec)

    max_bit = max(
        (
            int(bit)
            for round_record in experiment_order.values()
            for bit in round_record.values()
        ),
        default=0,
    )
    if max_bit > len(codebook.columns) - 1:
        raise ValueError(
            "The codebook does not contain enough bit columns for the experiment order file. "
            f"Maximum bit index is {max_bit}, but the codebook only has "
            f"{len(codebook.columns) - 1} bit columns."
        )

    return {
        "mode": mode,
        "sequence": sequence,
        "rounds": [int(round_id) for round_id in experiment_order],
        "experiment_order": experiment_order,
        "reference_tile": reference_tile,
        "enable_drift_correction": enable_drift_correction,
        "simulate_pump": simulate_pump,
        "simulate_valves": simulate_valves,
        "num_simulated_valves": num_simulated_valves,
        "pump_com_port": pump_com_port,
        "valve_com_port": valve_com_port,
        "microscope_type": microscope_type,
        "numerical_aperture": numerical_aperture,
        "refractive_index": refractive_index,
        "channel_specs": resolved_channel_specs,
        "fiducial_channel_name": fiducial_channel_name,
        "fiducial_channel_index": fiducial_channel_index,
        "tile_overlap": normalized_tile_overlap,
        "fluidics_program": fluidics_program,
        "exp_order": exp_order,
        "codebook": codebook,
        "illumination_profiles": illumination_profiles,
        "use_uniform_illumination": use_uniform_illumination,
        "core_metadata": core_metadata,
        "exp_order_path": exp_order_path,
        "codebook_path": codebook_path,
        "illumination_profiles_path": illumination_profiles_path,
    }


def build_merfish_metadata(
    normalized_ui_state: dict[str, Any],
) -> tuple[dict[str, Any], MDASequence]:
    """Build normalized MERFISH metadata from normalized widget state.

    Parameters
    ----------
    normalized_ui_state : dict[str, Any]
        Validated widget state returned by ``normalize_merfish_ui_state``.

    Returns
    -------
    tuple[dict[str, Any], MDASequence]
        Normalized MERFISH metadata and the base image sequence.
    """

    mode = normalized_ui_state["mode"]
    sequence = normalized_ui_state["sequence"]
    rounds = normalized_ui_state["rounds"]
    experiment_order = normalized_ui_state["experiment_order"]
    reference_tile = int(normalized_ui_state["reference_tile"])
    enable_drift_correction = bool(normalized_ui_state["enable_drift_correction"])
    simulate_pump = bool(normalized_ui_state["simulate_pump"])
    simulate_valves = bool(normalized_ui_state["simulate_valves"])
    num_simulated_valves = int(normalized_ui_state["num_simulated_valves"])
    pump_com_port = str(normalized_ui_state["pump_com_port"])
    valve_com_port = str(normalized_ui_state["valve_com_port"])
    microscope_type = str(normalized_ui_state["microscope_type"])
    numerical_aperture = float(normalized_ui_state["numerical_aperture"])
    refractive_index = float(normalized_ui_state["refractive_index"])
    specs = list(normalized_ui_state["channel_specs"])
    tile_overlap = float(normalized_ui_state["tile_overlap"])
    fluidics_program = normalized_ui_state["fluidics_program"]
    exp_order = normalized_ui_state["exp_order"]
    codebook = normalized_ui_state["codebook"]
    illumination_profiles = normalized_ui_state["illumination_profiles"]
    use_uniform_illumination = bool(normalized_ui_state["use_uniform_illumination"])
    core_metadata = normalized_ui_state["core_metadata"]
    exp_order_path = normalized_ui_state["exp_order_path"]
    codebook_path = normalized_ui_state["codebook_path"]
    illumination_profiles_path = normalized_ui_state["illumination_profiles_path"]

    metadata: dict[str, Any] = {
        "run_mode": mode.value,
        "rounds": rounds,
        "experiment_order": experiment_order,
        "reference_tile": int(reference_tile),
        "enable_drift_correction": bool(enable_drift_correction),
        "simulate_pump": bool(simulate_pump),
        "simulate_valves": bool(simulate_valves),
        "num_simulated_valves": int(num_simulated_valves),
        "pump_com_port": pump_com_port.strip() or "COM3",
        "valve_com_port": valve_com_port.strip() or "COM4",
        "pump_id": 30,
        "flip_flow_direction": False,
        "settle_time_s": 5.0,
        "fiducial_channel_name": normalized_ui_state["fiducial_channel_name"],
        "fiducial_channel_index": int(normalized_ui_state["fiducial_channel_index"]),
        "channel_specs": [dict(spec) for spec in specs],
        "microscope_type": microscope_type,
        "numerical_aperture": float(numerical_aperture),
        "refractive_index": float(refractive_index),
        "binning": int(core_metadata["binning"]),
        "default_pixel_size_um": float(core_metadata["pixel_size_um"]),
        "camera_model": core_metadata["camera_model"],
        "pixel_size_affine": (
            list(core_metadata["pixel_size_affine"])
            if core_metadata["pixel_size_affine"] is not None
            else None
        ),
        "z_step_um": float(core_metadata["z_step_um"]),
        "voxel_size_zyx_um": [
            float(core_metadata["z_step_um"]),
            float(core_metadata["pixel_size_um"]),
            float(core_metadata["pixel_size_um"]),
        ],
        "tile_overlap": tile_overlap,
        "e_per_adu": float(core_metadata["e_per_adu"]),
        "camera_offset_adu": float(core_metadata["camera_offset_adu"]),
        "affine_zyx_px": core_metadata["affine_zyx_px"],
    }
    metadata.update(
        input_file_metadata(
            exp_order=exp_order,
            codebook=codebook,
            exp_order_path=exp_order_path,
            codebook_path=codebook_path,
            illumination_profiles_path=illumination_profiles_path,
            use_uniform_illumination=use_uniform_illumination,
        )
    )
    if fluidics_program is not None:
        metadata["fluidics_program_records"] = fluidics_program.to_dict(
            orient="records"
        )
    if not use_uniform_illumination and illumination_profiles is not None:
        metadata["illumination_profiles"] = illumination_profiles.tolist()
    return metadata, sequence


def prepare_merfish_acquisition(
    *,
    base_sequence: MDASequence,
    merfish_metadata: dict[str, Any],
    save_path: str | Path | None,
    overwrite: bool = True,
    setup_payload: dict[str, Any] | None = None,
) -> tuple[list[MDAEvent], dict[str, Any], Path | None]:
    """Prepare the MERFISH run as normalized metadata plus events.

    Parameters
    ----------
    base_sequence : MDASequence
        Base MDA sequence from the upstream widget.
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    save_path : str or Path or None
        Upstream save path selected by the MDA widget.
    overwrite : bool, optional
        Whether to overwrite an existing experiment root.
    setup_payload : dict[str, Any] or None, optional
        Additional JSON-serializable setup payload stamped onto the initial
        setup action.

    Returns
    -------
    tuple[list[MDAEvent], dict[str, Any], Path | None]
        Prepared event list, normalized runtime metadata, and resolved
        experiment root.
    """

    run_mode = RunMode(str(merfish_metadata["run_mode"]))
    if run_mode is RunMode.FLUIDICS_ONLY:
        return _prepare_fluidics_only_acquisition(
            merfish_metadata,
            setup_payload=setup_payload,
        )
    return _prepare_imaging_acquisition(
        base_sequence=base_sequence,
        merfish_metadata=merfish_metadata,
        save_path=cast("str | Path", save_path),
        overwrite=overwrite,
        setup_payload=setup_payload,
    )


def _prepare_fluidics_only_acquisition(
    merfish_metadata: dict[str, Any],
    *,
    setup_payload: dict[str, Any] | None = None,
) -> tuple[list[MDAEvent], dict[str, Any], None]:
    """Prepare a fluidics-only MERFISH run.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    setup_payload : dict[str, Any] or None, optional
        Additional JSON-serializable setup payload stamped onto the initial
        setup action.

    Returns
    -------
    tuple[list[MDAEvent], dict[str, Any], None]
        Prepared event list, normalized runtime metadata, and ``None`` for the
        experiment root.
    """

    runtime_metadata = dict(merfish_metadata)
    events = build_merfish_events(
        MDASequence(),
        rounds=[int(round_id) for round_id in runtime_metadata["rounds"]],
        merfish_metadata=runtime_metadata,
        setup_payload=setup_payload,
    )
    return events, runtime_metadata, None


def _prepare_imaging_acquisition(
    *,
    base_sequence: MDASequence,
    merfish_metadata: dict[str, Any],
    save_path: str | Path,
    overwrite: bool,
    setup_payload: dict[str, Any] | None = None,
) -> tuple[list[MDAEvent], dict[str, Any], Path]:
    """Prepare an imaging MERFISH run with normalized runtime metadata.

    Parameters
    ----------
    base_sequence : MDASequence
        Base MDA sequence from the upstream widget.
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    save_path : str or Path
        Upstream save path selected by the MDA widget.
    overwrite : bool
        Whether to overwrite an existing experiment root.
    setup_payload : dict[str, Any] or None, optional
        Additional JSON-serializable setup payload stamped onto the initial
        setup action.

    Returns
    -------
    tuple[list[MDAEvent], dict[str, Any], Path]
        Prepared event list, normalized runtime metadata, and experiment root.
    """

    experiment_root = resolve_experiment_root(save_path, overwrite=overwrite)
    sequence_metadata = dict(merfish_metadata)
    sequence_metadata["num_tiles"] = tile_count_from_sequence(base_sequence)
    sequence_metadata["num_z_planes"] = max(int(base_sequence.sizes.get(Axis.Z, 0)), 1)
    sequence_metadata["experiment_root"] = str(experiment_root)
    sequence_metadata["datastore_root"] = str(experiment_root / "qi2labdatastore")
    events = build_merfish_events(
        base_sequence,
        rounds=[int(round_id) for round_id in sequence_metadata["rounds"]],
        merfish_metadata=sequence_metadata,
        setup_payload=setup_payload,
    )
    return events, sequence_metadata, experiment_root




