from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy import asarray, clip, finfo, iinfo, maximum
from numpy import ndarray as NDArray
from tests.merfish_builders import (
    DEMO_CHANNEL_SPECS,
    write_demo_codebook,
    write_demo_exp_order,
    write_demo_illumination_profiles,
    write_fluidics_program,
    write_refresh_fluidics_program,
)
from useq import MDASequence

from merfish3d_wfacq.core_metadata import derive_core_metadata
from merfish3d_wfacq.dispatch import prepare_merfish_dispatch
from merfish3d_wfacq.engine import MerfishMDAEngine
from merfish3d_wfacq.sequence import (
    MERFISH_EVENT_TARGET_KEY,
    RunMode,
    channel_index_key,
)
from merfish3d_wfacq.ui_state import build_merfish_ui_state
from merfish3d_wfacq.workflow import normalize_merfish_ui_state

TEST_CORE_METADATA: dict[str, Any] = {
    "camera_model": "DemoCamera",
    "pixel_size_um": 0.108,
    "pixel_size_affine": (0.108, 0.0, 0.0, 0.0, 0.108, 0.0),
    "affine_zyx_px": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "binning": 1,
    "e_per_adu": 0.54,
    "camera_offset_adu": 100.0,
    "z_step_um": 1.0,
}

CHANNEL_WAVELENGTHS_BY_NAME = {
    "DAPI": (0.405, 0.450),
    "FITC": (0.488, 0.520),
    "Fiducial-488": (0.488, 0.520),
    "Rhodamine": (0.561, 0.590),
    "Readout-561": (0.561, 0.590),
    "Cy5": (0.647, 0.680),
    "Readout-647": (0.647, 0.680),
}


def channel_wavelengths_for_name(config_name: str) -> tuple[float, float]:
    return CHANNEL_WAVELENGTHS_BY_NAME[str(config_name)]



def wavelength_rows_from_channel_specs(
    channel_specs: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    ordered_specs = sorted(channel_specs, key=channel_index_key)
    return [
        (
            str(round(float(spec["excitation_um"]) * 1000)),
            str(round(float(spec["emission_um"]) * 1000)),
        )
        for spec in ordered_specs
    ]


def sequence_from_channel_specs(
    channel_specs: list[dict[str, Any]],
    *,
    stage_positions: list[tuple[float, float, float]],
    z_plan: dict[str, Any] | None = None,
) -> MDASequence:
    ordered_specs = sorted(channel_specs, key=channel_index_key)
    return MDASequence(
        channel_group="Channel",
        channels=[
            {"config": str(spec["config_name"]), "exposure": 10.0 * (index + 1)}
            for index, spec in enumerate(ordered_specs)
        ],
        stage_positions=stage_positions,
        z_plan=z_plan,
    )


def prepare_dispatch_inputs(
    *,
    mode: RunMode,
    sequence: MDASequence | None,
    fluidics_program: pd.DataFrame | None,
    exp_order: pd.DataFrame | None,
    codebook: pd.DataFrame | None,
    illumination_profiles: np.ndarray | None,
    use_uniform_illumination: bool,
    core_metadata: dict[str, Any] | None = None,
    tile_overlap: float | None = 0.2,
    selected_single_round: int | None = None,
    save_path: Path | None = None,
    overwrite: bool = True,
    reference_tile: int = 0,
    enable_drift_correction: bool = True,
    simulate_pump: bool = True,
    simulate_valves: bool = True,
    num_simulated_valves: int = 4,
    pump_com_port: str = "COM3",
    valve_com_port: str = "COM4",
    microscope_type: str = "3D",
    numerical_aperture: float = 1.35,
    refractive_index: float = 1.51,
    exp_order_path: Path | None = None,
    codebook_path: Path | None = None,
    illumination_profiles_path: Path | None = None,
) -> dict[str, Any]:
    channel_specs: list[dict[str, Any]] = []
    if sequence is not None:
        channels = list(sequence.channels)
        if illumination_profiles is not None and illumination_profiles.shape[0] != len(channels):
            raise ValueError("Illumination profiles must match the number of sequence channels.")
        if exp_order is not None:
            exp_columns = [column for column in exp_order.columns if column != "round"]
            channel_names = [str(channel.config or f"channel_{index}") for index, channel in enumerate(channels)]
            if set(exp_columns) != set(channel_names):
                raise ValueError("exp_order columns must match the sequence channel names for test setup.")
        channel_specs = [
            {
                "channel_index": int(index),
                "config_name": str(channel.config or f"channel_{index}"),
                "excitation_um": channel_wavelengths_for_name(
                    str(channel.config or f"channel_{index}")
                )[0],
                "emission_um": channel_wavelengths_for_name(
                    str(channel.config or f"channel_{index}")
                )[1],
            }
            for index, channel in enumerate(channels)
        ]

    ui_state = build_merfish_ui_state(
        mode=mode,
        sequence=sequence,
        tile_overlap=tile_overlap,
        tile_overlap_error=None,
        wavelength_rows=(wavelength_rows_from_channel_specs(channel_specs) if channel_specs else None),
        selected_single_round=selected_single_round,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        codebook=codebook,
        illumination_profiles=illumination_profiles,
        use_uniform_illumination=use_uniform_illumination,
        core_metadata=dict(core_metadata or TEST_CORE_METADATA),
        core_metadata_error=None,
        reference_tile=reference_tile,
        enable_drift_correction=enable_drift_correction,
        simulate_pump=simulate_pump,
        simulate_valves=simulate_valves,
        num_simulated_valves=num_simulated_valves,
        pump_com_port=pump_com_port,
        valve_com_port=valve_com_port,
        microscope_type=microscope_type,
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
        exp_order_path=exp_order_path,
        codebook_path=codebook_path,
        illumination_profiles_path=illumination_profiles_path,
    )
    normalized_ui_state = normalize_merfish_ui_state(ui_state)
    events, output = prepare_merfish_dispatch(
        normalized_ui_state=normalized_ui_state,
        save_path=save_path,
        overwrite=overwrite,
    )
    runtime_metadata = dict(events[0].action.data["merfish_metadata"])
    return {
        "events": events,
        "output": output,
        "runtime_metadata": runtime_metadata,
        "normalized_ui_state": normalized_ui_state,
    }


def demo_stage_positions() -> list[tuple[float, float, float]]:
    step_um = 4.0 * (1.0 - 0.2)
    positions: list[tuple[float, float, float]] = []
    for row_index in range(3):
        for column_index in range(3):
            positions.append((column_index * step_um, row_index * step_um, 0.0))
    return positions


def demo_round_drifts() -> dict[int, float]:
    return {
        1: 0.00,
        2: 0.13,
        3: -0.08,
        4: 0.21,
        5: -0.16,
        6: 0.34,
        7: -0.27,
        8: 0.42,
    }


def prepared_targets_by_image_path(events: list[Any]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for event in events:
        target = event.metadata.get(MERFISH_EVENT_TARGET_KEY) if getattr(event, "metadata", None) else None
        if target is None:
            continue
        targets.setdefault(str(target["image_relpath"]), dict(target))
    return targets


def prepared_targets_by_folder(events: list[Any]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for event in events:
        target = event.metadata.get(MERFISH_EVENT_TARGET_KEY) if getattr(event, "metadata", None) else None
        if target is None:
            continue
        targets.setdefault(str(target["folder_relpath"]), dict(target))
    return targets


class DeterministicDriftEstimator:
    def __init__(self, *, absolute_offsets_um: dict[int, float]) -> None:
        self._absolute_offsets_um = {
            int(round_label): float(offset)
            for round_label, offset in absolute_offsets_um.items()
        }
        self.reference_stacks_seen: list[np.ndarray] = []
        self.moving_stacks_seen: list[np.ndarray] = []
        self._call_count = 0

    def estimate(
        self,
        reference_stack: np.ndarray,
        moving_stack: np.ndarray,
        *,
        z_step_um: float,
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        assert z_step_um == pytest.approx(1.0)
        assert current_offset_um == pytest.approx(0.0)
        assert moving_stack.shape[0] == 3
        self.reference_stacks_seen.append(asarray(reference_stack).copy())
        self.moving_stacks_seen.append(asarray(moving_stack).copy())
        self._call_count += 1
        absolute_offset_um = self._absolute_offsets_um[self._call_count + 1]
        return {
            "shift_planes": 0,
            "shift_um": absolute_offset_um,
            "absolute_offset_um": absolute_offset_um,
        }


class DemoCameraFrameRecorder:
    def __init__(self) -> None:
        self.raw_stacks: dict[str, list[np.ndarray]] = {}
        self.stage_positions: dict[str, list[float | None]] = {}

    def sequenceStarted(self, _sequence: object, _meta: dict[str, Any]) -> None:
        self.raw_stacks.clear()
        self.stage_positions.clear()

    def frameReady(self, img: np.ndarray, event: object, meta: dict[str, Any]) -> None:
        target = event.metadata[MERFISH_EVENT_TARGET_KEY]
        key = str(target["image_relpath"])
        self.raw_stacks.setdefault(key, []).append(asarray(img).copy())
        if key not in self.stage_positions:
            position = meta.get("position", {})
            self.stage_positions[key] = [
                float(position.get("z")) if position.get("z") is not None else None,
                float(position.get("y")) if position.get("y") is not None else None,
                float(position.get("x")) if position.get("x") is not None else None,
            ]

    def sequenceFinished(self, _sequence: object) -> None:
        return None


def expected_corrected_demo_camera_stack(
    raw_stack: NDArray[Any],
    *,
    offset_adu: float,
    e_per_adu: float,
    profile: NDArray[Any],
) -> np.ndarray:
    corrected = maximum(raw_stack.astype(np.float32) - offset_adu, 0.0)
    corrected *= float(e_per_adu)
    corrected = corrected / maximum(profile.astype(np.float32), finfo(np.float32).eps)
    corrected = clip(corrected, 0, iinfo(np.uint16).max)
    return corrected.astype(np.uint16)


def run_demo_multiround_acquisition(
    workspace_tmp_path: Path,
    demo_core: Any,
    *,
    channel_specs: list[dict[str, Any]] | None = None,
    exp_order_column_order: list[str] | None = None,
    tile_overlap: float | None = 0.2,
) -> dict[str, Any]:
    selected_specs = sorted(
        channel_specs or DEMO_CHANNEL_SPECS,
        key=channel_index_key,
    )
    workspace_tmp_path.mkdir(parents=True, exist_ok=True)
    codebook_path = workspace_tmp_path / "codebook.csv"
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    fluidics_path = workspace_tmp_path / "fluidics.csv"
    illumination_path = workspace_tmp_path / "illumination_profiles.ome.tif"
    image_shape = (int(demo_core.getImageHeight()), int(demo_core.getImageWidth()))

    codebook = write_demo_codebook(codebook_path)
    exp_order = write_demo_exp_order(
        exp_order_path,
        channel_specs=selected_specs,
        column_order=exp_order_column_order,
    )
    fluidics_program = write_fluidics_program(fluidics_path, list(range(1, 9)))
    illumination_profiles = write_demo_illumination_profiles(
        illumination_path,
        image_shape,
        channel_specs=selected_specs,
    )

    base_sequence = sequence_from_channel_specs(
        selected_specs,
        stage_positions=demo_stage_positions(),
        z_plan={"range": 2, "step": 1},
    )
    core_metadata = derive_core_metadata(demo_core, base_sequence)
    dispatch = prepare_dispatch_inputs(
        mode=RunMode.ITERATIVE,
        sequence=base_sequence,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        codebook=codebook,
        illumination_profiles=illumination_profiles,
        use_uniform_illumination=False,
        core_metadata=core_metadata,
        tile_overlap=tile_overlap,
        save_path=workspace_tmp_path / "r" / "run.ome.zarr",
        overwrite=False,
        reference_tile=0,
        exp_order_path=exp_order_path,
        codebook_path=codebook_path,
        illumination_profiles_path=illumination_path,
    )
    writer = dispatch["output"]
    runtime_metadata = dispatch["runtime_metadata"]
    events = dispatch["events"]

    round_drifts = demo_round_drifts()
    drift_estimator = DeterministicDriftEstimator(absolute_offsets_um=round_drifts)
    engine = MerfishMDAEngine(demo_core, drift_estimator=drift_estimator)
    raw_recorder = DemoCameraFrameRecorder()
    demo_core.register_mda_engine(engine)
    demo_core.run_mda(events, output=[writer, raw_recorder], block=True)

    assert writer is not None
    prepared_targets = prepared_targets_by_image_path(events)
    prepared_folders = prepared_targets_by_folder(events)
    channel_profile_by_name = {
        str(spec["config_name"]): illumination_profiles[int(spec["channel_index"])]
        for spec in runtime_metadata["channel_specs"]
    }
    return {
        "events": events,
        "datastore_root": writer.output_path,
        "runtime_metadata": runtime_metadata,
        "prepared_targets": prepared_targets,
        "prepared_folders": prepared_folders,
        "channel_profile_by_name": channel_profile_by_name,
        "codebook": codebook,
        "exp_order": exp_order,
        "exp_order_path": exp_order_path,
        "illumination_profiles": illumination_profiles,
        "camera_model": core_metadata["camera_model"],
        "affine_zyx_px": core_metadata["affine_zyx_px"],
        "round_drifts": round_drifts,
        "raw_recorder": raw_recorder,
        "drift_estimator": drift_estimator,
    }


def prepare_refresh_demo_acquisition(
    workspace_tmp_path: Path,
    demo_core: Any,
) -> dict[str, Any]:
    codebook_path = workspace_tmp_path / "codebook.csv"
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    fluidics_path = workspace_tmp_path / "fluidics.csv"
    illumination_path = workspace_tmp_path / "illumination_profiles.ome.tif"
    image_shape = (int(demo_core.getImageHeight()), int(demo_core.getImageWidth()))

    codebook = write_demo_codebook(codebook_path)
    exp_order = write_demo_exp_order(exp_order_path).iloc[[0]].reset_index(drop=True)
    fluidics_program = write_refresh_fluidics_program(fluidics_path)
    illumination_profiles = write_demo_illumination_profiles(illumination_path, image_shape)

    base_sequence = sequence_from_channel_specs(
        list(DEMO_CHANNEL_SPECS),
        stage_positions=[(0.0, 0.0, 0.0)],
        z_plan={"range": 0, "step": 1},
    )
    core_metadata = derive_core_metadata(demo_core, base_sequence)
    dispatch = prepare_dispatch_inputs(
        mode=RunMode.ITERATIVE,
        sequence=base_sequence,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        codebook=codebook,
        illumination_profiles=illumination_profiles,
        use_uniform_illumination=False,
        core_metadata=core_metadata,
        tile_overlap=0.2,
        save_path=workspace_tmp_path / "refresh_run.ome.zarr",
        overwrite=True,
        reference_tile=0,
        exp_order_path=exp_order_path,
        codebook_path=codebook_path,
        illumination_profiles_path=illumination_path,
    )
    return {
        "events": dispatch["events"],
        "writer": dispatch["output"],
        "raw_recorder": DemoCameraFrameRecorder(),
    }


