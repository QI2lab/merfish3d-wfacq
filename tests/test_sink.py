import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.merfish_builders import (
    DEFAULT_CHANNEL_SPECS,
    read_zarr3_array,
    write_codebook,
    write_exp_order,
    write_illumination_profiles,
)
from tests.merfish_test_utils import (
    TEST_CORE_METADATA,
    prepare_dispatch_inputs,
    sequence_from_channel_specs,
)
from useq import CustomAction

from merfish3d_wfacq.illumination import resolved_illumination_profiles
from merfish3d_wfacq.sequence import (
    MERFISH_EVENT_TARGET_KEY,
    RunMode,
    channel_index_key,
)
from merfish3d_wfacq.sink import MerfishFrameProcessor, Qi2labDatastoreWriter


def _writer_inputs(
    workspace_tmp_path: Path,
    *,
    run_mode: RunMode = RunMode.SINGLE_ROUND,
    rounds: list[int] | None = None,
    experiment_order: dict[int, list[int]] | dict[int, dict[str, int]] | None = None,
    use_uniform_illumination: bool = False,
    e_per_adu: float = 1.0,
    camera_offset_adu: float = 0.0,
    channel_specs: list[dict[str, Any]] | None = None,
    illumination_profiles: np.ndarray | None = None,
) -> tuple[Any, list[Any], dict[str, Any]]:
    selected_rounds = rounds or [1]
    selected_specs = sorted(
        channel_specs or DEFAULT_CHANNEL_SPECS,
        key=channel_index_key,
    )
    selected_order = experiment_order or {1: [1, 2]}
    codebook_path = workspace_tmp_path / "codebook.csv"
    exp_order_path = workspace_tmp_path / "exp_order.csv"
    illumination_path = workspace_tmp_path / "illumination.ome.tif"

    codebook = write_codebook(codebook_path, 8)
    exp_order = write_exp_order(
        exp_order_path,
        selected_order,
        channel_specs=selected_specs,
    )
    fluidics_program = None
    if run_mode in {RunMode.ITERATIVE, RunMode.SINGLE_ROUND}:
        fluidics_program = pd.DataFrame(
            {
                "round": [int(round_id) for round_id in selected_rounds for _ in (0, 1)],
                "source": [
                    value
                    for round_id in selected_rounds
                    for value in (f"B{int(round_id):02d}", "RUN")
                ],
                "time": [0.1, 0.0] * len(selected_rounds),
                "pump": [10.0, 0.0] * len(selected_rounds),
            }
        )

    profile_stack = illumination_profiles
    if not use_uniform_illumination:
        if profile_stack is None:
            profile_stack = np.ones((len(selected_specs), 4, 4), dtype=np.float32)
        write_illumination_profiles(illumination_path, profile_stack)

    base_sequence = sequence_from_channel_specs(
        selected_specs,
        stage_positions=[(1.0, 2.0, 3.0)],
        z_plan={"range": 2, "step": 1},
    )
    core_metadata = {
        **TEST_CORE_METADATA,
        "e_per_adu": float(e_per_adu),
        "camera_offset_adu": float(camera_offset_adu),
    }
    dispatch = prepare_dispatch_inputs(
        mode=run_mode,
        sequence=base_sequence,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        codebook=codebook,
        illumination_profiles=(None if use_uniform_illumination else profile_stack),
        use_uniform_illumination=use_uniform_illumination,
        core_metadata=core_metadata,
        tile_overlap=0.2,
        selected_single_round=(
            selected_rounds[0] if run_mode is RunMode.SINGLE_ROUND else None
        ),
        save_path=workspace_tmp_path / "run.ome.zarr",
        exp_order_path=exp_order_path,
        codebook_path=codebook_path,
        illumination_profiles_path=(
            None if use_uniform_illumination else illumination_path
        ),
    )
    return base_sequence, dispatch["events"], dispatch["runtime_metadata"]


def test_datastore_writer_creates_v06_layout(workspace_tmp_path: Path) -> None:
    base_sequence, events, metadata = _writer_inputs(workspace_tmp_path)
    image_events = [
        event
        for event in events
        if not isinstance(getattr(event, "action", None), CustomAction)
    ]

    writer = Qi2labDatastoreWriter(merfish_metadata=metadata)
    image_info = {
        "width": 4,
        "height": 4,
        "dtype": "uint16",
        "pixel_size_um": 0.108,
        "camera_label": "DemoCamera",
    }
    writer.set_illumination_profiles(resolved_illumination_profiles(metadata, image_info))
    writer.sequenceStarted(base_sequence, {"image_infos": [image_info]})

    prewrite_attrs = json.loads(
        (
            workspace_tmp_path
            / "run"
            / "qi2labdatastore"
            / "fiducial"
            / "tile0000"
            / "round001"
            / "attributes.json"
        ).read_text(encoding="utf-8")
    )
    assert prewrite_attrs["stage_zyx_um"] == [None, None, None]
    assert prewrite_attrs["affine_zyx_px"] == metadata["affine_zyx_px"]

    for index, event in enumerate(image_events):
        writer.frameReady(
            np.full((4, 4), 100 + index, dtype=np.uint16),
            event,
            {
                "runner_time_ms": 1234.0,
                "exposure_ms": event.exposure or 10.0,
                "position": {"x": 7.0, "y": 8.0, "z": 9.0},
            },
        )
    assert writer._streams == {}
    writer.sequenceFinished(base_sequence)

    datastore_root = workspace_tmp_path / "run" / "qi2labdatastore"

    assert (
        json.loads((datastore_root / "datastore_state.json").read_text(encoding="utf-8"))["Version"]
        == 0.6
    )
    calibrations = json.loads(
        (datastore_root / "calibrations" / "attributes.json").read_text(
            encoding="utf-8"
        )
    )
    assert calibrations["voxel_size_zyx_um"] == pytest.approx([1.0, 0.108, 0.108])
    assert calibrations["tile_overlap"] == pytest.approx(0.2)
    assert calibrations["e_per_ADU"] == pytest.approx(1.0)
    assert calibrations["camera_offset_adu"] == pytest.approx(0.0)
    assert calibrations["channels_in_data"] == [
        "Fiducial-488",
        "Readout-561",
        "Readout-647",
    ]
    assert (datastore_root / "calibrations" / "shading_maps").exists()
    assert (
        datastore_root
        / "fiducial"
        / "tile0000"
        / "round001"
        / "corrected_data.ome.zarr"
        / "zarr.json"
    ).exists()
    assert (
        datastore_root
        / "readouts"
        / "tile0000"
        / "bit001"
        / "corrected_data.ome.zarr"
        / "zarr.json"
    ).exists()

    readout_zarr = json.loads(
        (
            datastore_root
            / "readouts"
            / "tile0000"
            / "bit001"
            / "corrected_data.ome.zarr"
            / "zarr.json"
        ).read_text(encoding="utf-8")
    )
    frame_metadata = readout_zarr["attributes"]["ome_writers"]["frame_metadata"]
    assert frame_metadata[0]["applied_z_offset_um"] == pytest.approx(0.0)
    assert frame_metadata[0]["delta_t"] == pytest.approx(1.234)
    assert frame_metadata[0]["exposure_time"] == pytest.approx(0.02)
    assert frame_metadata[0]["position_z"] == pytest.approx(9.0)
    assert frame_metadata[0]["position_y"] == pytest.approx(8.0)
    assert frame_metadata[0]["position_x"] == pytest.approx(7.0)
    fiducial_attrs = json.loads(
        (
            datastore_root / "fiducial" / "tile0000" / "round001" / "attributes.json"
        ).read_text(encoding="utf-8")
    )
    assert fiducial_attrs["bit_label"] == 0
    assert fiducial_attrs["stage_zyx_um"] == [9.0, 8.0, 7.0]
    assert fiducial_attrs["applied_z_offset_um"] == pytest.approx(0.0)
    readout_attrs = json.loads(
        (
            datastore_root / "readouts" / "tile0000" / "bit001" / "attributes.json"
        ).read_text(encoding="utf-8")
    )
    assert readout_attrs["round_linker"] == 1
    assert readout_attrs["channel_config"] == "Readout-561"
    assert readout_attrs["excitation_um"] == pytest.approx(0.561)
    assert readout_attrs["emission_um"] == pytest.approx(0.590)
    assert readout_attrs["applied_z_offset_um"] == pytest.approx(0.0)


def test_frame_processor_applies_gain_offset_and_shading_correction(
    workspace_tmp_path: Path,
) -> None:
    custom_profiles = np.stack(
        [
            np.ones((4, 4), dtype=np.float32),
            np.full((4, 4), 3.0, dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
        ],
        axis=0,
    )
    base_sequence, events, metadata = _writer_inputs(
        workspace_tmp_path,
        e_per_adu=2.0,
        camera_offset_adu=10.0,
        illumination_profiles=custom_profiles,
    )
    image_events = [
        event
        for event in events
        if not isinstance(getattr(event, "action", None), CustomAction)
    ]

    processor = MerfishFrameProcessor(merfish_metadata=metadata)
    image_info = {
        "width": 4,
        "height": 4,
        "dtype": "uint16",
        "pixel_size_um": 0.108,
        "camera_label": "DemoCamera",
    }
    processor.sequenceStarted(base_sequence, {"image_infos": [image_info]})

    for event in image_events:
        processor.frameReady(
            np.full((4, 4), 100, dtype=np.uint16),
            event,
            {
                "runner_time_ms": 1.0,
                "exposure_ms": 10.0,
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        )
    assert processor.writer._streams == {}
    processor.sequenceFinished(base_sequence)

    corrected_stack = read_zarr3_array(
        workspace_tmp_path
        / "run"
        / "qi2labdatastore"
        / "readouts"
        / "tile0000"
        / "bit001"
        / "corrected_data.ome.zarr"
    )
    assert corrected_stack.shape == (3, 4, 4)
    assert np.all(corrected_stack == 60)


def test_frame_processor_supports_uniform_illumination_profiles(
    workspace_tmp_path: Path,
) -> None:
    base_sequence, events, metadata = _writer_inputs(
        workspace_tmp_path, use_uniform_illumination=True
    )
    image_events = [
        event
        for event in events
        if not isinstance(getattr(event, "action", None), CustomAction)
    ]

    processor = MerfishFrameProcessor(merfish_metadata=metadata)
    image_info = {
        "width": 4,
        "height": 4,
        "dtype": "uint16",
        "pixel_size_um": 0.108,
        "camera_label": "DemoCamera",
    }
    processor.sequenceStarted(base_sequence, {"image_infos": [image_info]})

    readout_event = next(
        event
        for event in image_events
        if int(event.metadata[MERFISH_EVENT_TARGET_KEY]["bit_label"]) > 0
    )
    processor.frameReady(
        np.full((4, 4), 25, dtype=np.uint16),
        readout_event,
        {
            "runner_time_ms": 1.0,
            "exposure_ms": 10.0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    )
    processor.sequenceFinished(base_sequence)

    corrected_stack = read_zarr3_array(
        processor.output_path
        / readout_event.metadata[MERFISH_EVENT_TARGET_KEY]["image_relpath"]
    )
    z_index = int(readout_event.index.get("z", 0))
    assert np.all(corrected_stack[z_index] == 25)

