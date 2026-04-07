import json
import shutil
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import _mmcore_plus
from tests.merfish_builders import DEMO_CHANNEL_SPECS, read_zarr3_array
from tests.merfish_test_utils import (
    expected_corrected_demo_camera_stack,
    prepare_refresh_demo_acquisition,
    run_demo_multiround_acquisition,
)

from merfish3d_wfacq.engine import MerfishMDAEngine
from merfish3d_wfacq.sequence import channel_index_key

PERMUTED_DEMO_CHANNEL_SPECS = [
    {
        "channel_index": 0,
        "config_name": "Cy5",
        "role": "readout",
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
    {
        "channel_index": 1,
        "config_name": "DAPI",
        "role": "fiducial",
        "excitation_um": 0.405,
        "emission_um": 0.450,
    },
    {
        "channel_index": 2,
        "config_name": "Rhodamine",
        "role": "readout",
        "excitation_um": 0.561,
        "emission_um": 0.590,
    },
]

INTEGRATION_CASES = [
    {
        "id": "canonical",
        "channel_specs": list(DEMO_CHANNEL_SPECS),
        "exp_order_column_order": ["DAPI", "Rhodamine", "Cy5"],
        "tile_overlap": 0.2,
        "expected_tile_overlap": 0.2,
    },
    {
        "id": "permuted_no_overlap",
        "channel_specs": list(PERMUTED_DEMO_CHANNEL_SPECS),
        "exp_order_column_order": ["Rhodamine", "Cy5", "DAPI"],
        "tile_overlap": None,
        "expected_tile_overlap": 0.0,
    },
]


def _integration_case_id(case: dict[str, Any]) -> str:
    return str(case["id"])


def _integration_case_workspace(case_id: str) -> Path:
    workspace = (
        Path(__file__).resolve().parents[1]
        / "pytest_tmp_env"
        / "integration_cases"
        / str(case_id)
    )
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture(scope="module", params=INTEGRATION_CASES, ids=_integration_case_id)
def demo_multiround_result(
    request: pytest.FixtureRequest,
) -> object:
    case = dict(request.param)
    workspace_tmp_path = _integration_case_workspace(str(case["id"]))
    _mmcore_plus._instance = None
    demo_core = CMMCorePlus()
    demo_core.loadSystemConfiguration("MMConfig_demo.cfg")
    try:
        result = run_demo_multiround_acquisition(
            workspace_tmp_path,
            demo_core,
            channel_specs=case["channel_specs"],
            exp_order_column_order=case["exp_order_column_order"],
            tile_overlap=case["tile_overlap"],
        )
        result["expected_tile_overlap"] = case["expected_tile_overlap"]
        yield result
    finally:
        try:
            demo_core.waitForSystem()
        except Exception:
            pass
        try:
            demo_core.unloadAllDevices()
        except Exception:
            pass
        _mmcore_plus._instance = None

def _specs_by_name(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["config_name"]): dict(spec)
        for spec in result["runtime_metadata"]["channel_specs"]
    }


def _expected_bit_linker(result: dict[str, Any], round_label: int) -> list[int]:
    row = result["exp_order"].set_index("round").loc[int(round_label)]
    readout_channel_names = [
        str(spec["config_name"])
        for spec in sorted(
            result["runtime_metadata"]["channel_specs"],
            key=channel_index_key,
        )
        if str(spec["role"]) == "readout"
    ]
    return [int(row[channel_name]) for channel_name in readout_channel_names]


def test_16_bit_multiround_acquisition_writes_exact_qi2lab_datastore(
    demo_multiround_result: object,
) -> None:
    result = demo_multiround_result
    datastore_root = result["datastore_root"]
    recorder = result["raw_recorder"]
    specs_by_name = _specs_by_name(result)
    runtime_metadata = result["runtime_metadata"]
    offset_adu = float(runtime_metadata["camera_offset_adu"])
    e_per_adu = float(runtime_metadata["e_per_adu"])
    calibrations = json.loads(
        (datastore_root / "calibrations" / "attributes.json").read_text(
            encoding="utf-8"
        )
    )

    assert calibrations["num_rounds"] == 8
    assert calibrations["num_bits"] == 16
    assert calibrations["num_tiles"] == 9
    assert calibrations["tile_overlap"] == pytest.approx(
        demo_multiround_result["expected_tile_overlap"]
    )
    assert calibrations["codebook"] == result["codebook"].values.tolist()
    assert calibrations["codebook_table"] == {
        "columns": [str(column) for column in result["codebook"].columns],
        "records": result["codebook"].to_dict(orient="records"),
    }
    assert calibrations["exp_order"] == result["exp_order"].drop(
        columns=["round"]
    ).values.tolist()
    assert calibrations["experiment_order"] == {
        "columns": [str(column) for column in result["exp_order"].columns],
        "records": result["exp_order"].to_dict(orient="records"),
    }
    assert calibrations["codebook_path"].endswith("codebook.csv")
    assert Path(calibrations["exp_order_path"]).name == result["exp_order_path"].name
    assert calibrations["illumination_profiles_path"].endswith(
        "illumination_profiles.ome.tif"
    )

    actual_image_paths = sorted(
        str(path.relative_to(datastore_root))
        for path in datastore_root.rglob("corrected_data.ome.zarr")
    )
    assert actual_image_paths == sorted(result["prepared_targets"])
    actual_folder_paths = sorted(
        str(path.parent.relative_to(datastore_root))
        for path in datastore_root.rglob("attributes.json")
        if path.parent.relative_to(datastore_root).parts[0] in {"fiducial", "readouts"}
    )
    assert actual_folder_paths == sorted(result["prepared_folders"])

    fiducial_tiles = sorted((datastore_root / "fiducial").glob("tile*"))
    readout_tiles = sorted((datastore_root / "readouts").glob("tile*"))
    assert [path.name for path in fiducial_tiles] == [
        f"tile{tile:04d}" for tile in range(9)
    ]
    assert [path.name for path in readout_tiles] == [
        f"tile{tile:04d}" for tile in range(9)
    ]

    for image_relpath, target in result["prepared_targets"].items():
        target_root = datastore_root / target["folder_relpath"]
        image_root = datastore_root / image_relpath
        target_attrs = json.loads(
            (target_root / "attributes.json").read_text(encoding="utf-8")
        )
        expected_stage_zyx = recorder.stage_positions[image_relpath]
        expected_drift = result["round_drifts"][int(target["round_label"])]
        channel_spec = specs_by_name[str(target["channel_config"])]

        assert target_attrs["stage_zyx_um"] == expected_stage_zyx
        assert target_attrs["affine_zyx_px"] == result["affine_zyx_px"]
        assert target_attrs["excitation_um"] == pytest.approx(
            float(channel_spec["excitation_um"])
        )
        assert target_attrs["emission_um"] == pytest.approx(
            float(channel_spec["emission_um"])
        )
        assert target_attrs["applied_z_offset_um"] == pytest.approx(expected_drift)
        if str(target["image_kind"]) == "fiducial":
            assert target_attrs["bit_label"] == 0
            assert target_attrs["bit_linker"] == _expected_bit_linker(
                result, int(target["round_label"])
            )
        else:
            assert target_attrs["round_linker"] == int(target["round_label"])

        target_zarr = json.loads((image_root / "zarr.json").read_text(encoding="utf-8"))
        frame_metadata = target_zarr["attributes"]["ome_writers"]["frame_metadata"]
        assert frame_metadata[0]["bit"] == int(target["bit_label"])
        assert frame_metadata[0]["applied_z_offset_um"] == pytest.approx(
            expected_drift
        )
        assert frame_metadata[0]["position_z"] == pytest.approx(expected_stage_zyx[0])
        assert frame_metadata[0]["position_y"] == pytest.approx(expected_stage_zyx[1])
        assert frame_metadata[0]["position_x"] == pytest.approx(expected_stage_zyx[2])
        target_array = read_zarr3_array(image_root)
        np.testing.assert_array_equal(
            target_array,
            expected_corrected_demo_camera_stack(
                np.stack(recorder.raw_stacks[image_relpath], axis=0),
                offset_adu=offset_adu,
                e_per_adu=e_per_adu,
                profile=result["channel_profile_by_name"][target["channel_config"]],
            ),
        )

    round1_reference = read_zarr3_array(
        datastore_root
        / "fiducial"
        / "tile0000"
        / "round001"
        / "corrected_data.ome.zarr"
    )
    assert len(result["drift_estimator"].reference_stacks_seen) == 7
    for reference_stack in result["drift_estimator"].reference_stacks_seen:
        np.testing.assert_array_equal(reference_stack, round1_reference)


def test_16_bit_multiround_acquisition_is_readable_by_qi2lab_datastore_api(
    demo_multiround_result: object,
    qi2lab_datastore_cls: type[object],
) -> None:
    result = demo_multiround_result
    datastore = qi2lab_datastore_cls(result["datastore_root"])
    recorder = result["raw_recorder"]
    specs_by_name = _specs_by_name(result)
    runtime_metadata = result["runtime_metadata"]
    offset_adu = float(runtime_metadata["camera_offset_adu"])
    e_per_adu = float(runtime_metadata["e_per_adu"])
    fiducial_targets = {
        (int(target["tile_index"]), int(target["round_label"])): target
        for target in result["prepared_targets"].values()
        if str(target["image_kind"]) == "fiducial"
    }
    readout_targets = {
        (int(target["tile_index"]), int(target["bit_label"])): target
        for target in result["prepared_targets"].values()
        if str(target["image_kind"]) == "readout"
    }

    assert datastore.datastore_state["Version"] == 0.6
    assert datastore.datastore_state["Corrected"] is True
    assert datastore.num_rounds == 8
    assert datastore.num_bits == 16
    assert datastore.num_tiles == 9
    assert datastore.microscope_type == "3D"
    assert datastore.camera_model == result["camera_model"]
    assert datastore.tile_overlap == pytest.approx(
        demo_multiround_result["expected_tile_overlap"]
    )
    assert datastore.e_per_ADU == pytest.approx(e_per_adu)
    assert datastore.na == pytest.approx(1.35)
    assert datastore.ri == pytest.approx(1.51)
    assert datastore.binning == 1
    assert list(datastore.channels_in_data) == [
        str(spec["config_name"])
        for spec in sorted(
            result["runtime_metadata"]["channel_specs"],
            key=channel_index_key,
        )
    ]
    np.testing.assert_allclose(np.asarray(datastore.voxel_size_zyx_um), [1.0, 1.0, 1.0])
    pd.testing.assert_frame_equal(
        datastore.codebook.reset_index(drop=True),
        result["codebook"].reset_index(drop=True),
    )
    np.testing.assert_array_equal(
        datastore.experiment_order.to_numpy(dtype=int),
        result["exp_order"].drop(columns=["round"]).to_numpy(dtype=int),
    )

    for (tile_index, round_label), target in fiducial_targets.items():
        stage_position, affine_zyx_px = datastore.load_local_stage_position_zyx_um(
            tile=tile_index, round=round_label - 1
        )
        np.testing.assert_allclose(
            stage_position,
            recorder.stage_positions[str(target["image_relpath"])],
        )
        np.testing.assert_allclose(np.asarray(affine_zyx_px), result["affine_zyx_px"])
        channel_spec = specs_by_name[str(target["channel_config"])]
        assert datastore.load_local_wavelengths_um(
            tile=tile_index, round=round_label - 1
        ) == pytest.approx(
            (
                float(channel_spec["excitation_um"]),
                float(channel_spec["emission_um"]),
            )
        )
        np.testing.assert_array_equal(
            np.asarray(
                datastore.load_local_corrected_image(
                    tile=tile_index, round=round_label - 1, return_future=False
                )
            ),
            expected_corrected_demo_camera_stack(
                np.stack(recorder.raw_stacks[str(target["image_relpath"])], axis=0),
                offset_adu=offset_adu,
                e_per_adu=e_per_adu,
                profile=result["channel_profile_by_name"][target["channel_config"]],
            ),
        )

    for (tile_index, bit_label), target in readout_targets.items():
        channel_spec = specs_by_name[str(target["channel_config"])]
        assert datastore.load_local_wavelengths_um(
            tile=tile_index, bit=bit_label - 1
        ) == pytest.approx(
            (
                float(channel_spec["excitation_um"]),
                float(channel_spec["emission_um"]),
            )
        )
        np.testing.assert_array_equal(
            np.asarray(
                datastore.load_local_corrected_image(
                    tile=tile_index, bit=bit_label - 1, return_future=False
                )
            ),
            expected_corrected_demo_camera_stack(
                np.stack(recorder.raw_stacks[str(target["image_relpath"])], axis=0),
                offset_adu=offset_adu,
                e_per_adu=e_per_adu,
                profile=result["channel_profile_by_name"][target["channel_config"]],
            ),
        )


def test_refresh_pauses_engine_then_continues(
    demo_core: Any,
    workspace_tmp_path: Path,
) -> None:
    acquisition = prepare_refresh_demo_acquisition(workspace_tmp_path, demo_core)
    events = acquisition["events"]
    writer = acquisition["writer"]

    refresh_called = Event()
    allow_refresh = Event()
    refresh_payload: dict[str, Any] = {}
    statuses: list[str] = []
    logs: list[str] = []
    raw_recorder = acquisition["raw_recorder"]
    failures: list[BaseException] = []

    def _refresh_handler(payload: dict[str, Any]) -> bool:
        refresh_payload.update(payload)
        refresh_called.set()
        return allow_refresh.wait(timeout=20)

    def _run_mda() -> None:
        try:
            demo_core.run_mda(events, output=[writer, raw_recorder], block=True)
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    engine = MerfishMDAEngine(
        demo_core,
        log_callback=logs.append,
        status_callback=statuses.append,
        refresh_handler=_refresh_handler,
    )
    demo_core.register_mda_engine(engine)

    worker = Thread(target=_run_mda, daemon=True)
    worker.start()

    assert refresh_called.wait(timeout=20), "REFRESH step was never reached."
    assert worker.is_alive()
    assert refresh_payload == {"round": 1, "source": "REFRESH", "pause_seconds": 0.0}
    assert raw_recorder.raw_stacks == {}
    assert "Waiting for operator REFRESH confirmation." in statuses

    allow_refresh.set()
    worker.join(timeout=60)

    if failures:
        raise failures[0]
    assert not worker.is_alive(), "Acquisition thread did not complete after REFRESH."
    assert writer.output_path is not None
    assert raw_recorder.raw_stacks
    assert any("Refresh confirmed." in message for message in logs)
    assert any(writer.output_path.rglob("corrected_data.ome.zarr"))




