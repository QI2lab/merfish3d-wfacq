from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.merfish_builders import FITC_RHODAMINE_CY5_SEQUENCE, TWO_ROUND_EXP_ORDER
from useq import CustomAction, MDASequence

from merfish3d_wfacq.sequence import MERFISH_SETUP_ACTION_NAME, RunMode
from merfish3d_wfacq.utils.data_io import read_exp_order
from merfish3d_wfacq.workflow import (
    build_merfish_metadata,
    normalize_merfish_ui_state,
    prepare_merfish_acquisition,
)

CORE_METADATA: dict[str, Any] = {
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

FITC_RHODAMINE_CY5_CHANNEL_SPECS: list[dict[str, Any]] = [
    {
        "channel_index": 0,
        "config_name": "FITC",
        "role": "fiducial",
        "excitation_um": 0.488,
        "emission_um": 0.520,
    },
    {
        "channel_index": 1,
        "config_name": "Rhodamine",
        "role": "readout",
        "excitation_um": 0.561,
        "emission_um": 0.590,
    },
    {
        "channel_index": 2,
        "config_name": "Cy5",
        "role": "readout",
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
]

PERMUTED_CHANNEL_SPECS: list[dict[str, Any]] = [
    {
        "channel_index": 0,
        "config_name": "Cy5",
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
    {
        "channel_index": 1,
        "config_name": "FITC",
        "excitation_um": 0.488,
        "emission_um": 0.520,
    },
    {
        "channel_index": 2,
        "config_name": "Rhodamine",
        "excitation_um": 0.561,
        "emission_um": 0.590,
    },
]

ITERATIVE_FLUIDICS_PROGRAM = pd.DataFrame(
    {
        "round": [1, 1, 2, 2],
        "source": ["B01", "RUN", "B02", "RUN"],
        "time": [0.1, 0.0, 0.1, 0.0],
        "pump": [10.0, 0.0, 10.0, 0.0],
    }
)

ITERATIVE_CODEBOOK = pd.DataFrame(
    {
        "gene_id": ["gene"],
        "bit01": [1],
        "bit02": [0],
        "bit03": [1],
        "bit04": [0],
    }
)

ITERATIVE_ILLUMINATION_PROFILES = np.ones((3, 4, 4), dtype=np.float32)

FLUIDICS_ONLY_PROGRAM_NO_RUN = pd.DataFrame(
    {
        "round": [1, 2],
        "source": ["B01", "B02"],
        "time": [0.1, 0.2],
        "pump": [10.0, 12.0],
    }
)


def _sequence() -> MDASequence:
    return FITC_RHODAMINE_CY5_SEQUENCE.model_copy(deep=True)


def _two_round_exp_order() -> pd.DataFrame:
    return TWO_ROUND_EXP_ORDER.copy(deep=True)


def _ui_state(
    *,
    mode: RunMode,
    sequence: MDASequence | None,
    fluidics_program: pd.DataFrame | None = None,
    exp_order: pd.DataFrame | None = None,
    codebook: pd.DataFrame | None = None,
    illumination_profiles: np.ndarray | None = None,
    use_uniform_illumination: bool = False,
    tile_overlap: float | None = 0.2,
    tile_overlap_error: str | None = None,
    selected_single_round: int | None = None,
    channel_specs: list[dict[str, Any]] | None = None,
    reference_tile: int = 4,
) -> dict[str, object]:
    return {
        "mode": mode,
        "sequence": sequence,
        "fluidics_program": fluidics_program,
        "exp_order": exp_order,
        "codebook": codebook,
        "illumination_profiles": illumination_profiles,
        "use_uniform_illumination": use_uniform_illumination,
        "core_metadata": CORE_METADATA,
        "core_metadata_error": None,
        "tile_overlap": tile_overlap,
        "tile_overlap_error": tile_overlap_error,
        "selected_single_round": selected_single_round,
        "channel_specs": list(channel_specs or FITC_RHODAMINE_CY5_CHANNEL_SPECS),
        "reference_tile": int(reference_tile),
        "enable_drift_correction": True,
        "simulate_pump": True,
        "simulate_valves": True,
        "num_simulated_valves": 4,
        "pump_com_port": "COM3",
        "valve_com_port": "COM4",
        "microscope_type": "3D",
        "numerical_aperture": 1.35,
        "refractive_index": 1.51,
        "exp_order_path": None,
        "codebook_path": None,
        "illumination_profiles_path": None,
    }


def test_normalize_merfish_ui_state_accepts_uniform_illumination_without_profile() -> None:
    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=RunMode.ITERATIVE,
            sequence=_sequence(),
            fluidics_program=ITERATIVE_FLUIDICS_PROGRAM,
            exp_order=_two_round_exp_order(),
            codebook=ITERATIVE_CODEBOOK,
            illumination_profiles=None,
            use_uniform_illumination=True,
            channel_specs=FITC_RHODAMINE_CY5_CHANNEL_SPECS,
        )
    )

    assert bool(normalized_ui_state["use_uniform_illumination"]) is True


def test_normalize_merfish_ui_state_accepts_fluidics_only_without_run_commands() -> None:
    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=RunMode.FLUIDICS_ONLY,
            sequence=None,
            fluidics_program=FLUIDICS_ONLY_PROGRAM_NO_RUN,
            exp_order=None,
            codebook=None,
            illumination_profiles=None,
            use_uniform_illumination=True,
            channel_specs=[],
        )
    )

    assert normalized_ui_state["mode"] is RunMode.FLUIDICS_ONLY
    assert normalized_ui_state["rounds"] == [1, 2]
    assert normalized_ui_state["experiment_order"] == {}


def test_build_merfish_metadata_uses_normalized_core_and_widget_values() -> None:
    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=RunMode.ITERATIVE,
            sequence=_sequence(),
            fluidics_program=ITERATIVE_FLUIDICS_PROGRAM,
            exp_order=_two_round_exp_order(),
            codebook=ITERATIVE_CODEBOOK,
            illumination_profiles=ITERATIVE_ILLUMINATION_PROFILES,
            channel_specs=FITC_RHODAMINE_CY5_CHANNEL_SPECS,
        )
    )
    metadata, sequence = build_merfish_metadata(normalized_ui_state)

    assert sequence == _sequence()
    assert metadata["rounds"] == [1, 2]
    assert metadata["experiment_order"] == {
        1: {"FITC": 0, "Rhodamine": 1, "Cy5": 2},
        2: {"FITC": 0, "Rhodamine": 3, "Cy5": 4},
    }
    assert metadata["fluidics_program_records"] == [
        {"round": 1, "source": "B01", "time": 0.1, "pump": 10.0},
        {"round": 1, "source": "RUN", "time": 0.0, "pump": 0.0},
        {"round": 2, "source": "B02", "time": 0.1, "pump": 10.0},
        {"round": 2, "source": "RUN", "time": 0.0, "pump": 0.0},
    ]
    assert metadata["default_pixel_size_um"] == 0.108
    assert metadata["z_step_um"] == 1.0
    assert metadata["tile_overlap"] == 0.2
    assert metadata["e_per_adu"] == 0.54
    assert metadata["camera_offset_adu"] == 100.0
    assert metadata["affine_zyx_px"] == CORE_METADATA["affine_zyx_px"]
    assert metadata["illumination_profiles_mode"] == "measured"


def test_normalize_merfish_ui_state_infers_fiducial_from_exp_order_name() -> None:
    sequence = MDASequence(
        channels=[
            {"config": "Cy5", "exposure": 10.0},
            {"config": "FITC", "exposure": 10.0},
            {"config": "Rhodamine", "exposure": 10.0},
        ],
        stage_positions=[(0.0, 0.0, 0.0)],
    )
    exp_order = pd.DataFrame(
        {
            "round": [1, 2],
            "FITC": [0, 0],
            "Rhodamine": [1, 3],
            "Cy5": [2, 4],
        }
    )

    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=RunMode.ITERATIVE,
            sequence=sequence,
            fluidics_program=ITERATIVE_FLUIDICS_PROGRAM,
            exp_order=exp_order,
            codebook=ITERATIVE_CODEBOOK,
            illumination_profiles=ITERATIVE_ILLUMINATION_PROFILES,
            channel_specs=PERMUTED_CHANNEL_SPECS,
        )
    )
    metadata, _ = build_merfish_metadata(normalized_ui_state)

    assert normalized_ui_state["fiducial_channel_name"] == "FITC"
    assert normalized_ui_state["fiducial_channel_index"] == 1
    assert [spec["role"] for spec in normalized_ui_state["channel_specs"]] == [
        "readout",
        "fiducial",
        "readout",
    ]
    assert metadata["fiducial_channel_name"] == "FITC"
    assert metadata["fiducial_channel_index"] == 1


def test_build_merfish_metadata_marks_uniform_illumination_mode() -> None:
    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=RunMode.ITERATIVE,
            sequence=_sequence(),
            fluidics_program=ITERATIVE_FLUIDICS_PROGRAM,
            exp_order=_two_round_exp_order(),
            codebook=ITERATIVE_CODEBOOK,
            illumination_profiles=None,
            use_uniform_illumination=True,
            channel_specs=FITC_RHODAMINE_CY5_CHANNEL_SPECS,
        )
    )
    metadata, _ = build_merfish_metadata(normalized_ui_state)

    assert metadata["illumination_profiles_mode"] == "uniform"
    assert metadata["illumination_profiles_path"] == "<uniform>"
    assert "illumination_profiles" not in metadata


@pytest.mark.parametrize(
    ("run_mode", "save_name", "expect_writer"),
    [
        (RunMode.ITERATIVE, "synthetic_run.ome.zarr", True),
        (RunMode.FLUIDICS_ONLY, None, False),
    ],
)
def test_prepare_merfish_acquisition_handles_imaging_and_fluidics_only_modes(
    workspace_tmp_path: Path,
    run_mode: RunMode,
    save_name: str | None,
    expect_writer: bool,
) -> None:
    normalized_ui_state = normalize_merfish_ui_state(
        _ui_state(
            mode=run_mode,
            sequence=(
                _sequence()
                if run_mode is not RunMode.FLUIDICS_ONLY
                else None
            ),
            fluidics_program=(
                ITERATIVE_FLUIDICS_PROGRAM
                if run_mode is not RunMode.FLUIDICS_ONLY
                else FLUIDICS_ONLY_PROGRAM_NO_RUN
            ),
            exp_order=(
                _two_round_exp_order() if run_mode is not RunMode.FLUIDICS_ONLY else None
            ),
            codebook=(
                ITERATIVE_CODEBOOK if run_mode is not RunMode.FLUIDICS_ONLY else None
            ),
            illumination_profiles=(
                ITERATIVE_ILLUMINATION_PROFILES
                if run_mode is not RunMode.FLUIDICS_ONLY
                else None
            ),
            use_uniform_illumination=(run_mode is RunMode.FLUIDICS_ONLY),
            channel_specs=(
                FITC_RHODAMINE_CY5_CHANNEL_SPECS
                if run_mode is not RunMode.FLUIDICS_ONLY
                else []
            ),
            reference_tile=0,
        )
    )
    merfish_metadata, base_sequence = build_merfish_metadata(normalized_ui_state)
    save_path = workspace_tmp_path / save_name if save_name is not None else None
    setup_payload = {"drift_reference_store_id": "runtime"} if expect_writer else None

    events, runtime_metadata, experiment_root = prepare_merfish_acquisition(
        base_sequence=base_sequence,
        merfish_metadata=merfish_metadata,
        save_path=save_path,
        overwrite=True,
        setup_payload=setup_payload,
    )

    assert isinstance(events, list)
    assert isinstance(runtime_metadata, dict)
    assert events
    assert isinstance(events[0].action, CustomAction)
    assert events[0].action.name == MERFISH_SETUP_ACTION_NAME

    setup_metadata = dict(events[0].action.data["merfish_metadata"])
    if expect_writer:
        assert experiment_root == workspace_tmp_path / "synthetic_run"
        assert setup_metadata["experiment_root"] == str(experiment_root)
        assert setup_metadata["datastore_root"] == str(
            experiment_root / "qi2labdatastore"
        )
        assert runtime_metadata["experiment_root"] == str(experiment_root)
    else:
        assert experiment_root is None
        assert "experiment_root" not in setup_metadata
        assert "datastore_root" not in setup_metadata

    image_events = [
        event
        for event in events
        if not isinstance(getattr(event, "action", None), CustomAction)
    ]
    assert bool(image_events) is expect_writer


@pytest.mark.parametrize(
    "real_fluidics_case",
    ["initial", "full"],
    indirect=True,
)
def test_normalize_merfish_ui_state_uses_run_anchored_fluidics_rounds(
    workspace_tmp_path: Path,
    real_fluidics_case: dict[str, object],
) -> None:
    exp_order_path = workspace_tmp_path / "exp_order.csv"
    exp_order_path.write_text(str(real_fluidics_case["exp_order_text"]), encoding="utf-8")
    exp_order = read_exp_order(exp_order_path)
    ui_state = _ui_state(
        mode=RunMode.ITERATIVE,
        sequence=_sequence(),
        fluidics_program=real_fluidics_case["program"],
        exp_order=exp_order,
        codebook=pd.DataFrame(
            {
                "gene_id": ["gene"],
                **{f"bit{index:02d}": [index % 2] for index in range(1, 17)},
            }
        ),
        illumination_profiles=np.ones((3, 4, 4), dtype=np.float32),
        channel_specs=FITC_RHODAMINE_CY5_CHANNEL_SPECS,
    )

    if real_fluidics_case["iterative_error"] is None:
        normalized_ui_state = normalize_merfish_ui_state(ui_state)
        assert normalized_ui_state["rounds"] == real_fluidics_case["expected_imaging_rounds"]
    else:
        with pytest.raises(ValueError) as exc_info:
            normalize_merfish_ui_state(ui_state)
        assert str(real_fluidics_case["iterative_error"]) in str(exc_info.value)






