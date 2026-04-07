from pathlib import Path

import pytest

from merfish3d_wfacq.utils.data_io import (
    append_index_filepath,
    fluidics_rounds,
    imaging_rounds,
    infer_fiducial_channel_name,
    read_exp_order,
    read_fluidics_program,
    validate_round_mappings,
)


def test_read_fluidics_program_and_validate_round_mappings(workspace_tmp_path: Path) -> None:
    fluidics_path = workspace_tmp_path / "fluidics.csv"
    fluidics_path.write_text(
        "round,source,time,pump\n1,B01,0.5,10\n1,RUN,0,0\n2,B02,0.5,12\n2,RUN,0,0\n",
        encoding="utf-8",
    )
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,FITC,Rhodamine,Cy5\n1,0,5,6\n2,0,7,8\n",
        encoding="utf-8",
    )

    program = read_fluidics_program(fluidics_path)
    exp_order = read_exp_order(exp_order_path)
    round_to_bits = validate_round_mappings(
        run_mode="iterative",
        fluidics_program=program,
        exp_order=exp_order,
        expected_channel_names=["FITC", "Rhodamine", "Cy5"],
    )

    assert program["source"].tolist() == ["B01", "RUN", "B02", "RUN"]
    assert round_to_bits == {
        1: {"FITC": 0, "Rhodamine": 5, "Cy5": 6},
        2: {"FITC": 0, "Rhodamine": 7, "Cy5": 8},
    }


def test_read_exp_order_rejects_duplicate_rounds(workspace_tmp_path: Path) -> None:
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,FITC,Rhodamine,Cy5\n1,0,5,6\n1,0,7,8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate round"):
        read_exp_order(exp_order_path)


def test_validate_round_mappings_allows_permuted_channel_columns(workspace_tmp_path: Path) -> None:
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,Cy5,FITC,Rhodamine\n1,6,0,5\n",
        encoding="utf-8",
    )
    exp_order = read_exp_order(exp_order_path)

    round_to_bits = validate_round_mappings(
        run_mode="single_round",
        fluidics_program=None,
        exp_order=exp_order,
        selected_single_round=1,
        expected_channel_names=["FITC", "Rhodamine", "Cy5"],
    )

    assert round_to_bits == {1: {"Cy5": 6, "FITC": 0, "Rhodamine": 5}}


def test_validate_round_mappings_rejects_channel_name_mismatch(workspace_tmp_path: Path) -> None:
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,FITC,Rhodamine,Alexa750\n1,0,5,6\n",
        encoding="utf-8",
    )
    exp_order = read_exp_order(exp_order_path)

    with pytest.raises(ValueError, match="channel columns must match the active MDA channel names"):
        validate_round_mappings(
            run_mode="single_round",
            fluidics_program=None,
            exp_order=exp_order,
            selected_single_round=1,
            expected_channel_names=["FITC", "Rhodamine", "Cy5"],
        )


def test_validate_round_mappings_requires_unique_all_zero_fiducial_column(workspace_tmp_path: Path) -> None:
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,FITC,Rhodamine,Cy5\n1,9,5,6\n",
        encoding="utf-8",
    )
    exp_order = read_exp_order(exp_order_path)

    with pytest.raises(ValueError, match="exactly one fiducial channel column"):
        validate_round_mappings(
            run_mode="single_round",
            fluidics_program=None,
            exp_order=exp_order,
            selected_single_round=1,
            expected_channel_names=["FITC", "Rhodamine", "Cy5"],
        )


def test_infer_fiducial_channel_name_uses_unique_all_zero_column(workspace_tmp_path: Path) -> None:
    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(
        "round,Cy5,FITC,Rhodamine\n1,6,0,5\n2,8,0,7\n",
        encoding="utf-8",
    )
    exp_order = read_exp_order(exp_order_path)

    assert infer_fiducial_channel_name(exp_order) == "FITC"


def test_append_index_filepath_handles_ome_zarr_suffix(workspace_tmp_path: Path) -> None:
    first = workspace_tmp_path / "acq.ome.zarr"
    first.mkdir()

    second = append_index_filepath(first)
    assert second.name == "acq-1.ome.zarr"


@pytest.mark.parametrize(
    "real_fluidics_case",
    ["initial", "full"],
    indirect=True,
)
def test_real_fluidics_files_anchor_imaging_rounds_to_run_commands(
    workspace_tmp_path: Path,
    real_fluidics_case: dict[str, object],
) -> None:
    program = real_fluidics_case["program"]

    assert fluidics_rounds(program) == real_fluidics_case["expected_fluidics_rounds"]
    assert imaging_rounds(program) == real_fluidics_case["expected_imaging_rounds"]

    exp_order_path = workspace_tmp_path / "experiment_order.csv"
    exp_order_path.write_text(str(real_fluidics_case["exp_order_text"]), encoding="utf-8")
    exp_order = read_exp_order(exp_order_path)

    if real_fluidics_case["iterative_error"] is not None:
        with pytest.raises(ValueError, match=str(real_fluidics_case["iterative_error"])):
            validate_round_mappings(
                run_mode="iterative",
                fluidics_program=program,
                exp_order=exp_order,
                expected_channel_names=["FITC", "Rhodamine", "Cy5"],
            )
    else:
        round_to_bits = validate_round_mappings(
            run_mode="iterative",
            fluidics_program=program,
            exp_order=exp_order,
            expected_channel_names=["FITC", "Rhodamine", "Cy5"],
        )
        assert sorted(round_to_bits) == real_fluidics_case["expected_imaging_rounds"]



