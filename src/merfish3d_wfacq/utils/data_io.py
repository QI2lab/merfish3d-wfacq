#!/usr/bin/env python

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

FLUIDICS_REQUIRED_COLUMNS = ("round", "source", "time", "pump")
EXP_ORDER_REQUIRED_COLUMNS = ("round",)


def _native_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python values when possible."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with stripped lowercase column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized column names.
    """

    rename = {col: str(col).strip().lower() for col in df.columns}
    return df.rename(columns=rename)


def _normalize_exp_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return an experiment-order table with stripped channel names.

    Parameters
    ----------
    df : pd.DataFrame
        Input experiment-order table.

    Returns
    -------
    pd.DataFrame
        Dataframe with an exact ``round`` column and stripped channel names.
    """

    rename: dict[Any, str] = {}
    seen: set[str] = set()
    for column in df.columns:
        stripped = str(column).strip()
        normalized = "round" if stripped.lower() == "round" else stripped
        if not normalized:
            raise ValueError("The experiment order file contains an empty column name.")
        if normalized in seen:
            raise ValueError(
                f"The experiment order file contains duplicate column names after normalization: {normalized!r}"
            )
        seen.add(normalized)
        rename[column] = normalized
    return df.rename(columns=rename)


def _read_delimited_table(path: str | Path) -> pd.DataFrame:
    """Read a delimited table with pandas auto-detected separators.

    Parameters
    ----------
    path : str or Path
        Path to the delimited table.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    return pd.read_csv(path, sep=None, engine="python")


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from string-like dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with stripped string columns.
    """

    for column in df.select_dtypes(include=["object", "string"]):
        df[column] = df[column].astype(str).str.strip()
    return df


def _coerce_round_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a dataframe round column to integer dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a ``round`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with an integer ``round`` column.
    """

    df["round"] = pd.to_numeric(df["round"], errors="raise").astype(int)
    return df


def _coerce_optional_int(value: Any) -> Any:
    """Convert numeric values to ints when they are whole numbers.

    Parameters
    ----------
    value : Any
        Candidate numeric value.

    Returns
    -------
    Any
        Integer-coerced value when possible.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def json_value(value: Any) -> Any:
    """Convert pandas, numpy, and path objects into JSON-safe values.

    Parameters
    ----------
    value : Any
        Value to normalize for JSON serialization.

    Returns
    -------
    Any
        JSON-safe representation of ``value``.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "records": value.to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    return value


def read_metadata(fname: str | Path) -> dict[str, Any]:
    """Read a single-row metadata CSV into a dictionary."""

    frame = _read_delimited_table(fname)
    if frame.empty:
        return {}
    return {str(k): _native_scalar(v) for k, v in frame.iloc[0].to_dict().items()}


def read_config_file(config_path: str | Path) -> dict[str, Any]:
    """Read a two-column key/value CSV configuration file."""

    series = pd.read_csv(
        config_path, sep=None, engine="python", header=None, index_col=0
    ).squeeze("columns")
    return {str(k): _native_scalar(v) for k, v in series.to_dict().items()}


def read_fluidics_program(program_path: str | Path) -> pd.DataFrame:
    """Read and validate a fluidics program CSV using pandas."""

    program = _normalize_columns(_read_delimited_table(program_path))
    return normalize_fluidics_program(program)


def normalize_fluidics_program(program: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a fluidics program dataframe."""

    program = _normalize_columns(program)
    missing = [column for column in FLUIDICS_REQUIRED_COLUMNS if column not in program]
    if missing:
        raise ValueError(
            f"Fluidics program is missing required column(s): {', '.join(missing)}"
        )

    program = program.loc[:, list(FLUIDICS_REQUIRED_COLUMNS)].copy()
    program = program.dropna(subset=list(FLUIDICS_REQUIRED_COLUMNS))
    program = _strip_object_columns(program)
    program = _coerce_round_column(program)
    program["source"] = program["source"].astype(str).str.upper()
    program["time"] = pd.to_numeric(program["time"], errors="raise").astype(float)
    program["pump"] = pd.to_numeric(program["pump"], errors="raise").astype(float)
    program = program.reset_index(drop=True)

    if program.empty:
        raise ValueError("Fluidics program does not contain any executable steps.")

    return program


def read_exp_order(exp_order_path: str | Path) -> pd.DataFrame:
    """Read a qi2lab experiment-order table using pandas."""

    exp_order = _normalize_exp_order_columns(_read_delimited_table(exp_order_path))
    missing = [
        column for column in EXP_ORDER_REQUIRED_COLUMNS if column not in exp_order
    ]
    if missing:
        raise ValueError(
            f"The experiment order file is missing required column(s): {', '.join(missing)}"
        )

    exp_order = exp_order.copy()
    channel_columns = [column for column in exp_order.columns if column != "round"]
    if not channel_columns:
        raise ValueError(
            "The experiment order file must contain at least one channel column after the round column."
        )

    exp_order = exp_order.dropna(subset=["round"])
    exp_order = _strip_object_columns(exp_order)
    exp_order = _coerce_round_column(exp_order)
    if exp_order[channel_columns].isnull().any(axis=None):
        raise ValueError("The experiment order file contains missing channel-bit values.")

    for column in channel_columns:
        exp_order[column] = pd.to_numeric(exp_order[column], errors="raise").astype(int)

    exp_order = exp_order.sort_values("round").reset_index(drop=True)

    if exp_order.empty:
        raise ValueError("The experiment order file does not contain any round/channel mappings.")
    if exp_order["round"].duplicated().any():
        duplicate_rounds = sorted(
            exp_order.loc[exp_order["round"].duplicated(), "round"].unique()
        )
        raise ValueError(
            f"The experiment order file contains duplicate round values: {duplicate_rounds}"
        )

    return exp_order


def read_codebook(codebook_path: str | Path) -> pd.DataFrame:
    """Read a MERFISH codebook using pandas."""

    codebook = _normalize_columns(_read_delimited_table(codebook_path))
    codebook = _strip_object_columns(codebook)
    codebook = codebook.dropna(how="all").reset_index(drop=True)
    if codebook.empty:
        raise ValueError("codebook does not contain any rows.")
    if len(codebook.columns) < 2:
        raise ValueError(
            "codebook must contain at least one identifier column and one bit column."
        )
    return codebook


def read_illumination_profiles(illumination_path: str | Path) -> np.ndarray:
    """Read channel illumination profiles from a TIFF/OME-TIFF file."""

    profiles = np.asarray(tifffile.imread(illumination_path), dtype=np.float32)
    if profiles.ndim == 2:
        profiles = np.expand_dims(profiles, axis=0)
    if profiles.ndim != 3:
        raise ValueError(
            "Illumination profiles must be a 2D YX image or 3D CYX stack, "
            f"received shape {profiles.shape}."
        )
    if not np.isfinite(profiles).all():
        raise ValueError("Illumination profiles contain non-finite values.")
    if np.any(profiles <= 0):
        raise ValueError(
            "Illumination profiles must be strictly positive for flatfield correction."
        )
    return profiles


def experiment_order_mapping(exp_order: pd.DataFrame) -> dict[int, dict[str, int]]:
    """Return round-to-channel-bit mappings from a normalized experiment-order table."""

    channel_columns = [column for column in exp_order.columns if column != "round"]
    return {
        int(row["round"]): {
            str(column): int(row[column]) for column in channel_columns
        }
        for row in exp_order.to_dict(orient="records")
    }


def infer_fiducial_channel_name(exp_order: pd.DataFrame) -> str:
    """Return the fiducial channel name from a normalized experiment-order table.

    Parameters
    ----------
    exp_order : pd.DataFrame
        Normalized experiment-order table.

    Returns
    -------
    str
        Channel name whose bit label is ``0`` for every round.

    Raises
    ------
    ValueError
        If the table does not contain exactly one all-zero channel column.
    """

    channel_columns = [column for column in exp_order.columns if column != "round"]
    zero_columns = [column for column in channel_columns if (exp_order[column] == 0).all()]
    if len(zero_columns) != 1:
        raise ValueError(
            "The experiment order file must contain exactly one fiducial channel column with 0 "
            f"for every round. Found {zero_columns}."
        )
    return str(zero_columns[0])


def fluidics_rounds(program: pd.DataFrame | None) -> list[int]:
    """Return all distinct round labels present in a fluidics program."""

    if program is None or program.empty:
        return []
    return [int(round_id) for round_id in program["round"].drop_duplicates().tolist()]


def imaging_rounds(program: pd.DataFrame | None) -> list[int]:
    """Return fluidics rounds that actually trigger imaging via ``RUN``."""

    if program is None or program.empty:
        return []
    source = program["source"].astype(str).str.strip().str.upper()
    run_rows = program.loc[source == "RUN", "round"]
    return [int(round_id) for round_id in run_rows.drop_duplicates().tolist()]


def validate_round_mappings(
    *,
    run_mode: str,
    fluidics_program: pd.DataFrame | None,
    exp_order: pd.DataFrame | None,
    selected_single_round: int | None = None,
    expected_channel_names: list[str] | None = None,
) -> dict[int, dict[str, int]]:
    """Validate experiment-order mappings for the requested run mode."""

    executable_rounds = imaging_rounds(fluidics_program)

    if run_mode == "fluidics_only":
        return {}

    if exp_order is None or exp_order.empty:
        raise ValueError("An experiment order file is required for imaging runs.")

    channel_columns = [column for column in exp_order.columns if column != "round"]
    if expected_channel_names is not None:
        normalized_channel_names = [str(name).strip() for name in expected_channel_names]
        expected_set = set(normalized_channel_names)
        found_set = set(channel_columns)
        if found_set != expected_set:
            missing = sorted(expected_set - found_set)
            extra = sorted(found_set - expected_set)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            detail_text = "; ".join(details)
            raise ValueError(
                "Experiment order file channel columns must match the active MDA channel names "
                f"regardless of order. Expected names {normalized_channel_names}, found {channel_columns}. {detail_text}"
            )
    infer_fiducial_channel_name(exp_order)

    round_to_bits = experiment_order_mapping(exp_order)

    if run_mode == "iterative":
        if fluidics_program is None or fluidics_program.empty:
            raise ValueError("A fluidics program is required for iterative imaging.")
        if not executable_rounds:
            raise ValueError(
                "The loaded fluidics program does not contain any RUN commands for imaging."
            )
        if set(executable_rounds) != set(round_to_bits):
            raise ValueError(
                "Experiment order file rounds must exactly match the fluidics-program rounds "
                f"with RUN commands for iterative imaging. Fluidics RUN rounds={executable_rounds}, "
                f"exp_order rounds={sorted(round_to_bits)}."
            )
        return round_to_bits

    if run_mode == "single_round":
        if selected_single_round is None:
            raise ValueError("A single-round imaging run requires a selected round.")
        if selected_single_round not in round_to_bits:
            raise ValueError(
                f"Selected round {selected_single_round} is not present in the experiment order file."
            )
        if executable_rounds and selected_single_round not in executable_rounds:
            raise ValueError(
                f"Selected round {selected_single_round} does not contain a RUN command in the fluidics program."
            )
        return {selected_single_round: round_to_bits[selected_single_round]}

    raise ValueError(f"Unsupported MERFISH run mode: {run_mode!r}")


def write_metadata(data_dict: Mapping[str, Any] | Any, save_path: str | Path) -> None:
    """Write a one-row metadata CSV using pandas."""

    if isinstance(data_dict, Mapping):
        data = dict(data_dict)
    else:
        raise TypeError("write_metadata expects a mapping.")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([data]).to_csv(save_path, index=False)


def write_json(data: Any, save_path: str | Path) -> None:
    """Write JSON metadata with numpy/pandas/path coercion."""

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(save_path).open("w", encoding="utf-8") as handle:
        json.dump(json_value(data), handle, indent=2, sort_keys=True)


def time_stamp() -> str:
    """Return the current timestamp for fluidics log messages.

    Returns
    -------
    str
        Current timestamp formatted for log output.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_index_filepath(filepath: str | Path) -> Path:
    """Append a numeric suffix to a file or directory path if it already exists."""

    path = Path(filepath)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    parent = path.parent

    candidate = path
    index = 1
    while candidate.exists():
        name = f"{stem}-{index}{suffix}"
        candidate = parent / name
        index += 1
    return candidate


