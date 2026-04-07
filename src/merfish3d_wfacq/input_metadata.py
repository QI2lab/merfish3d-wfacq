from pathlib import Path
from typing import Any

import pandas as pd


def table_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Serialize one dataframe into columns-plus-records form.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-safe table payload with ``columns`` and ``records`` keys.
    """

    return {
        "columns": [str(column) for column in frame.columns],
        "records": frame.to_dict(orient="records"),
    }


def table_values(
    payload: dict[str, Any], *, exclude_columns: set[str] | None = None
) -> list[list[Any]]:
    """Return row-major table values from a serialized table payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Serialized table payload with ``columns`` and ``records`` keys.
    exclude_columns : set[str] or None, optional
        Column names to omit from the returned rows.

    Returns
    -------
    list[list[Any]]
        Table values in row-major order.
    """

    excluded = exclude_columns or set()
    columns = [str(column) for column in payload["columns"] if str(column) not in excluded]
    records = list(payload["records"])
    return [[record[column] for column in columns] for record in records]


def input_file_metadata(
    *,
    exp_order: pd.DataFrame | None,
    codebook: pd.DataFrame | None,
    exp_order_path: str | Path | None,
    codebook_path: str | Path | None,
    illumination_profiles_path: str | Path | None,
    use_uniform_illumination: bool,
) -> dict[str, Any]:
    """Build the serialized file/table metadata used at runtime.

    Parameters
    ----------
    exp_order : pd.DataFrame or None
        Experiment-order table.
    codebook : pd.DataFrame or None
        Codebook table.
    exp_order_path : str or Path or None
        Source path of the experiment-order table.
    codebook_path : str or Path or None
        Source path of the codebook table.
    illumination_profiles_path : str or Path or None
        Source path of the illumination profiles.
    use_uniform_illumination : bool
        Whether the run uses generated all-ones profiles.

    Returns
    -------
    dict[str, Any]
        Serialized input metadata for runtime setup and datastore metadata.
    """

    metadata: dict[str, Any] = {
        "exp_order_path": str(exp_order_path) if exp_order_path else None,
        "codebook_path": str(codebook_path) if codebook_path else None,
        "illumination_profiles_mode": ("uniform" if use_uniform_illumination else "measured"),
        "illumination_profiles_path": (
            "<uniform>"
            if use_uniform_illumination
            else str(illumination_profiles_path)
            if illumination_profiles_path
            else "<in-memory>"
        ),
    }
    if exp_order is not None:
        metadata["experiment_order_table"] = table_payload(exp_order)
    if codebook is not None:
        metadata["codebook_table"] = table_payload(codebook)
    return metadata


def calibration_input_metadata(merfish_metadata: dict[str, Any]) -> dict[str, Any]:
    """Build datastore calibration payload fields from runtime metadata.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH runtime metadata.

    Returns
    -------
    dict[str, Any]
        Serialized input tables, flattened values, and source paths.
    """

    codebook_table = dict(merfish_metadata["codebook_table"])
    experiment_order_table = dict(merfish_metadata["experiment_order_table"])
    return {
        "codebook": table_values(codebook_table),
        "codebook_table": codebook_table,
        "exp_order": table_values(experiment_order_table, exclude_columns={"round"}),
        "experiment_order": experiment_order_table,
        "illumination_profiles_path": str(Path(str(merfish_metadata["illumination_profiles_path"]))),
        "codebook_path": str(Path(str(merfish_metadata["codebook_path"]))),
        "exp_order_path": str(Path(str(merfish_metadata["exp_order_path"]))),
    }
