#!/usr/bin/env python

"""Utilities to execute fluidics programs without any GUI dependency."""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from merfish3d_wfacq.utils.data_io import time_stamp

VALVE_LOOKUP: dict[str, tuple[int, int]] = {
    "B01": (0, 1),
    "B02": (0, 2),
    "B03": (0, 3),
    "B04": (0, 4),
    "B05": (0, 5),
    "B06": (0, 6),
    "B07": (0, 7),
    "B08": (1, 1),
    "B09": (1, 2),
    "B10": (1, 3),
    "B11": (1, 4),
    "B12": (1, 5),
    "B13": (1, 6),
    "B14": (1, 7),
    "B15": (2, 1),
    "B16": (2, 2),
    "B17": (2, 3),
    "B18": (2, 4),
    "B19": (2, 5),
    "B20": (2, 6),
    "B21": (2, 7),
    "B22": (3, 0),
    "B23": (3, 2),
    "B24": (3, 3),
    "SSC": (3, 0),
    "READOUT WASH": (3, 4),
    "IMAGING BUFFER": (3, 5),
    "CLEAVE": (3, 7),
}

RefreshHandler = Callable[[dict[str, Any]], bool]
LogHandler = Callable[[str], None]
SleepHandler = Callable[[float], None]
FluidicsRow = Mapping[str, Any]


def lookup_valve(source_name: str) -> tuple[int, int] | None:
    """Convert a fluidics source name to an MVP valve/unit tuple.

    Parameters
    ----------
    source_name : str
        Fluidics source name from the program table.

    Returns
    -------
    tuple[int, int] or None
        ``(unit, port)`` mapping for the source, if known.
    """

    return VALVE_LOOKUP.get(str(source_name).strip().upper())


def _log(log_fn: LogHandler | None, message: str) -> None:
    """Emit a timestamped fluidics log line through the injected logger.

    Parameters
    ----------
    log_fn : LogHandler or None
        Logger callback used for fluidics messages.
    message : str
        Message to emit.
    """

    if log_fn is not None:
        log_fn(f"{time_stamp()}: {message}")


def _set_active_valve(mvp_controller: Any, active_unit: int, valve_number: int) -> None:
    """Route the active manifold to the requested valve position.

    Parameters
    ----------
    mvp_controller : Any
        Valve controller object.
    active_unit : int
        Unit index that should be opened.
    valve_number : int
        Port number to activate on the selected unit.
    """

    for valve_id in range(4):
        target_port = valve_number if valve_id == active_unit else 0
        mvp_controller.changePort(valve_ID=valve_id, port_ID=target_port)


def _row_value(row: FluidicsRow | pd.Series, key: str) -> Any:
    """Return one value from a dataframe row or mapping row.

    Parameters
    ----------
    row : FluidicsRow or pd.Series
        One fluidics-program row.
    key : str
        Column name to read.

    Returns
    -------
    Any
        Retrieved value.
    """

    if isinstance(row, Mapping):
        return row.get(key)
    return row.get(key)


def _pump_rate_from_row(row: FluidicsRow | pd.Series) -> float:
    """Resolve the commanded pump rate for one fluidics row.

    Parameters
    ----------
    row : FluidicsRow or pd.Series
        One fluidics-program row.

    Returns
    -------
    float
        Pump rate to send to the pump controller.
    """

    pump_value = _row_value(row, "pump")
    if pump_value is not None and pd.notna(pump_value):
        return float(pump_value)

    volume = _row_value(row, "volume")
    time_min = _row_value(row, "time")
    if volume is None or pd.isna(volume) or time_min is None or pd.isna(time_min):
        raise ValueError(
            "Fluidics row must contain either 'pump' or ('volume' and 'time')."
        )

    pump_amount_ml = float(volume)
    pump_time_min = float(time_min)
    flow_ml_per_min = round(pump_amount_ml / pump_time_min, 2)
    flow_lookup = {
        1.00: 48.0,
        0.50: 11.0,
        0.40: 10.0,
        0.36: 9.5,
        0.33: 9.0,
        0.22: 5.0,
        0.20: 4.0,
    }
    if flow_ml_per_min not in flow_lookup:
        raise ValueError(
            f"Unable to determine a pump rate for volume={pump_amount_ml} mL, "
            f"time={pump_time_min} min."
        )
    return flow_lookup[flow_ml_per_min]


def run_fluidic_program_wf(
    r_idx: int,
    program_rows: Sequence[FluidicsRow] | pd.DataFrame,
    mvp_controller: Any,
    pump_controller: Any,
    *,
    refresh_handler: RefreshHandler | None = None,
    log_fn: LogHandler | None = print,
    sleep_fn: SleepHandler = time.sleep,
    settle_time_s: float = 5.0,
) -> bool:
    """Run a fluidics program for a given round using injected callbacks.

    Parameters
    ----------
    r_idx : int
        Round label to execute.
    program_rows : Sequence[FluidicsRow] or pd.DataFrame
        Fluidics program rows for all rounds.
    mvp_controller : Any
        Valve controller object.
    pump_controller : Any
        Pump controller object.
    refresh_handler : RefreshHandler or None, optional
        Callback used to confirm ``REFRESH`` steps.
    log_fn : LogHandler or None, optional
        Logger used for fluidics progress messages.
    sleep_fn : SleepHandler, optional
        Sleep function used for waits and pauses.
    settle_time_s : float, optional
        Settling delay after valve changes.

    Returns
    -------
    bool
        ``True`` when the round completes successfully.
    """

    if isinstance(program_rows, pd.DataFrame):
        current_program = program_rows.loc[program_rows["round"] == int(r_idx)].to_dict(
            orient="records"
        )
    else:
        current_program = [
            dict(row)
            for row in program_rows
            if int(_row_value(row, "round")) == int(r_idx)
        ]
    if not current_program:
        raise ValueError(f"Fluidics program does not contain round {r_idx}.")

    _log(log_fn, f"Executing iterative round {r_idx}.")
    for row in current_program:
        source_name = str(_row_value(row, "source")).strip().upper()
        pump_time_min = float(_row_value(row, "time"))

        if source_name == "RUN":
            pump_controller.stopFlow()
            _log(log_fn, "Fluidics round done, running imaging.")
            break

        if source_name == "PAUSE":
            pump_controller.stopFlow()
            _log(log_fn, f"Pausing for {pump_time_min * 60:.1f} seconds.")
            sleep_fn(max(0.0, pump_time_min * 60))
            continue

        if source_name == "REFRESH":
            pump_controller.stopFlow()
            if refresh_handler is not None:
                approved = bool(
                    refresh_handler(
                        {
                            "round": int(r_idx),
                            "source": source_name,
                            "pause_seconds": pump_time_min * 60,
                        }
                    )
                )
                if not approved:
                    raise RuntimeError("Operator rejected REFRESH confirmation.")
            _log(log_fn, "Refresh confirmed.")
            continue

        valve_position = lookup_valve(source_name)
        if valve_position is None:
            raise ValueError(f"Unknown fluidics source: {source_name!r}")

        mvp_unit, valve_number = valve_position
        _set_active_valve(mvp_controller, mvp_unit, valve_number)
        if settle_time_s > 0:
            sleep_fn(settle_time_s)

        pump_rate = _pump_rate_from_row(row)
        _log(
            log_fn,
            f"Source={source_name}; MVP unit={mvp_unit}; "
            f"Valve={valve_number}; Pump rate={pump_rate}",
        )
        pump_controller.startFlow(pump_rate, direction="Forward")
        sleep_fn(max(0.0, pump_time_min * 60))
        pump_controller.stopFlow()

    return True
