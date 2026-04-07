from collections.abc import Callable
from typing import Any

from merfish3d_wfacq.hardware.APump import APump
from merfish3d_wfacq.hardware.HamiltonMVP import HamiltonMVP
from merfish3d_wfacq.utils.callbacks import emit_callback
from merfish3d_wfacq.utils.fluidics_control import run_fluidic_program_wf


class FluidicsController:
    """High-level fluidics controller that wraps the pump and valve hardware."""

    def __init__(
        self,
        merfish_metadata: dict[str, Any],
        fluidics_program: list[dict[str, Any]],
        *,
        request_refresh: Callable[[dict[str, Any]], bool] | None = None,
        log_callback: Callable[[str], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
    ) -> None:
        """Initialize the controller from normalized MERFISH metadata.

        Parameters
        ----------
        merfish_metadata : dict[str, Any]
            Normalized MERFISH metadata for the run.
        fluidics_program : list[dict[str, Any]]
            Normalized fluidics program rows.
        request_refresh : Callable[[dict[str, Any]], bool] or None, optional
            Callback used to confirm ``REFRESH`` steps.
        log_callback : Callable[[str], Any] or None, optional
            Callback used for fluidics log messages.
        status_callback : Callable[[str], Any] or None, optional
            Callback used for operator-facing status messages.
        """

        self._metadata = dict(merfish_metadata)
        self._program = [dict(record) for record in fluidics_program]
        self._request_refresh_callback = request_refresh
        self._log_callback = log_callback
        self._status_callback = status_callback
        self._pump: APump | None = None
        self._valves: HamiltonMVP | None = None

    def connect(self) -> None:
        """Connect the configured pump and valve hardware if needed."""

        if self._pump is None:
            self._pump = APump(
                {
                    "pump_com_port": str(self._metadata["pump_com_port"]),
                    "pump_ID": int(self._metadata["pump_id"]),
                    "verbose": True,
                    "simulate_pump": bool(self._metadata["simulate_pump"]),
                    "serial_verbose": False,
                    "flip_flow_direction": bool(self._metadata["flip_flow_direction"]),
                }
            )
            self._pump.enableRemoteControl(True)

        if self._valves is None:
            num_simulated = (
                max(4, int(self._metadata["num_simulated_valves"]))
                if bool(self._metadata["simulate_valves"])
                else 0
            )
            self._valves = HamiltonMVP(
                com_port=str(self._metadata["valve_com_port"]),
                num_simulated_valves=num_simulated,
                verbose=False,
            )

    def execute_round(self, round_id: int) -> None:
        """Execute one fluidics round from the normalized program.

        Parameters
        ----------
        round_id : int
            Round label to execute.
        """

        self.connect()
        run_fluidic_program_wf(
            round_id,
            self._program,
            self._valves,
            self._pump,
            refresh_handler=self._request_refresh,
            log_fn=self._log,
            settle_time_s=float(self._metadata["settle_time_s"]),
        )

    def close(self) -> None:
        """Close any active hardware connections."""

        if self._valves is not None:
            self._valves.close()
            self._valves = None
        if self._pump is not None:
            self._pump.close()
            self._pump = None

    @property
    def pump(self) -> APump | None:
        """Return the active pump handle, if connected.

        Returns
        -------
        APump or None
            Active pump handle.
        """

        return self._pump

    def _request_refresh(self, payload: dict[str, Any]) -> bool:
        """Request operator confirmation for a REFRESH step.

        Parameters
        ----------
        payload : dict[str, Any]
            Normalized REFRESH payload for the current round.

        Returns
        -------
        bool
            ``True`` when the operator confirms the refresh step.
        """

        if self._request_refresh_callback is None:
            raise RuntimeError(
                "A REFRESH step was requested but no operator confirmation handler is configured."
            )
        emit_callback(self._status_callback, "Waiting for operator REFRESH confirmation.")
        return bool(self._request_refresh_callback(payload))

    def _log(self, message: str) -> None:
        """Forward a log message to the configured callback.

        Parameters
        ----------
        message : str
            Message to emit.
        """

        emit_callback(self._log_callback, message)
