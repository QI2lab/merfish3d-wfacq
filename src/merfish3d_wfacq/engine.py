from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
from pymmcore_plus.mda import MDAEngine
from useq import CustomAction, MDAEvent

from merfish3d_wfacq.drift import ZDriftEstimator
from merfish3d_wfacq.fluidics import FluidicsController
from merfish3d_wfacq.runtime_state import (
    get_runtime_state,
    unregister_runtime_state,
)
from merfish3d_wfacq.sequence import (
    MERFISH_EVENT_PLANNED_Z_UM_KEY,
    MERFISH_SETUP_ACTION_NAME,
    RunMode,
)
from merfish3d_wfacq.utils.callbacks import emit_callback

FLUIDICS_ACTION_NAME = "fluidics"
DRIFT_ACTION_NAME = "drift_correct"


class MerfishMDAEngine(MDAEngine):
    """Custom MDA engine that injects MERFISH fluidics and drift steps."""

    def __init__(
        self,
        mmc: Any,
        *,
        log_callback: Callable[[str], Any] | None = None,
        status_callback: Callable[[str], Any] | None = None,
        refresh_handler: Callable[[dict[str, Any]], bool] | None = None,
        drift_estimator: ZDriftEstimator | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the engine with MERFISH action handlers.

        Parameters
        ----------
        mmc : Any
            Active ``CMMCorePlus`` instance.
        log_callback : Callable[[str], Any] or None, optional
            Callback used for log messages.
        status_callback : Callable[[str], Any] or None, optional
            Callback used for status messages.
        refresh_handler : Callable[[dict[str, Any]], bool] or None, optional
            Callback used to confirm ``REFRESH`` fluidics steps.
        drift_estimator : ZDriftEstimator or None, optional
            Drift estimator used for fiducial stack matching.
        **kwargs : Any
            Additional ``MDAEngine`` keyword arguments.
        """

        kwargs.setdefault("use_hardware_sequencing", False)
        super().__init__(mmc, **kwargs)
        self._merfish_metadata: dict[str, Any] | None = None
        self._log_callback = log_callback
        self._status_callback = status_callback
        self._refresh_handler = refresh_handler
        self._drift_estimator = drift_estimator or ZDriftEstimator()
        self._fluidics_controller: FluidicsController | None = None
        self._drift_reference_store_id: str | None = None
        self._drift_reference_store: dict[str, Any] | None = None
        self._offset_z_um = 0.0
        self._prepared_events: list[MDAEvent] | None = None
        self._current_event_index = -1
        self.include_frame_position_metadata = True

    def setup_sequence(self, sequence: Any) -> dict[str, Any] | None:
        """Reset MERFISH runtime state before a new sequence starts.

        Parameters
        ----------
        sequence : Any
            Sequence about to be executed by the engine.

        Returns
        -------
        dict[str, Any] or None
            Sequence-start metadata returned by ``MDAEngine``.
        """

        self._offset_z_um = 0.0
        self._fluidics_controller = None
        self._merfish_metadata = None
        self._drift_reference_store_id = None
        self._drift_reference_store = None
        self._prepared_events = None
        self._current_event_index = -1
        return super().setup_sequence(sequence)

    def event_iterator(
        self, events: list[MDAEvent] | Iterable[MDAEvent]
    ) -> Iterator[MDAEvent]:
        """Track prepared MERFISH event lists and pass through upstream runs.

        Parameters
        ----------
        events : list[MDAEvent] or Iterable[MDAEvent]
            Event source for the current run. MERFISH runs provide a
            prepared ``list[MDAEvent]``, while plain upstream MDA runs may
            provide other iterables.

        Yields
        ------
        MDAEvent
            Events in execution order.
        """

        if isinstance(events, list):
            self._prepared_events = events
            self._current_event_index = -1
            for index, event in enumerate(events):
                self._current_event_index = index
                yield event
            return
        self._prepared_events = None
        self._current_event_index = -1
        yield from events

    def teardown_sequence(self, sequence: Any) -> None:
        """Release MERFISH runtime resources after a sequence finishes.

        Parameters
        ----------
        sequence : Any
            Sequence that just finished execution.
        """

        had_merfish_controller = self._fluidics_controller is not None
        if self._fluidics_controller is not None:
            self._fluidics_controller.close()
            self._fluidics_controller = None
        self._merfish_metadata = None
        unregister_runtime_state(self._drift_reference_store_id)
        self._drift_reference_store_id = None
        self._drift_reference_store = None
        self._prepared_events = None
        self._current_event_index = -1
        if had_merfish_controller:
            self._status("MERFISH acquisition finished.")
        super().teardown_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Execute MERFISH custom actions before standard setup.

        Parameters
        ----------
        event : MDAEvent
            Event about to be configured on the microscope.
        """

        action = event.action
        if isinstance(action, CustomAction):
            if action.name == MERFISH_SETUP_ACTION_NAME:
                self._execute_setup_action(action.data)
            elif action.name == FLUIDICS_ACTION_NAME:
                self._execute_fluidics_action(action.data)
            elif action.name == DRIFT_ACTION_NAME:
                self._execute_drift_action(action.data)
            return

        super().setup_event(event)

    def exec_event(self, event: MDAEvent) -> Iterable[Any]:
        """Skip execution for custom actions and delegate image events upstream.

        Parameters
        ----------
        event : MDAEvent
            Event to execute.

        Returns
        -------
        Iterable[Any]
            Upstream execution result for image events.
        """

        action = event.action
        if isinstance(action, CustomAction):
            return ()
        return super().exec_event(event) or ()

    def _execute_setup_action(self, data: dict[str, Any]) -> None:
        """Load normalized MERFISH metadata for the current run.

        Parameters
        ----------
        data : dict[str, Any]
            Setup payload stamped onto the initial custom action.
        """

        self._merfish_metadata = dict(data["merfish_metadata"])
        self._drift_reference_store_id = data.get("drift_reference_store_id")
        self._drift_reference_store = (
            get_runtime_state(self._drift_reference_store_id)
            if self._drift_reference_store_id is not None
            else None
        )
        self._offset_z_um = 0.0
        self._status("Preparing MERFISH acquisition.")
        if RunMode(str(self._merfish_metadata["run_mode"])) is not RunMode.SINGLE_ROUND:
            self._fluidics_controller = FluidicsController(
                self._merfish_metadata,
                [
                    dict(record)
                    for record in self._merfish_metadata["fluidics_program_records"]
                ],
                request_refresh=self._refresh_handler,
                log_callback=self._log_callback,
                status_callback=self._status_callback,
            )

    def _execute_fluidics_action(self, data: dict[str, Any]) -> None:
        """Execute one prepared fluidics step.

        Parameters
        ----------
        data : dict[str, Any]
            Fluidics action payload for the current round.
        """

        round_label = int(data["round"])
        self._status(f"Running fluidics round {round_label}.")
        self._fluidics_controller.execute_round(round_label)

    def _execute_drift_action(self, data: dict[str, Any]) -> None:
        """Acquire drift data and update future event z positions.

        Parameters
        ----------
        data : dict[str, Any]
            Drift action payload for the current round.
        """

        reference_stack = self._as_stack(self._reference_stack_from_store())
        current_stack = self._acquire_drift_stack(data)
        z_positions = [float(z) for z in data["z_positions"]]
        if reference_stack.shape[0] == 1:
            result = self._drift_estimator.estimate_from_reference_plane(
                reference_stack[0],
                current_stack,
                reference_z_um=float(data["reference_z_um"]),
                moving_z_positions_um=[
                    float(z_pos) + float(self._offset_z_um) for z_pos in z_positions
                ],
                current_offset_um=float(self._offset_z_um),
            )
        else:
            if current_stack.shape[0] < 2:
                raise RuntimeError(
                    "MERFISH drift correction requires a multi-plane drift stack."
                )
            z_step = abs(z_positions[1] - z_positions[0])
            result = self._drift_estimator.estimate(
                reference_stack,
                current_stack,
                z_step_um=z_step,
                current_offset_um=0.0,
            )
        self._offset_z_um = float(result["absolute_offset_um"])
        self._apply_future_drift_offset(
            time_index=int(data["time_index"]),
            offset_z_um=self._offset_z_um,
        )
        self._status(f"Updated z drift offset to {self._offset_z_um:.3f} um.")

    def _acquire_drift_stack(self, data: dict[str, Any]) -> np.ndarray:
        """Acquire the prepared drift stack for one round.

        Parameters
        ----------
        data : dict[str, Any]
            Drift action payload for the current round.

        Returns
        -------
        np.ndarray
            Acquired drift stack with shape ``(z, y, x)``.
        """

        core = self.mmcore
        x_pos = data["x_pos"]
        y_pos = data["y_pos"]
        if x_pos is not None and y_pos is not None:
            core.setXYPosition(float(x_pos), float(y_pos))
        core.setConfig(str(data["channel_group"]), str(data["channel_config"]))
        exposure_ms = data["exposure_ms"]
        if exposure_ms is not None:
            core.setExposure(float(exposure_ms))
        core.waitForSystem()

        images = []
        for z_pos in data["z_positions"]:
            target_z = float(z_pos) + self._offset_z_um
            if core.getFocusDevice():
                core.setZPosition(target_z)
                core.waitForSystem()
            core.snapImage()
            images.append(np.asarray(core.getImage()).copy())

        return np.stack(images, axis=0)

    def _reference_stack_from_store(self) -> np.ndarray:
        """Return the captured round-1 fiducial reference stack.

        Returns
        -------
        np.ndarray
            Corrected round-1 fiducial reference data captured in memory.
        """

        frames = [] if self._drift_reference_store is None else self._drift_reference_store["frames"]
        if not frames:
            raise RuntimeError(
                "MERFISH drift correction requires the round-1 fiducial reference "
                "stack, but no reference frames were captured before drift execution."
            )
        if len(frames) == 1:
            return np.asarray(frames[0])
        return np.stack([np.asarray(frame) for frame in frames], axis=0)

    def _as_stack(self, image: np.ndarray) -> np.ndarray:
        """Coerce a reference image into ``(z, y, x)`` stack shape.

        Parameters
        ----------
        image : np.ndarray
            Reference image or stack.

        Returns
        -------
        np.ndarray
            Stack with shape ``(z, y, x)``.
        """

        array = np.asarray(image)
        if array.ndim == 2:
            return np.expand_dims(array, axis=0)
        if array.ndim != 3:
            raise ValueError(
                "MERFISH drift correction expects reference data with shape (z, y, x)."
            )
        return array

    def _log(self, message: str) -> None:
        """Forward a log message through the configured callback.

        Parameters
        ----------
        message : str
            Message to emit.
        """

        emit_callback(self._log_callback, message)

    def _status(self, message: str) -> None:
        """Forward a status message through the configured callback.

        Parameters
        ----------
        message : str
            Message to emit.
        """

        emit_callback(self._status_callback, message)

    def _apply_future_drift_offset(self, *, time_index: int, offset_z_um: float) -> None:
        """Rewrite future prepared events with the updated z offset.

        Parameters
        ----------
        time_index : int
            Time index after which future events should be updated.
        offset_z_um : float
            New absolute z offset in microns.
        """

        if self._prepared_events is None:
            return
        for index in range(self._current_event_index + 1, len(self._prepared_events)):
            event = self._prepared_events[index]
            action = event.action
            if isinstance(action, CustomAction):
                continue
            if int(event.index.get("t", 0)) < int(time_index):
                continue
            planned_z_um = event.metadata[MERFISH_EVENT_PLANNED_Z_UM_KEY]
            if planned_z_um is None:
                continue
            self._prepared_events[index] = event.replace(
                z_pos=float(planned_z_um) + float(offset_z_um)
            )
