from typing import Any

import numpy as np
import pytest
from tests.merfish_builders import (
    write_codebook,
    write_exp_order,
    write_fluidics_program,
)
from tests.merfish_test_utils import (
    TEST_CORE_METADATA,
    prepare_dispatch_inputs,
    sequence_from_channel_specs,
)
from useq import CustomAction, MDASequence

from merfish3d_wfacq.engine import (
    DRIFT_ACTION_NAME,
    FLUIDICS_ACTION_NAME,
    MerfishMDAEngine,
)
from merfish3d_wfacq.runtime_state import register_runtime_state
from merfish3d_wfacq.sequence import MERFISH_SETUP_ACTION_NAME, ImageKind, RunMode

ENGINE_CHANNEL_SPECS = [
    {
        "channel_index": 0,
        "config_name": "DAPI",
        "role": ImageKind.FIDUCIAL.value,
        "excitation_um": 0.405,
        "emission_um": 0.450,
    },
    {
        "channel_index": 1,
        "config_name": "Cy5",
        "role": ImageKind.READOUT.value,
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
]


def _prepare_engine_dispatch(
    workspace_tmp_path: Any,
    *,
    mode: RunMode,
    rounds: list[int],
    z_plan: dict[str, Any] | None = None,
    stage_positions: list[tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    fluidics_path = workspace_tmp_path / "fluidics.csv"
    fluidics_program = write_fluidics_program(fluidics_path, rounds)
    if mode is RunMode.FLUIDICS_ONLY:
        return prepare_dispatch_inputs(
            mode=mode,
            sequence=None,
            fluidics_program=fluidics_program,
            exp_order=None,
            codebook=None,
            illumination_profiles=None,
            use_uniform_illumination=True,
            core_metadata=TEST_CORE_METADATA,
            save_path=None,
        )

    codebook_path = workspace_tmp_path / "codebook.csv"
    exp_order_path = workspace_tmp_path / "exp_order.csv"
    experiment_order = {
        int(round_label): {"DAPI": 0, "Cy5": int(round_index + 1)}
        for round_index, round_label in enumerate(rounds)
    }
    codebook = write_codebook(codebook_path, 8)
    exp_order = write_exp_order(
        exp_order_path,
        experiment_order,
        channel_specs=ENGINE_CHANNEL_SPECS,
    )
    sequence = sequence_from_channel_specs(
        ENGINE_CHANNEL_SPECS,
        stage_positions=stage_positions or [(0.0, 0.0, 1.0)],
        z_plan=z_plan,
    )
    return prepare_dispatch_inputs(
        mode=mode,
        sequence=sequence,
        fluidics_program=fluidics_program,
        exp_order=exp_order,
        codebook=codebook,
        illumination_profiles=None,
        use_uniform_illumination=True,
        core_metadata=TEST_CORE_METADATA,
        tile_overlap=0.2,
        save_path=workspace_tmp_path / "synthetic_run.ome.zarr",
        exp_order_path=exp_order_path,
        codebook_path=codebook_path,
    )


def _events_with_runtime_store(events: list[Any], state_id: str) -> list[Any]:
    setup_event = events[0]
    setup_data = dict(setup_event.action.data)
    setup_data["drift_reference_store_id"] = state_id
    return [
        setup_event.replace(
            action=CustomAction(name=setup_event.action.name, data=setup_data)
        ),
        *events[1:],
    ]



class _ReferenceStackEstimator:
    def __init__(self, expected_reference: np.ndarray) -> None:
        self.expected_reference = expected_reference

    def estimate(
        self,
        reference_stack: np.ndarray,
        moving_stack: np.ndarray,
        *,
        z_step_um: float,
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        np.testing.assert_array_equal(reference_stack, self.expected_reference)
        assert moving_stack.shape[0] == 3
        assert z_step_um == 1.0
        assert current_offset_um == 0.0
        return {"shift_planes": 0, "shift_um": 0.75, "absolute_offset_um": 0.75}


class _ReferencePlaneEstimator:
    def __init__(self, expected_reference_plane: np.ndarray) -> None:
        self.expected_reference_plane = expected_reference_plane

    def estimate_from_reference_plane(
        self,
        reference_plane: np.ndarray,
        moving_stack: np.ndarray,
        *,
        reference_z_um: float,
        moving_z_positions_um: list[float],
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        np.testing.assert_array_equal(reference_plane, self.expected_reference_plane)
        assert moving_stack.shape[0] == 5
        assert reference_z_um == pytest.approx(1.0)
        assert moving_z_positions_um == pytest.approx([-1.0, 0.0, 1.0, 2.0, 3.0])
        assert current_offset_um == pytest.approx(0.0)
        return {"shift_planes": 3, "shift_um": 1.0, "absolute_offset_um": 1.0}

    def estimate(
        self,
        reference_stack: np.ndarray,
        moving_stack: np.ndarray,
        *,
        z_step_um: float,
        current_offset_um: float = 0.0,
    ) -> dict[str, float | int]:
        raise AssertionError("2D drift should use the reference-plane path.")


class _CountingFrameRecorder:
    def __init__(self) -> None:
        self.frames = 0

    def sequenceStarted(self, _sequence: object, _meta: dict[str, object]) -> None:
        self.frames = 0

    def frameReady(
        self, _img: np.ndarray, _event: object, _meta: dict[str, object]
    ) -> None:
        self.frames += 1

    def sequenceFinished(self, _sequence: object) -> None:
        return None
def test_merfish_events_contain_prepared_setup_fluidics_and_drift_actions(
    workspace_tmp_path: Any,
) -> None:
    dispatch = _prepare_engine_dispatch(
        workspace_tmp_path,
        mode=RunMode.ITERATIVE,
        rounds=[1, 2],
    )
    events = dispatch["events"]
    action_names = [
        event.action.name
        for event in events
        if isinstance(getattr(event, "action", None), CustomAction)
    ]

    assert action_names.count(MERFISH_SETUP_ACTION_NAME) == 1
    assert action_names.count(FLUIDICS_ACTION_NAME) == 2
    assert action_names.count(DRIFT_ACTION_NAME) == 1

    image_events = [
        event
        for event in events
        if not isinstance(getattr(event, "action", None), CustomAction)
    ]
    assert image_events
    assert all(int(event.index.get("t", -1)) in {0, 1} for event in image_events)
    assert {event.metadata["merfish_image_kind"] for event in image_events} == {
        ImageKind.FIDUCIAL.value,
        ImageKind.READOUT.value,
    }


def test_fluidics_only_mode_strips_image_events(workspace_tmp_path: Any) -> None:
    events = _prepare_engine_dispatch(
        workspace_tmp_path,
        mode=RunMode.FLUIDICS_ONLY,
        rounds=[3, 4],
    )["events"]

    assert len(events) == 3
    assert events[0].action.name == MERFISH_SETUP_ACTION_NAME
    assert [event.action.name for event in events[1:]] == [
        FLUIDICS_ACTION_NAME,
        FLUIDICS_ACTION_NAME,
    ]


def test_drift_action_uses_shared_round_one_reference_store(
    workspace_tmp_path: Any, demo_core: Any
) -> None:
    reference_stack = np.full((3, 4, 4), 324, dtype=np.uint16)
    state_id = register_runtime_state({"reference_tile": 0, "frames": [*reference_stack]})
    events = _events_with_runtime_store(
        _prepare_engine_dispatch(
            workspace_tmp_path,
            mode=RunMode.ITERATIVE,
            rounds=[1, 2],
            z_plan={"range": 2, "step": 1},
        )["events"],
        state_id,
    )

    engine = MerfishMDAEngine(
        demo_core,
        drift_estimator=_ReferenceStackEstimator(reference_stack),
    )
    engine.setup_sequence(MDASequence())
    setup_event = events[0]
    drift_event_index, drift_event = next(
        (index, event)
        for index, event in enumerate(events)
        if isinstance(getattr(event, "action", None), CustomAction)
        and event.action.name == DRIFT_ACTION_NAME
    )
    engine._prepared_events = list(events)
    engine._current_event_index = drift_event_index
    engine.setup_event(setup_event)

    before_round_two_images = [
        event
        for event in engine._prepared_events
        if not isinstance(getattr(event, "action", None), CustomAction)
        and int(event.index.get("t", 0)) == 1
    ]
    assert before_round_two_images
    assert all(
        event.z_pos == event.metadata["planned_z_um"] for event in before_round_two_images
    )

    engine._execute_drift_action(dict(drift_event.action.data))

    assert engine._offset_z_um == 0.75
    after_round_two_images = [
        event
        for event in engine._prepared_events
        if not isinstance(getattr(event, "action", None), CustomAction)
        and int(event.index.get("t", 0)) >= 1
    ]
    assert after_round_two_images
    assert all(
        event.z_pos == pytest.approx(float(event.metadata["planned_z_um"]) + 0.75)
        for event in after_round_two_images
    )


def test_2d_drift_uses_sweep_plan_and_materializes_prepared_iterable(
    workspace_tmp_path: Any, demo_core: Any
) -> None:
    reference_plane = np.full((4, 4), 324, dtype=np.uint16)
    state_id = register_runtime_state({"reference_tile": 0, "frames": [reference_plane]})
    events = _events_with_runtime_store(
        _prepare_engine_dispatch(
            workspace_tmp_path,
            mode=RunMode.ITERATIVE,
            rounds=[1, 2],
            z_plan=None,
        )["events"],
        state_id,
    )

    drift_event = next(
        event
        for event in events
        if isinstance(getattr(event, "action", None), CustomAction)
        and event.action.name == DRIFT_ACTION_NAME
    )
    assert drift_event.action.data["reference_z_um"] == pytest.approx(1.0)
    assert drift_event.action.data["z_positions"] == pytest.approx(
        [-1.0, 0.0, 1.0, 2.0, 3.0]
    )

    engine = MerfishMDAEngine(
        demo_core,
        drift_estimator=_ReferencePlaneEstimator(reference_plane),
    )
    engine.setup_sequence(MDASequence())
    prepared_events = list(engine.event_iterator(list(events)))
    assert isinstance(engine._prepared_events, list)

    setup_event = prepared_events[0]
    drift_event_index = next(
        index
        for index, event in enumerate(prepared_events)
        if isinstance(getattr(event, "action", None), CustomAction)
        and event.action.name == DRIFT_ACTION_NAME
    )
    engine._current_event_index = drift_event_index
    engine.setup_event(setup_event)

    before_round_two_images = [
        event
        for event in engine._prepared_events
        if not isinstance(getattr(event, "action", None), CustomAction)
        and int(event.index.get("t", 0)) == 1
    ]
    assert before_round_two_images
    assert all(
        event.z_pos == event.metadata["planned_z_um"] for event in before_round_two_images
    )

    engine._execute_drift_action(dict(prepared_events[drift_event_index].action.data))

    assert engine._offset_z_um == pytest.approx(1.0)
    after_round_two_images = [
        event
        for event in engine._prepared_events
        if not isinstance(getattr(event, "action", None), CustomAction)
        and int(event.index.get("t", 0)) >= 1
    ]
    assert after_round_two_images
    assert all(
        event.z_pos == pytest.approx(float(event.metadata["planned_z_um"]) + 1.0)
        for event in after_round_two_images
    )


def test_engine_builds_fluidics_controller_from_setup_action_metadata(
    demo_core: Any,
    workspace_tmp_path: Any,
) -> None:
    events = _prepare_engine_dispatch(
        workspace_tmp_path,
        mode=RunMode.ITERATIVE,
        rounds=[1, 2],
    )["events"]
    engine = MerfishMDAEngine(demo_core)

    engine.setup_sequence(MDASequence())
    engine.setup_event(events[0])

    assert engine._fluidics_controller is not None
    assert [row["source"] for row in engine._fluidics_controller._program] == [
        "B01",
        "RUN",
        "B02",
        "RUN",
    ]


def test_plain_mda_grid_run_uses_base_engine_behavior(demo_core: Any) -> None:
    sequence = MDASequence(
        channels=[{"config": "DAPI", "exposure": 10.0}],
        stage_positions=[
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (5.0, 5.0, 0.0),
        ],
        z_plan={"range": 0, "step": 1},
    )
    recorder = _CountingFrameRecorder()
    engine = MerfishMDAEngine(demo_core)
    demo_core.register_mda_engine(engine)

    demo_core.run_mda(sequence, output=recorder, block=True)

    assert recorder.frames == 4


