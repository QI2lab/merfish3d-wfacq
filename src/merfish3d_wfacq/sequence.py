from copy import deepcopy
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from useq import Axis, CustomAction, MDAEvent, MDASequence

MERFISH_SETUP_ACTION_NAME = "merfish_setup"
MERFISH_EVENT_TARGET_KEY = "merfish_target"
MERFISH_EVENT_TARGET_CLOSE_KEY = "merfish_target_close_after_write"
MERFISH_EVENT_IMAGE_KIND_KEY = "merfish_image_kind"
MERFISH_EVENT_ROUND_LABEL_KEY = "merfish_round_label"
MERFISH_EVENT_BIT_LABEL_KEY = "merfish_bit_label"
MERFISH_EVENT_PLANNED_Z_UM_KEY = "planned_z_um"
DRIFT_SWEEP_PLANES_2D = 5


class RunMode(StrEnum):
    """Supported MERFISH acquisition modes."""

    FLUIDICS_ONLY = "fluidics_only"
    ITERATIVE = "iterative"
    SINGLE_ROUND = "single_round"


class ImageKind(StrEnum):
    """Logical image targets written into the datastore."""

    FIDUCIAL = "fiducial"
    READOUT = "readout"


def _axis_count(sequence: Any, axis: Axis) -> int:
    """Return a positive sequence axis count.

    Parameters
    ----------
    sequence : Any
        Sequence-like object exposing ``sizes``.
    axis : Axis
        Axis to inspect.

    Returns
    -------
    int
        Positive axis count.
    """

    value = sequence.sizes.get(axis, sequence.sizes.get(axis.value, 0))
    return max(int(value), 1)


def _event_axis_index(event: Any, axis: Axis) -> int:
    """Return one event axis index using ``useq.Axis`` keys.

    Parameters
    ----------
    event : Any
        Event-like object exposing ``index``.
    axis : Axis
        Axis to inspect.

    Returns
    -------
    int
        Event index for the requested axis.
    """

    return int(event.index.get(axis, event.index.get(axis.value, 0)))


def channel_index_key(record: dict[str, Any]) -> int:
    """Return the integer channel index from a channel-spec record.

    Parameters
    ----------
    record : dict[str, Any]
        Channel-spec-like mapping with a ``channel_index`` field.

    Returns
    -------
    int
        Integer channel index for stable ordering.
    """

    return int(record["channel_index"])


def tile_count_from_sequence(sequence: Any) -> int:
    """Return the nominal tile count from the sequence p/g axes."""

    p_count = _axis_count(sequence, Axis.POSITION)
    g_count = _axis_count(sequence, Axis.GRID)
    return p_count * g_count


def tile_index_from_event(event: Any) -> int:
    """Return the linear tile index for an event from the sequence p/g axes."""

    sequence = event.sequence
    if sequence is None:
        return 0

    tile_axes = tuple(
        axis
        for axis in sequence.axis_order
        if axis in {Axis.POSITION, Axis.GRID, Axis.POSITION.value, Axis.GRID.value}
    )
    if not tile_axes:
        return 0

    tile_index = 0
    for axis in tile_axes:
        normalized_axis = Axis(str(axis))
        axis_size = _axis_count(sequence, normalized_axis)
        tile_index = tile_index * axis_size + _event_axis_index(event, normalized_axis)
    return tile_index


def build_merfish_events(
    base_sequence: MDASequence,
    *,
    rounds: list[int],
    merfish_metadata: dict[str, Any],
    setup_payload: dict[str, Any] | None = None,
) -> list[MDAEvent]:
    """Build the prepared MERFISH event list for `run_mda()`."""

    run_mode = RunMode(str(merfish_metadata["run_mode"]))
    prepared_sequence = _prepared_image_sequence(base_sequence, rounds)
    events = [
        MDAEvent(
            index={},
            sequence=prepared_sequence,
            action=CustomAction(
                name=MERFISH_SETUP_ACTION_NAME,
                data={
                    "merfish_metadata": deepcopy(merfish_metadata),
                    **(setup_payload or {}),
                },
            ),
        )
    ]

    round_actions = _round_action_plan(merfish_metadata, rounds)
    if run_mode is RunMode.FLUIDICS_ONLY:
        for round_record in round_actions:
            if bool(round_record["run_fluidics"]):
                events.append(
                    _custom_action_event(
                        prepared_sequence,
                        time_index=int(round_record["time_index"]),
                        name="fluidics",
                        data={"round": int(round_record["round_label"])},
                    )
                )
        return events

    reference_probe = _reference_probe_from_sequence(
        prepared_sequence,
        reference_tile=int(
            merfish_metadata["reference_tile"]
        ),
        fiducial_channel_index=int(merfish_metadata["fiducial_channel_index"]),
        z_step_um=float(merfish_metadata["z_step_um"]),
    )
    if (
        reference_probe is None
        and run_mode is not RunMode.FLUIDICS_ONLY
        and bool(merfish_metadata["enable_drift_correction"])
    ):
        raise RuntimeError(
            "MERFISH drift correction requires a round-1 fiducial reference event."
        )
    route_lookup = {
        (int(record["time_index"]), int(record["channel_index"])): record
        for record in route_records_for_rounds(merfish_metadata, rounds)
    }
    prepared_records = _prepared_image_records(
        list(prepared_sequence.iter_events()),
        route_lookup,
    )
    seen_target_keys: set[str] = set()
    for record in reversed(prepared_records):
        target_key = str(record["target_record"]["image_relpath"])
        record["close_after_write"] = target_key not in seen_target_keys
        seen_target_keys.add(target_key)

    current_time_index: int | None = None
    for record in prepared_records:
        event = record["event"]
        time_index = int(record["time_index"])
        if time_index != current_time_index:
            current_time_index = time_index
            round_record = round_actions[time_index]
            if bool(round_record["run_fluidics"]):
                events.append(
                    _custom_action_event(
                        prepared_sequence,
                        time_index=time_index,
                        name="fluidics",
                        data={"round": int(round_record["round_label"])},
                    )
                )
            if bool(round_record["run_drift"]) and reference_probe is not None:
                events.append(
                    _custom_action_event(
                        prepared_sequence,
                        time_index=time_index,
                        name="drift_correct",
                        data={
                            "time_index": time_index,
                            "round": int(round_record["round_label"]),
                            **reference_probe,
                        },
                    )
                )

        events.append(
            _image_event_with_target_metadata(
                event,
                record["route_record"],
                record["target_record"],
                close_after_write=bool(record["close_after_write"]),
            )
        )

    return events


def _prepared_image_sequence(
    base_sequence: MDASequence,
    rounds: list[int],
) -> MDASequence:
    """Return the image sequence with time expanded over MERFISH rounds.

    Parameters
    ----------
    base_sequence : MDASequence
        Base image sequence from the MDA widget.
    rounds : list[int]
        Imaging rounds included in the run.

    Returns
    -------
    MDASequence
        Image sequence with time expanded over MERFISH rounds.
    """

    axis_order = [axis for axis in base_sequence.axis_order if axis != Axis.TIME]
    axis_order.insert(0, Axis.TIME)
    return base_sequence.replace(
        axis_order=tuple(axis_order),
        time_plan={"interval": timedelta(0), "loops": len(rounds)},
        metadata=deepcopy(base_sequence.metadata),
    )


def _custom_action_event(
    sequence: MDASequence,
    *,
    time_index: int,
    name: str,
    data: dict[str, Any],
) -> MDAEvent:
    """Build one MERFISH custom-action event.

    Parameters
    ----------
    sequence : MDASequence
        Prepared sequence attached to the event.
    time_index : int
        Time index for the action.
    name : str
        Custom action name.
    data : dict[str, Any]
        Action payload.

    Returns
    -------
    MDAEvent
        Prepared custom-action event.
    """

    return MDAEvent(
        index={"t": int(time_index)},
        sequence=sequence,
        action=CustomAction(name=name, data=data),
    )


def _image_event_with_target_metadata(
    event: MDAEvent,
    route_record: dict[str, Any],
    target_record: dict[str, Any],
    *,
    close_after_write: bool,
) -> MDAEvent:
    """Stamp one prepared image event with normalized routing metadata.

    Parameters
    ----------
    event : MDAEvent
        Prepared image event.
    route_record : dict[str, Any]
        Normalized routing record for the event.
    target_record : dict[str, Any]
        Normalized datastore target for the event.
    close_after_write : bool
        Whether the target stream should close after this frame.

    Returns
    -------
    MDAEvent
        Image event carrying MERFISH routing metadata.
    """

    metadata = dict(event.metadata or {})
    metadata[MERFISH_EVENT_TARGET_KEY] = target_record
    metadata[MERFISH_EVENT_TARGET_CLOSE_KEY] = bool(close_after_write)
    metadata[MERFISH_EVENT_IMAGE_KIND_KEY] = str(route_record["image_kind"])
    metadata[MERFISH_EVENT_ROUND_LABEL_KEY] = int(route_record["round_label"])
    metadata[MERFISH_EVENT_BIT_LABEL_KEY] = int(route_record["bit_label"])
    metadata[MERFISH_EVENT_PLANNED_Z_UM_KEY] = (
        float(event.z_pos) if event.z_pos is not None else None
    )
    return event.replace(metadata=metadata)


def _prepared_image_records(
    prepared_events: list[MDAEvent],
    route_lookup: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return normalized per-event records for image-event stamping.

    Parameters
    ----------
    prepared_events : list[MDAEvent]
        Prepared image events from the expanded image sequence.
    route_lookup : dict[tuple[int, int], dict[str, Any]]
        Routing records keyed by time and channel index.

    Returns
    -------
    list[dict[str, Any]]
        Per-event records with cached time, route, and target metadata.
    """

    records: list[dict[str, Any]] = []
    for event in prepared_events:
        time_index = _event_axis_index(event, Axis.TIME)
        channel_index = _event_axis_index(event, Axis.CHANNEL)
        route_record = route_lookup[(time_index, channel_index)]
        target_record = target_record_for_route(
            route_record, tile_index=tile_index_from_event(event)
        )
        records.append(
            {
                "event": event,
                "time_index": time_index,
                "route_record": route_record,
                "target_record": target_record,
                "close_after_write": False,
            }
        )
    return records


def target_record_for_route(
    route_record: dict[str, Any], *, tile_index: int
) -> dict[str, Any]:
    """Build the normalized datastore target record for one image event.

    Parameters
    ----------
    route_record : dict[str, Any]
        Normalized routing record for the event.
    tile_index : int
        Tile index for the event.

    Returns
    -------
    dict[str, Any]
        Stamped datastore target record.
    """

    image_kind = str(route_record["image_kind"])
    round_label = int(route_record["round_label"])
    bit_label = int(route_record["bit_label"])
    if image_kind == ImageKind.FIDUCIAL.value:
        folder_relpath = (
            Path("fiducial") / f"tile{int(tile_index):04d}" / f"round{round_label:03d}"
        )
    else:
        folder_relpath = (
            Path("readouts") / f"tile{int(tile_index):04d}" / f"bit{bit_label:03d}"
        )
    return {
        "tile_index": int(tile_index),
        "image_kind": image_kind,
        "channel_index": int(route_record["channel_index"]),
        "channel_config": str(route_record["channel_config"]),
        "round_label": round_label,
        "bit_label": bit_label,
        "folder_relpath": str(folder_relpath),
        "image_relpath": str(folder_relpath / "corrected_data.ome.zarr"),
    }


def _reference_probe_from_sequence(
    sequence: MDASequence,
    *,
    reference_tile: int,
    fiducial_channel_index: int,
    z_step_um: float,
) -> dict[str, Any] | None:
    """Extract the round-1 fiducial probe used for drift correction.

    Parameters
    ----------
    sequence : MDASequence
        Prepared image sequence.
    reference_tile : int
        Tile index used as the drift reference.
    fiducial_channel_index : int
        Fiducial channel index.
    z_step_um : float
        Nominal z spacing in microns.

    Returns
    -------
    dict[str, Any] or None
        Drift-reference payload, or ``None`` when unavailable.
    """

    acquire_events = [
        event
        for event in sequence.iter_events()
        if _event_axis_index(event, Axis.TIME) == 0
        and tile_index_from_event(event) == int(reference_tile)
        and _event_axis_index(event, Axis.CHANNEL) == int(fiducial_channel_index)
    ]
    if not acquire_events:
        return None

    probe_event = acquire_events[0]
    z_positions = [
        float(event.z_pos) for event in acquire_events if event.z_pos is not None
    ]
    if not z_positions and probe_event.z_pos is not None:
        z_positions = [float(probe_event.z_pos)]

    reference_z_um = float(z_positions[0]) if z_positions else 0.0
    drift_z_positions = list(z_positions)
    if len(drift_z_positions) < 2:
        half_width = DRIFT_SWEEP_PLANES_2D // 2
        drift_z_positions = [
            reference_z_um + (index - half_width) * float(z_step_um)
            for index in range(DRIFT_SWEEP_PLANES_2D)
        ]

    return {
        "reference_tile": int(reference_tile),
        "x_pos": float(probe_event.x_pos) if probe_event.x_pos is not None else None,
        "y_pos": float(probe_event.y_pos) if probe_event.y_pos is not None else None,
        "reference_z_um": reference_z_um,
        "z_positions": drift_z_positions,
        "channel_group": probe_event.channel.group,
        "channel_config": probe_event.channel.config,
        "exposure_ms": probe_event.exposure,
    }


def ordered_channel_bits_for_rounds(
    channel_names: list[str],
    experiment_order: dict[int, dict[str, int]],
    rounds: list[int],
) -> list[dict[str, Any]]:
    """Return ordered per-round channel-bit records.

    Parameters
    ----------
    channel_names : list[str]
        Channel config names in the desired output order.
    experiment_order : dict[int, dict[str, int]]
        Round-to-channel-bit mapping.
    rounds : list[int]
        Imaging rounds included in the run.

    Returns
    -------
    list[dict[str, Any]]
        Ordered round records with channel bits matching ``channel_names``.
    """

    return [
        {
            "round_label": int(round_label),
            "channel_bits": [
                int(experiment_order[int(round_label)][channel_name])
                for channel_name in channel_names
            ],
        }
        for round_label in rounds
    ]


def route_records_for_rounds(
    merfish_metadata: dict[str, Any], rounds: list[int]
) -> list[dict[str, Any]]:
    """Build normalized routing records for all rounds and channels.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata.
    rounds : list[int]
        Imaging rounds included in the run.

    Returns
    -------
    list[dict[str, Any]]
        Routing records in time/channel order.
    """

    channel_specs = [
        {
            "channel_index": int(record["channel_index"]),
            "config_name": str(record["config_name"]),
            "role": str(record["role"]),
            "excitation_um": float(record["excitation_um"]),
            "emission_um": float(record["emission_um"]),
        }
        for record in merfish_metadata["channel_specs"]
    ]
    experiment_order = {
        int(round_id): {
            str(channel_name): int(bit)
            for channel_name, bit in channel_bits.items()
        }
        for round_id, channel_bits in merfish_metadata["experiment_order"].items()
    }

    ordered_channel_names = [
        str(spec["config_name"])
        for spec in sorted(channel_specs, key=channel_index_key)
    ]
    readout_channel_names = [
        str(spec["config_name"])
        for spec in channel_specs
        if str(spec["role"]) == ImageKind.READOUT.value
    ]
    route_records: list[dict[str, Any]] = []
    for time_index, round_record in enumerate(
        ordered_channel_bits_for_rounds(ordered_channel_names, experiment_order, rounds)
    ):
        round_label = int(round_record["round_label"])
        round_bits = dict(zip(ordered_channel_names, round_record["channel_bits"], strict=True))
        bit_linker = [int(round_bits[channel_name]) for channel_name in readout_channel_names]
        for channel_spec in channel_specs:
            channel_name = str(channel_spec["config_name"])
            bit_label = int(round_bits[channel_name])
            image_kind = str(channel_spec["role"])
            route_records.append(
                {
                    "time_index": int(time_index),
                    "channel_index": int(channel_spec["channel_index"]),
                    "channel_config": channel_name,
                    "image_kind": image_kind,
                    "round_label": round_label,
                    "bit_label": bit_label,
                    "excitation_um": float(channel_spec["excitation_um"]),
                    "emission_um": float(channel_spec["emission_um"]),
                    "linker": (
                        {"bit_linker": list(bit_linker)}
                        if image_kind == ImageKind.FIDUCIAL.value
                        else {"round_linker": round_label}
                    ),
                }
            )
    return route_records


def _round_action_plan(
    merfish_metadata: dict[str, Any], rounds: list[int]
) -> list[dict[str, Any]]:
    """Build the per-round MERFISH custom-action plan.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata.
    rounds : list[int]
        Imaging rounds included in the run.

    Returns
    -------
    list[dict[str, Any]]
        Per-round fluidics and drift actions.
    """

    run_mode = RunMode(str(merfish_metadata["run_mode"]))
    fluidics_active = run_mode is not RunMode.SINGLE_ROUND
    drift_active = run_mode is not RunMode.FLUIDICS_ONLY and bool(
        merfish_metadata["enable_drift_correction"]
    )
    return [
        {
            "time_index": int(time_index),
            "round_label": int(round_label),
            "run_fluidics": bool(fluidics_active),
            "run_drift": bool(drift_active and time_index > 0),
        }
        for time_index, round_label in enumerate(rounds)
    ]
