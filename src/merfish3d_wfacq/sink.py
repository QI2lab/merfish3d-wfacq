import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ome_writers import (
    AcquisitionSettings,
    OmeZarrFormat,
    StandardAxis,
    create_stream,
)
from pymmcore_plus.mda._sink import _frame_meta_to_ome
from useq import MDAEvent, MDASequence

from merfish3d_wfacq.datastore import prepare_qi2lab_datastore
from merfish3d_wfacq.illumination import resolved_illumination_profiles
from merfish3d_wfacq.sequence import (
    MERFISH_EVENT_PLANNED_Z_UM_KEY,
    MERFISH_EVENT_TARGET_CLOSE_KEY,
    MERFISH_EVENT_TARGET_KEY,
)
from merfish3d_wfacq.utils.data_io import json_value, write_json


def _stream_dimensions(
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    scales_um: dict[str, float],
) -> list[Any]:
    """Build ome-writers dimensions for one array layout.

    Parameters
    ----------
    axes : tuple[str, ...]
        Logical axis names for the target array.
    shape : tuple[int, ...]
        Array shape matching ``axes``.
    scales_um : dict[str, float]
        Physical scales for spatial axes in microns.

    Returns
    -------
    list[Any]
        ``ome-writers`` dimension models for the target array.
    """

    return [
        StandardAxis(axis_name).to_dimension(
            count=int(axis_size),
            scale=(float(scales_um[axis_name]) if axis_name in {"z", "y", "x"} else None),
        )
        for axis_name, axis_size in zip(axes, shape, strict=True)
    ]


class Qi2labDatastoreWriter:
    """MDA listener that routes prepared MERFISH image events to ome-writers."""

    def __init__(
        self,
        *,
        merfish_metadata: dict[str, Any],
        stream_factory: Callable[[AcquisitionSettings], Any] = create_stream,
        format_factory: Callable[..., OmeZarrFormat] = OmeZarrFormat,
    ) -> None:
        """Initialize the writer from normalized MERFISH metadata.

        Parameters
        ----------
        merfish_metadata : dict[str, Any]
            Normalized MERFISH metadata prepared before acquisition.
        stream_factory : Callable[[AcquisitionSettings], Any], optional
            Factory used to create ``ome-writers`` streams.
        format_factory : Callable[..., OmeZarrFormat], optional
            Factory used to create ``ome-writers`` format models.
        """

        self._merfish_metadata = dict(merfish_metadata)
        self._stream_factory = stream_factory
        self._format_factory = format_factory
        self._streams: dict[Path, Any] = {}
        self._voxel_size_zyx_um = (1.0, 1.0, 1.0)
        self._state: dict[str, Any] = {"Corrected": False}
        self._written_targets: set[str] = set()
        self._illumination_profiles: np.ndarray | None = None
        self.output_path: Path | None = None

    def set_illumination_profiles(self, illumination_profiles: np.ndarray) -> None:
        """Set resolved illumination profiles for datastore setup.

        Parameters
        ----------
        illumination_profiles : np.ndarray
            Resolved illumination-profile stack with shape ``(c, y, x)``.
        """

        self._illumination_profiles = np.asarray(illumination_profiles, dtype=np.float32)

    def sequenceStarted(self, sequence: MDASequence, meta: dict[str, Any]) -> None:
        """Prepare the datastore layout at sequence start.

        Parameters
        ----------
        sequence : MDASequence
            Sequence emitted by ``pymmcore-plus``.
        meta : dict[str, Any]
            Sequence-start summary metadata from ``pymmcore-plus``.
        """

        info = meta["image_infos"][0]
        if self._illumination_profiles is None:
            raise RuntimeError(
                "Resolved illumination profiles must be set before the datastore writer starts."
            )
        self._voxel_size_zyx_um = tuple(
            float(item) for item in self._merfish_metadata["voxel_size_zyx_um"]
        )
        self._written_targets.clear()
        self.output_path = Path(str(self._merfish_metadata["datastore_root"]))
        self.output_path = prepare_qi2lab_datastore(
            merfish_metadata=self._merfish_metadata,
            image_info=dict(info),
            illumination_profiles=np.asarray(self._illumination_profiles, dtype=np.float32),
            stream_factory=self._stream_factory,
            format_factory=self._format_factory,
        )
        self._state = json.loads(
            (self.output_path / "datastore_state.json").read_text(encoding="utf-8")
        )

    def frameReady(
        self, img: np.ndarray, event: MDAEvent, meta: dict[str, Any]
    ) -> None:
        """Append one corrected frame to its stamped datastore target.

        Parameters
        ----------
        img : np.ndarray
            Corrected image frame to write.
        event : MDAEvent
            Prepared image event carrying stamped routing metadata.
        meta : dict[str, Any]
            Frame metadata emitted by ``pymmcore-plus``.
        """

        self._write_target_frame(
            event.metadata[MERFISH_EVENT_TARGET_KEY], np.asarray(img), event, meta
        )

    def sequenceFinished(self, _sequence: MDASequence) -> None:
        """Finalize datastore state and close any open streams.

        Parameters
        ----------
        _sequence : MDASequence
            Sequence that just finished.
        """

        self._state["Corrected"] = bool(self._written_targets)
        write_json(self._state, self.output_path / "datastore_state.json")
        self._close_streams()

    def _write_target_frame(
        self,
        target: dict[str, Any],
        frame: np.ndarray,
        event: MDAEvent,
        runner_meta: dict[str, Any],
    ) -> None:
        """Write one frame into the appropriate datastore target.

        Parameters
        ----------
        target : dict[str, Any]
            Prepared target metadata for the frame.
        frame : np.ndarray
            Corrected image payload.
        event : MDAEvent
            Prepared image event being written.
        runner_meta : dict[str, Any]
            Frame metadata emitted by ``pymmcore-plus``.
        """

        target_root = self.output_path / Path(str(target["folder_relpath"]))
        target_attributes_path = target_root / "attributes.json"
        target_key = str(target["image_relpath"])
        image_path = self.output_path / Path(str(target["image_relpath"]))
        position = runner_meta["position"]
        stage_zyx_um = [
            float(position["z"]),
            float(position["y"]),
            float(position["x"]),
        ]
        planned_z = event.metadata[MERFISH_EVENT_PLANNED_Z_UM_KEY]
        applied_z_offset_um = (
            0.0
            if planned_z is None or event.z_pos is None
            else float(event.z_pos) - float(planned_z)
        )
        if target_key not in self._written_targets:
            target_attributes = json.loads(
                target_attributes_path.read_text(encoding="utf-8")
            )
            target_attributes["stage_zyx_um"] = stage_zyx_um
            target_attributes["applied_z_offset_um"] = applied_z_offset_um
            write_json(target_attributes, target_root / "attributes.json")

        self._append_target_frame(
            image_path,
            np.asarray(frame),
            z_count=int(self._merfish_metadata["num_z_planes"]),
            scales_um={
                "z": self._voxel_size_zyx_um[0],
                "y": self._voxel_size_zyx_um[1],
                "x": self._voxel_size_zyx_um[2],
            },
            frame_metadata=self._frame_metadata(
                target,
                runner_meta,
                applied_z_offset_um=applied_z_offset_um,
            ),
        )
        self._written_targets.add(target_key)
        if bool(event.metadata[MERFISH_EVENT_TARGET_CLOSE_KEY]):
            self._close_stream(image_path)

    def _frame_metadata(
        self,
        target: dict[str, Any],
        runner_meta: dict[str, Any],
        *,
        applied_z_offset_um: float,
    ) -> dict[str, Any]:
        """Build per-frame ome-writers metadata for one MERFISH image.

        Parameters
        ----------
        target : dict[str, Any]
            Prepared target metadata for the frame.
        runner_meta : dict[str, Any]
            Frame metadata emitted by ``pymmcore-plus``.
        applied_z_offset_um : float
            Drift offset applied to the written image event.

        Returns
        -------
        dict[str, Any]
            ``ome-writers`` frame metadata for the target image.
        """

        base_metadata = dict(_frame_meta_to_ome(runner_meta))
        base_metadata.update(
            {
                "round": int(target["round_label"]),
                "bit": int(target["bit_label"]),
                "tile_index": int(target["tile_index"]),
                "channel_index": int(target["channel_index"]),
                "channel": str(target["channel_config"]),
                "image_kind": str(target["image_kind"]),
                "applied_z_offset_um": float(applied_z_offset_um),
            }
        )
        return base_metadata

    def _append_target_frame(
        self,
        path: Path,
        frame: np.ndarray,
        *,
        z_count: int,
        scales_um: dict[str, float],
        frame_metadata: dict[str, Any],
    ) -> None:
        """Append one frame to the target ome-writers stream.

        Parameters
        ----------
        path : Path
            OME-Zarr path for the target stream.
        frame : np.ndarray
            Corrected image frame.
        z_count : int
            Number of z planes expected for the target stream.
        scales_um : dict[str, float]
            Spatial axis scales in microns.
        frame_metadata : dict[str, Any]
            Per-frame metadata passed to ``ome-writers``.
        """

        frame_array = np.asarray(frame)
        stream = self._stream_for_frame(path, frame_array, z_count, scales_um)
        stream.append(
            frame_array,
            frame_metadata={
                str(key): json_value(value)
                for key, value in frame_metadata.items()
            },
        )

    def _stream_for_array(
        self,
        path: Path,
        array: np.ndarray,
        axes: tuple[str, ...],
        scales_um: dict[str, float],
    ) -> Any:
        """Return a cached ome-writers stream for one array layout.

        Parameters
        ----------
        path : Path
            Path used for the target stream.
        array : np.ndarray
            Template array describing shape and dtype.
        axes : tuple[str, ...]
            Logical axis order for the target array.
        scales_um : dict[str, float]
            Spatial axis scales in microns.

        Returns
        -------
        Any
            Open ``ome-writers`` stream for the target path.
        """

        if path in self._streams:
            return self._streams[path]

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith(".ome.zarr"):
            format_model = self._format_factory(backend="tensorstore")
        else:
            format_model = self._format_factory(backend="tensorstore", suffix="")
        settings = AcquisitionSettings.model_validate(
            {
                "root_path": str(path),
                "overwrite": True,
                "format": format_model,
                "dimensions": _stream_dimensions(axes, tuple(array.shape), scales_um),
                "dtype": np.dtype(array.dtype).name,
            }
        )
        stream = self._stream_factory(settings)
        self._streams[path] = stream
        return stream

    def _stream_for_frame(
        self,
        path: Path,
        frame: np.ndarray,
        z_count: int,
        scales_um: dict[str, float],
    ) -> Any:
        """Return the ome-writers stream used for one corrected frame.

        Parameters
        ----------
        path : Path
            Path used for the target stream.
        frame : np.ndarray
            Corrected image frame that will be appended.
        z_count : int
            Number of z planes expected for the target stream.
        scales_um : dict[str, float]
            Spatial axis scales in microns.

        Returns
        -------
        Any
            Open ``ome-writers`` stream for the frame.
        """

        if z_count > 1:
            shape = (int(z_count), *frame.shape)
            axes = ("z", "y", "x")
        else:
            shape = tuple(frame.shape)
            axes = ("y", "x")
        template = np.empty(shape, dtype=frame.dtype)
        return self._stream_for_array(path, template, axes, scales_um)

    def _close_streams(self) -> None:
        """Close and clear all cached ome-writers streams."""

        for stream in self._streams.values():
            stream.close()
        self._streams.clear()

    def _close_stream(self, path: Path) -> None:
        """Close one cached ome-writers stream.

        Parameters
        ----------
        path : Path
            Path identifying the cached stream.
        """

        stream = self._streams.pop(path, None)
        if stream is not None:
            stream.close()


class MerfishFrameProcessor:
    """Forwarding MDA listener that corrects frames before writing them."""

    def __init__(
        self,
        *,
        merfish_metadata: dict[str, Any],
        drift_reference_store: dict[str, Any] | None = None,
        writer: Qi2labDatastoreWriter | None = None,
    ) -> None:
        """Initialize the processor and downstream datastore writer.

        Parameters
        ----------
        merfish_metadata : dict[str, Any]
            Normalized MERFISH metadata prepared before acquisition.
        drift_reference_store : dict[str, Any] or None, optional
            Shared store used to capture the round-1 fiducial reference stack.
        writer : Qi2labDatastoreWriter or None, optional
            Downstream writer used after frame correction.
        """

        self._merfish_metadata = dict(merfish_metadata)
        self._drift_reference_store = (
            drift_reference_store
            if drift_reference_store is not None
            else {
                "reference_tile": int(self._merfish_metadata["reference_tile"]),
                "frames": [],
            }
        )
        self.writer = writer or Qi2labDatastoreWriter(merfish_metadata=merfish_metadata)
        self._illumination_profiles: np.ndarray | None = None
        self._camera_dtype: np.dtype[Any] | None = None
        self._camera_offset_adu = float(
            self._merfish_metadata["camera_offset_adu"]
        )
        self._e_per_adu = float(
            self._merfish_metadata["e_per_adu"]
        )

    @property
    def output_path(self) -> Path | None:
        """Return the downstream datastore root path, if prepared.

        Returns
        -------
        Path or None
            Prepared datastore root path.
        """

        return self.writer.output_path

    @property
    def drift_reference_store(self) -> dict[str, Any]:
        """Return the shared in-memory drift reference store.

        Returns
        -------
        dict[str, Any]
            Shared drift reference store for the current run.
        """

        return self._drift_reference_store

    def set_illumination_profiles(self, illumination_profiles: np.ndarray) -> None:
        """Set resolved illumination profiles for datastore setup.

        Parameters
        ----------
        illumination_profiles : np.ndarray
            Resolved illumination-profile stack with shape ``(c, y, x)``.
        """

        self._illumination_profiles = np.asarray(illumination_profiles, dtype=np.float32)

    def sequenceStarted(self, sequence: MDASequence, meta: dict[str, Any]) -> None:
        """Cache correction assets and prepare the downstream writer.

        Parameters
        ----------
        sequence : MDASequence
            Sequence emitted by ``pymmcore-plus``.
        meta : dict[str, Any]
            Sequence-start summary metadata from ``pymmcore-plus``.
        """

        image_info = meta["image_infos"][0]
        self._illumination_profiles = resolved_illumination_profiles(
            self._merfish_metadata, image_info
        )
        self.writer.set_illumination_profiles(self._illumination_profiles)
        self._camera_dtype = np.dtype(image_info["dtype"])
        self._drift_reference_store["frames"] = []
        self.writer.sequenceStarted(sequence, meta)

    def frameReady(
        self, img: np.ndarray, event: MDAEvent, meta: dict[str, Any]
    ) -> None:
        """Correct one frame and forward it to the datastore writer.

        Parameters
        ----------
        img : np.ndarray
            Raw image frame from ``pymmcore-plus``.
        event : MDAEvent
            Prepared image event carrying stamped routing metadata.
        meta : dict[str, Any]
            Frame metadata emitted by ``pymmcore-plus``.
        """

        target = event.metadata[MERFISH_EVENT_TARGET_KEY]
        corrected = self._apply_corrections(np.asarray(img), int(target["channel_index"]))
        self._capture_reference_frame(target, corrected)
        self.writer.frameReady(corrected, event, meta)

    def sequenceFinished(self, sequence: MDASequence) -> None:
        """Finalize the downstream writer after acquisition.

        Parameters
        ----------
        sequence : MDASequence
            Sequence that just finished.
        """

        self.writer.sequenceFinished(sequence)

    def _apply_corrections(self, stack: np.ndarray, channel_index: int) -> np.ndarray:
        """Apply offset, gain, and flatfield correction to one frame.

        Parameters
        ----------
        stack : np.ndarray
            Raw image frame to correct.
        channel_index : int
            Channel index selecting the illumination profile.

        Returns
        -------
        np.ndarray
            Corrected frame cast back to the camera dtype.
        """

        corrected = np.maximum(
            np.asarray(stack, dtype=np.float32) - self._camera_offset_adu,
            0.0,
        )
        corrected *= self._e_per_adu

        profile = np.asarray(self._illumination_profiles[channel_index], dtype=np.float32)
        if tuple(profile.shape) != tuple(corrected.shape[-2:]):
            raise ValueError(
                "Illumination-profile YX shape must match the image shape. "
                f"Expected {corrected.shape[-2:]}, received {profile.shape}."
            )
        corrected = corrected / np.maximum(profile, np.finfo(np.float32).eps)

        if np.issubdtype(self._camera_dtype, np.integer):
            info = np.iinfo(self._camera_dtype)
            corrected = np.clip(corrected, info.min, info.max)
        return corrected.astype(self._camera_dtype, copy=False)

    def _capture_reference_frame(
        self, target: dict[str, Any], corrected: np.ndarray
    ) -> None:
        """Capture round-1 fiducial frames for the shared drift reference.

        Parameters
        ----------
        target : dict[str, Any]
            Prepared target metadata for the frame.
        corrected : np.ndarray
            Corrected frame that will be written.
        """

        if str(target["image_kind"]) != "fiducial":
            return
        if int(target["round_label"]) != 1:
            return
        if int(target["tile_index"]) != int(self._drift_reference_store["reference_tile"]):
            return
        self._drift_reference_store["frames"].append(np.asarray(corrected).copy())
