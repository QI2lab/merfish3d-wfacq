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

from merfish3d_wfacq.input_metadata import calibration_input_metadata
from merfish3d_wfacq.sequence import route_records_for_rounds, target_record_for_route
from merfish3d_wfacq.utils.data_io import append_index_filepath, json_value, write_json


def resolve_experiment_root(
    requested_root: str | Path, *, overwrite: bool = False
) -> Path:
    """Resolve the experiment root directory for a MERFISH run.

    Parameters
    ----------
    requested_root : str or Path
        Requested save location from the MDA workflow.
    overwrite : bool, optional
        Whether to reuse an existing root path.

    Returns
    -------
    Path
        Experiment root with OME suffixes removed and collisions avoided.
    """

    root = Path(requested_root)
    if root.suffix or root.name.endswith((".ome.zarr", ".ome.tif", ".ome.tiff")):
        root = root.parent / _stem_without_ome_suffix(root)
    if not overwrite and root.exists():
        root = append_index_filepath(root)
    return root


def verify_static_metadata(
    merfish_metadata: dict[str, Any], image_info: dict[str, Any]
) -> None:
    """Verify summary metadata against pre-acquisition core metadata.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata prepared before acquisition.
    image_info : dict[str, Any]
        Summary image info emitted by ``pymmcore-plus``.
    """

    pixel_size_um = float(image_info["pixel_size_um"])
    expected_pixel_size = float(merfish_metadata["default_pixel_size_um"])
    if not np.isclose(
        float(pixel_size_um),
        float(expected_pixel_size),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Summary pixel size does not match the pre-acquisition Micro-Manager "
            "core metadata captured in the sequence."
        )

    raw_affine = image_info.get("pixel_size_affine")
    expected_affine = merfish_metadata["pixel_size_affine"]
    if raw_affine is None:
        return

    observed = np.asarray(raw_affine, dtype=np.float32)
    expected = np.asarray(expected_affine, dtype=np.float32)
    if observed.shape != expected.shape or not np.allclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Summary PixelSizeAffine does not match the pre-acquisition "
            "Micro-Manager core metadata captured in the sequence."
        )


def prepare_qi2lab_datastore(
    *,
    merfish_metadata: dict[str, Any],
    image_info: dict[str, Any],
    illumination_profiles: np.ndarray,
    stream_factory: Callable[[AcquisitionSettings], Any] = create_stream,
    format_factory: Callable[..., OmeZarrFormat] = OmeZarrFormat,
) -> Path:
    """Create and pre-populate the qi2lab datastore layout.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    image_info : dict[str, Any]
        Summary image info emitted at sequence start.
    illumination_profiles : np.ndarray
        Resolved illumination-profile stack with shape ``(c, y, x)``.
    stream_factory : Callable[[AcquisitionSettings], Any], optional
        Factory used to create ome-writers streams.
    format_factory : Callable[..., OmeZarrFormat], optional
        Factory used to build ome-writers format models.

    Returns
    -------
    Path
        Root path of the initialized datastore.
    """

    experiment_root = Path(str(merfish_metadata["experiment_root"]))
    datastore_root = Path(str(merfish_metadata["datastore_root"]))

    verify_static_metadata(merfish_metadata, image_info)
    voxel_size_zyx_um = tuple(
        float(value) for value in merfish_metadata["voxel_size_zyx_um"]
    )
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
    illumination_profiles = np.asarray(illumination_profiles, dtype=np.float32)
    rounds = [int(round_id) for round_id in merfish_metadata["rounds"]]
    num_tiles = int(merfish_metadata["num_tiles"])
    route_records = route_records_for_rounds(merfish_metadata, rounds)

    experiment_root.mkdir(parents=True, exist_ok=True)
    _initialize_datastore_root(datastore_root)
    write_json(_datastore_state(corrected=False), datastore_root / "datastore_state.json")
    write_json(
        _calibration_attributes(
            merfish_metadata=merfish_metadata,
            image_info=image_info,
            voxel_size_zyx_um=voxel_size_zyx_um,
            channel_specs=channel_specs,
            rounds=rounds,
        ),
        datastore_root / "calibrations" / "attributes.json",
    )
    _write_shading_maps(
        datastore_root=datastore_root,
        shading_maps=illumination_profiles,
        stream_factory=stream_factory,
        format_factory=format_factory,
    )

    _initialize_tiles(
        datastore_root=datastore_root,
        num_tiles=num_tiles,
        route_records=route_records,
        voxel_size_zyx_um=voxel_size_zyx_um,
        affine_zyx_px=merfish_metadata["affine_zyx_px"],
        shading_correction=True,
        camera_offset_adu=float(merfish_metadata["camera_offset_adu"]),
        e_per_adu=float(merfish_metadata["e_per_adu"]),
    )
    return datastore_root


def _initialize_datastore_root(datastore_root: Path) -> None:
    """Create the static top-level qi2lab datastore folders.

    Parameters
    ----------
    datastore_root : Path
        Root directory of the datastore being initialized.
    """

    datastore_root.mkdir(parents=True, exist_ok=True)
    for folder in (
        "calibrations",
        "fiducial",
        "readouts",
        "feature_predictor_localizations",
        "decoded",
        "fused",
        "segmentation",
        "mtx_output",
    ):
        (datastore_root / folder).mkdir(parents=True, exist_ok=True)


def _write_shading_maps(
    *,
    datastore_root: Path,
    shading_maps: np.ndarray,
    stream_factory: Callable[[AcquisitionSettings], Any],
    format_factory: Callable[..., OmeZarrFormat],
) -> None:
    """Write illumination profiles into the calibration store.

    Parameters
    ----------
    datastore_root : Path
        Root directory of the datastore being initialized.
    shading_maps : np.ndarray
        Illumination profile stack with shape ``(c, y, x)``.
    stream_factory : Callable[[AcquisitionSettings], Any]
        Factory used to create ``ome-writers`` streams.
    format_factory : Callable[..., OmeZarrFormat]
        Factory used to build ``ome-writers`` format models.
    """

    path = datastore_root / "calibrations" / "shading_maps"
    settings = AcquisitionSettings.model_validate(
        {
            "root_path": str(path),
            "overwrite": True,
            "format": format_factory(backend="tensorstore", suffix=""),
            "dimensions": [
                StandardAxis("c").to_dimension(count=int(shading_maps.shape[0])),
                StandardAxis("y").to_dimension(count=int(shading_maps.shape[1])),
                StandardAxis("x").to_dimension(count=int(shading_maps.shape[2])),
            ],
            "dtype": np.dtype(np.float32).name,
        }
    )
    with stream_factory(settings) as stream:
        for plane in np.asarray(shading_maps, dtype=np.float32):
            stream.append(plane)


def _stem_without_ome_suffix(path: Path) -> str:
    """Return a path stem with any OME suffix removed.

    Parameters
    ----------
    path : Path
        Candidate file or directory path.

    Returns
    -------
    str
        Path stem without any OME image suffix.
    """

    name = path.name
    for suffix in (".ome.zarr", ".ome.tiff", ".ome.tif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _datastore_state(*, corrected: bool) -> dict[str, Any]:
    """Return the top-level qi2lab datastore state payload.

    Parameters
    ----------
    corrected : bool
        Whether corrected image data has been written.

    Returns
    -------
    dict[str, Any]
        Serialized datastore state payload.
    """

    return {
        "Version": 0.6,
        "Initialized": True,
        "Calibrations": True,
        "Corrected": bool(corrected),
        "LocalRegistered": False,
        "GlobalRegistered": False,
        "Fused": False,
        "SegmentedCells": False,
        "DecodedSpots": False,
        "FilteredSpots": False,
        "RefinedSpots": False,
        "mtxOutput": False,
        "BaysorPath": "",
        "BaysorOptions": "",
        "JuliaThreads": "0",
    }


def _calibration_attributes(
    *,
    merfish_metadata: dict[str, Any],
    image_info: dict[str, Any],
    voxel_size_zyx_um: tuple[float, float, float],
    channel_specs: list[dict[str, Any]],
    rounds: list[int],
) -> dict[str, Any]:
    """Build calibration attributes for datastore initialization.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    image_info : dict[str, Any]
        Summary image info emitted at sequence start.
    voxel_size_zyx_um : tuple[float, float, float]
        Voxel size in microns.
    channel_specs : list[dict[str, Any]]
        Normalized channel specifications.
    rounds : list[int]
        Imaging rounds included in the run.

    Returns
    -------
    dict[str, Any]
        Calibration attributes written to the datastore.
    """

    input_metadata = calibration_input_metadata(merfish_metadata)
    bit_values = [
        int(bit)
        for channel_bits in merfish_metadata["experiment_order"].values()
        for bit in channel_bits.values()
        if int(bit) > 0
    ]

    return {
        **input_metadata,
        "channels_in_data": [str(spec["config_name"]) for spec in channel_specs],
        "channel_specs": [dict(spec) for spec in channel_specs],
        "voxel_size_zyx_um": list(voxel_size_zyx_um),
        "num_rounds": len(rounds),
        "num_bits": int(max(bit_values)),
        "num_tiles": int(merfish_metadata["num_tiles"]),
        "fiducial_channel_index": int(merfish_metadata["fiducial_channel_index"]),
        "microscope_type": merfish_metadata["microscope_type"],
        "tile_overlap": float(merfish_metadata["tile_overlap"]),
        "e_per_ADU": float(merfish_metadata["e_per_adu"]),
        "camera_offset_adu": float(merfish_metadata["camera_offset_adu"]),
        "na": float(merfish_metadata["numerical_aperture"]),
        "ri": float(merfish_metadata["refractive_index"]),
        "binning": int(merfish_metadata["binning"]),
        "camera_model": str(merfish_metadata["camera_model"]),
        "psf_manifest": {},
        "available_calibration_references": {
            "shading_maps": True,
            "noise_map": False,
            "psf_data": 0,
        },
    }


def _initialize_tiles(
    *,
    datastore_root: Path,
    num_tiles: int,
    route_records: list[dict[str, Any]],
    voxel_size_zyx_um: tuple[float, float, float],
    affine_zyx_px: Any,
    shading_correction: bool,
    camera_offset_adu: float,
    e_per_adu: float,
) -> None:
    """Create per-tile fiducial and readout attribute files.

    Parameters
    ----------
    datastore_root : Path
        Root directory of the datastore being initialized.
    num_tiles : int
        Number of image tiles in the experiment.
    route_records : list[dict[str, Any]]
        Normalized routing records shared with event generation.
    voxel_size_zyx_um : tuple[float, float, float]
        Voxel size in microns.
    affine_zyx_px : Any
        Camera-to-stage affine expressed in pixels.
    shading_correction : bool
        Whether shading correction is enabled.
    camera_offset_adu : float
        Camera offset in ADU.
    e_per_adu : float
        Camera gain in electrons per ADU.
    """

    for tile_index in range(int(num_tiles)):
        for route_record in route_records:
            target_record = target_record_for_route(
                route_record, tile_index=int(tile_index)
            )
            target_root = datastore_root / Path(str(target_record["folder_relpath"]))
            target_root.mkdir(parents=True, exist_ok=True)
            write_json(
                _target_attributes(
                    target_record=target_record,
                    route_record=route_record,
                    voxel_size_zyx_um=voxel_size_zyx_um,
                    affine_zyx_px=affine_zyx_px,
                    shading_correction=shading_correction,
                    camera_offset_adu=camera_offset_adu,
                    e_per_adu=e_per_adu,
                ),
                target_root / "attributes.json",
            )


def _target_attributes(
    *,
    target_record: dict[str, Any],
    route_record: dict[str, Any],
    voxel_size_zyx_um: tuple[float, float, float],
    affine_zyx_px: Any,
    shading_correction: bool,
    camera_offset_adu: float,
    e_per_adu: float,
) -> dict[str, Any]:
    """Build the attribute payload for one datastore target.

    Parameters
    ----------
    target_record : dict[str, Any]
        Normalized datastore target record.
    route_record : dict[str, Any]
        Normalized routing record shared with event generation.
    voxel_size_zyx_um : tuple[float, float, float]
        Voxel size in microns.
    affine_zyx_px : Any
        Camera-to-stage affine expressed in pixels.
    shading_correction : bool
        Whether shading correction is enabled.
    camera_offset_adu : float
        Camera offset in ADU.
    e_per_adu : float
        Camera gain in electrons per ADU.

    Returns
    -------
    dict[str, Any]
        Attribute payload written for the target.
    """

    channel_index = int(route_record["channel_index"])
    return {
        "tile_index": int(target_record["tile_index"]),
        "tile_id": f"tile{int(target_record['tile_index']):04d}",
        "image_kind": str(target_record["image_kind"]),
        "channel_index": channel_index,
        "channel_config": str(route_record["channel_config"]),
        "round_label": int(target_record["round_label"]),
        "bit_label": int(target_record["bit_label"]),
        "stage_zyx_um": [None, None, None],
        "voxel_size_zyx_um": list(voxel_size_zyx_um),
        "excitation_um": float(route_record["excitation_um"]),
        "emission_um": float(route_record["emission_um"]),
        "affine_zyx_px": affine_zyx_px,
        "psf_idx": channel_index,
        "gain_correction": True,
        "hotpixel_correction": False,
        "shading_correction": bool(shading_correction),
        "camera_offset_adu": float(camera_offset_adu),
        "e_per_adu": float(e_per_adu),
        "applied_z_offset_um": 0.0,
        **{str(key): json_value(value) for key, value in dict(route_record["linker"]).items()},
    }
