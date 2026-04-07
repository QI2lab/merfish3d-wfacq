from typing import Any

import numpy as np
from useq import MDASequence


class CoreMetadataError(RuntimeError):
    """Raised when required datastore metadata cannot be derived from MMCore."""


def derive_core_metadata(
    mmc: Any, sequence: MDASequence | None = None
) -> dict[str, Any]:
    """Extract datastore metadata from MMCore and the configured sequence.

    Parameters
    ----------
    mmc : Any
        Active ``CMMCorePlus`` instance.
    sequence : MDASequence or None, optional
        Current MDA sequence used to derive z spacing.

    Returns
    -------
    dict[str, Any]
        Normalized core metadata used by MERFISH acquisition setup.
    """

    camera_device = str(mmc.getCameraDevice() or "").strip()
    if not camera_device:
        raise CoreMetadataError("No camera device is configured in Micro-Manager core.")

    camera_model = _camera_model(mmc, camera_device)

    pixel_size_um = float(mmc.getPixelSizeUm(True))
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise CoreMetadataError(
            "Micro-Manager core did not provide a valid pixel size."
        )

    raw_affine = tuple(float(value) for value in mmc.getPixelSizeAffine(True))
    if len(raw_affine) != 6:
        raise CoreMetadataError(
            "Micro-Manager core returned an invalid PixelSizeAffine payload."
        )
    affine_zyx_px = pixel_size_affine_to_affine_zyx_px(raw_affine, pixel_size_um)

    binning = _required_camera_numeric_property(
        mmc,
        camera_device,
        aliases=("binning",),
        error_message="Camera binning is not available from Micro-Manager core.",
    )
    e_per_adu = _required_camera_numeric_property(
        mmc,
        camera_device,
        aliases=(
            "conversion factor coeff",
            "conversionfactorcoeff",
            "camera-conversion factor coeff",
            "photon conversion factor",
        ),
        error_message=(
            "Camera conversion factor (e-/ADU) is not available from "
            "Micro-Manager core."
        ),
    )
    camera_offset_adu = _required_camera_numeric_property(
        mmc,
        camera_device,
        aliases=(
            "conversion factor offset",
            "conversionfactoroffset",
            "camera-conversion factor offset",
            "offset",
        ),
        error_message=("Camera offset (ADU) is not available from Micro-Manager core."),
    )

    return {
        "camera_model": camera_model,
        "pixel_size_um": pixel_size_um,
        "pixel_size_affine": raw_affine,
        "affine_zyx_px": affine_zyx_px,
        "binning": int(binning),
        "e_per_adu": float(e_per_adu),
        "camera_offset_adu": float(camera_offset_adu),
        "z_step_um": _z_step_um(sequence),
    }


def pixel_size_affine_to_affine_zyx_px(
    pixel_size_affine: tuple[float, float, float, float, float, float]
    | list[float]
    | tuple[float, ...],
    pixel_size_um: float,
) -> list[list[float]]:
    """Convert MMCore PixelSizeAffine into the 4x4 qi2lab camera-stage affine."""

    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise CoreMetadataError("Pixel size must be finite and positive.")

    affine_values = np.asarray(pixel_size_affine, dtype=np.float32)
    if affine_values.shape != (6,):
        raise CoreMetadataError("PixelSizeAffine must contain exactly six elements.")
    affine_values = np.round(affine_values / float(pixel_size_um), 2)
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, affine_values[4], affine_values[3], 0.0],
            [0.0, affine_values[1], affine_values[0], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ).tolist()


def _camera_model(mmc: Any, camera_device: str) -> str:
    """Return the best available camera model string from MMCore.

    Parameters
    ----------
    mmc : Any
        Active ``CMMCorePlus`` instance.
    camera_device : str
        Active camera device label.

    Returns
    -------
    str
        Best available camera model string.
    """

    for property_name in ("CameraName", "Camera-CameraName"):
        try:
            value = str(mmc.getProperty(camera_device, property_name)).strip()
        except Exception:
            continue
        if value:
            return value
    return camera_device


def _required_camera_numeric_property(
    mmc: Any,
    camera_device: str,
    *,
    aliases: tuple[str, ...],
    error_message: str,
) -> float:
    """Return one required numeric camera property from MMCore.

    Parameters
    ----------
    mmc : Any
        Active ``CMMCorePlus`` instance.
    camera_device : str
        Active camera device label.
    aliases : tuple[str, ...]
        Candidate property aliases to search.
    error_message : str
        Error raised when the property cannot be found.

    Returns
    -------
    float
        Resolved numeric property value.
    """

    property_names = _property_name_map(mmc, camera_device)
    for alias in aliases:
        property_name = property_names.get(_normalize_property_name(alias))
        if property_name is None:
            continue
        try:
            value = float(mmc.getProperty(camera_device, property_name))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    raise CoreMetadataError(error_message)


def _property_name_map(mmc: Any, device_label: str) -> dict[str, str]:
    """Map normalized MMCore property names back to raw property labels.

    Parameters
    ----------
    mmc : Any
        Active ``CMMCorePlus`` instance.
    device_label : str
        Device label queried for properties.

    Returns
    -------
    dict[str, str]
        Mapping from normalized aliases to raw MMCore property names.
    """

    try:
        property_names = list(mmc.getDevicePropertyNames(device_label))
    except Exception:
        property_names = []
    return {_normalize_property_name(name): str(name) for name in property_names}


def _normalize_property_name(name: str) -> str:
    """Normalize a property name for alias matching.

    Parameters
    ----------
    name : str
        Raw property name.

    Returns
    -------
    str
        Lowercase alphanumeric alias.
    """

    return "".join(character for character in str(name).lower() if character.isalnum())


def _z_step_um(sequence: MDASequence | None) -> float:
    """Return the nominal z spacing encoded in a sequence.

    Parameters
    ----------
    sequence : MDASequence or None
        Sequence whose z plan should be inspected.

    Returns
    -------
    float
        Nominal z spacing in microns.
    """

    if sequence is None or sequence.z_plan is None:
        return 1.0
    try:
        positions = [float(position) for position in sequence.z_plan.positions()]
    except Exception:
        positions = []
    if len(positions) >= 2:
        return abs(positions[1] - positions[0])
    if getattr(sequence.z_plan, "step", None) is not None:
        return abs(float(sequence.z_plan.step))
    return 1.0
