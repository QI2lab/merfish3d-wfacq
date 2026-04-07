from typing import Any

import pytest
from useq import MDASequence

from merfish3d_wfacq.core_metadata import (
    CoreMetadataError,
    derive_core_metadata,
    pixel_size_affine_to_affine_zyx_px,
)


class _FakeCore:
    def __init__(self) -> None:
        self.properties = {
            ("Camera", "CameraName"): "FakeHamamatsu",
            ("Camera", "Binning"): "2",
            ("Camera", "CONVERSION FACTOR COEFF"): "0.54",
            ("Camera", "CONVERSION FACTOR OFFSET"): "100",
        }

    def getCameraDevice(self) -> str:
        return "Camera"

    def getPixelSizeUm(self, _cached: bool = False) -> float:
        return 0.108

    def getPixelSizeAffine(self, _cached: bool = False) -> tuple[float, ...]:
        return (0.108, 0.0, 0.0, 0.0, 0.108, 0.0)

    def getDevicePropertyNames(self, _device: str) -> list[str]:
        return [
            "CameraName",
            "Binning",
            "CONVERSION FACTOR COEFF",
            "CONVERSION FACTOR OFFSET",
        ]

    def getProperty(self, device: str, property_name: str) -> Any:
        return self.properties[(device, property_name)]


def test_pixel_size_affine_is_converted_like_create_datastore() -> None:
    assert pixel_size_affine_to_affine_zyx_px(
        (0.108, 0.0, 0.0, 0.0, 0.108, 0.0), 0.108
    ) == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_derive_core_metadata_reads_required_fields_from_demo_mmcore(
    demo_core: Any,
) -> None:
    sequence = MDASequence(z_plan={"range": 2, "step": 1})
    metadata = derive_core_metadata(demo_core, sequence)
    camera_device = str(demo_core.getCameraDevice())

    assert metadata["camera_model"] == str(
        demo_core.getProperty(camera_device, "CameraName")
    )
    assert metadata["pixel_size_um"] == pytest.approx(
        float(demo_core.getPixelSizeUm(True))
    )
    assert metadata["binning"] == int(
        float(demo_core.getProperty(camera_device, "Binning"))
    )
    assert metadata["e_per_adu"] == pytest.approx(
        float(demo_core.getProperty(camera_device, "Photon Conversion Factor"))
    )
    assert metadata["camera_offset_adu"] == pytest.approx(
        float(demo_core.getProperty(camera_device, "Offset"))
    )
    assert metadata["z_step_um"] == pytest.approx(1.0)
    assert metadata["affine_zyx_px"] == pixel_size_affine_to_affine_zyx_px(
        tuple(float(value) for value in demo_core.getPixelSizeAffine(True)),
        float(demo_core.getPixelSizeUm(True)),
    )


def test_derive_core_metadata_errors_when_core_is_missing_conversion_factor() -> None:
    core = _FakeCore()
    del core.properties[("Camera", "CONVERSION FACTOR COEFF")]

    with pytest.raises(CoreMetadataError):
        derive_core_metadata(core, MDASequence())
