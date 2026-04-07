from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorstore as ts
import tifffile
from numpy import asarray
from useq import MDASequence

from merfish3d_wfacq.sequence import ImageKind, channel_index_key
from merfish3d_wfacq.utils.data_io import (
    read_codebook,
    read_fluidics_program,
)

DEFAULT_CHANNEL_SPECS = [
    {
        "channel_index": 0,
        "config_name": "Fiducial-488",
        "role": ImageKind.FIDUCIAL.value,
        "excitation_um": 0.488,
        "emission_um": 0.520,
    },
    {
        "channel_index": 1,
        "config_name": "Readout-561",
        "role": ImageKind.READOUT.value,
        "excitation_um": 0.561,
        "emission_um": 0.590,
    },
    {
        "channel_index": 2,
        "config_name": "Readout-647",
        "role": ImageKind.READOUT.value,
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
]

DEMO_CHANNEL_SPECS = [
    {
        "channel_index": 0,
        "config_name": "DAPI",
        "role": ImageKind.FIDUCIAL.value,
        "excitation_um": 0.405,
        "emission_um": 0.450,
    },
    {
        "channel_index": 1,
        "config_name": "Rhodamine",
        "role": ImageKind.READOUT.value,
        "excitation_um": 0.561,
        "emission_um": 0.590,
    },
    {
        "channel_index": 2,
        "config_name": "Cy5",
        "role": ImageKind.READOUT.value,
        "excitation_um": 0.647,
        "emission_um": 0.680,
    },
]


FITC_RHODAMINE_CY5_SEQUENCE = MDASequence(
    channels=[
        {"config": "FITC", "exposure": 10.0},
        {"config": "Rhodamine", "exposure": 10.0},
        {"config": "Cy5", "exposure": 10.0},
    ],
    stage_positions=[(0.0, 0.0, 0.0)],
)

TWO_ROUND_EXP_ORDER = pd.DataFrame(
    {
        "round": [1, 2],
        "FITC": [0, 0],
        "Rhodamine": [1, 3],
        "Cy5": [2, 4],
    }
)


def _normalize_experiment_order(
    experiment_order: dict[int, dict[str, int]] | dict[int, list[int]],
    channel_specs: list[dict[str, Any]],
) -> dict[int, dict[str, int]]:
    """Normalize test experiment-order input to channel-keyed mappings."""

    ordered_specs = sorted(channel_specs, key=channel_index_key)
    fiducial_specs = [
        spec for spec in ordered_specs if str(spec.get("role")) == ImageKind.FIDUCIAL.value
    ]
    if len(fiducial_specs) != 1:
        raise ValueError(
            "Test channel specs must include exactly one fiducial channel role."
        )
    fiducial_name = str(fiducial_specs[0]["config_name"])
    readout_specs = [
        spec for spec in ordered_specs if str(spec.get("role")) != ImageKind.FIDUCIAL.value
    ]

    normalized: dict[int, dict[str, int]] = {}
    for round_id, round_bits in experiment_order.items():
        if isinstance(round_bits, dict):
            normalized[int(round_id)] = {
                str(channel_name): int(bit)
                for channel_name, bit in round_bits.items()
            }
            continue

        readout_bits = [int(bit) for bit in round_bits]
        normalized[int(round_id)] = {
            fiducial_name: 0,
            **{
                str(spec["config_name"]): int(bit)
                for spec, bit in zip(readout_specs, readout_bits, strict=True)
            },
        }
    return normalized


def write_codebook(path: Path, num_bits: int) -> pd.DataFrame:
    data = {"gene_id": ["gene_a"]}
    for bit_index in range(1, num_bits + 1):
        data[f"bit{bit_index:02d}"] = [0]
    frame = pd.DataFrame(data)
    frame.to_csv(path, index=False)
    return frame


def write_demo_codebook(path: Path) -> pd.DataFrame:
    rows = ["gene_id," + ",".join(f"bit{bit:02d}" for bit in range(1, 17))]
    rows.append("gene_a," + ",".join("1" if bit % 2 else "0" for bit in range(1, 17)))
    rows.append(
        "gene_b," + ",".join("1" if bit % 3 == 0 else "0" for bit in range(1, 17))
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return read_codebook(path)


def write_exp_order(
    path: Path,
    experiment_order: dict[int, dict[str, int]] | dict[int, list[int]],
    *,
    channel_specs: list[dict[str, Any]] | None = None,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    specs = channel_specs or list(DEFAULT_CHANNEL_SPECS)
    normalized_experiment_order = _normalize_experiment_order(experiment_order, specs)
    default_column_order = [
        str(spec["config_name"])
        for spec in sorted(specs, key=channel_index_key)
    ]
    channel_names = list(column_order or default_column_order)
    columns = ["round", *channel_names]
    rows = []
    for round_id, channel_bits in normalized_experiment_order.items():
        row = {"round": int(round_id)}
        for channel_name in channel_names:
            row[channel_name] = int(channel_bits[channel_name])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    return frame


def write_demo_exp_order(
    path: Path,
    *,
    channel_specs: list[dict[str, Any]] | None = None,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    return write_exp_order(
        path,
        {
            round_label: {
                "DAPI": 0,
                "Rhodamine": 2 * round_label - 1,
                "Cy5": 2 * round_label,
            }
            for round_label in range(1, 9)
        },
        channel_specs=channel_specs or list(DEMO_CHANNEL_SPECS),
        column_order=column_order,
    )


def write_fluidics_program(path: Path, rounds: list[int]) -> pd.DataFrame:
    records = []
    for round_id in rounds:
        records.extend(
            [
                {
                    "round": int(round_id),
                    "source": f"B{int(round_id):02d}",
                    "time": 0.1,
                    "pump": 10.0,
                },
                {"round": int(round_id), "source": "RUN", "time": 0.0, "pump": 0.0},
            ]
        )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(path, index=False)
    return frame


def write_refresh_fluidics_program(path: Path) -> pd.DataFrame:
    path.write_text(
        "\n".join(
            [
                "round,source,time,pump",
                "1,B01,0,10",
                "1,REFRESH,0,0",
                "1,RUN,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return read_fluidics_program(path)


def write_illumination_profiles(path: Path, profiles: np.ndarray) -> np.ndarray:
    tifffile.imwrite(path, profiles, photometric="minisblack", metadata={"axes": "CYX"})
    return profiles


def write_demo_illumination_profiles(
    path: Path,
    shape: tuple[int, int],
    *,
    channel_specs: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    specs = channel_specs or list(DEMO_CHANNEL_SPECS)
    profile_value_by_channel = {
        "DAPI": 1.0,
        "FITC": 1.0,
        "Fiducial-488": 1.0,
        "Rhodamine": 2.0,
        "Readout-561": 2.0,
        "Cy5": 3.0,
        "Readout-647": 3.0,
    }
    profiles = np.stack(
        [
            np.full(
                shape,
                profile_value_by_channel[str(spec["config_name"])],
                dtype=np.float32,
            )
            for spec in sorted(specs, key=channel_index_key)
        ],
        axis=0,
    )
    return write_illumination_profiles(path, profiles)


def read_zarr3_array(image_path: Path) -> np.ndarray:
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(image_path / "0")},
        },
        open=True,
    ).result()
    return asarray(store.read().result())



