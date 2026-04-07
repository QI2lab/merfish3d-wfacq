from typing import Any

import numpy as np


def resolved_illumination_profiles(
    merfish_metadata: dict[str, Any], image_info: dict[str, Any]
) -> np.ndarray:
    """Return measured or generated illumination profiles for one run.

    Parameters
    ----------
    merfish_metadata : dict[str, Any]
        Normalized MERFISH metadata for the run.
    image_info : dict[str, Any]
        Summary image info emitted at sequence start.

    Returns
    -------
    np.ndarray
        Illumination-profile stack with shape ``(c, y, x)``.
    """

    if str(merfish_metadata["illumination_profiles_mode"]) == "uniform":
        return np.ones(
            (
                len(merfish_metadata["channel_specs"]),
                int(image_info["height"]),
                int(image_info["width"]),
            ),
            dtype=np.float32,
        )
    return np.asarray(merfish_metadata["illumination_profiles"], dtype=np.float32)
