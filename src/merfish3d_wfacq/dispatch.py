from pathlib import Path
from typing import Any

from useq import MDAEvent

from merfish3d_wfacq.runtime_state import create_drift_reference_runtime
from merfish3d_wfacq.sequence import RunMode
from merfish3d_wfacq.sink import MerfishFrameProcessor
from merfish3d_wfacq.workflow import (
    build_merfish_metadata,
    prepare_merfish_acquisition,
)


def prepare_merfish_dispatch(
    *,
    normalized_ui_state: dict[str, Any],
    save_path: str | Path | None,
    overwrite: bool = True,
) -> tuple[list[MDAEvent], MerfishFrameProcessor | None]:
    """Prepare the event list and output listener for one MERFISH run.

    Parameters
    ----------
    normalized_ui_state : dict[str, Any]
        Validated widget state returned by
        ``merfish3d_wfacq.workflow.normalize_merfish_ui_state``.
    save_path : str or Path or None
        Upstream save path selected by the MDA widget.
    overwrite : bool, optional
        Whether to overwrite an existing experiment root.

    Returns
    -------
    tuple[list[MDAEvent], MerfishFrameProcessor or None]
        Prepared event list plus the output listener used by ``run_mda``.
    """

    merfish_metadata, base_sequence = build_merfish_metadata(normalized_ui_state)
    setup_payload: dict[str, Any] | None = None
    drift_reference_store: dict[str, Any] | None = None

    if normalized_ui_state["mode"] is not RunMode.FLUIDICS_ONLY:
        drift_reference_store, setup_payload = create_drift_reference_runtime(
            int(merfish_metadata["reference_tile"])
        )

    sequence, runtime_metadata, _experiment_root = prepare_merfish_acquisition(
        base_sequence=base_sequence,
        merfish_metadata=merfish_metadata,
        save_path=save_path,
        overwrite=overwrite,
        setup_payload=setup_payload,
    )
    output = (
        None
        if normalized_ui_state["mode"] is RunMode.FLUIDICS_ONLY
        else MerfishFrameProcessor(
            merfish_metadata=runtime_metadata,
            drift_reference_store=drift_reference_store,
        )
    )
    return sequence, output
