# Preparation And Normalization

## Operator input files

The concrete file-format guidance lives in [Preparing input files](../input-files.md).

That page documents the runtime contracts for:

- the fluidics program CSV
- the experiment order file
- the codebook CSV

## `merfish3d_wfacq.ui_state`

This module contains widget-facing normalization helpers:

- `build_merfish_ui_state`
- `wavelength_rows_for_sequence`
- `channel_specs_from_sequence_wavelength_rows`
- `bit_mapping_preview`
- `fluidics_round_options`

These helpers convert the current upstream MDA state and the MERFISH widget state into a normalized Python dict for validation and dispatch.

## `merfish3d_wfacq.workflow`

This module validates and normalizes the MERFISH run configuration.

### `normalize_merfish_ui_state`

Purpose:

- validate the currently loaded widget state
- enforce the fluidics / experiment-order / codebook contract
- infer the fiducial channel from the unique all-zero experiment order file column
- return one normalized dict for the run

### `build_merfish_metadata`

Purpose:

- convert normalized widget state into the metadata used by sequence generation, engine setup, and datastore setup

### `prepare_merfish_acquisition`

Purpose:

- add runtime metadata such as resolved experiment root, datastore root, tile count, and z-plane count
- return the prepared event list plus runtime metadata

## `merfish3d_wfacq.dispatch`

### `prepare_merfish_dispatch`

Purpose:

- create the in-memory drift runtime store when needed
- build prepared events and runtime metadata
- construct the output listener used for `run_mda(...)`

Returns:

- prepared `list[MDAEvent]`
- `MerfishFrameProcessor | None`

This is the narrow non-Qt assembly layer between GUI state and execution.

