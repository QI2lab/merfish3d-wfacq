# Execution And Custom Actions

## `merfish3d_wfacq.sequence`

### `build_merfish_events`

This is the main experiment compiler.

Inputs:

- base `MDASequence`
- normalized MERFISH metadata
- resolved imaging rounds
- optional setup payload

Output:

- full prepared `list[MDAEvent]`

The prepared event list contains:

- a `merfish_setup` `CustomAction`
- optional `fluidics` `CustomAction`s
- optional `drift_correct` `CustomAction`s
- normal image `MDAEvent`s for fiducial and readout imaging

Each image event is stamped with the routing metadata needed downstream by the writer.

## `merfish3d_wfacq.engine.MerfishMDAEngine`

This engine subclasses `pymmcore_plus.mda.MDAEngine`.

Purpose:

- execute MERFISH `CustomAction`s
- otherwise delegate normal image events to the upstream engine
- update future prepared image-event `z_pos` values after drift correction

Supported custom action names:

- `merfish_setup`
- `fluidics`
- `drift_correct`

## `merfish3d_wfacq.fluidics.FluidicsController`

Purpose:

- execute one prepared fluidics round
- relay log/status callbacks
- request operator confirmation for `REFRESH` steps

## Drift runtime contract

- round-1 fiducial reference frames are captured in memory, not reloaded from datastore files
- drift correction updates future prepared image events in the event list
- REFRESH is the only accepted execution-to-GUI coupling
