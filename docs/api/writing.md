# Writing And Datastore Setup

## `merfish3d_wfacq.datastore`

### `prepare_qi2lab_datastore`

Purpose:

- create the qi2lab datastore layout at `sequenceStarted()`
- write static datastore metadata and target sidecars
- create shading-map outputs through `ome-writers`

This module is setup-only. It does not stream acquisition frames.

## `merfish3d_wfacq.sink.MerfishFrameProcessor`

Purpose:

- receive `frameReady(img, event, meta)` from `pymmcore-plus`
- apply camera offset, gain, and illumination correction
- capture the round-1 fiducial drift reference stack in memory
- forward corrected frames to the datastore writer

## `merfish3d_wfacq.sink.Qi2labDatastoreWriter`

Purpose:

- prepare the datastore at `sequenceStarted()`
- route frames using metadata already stamped onto prepared image events
- build standard frame metadata from the upstream payload
- append arrays using `ome-writers`
- close per-target streams after the final frame for that target

## Upstream contracts reused

- `CMMCorePlus.run_mda(...)`
- `sequenceStarted / frameReady / sequenceFinished`
- `pymmcore_plus.mda._sink._frame_meta_to_ome(...)`
- `ome_writers.create_stream(...).append(...)`

## Stage-position contract

The writer uses only the runner-delivered payload metadata for position:

- no MMCore position queries in the writer
- no planned-position fallback in the writer
- actual stage coordinates are taken from `meta["position"]`
