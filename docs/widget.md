# MERFISH Widget

The MERFISH dock extends the standard `pymmcore-gui` MDA workflow. Use the regular MDA widget to configure acquisition geometry, then use the MERFISH widget to add fluidics, experiment-order, codebook, and datastore metadata.

## Intended use

- Configure channels, stage positions, z plan, and save destination in the upstream MDA widget.
- Open the MERFISH dock and choose a run mode: `Fluidics only`, `Iterative imaging`, or `Single round test`.
- Load the fluidics CSV when the selected mode uses fluidics.
- Load an experiment order file for imaging modes. It must contain a `round` column, one column per active MDA channel config name, and exactly one all-zero channel column for the fiducial.
- Load a codebook for imaging modes.
- Load measured illumination profiles, or enable `Use uniform / unknown illumination` to generate all-ones profiles at runtime.
- Confirm the read-only core metadata fields populated from Micro-Manager: pixel size, camera model, binning, affine, gain, and offset.
- Confirm the Stage Explorer overlap and reference tile before drift-enabled runs.
- Click `Run acquisition` for iterative imaging, `Run single round imaging` for single-round tests, or `Run fluidics` for fluidics-only runs.
- Use `Abort` to cancel a running acquisition.
- When a `REFRESH` step is requested during fluidics, complete the manual step and confirm the dialog so execution can continue.

## Input file guidance

Use [Preparing input files](input-files.md) for the exact file-format rules.

The most important contracts are:

- in imaging modes, fluidics rounds only image when the program contains `RUN`
- in fluidics-only mode, every round in the fluidics program executes even if there are no `RUN` rows
- experiment order file channel names must exactly match the active MDA channel names
- experiment order file channel order does not matter
- the unique all-zero experiment order file column is the fiducial channel
- the codebook must contain enough bit columns for the experiment order

## File-loading contract

- The widget loads fluidics, experiment-order, codebook, and illumination inputs immediately when the user selects them.
- Parsed objects are kept in memory and passed into workflow helpers.
- Runtime execution does not reload those files from disk.

## Runtime handoff

- The widget collects normalized UI state through `merfish3d_wfacq.ui_state.build_merfish_ui_state`.
- That state is validated and normalized through `merfish3d_wfacq.workflow.normalize_merfish_ui_state`.
- Dispatch preparation is performed through `merfish3d_wfacq.dispatch.prepare_merfish_dispatch`.
- The final run is executed with `CMMCorePlus.run_mda(...)` using a prepared `list[MDAEvent]` and a `MerfishFrameProcessor` output listener.

