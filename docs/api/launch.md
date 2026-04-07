# Launch And GUI Integration

## Top-level entrypoints

### `merfish3d_wfacq.launch_merfish_app`

- Module: `merfish3d_wfacq`
- Defined in: `merfish3d_wfacq.app`
- Purpose: launch `pymmcore-gui` with the MERFISH dock attached

Parameters:

- `mm_config`: optional Micro-Manager config path, `None`, or `False`
- `mmcore`: optional existing `CMMCorePlus`
- `exec_app`: whether to enter the Qt event loop

Returns:

- `pymmcore_gui.MicroManagerGUI`

### `merfish3d_wfacq.enhance_main_window`

- Module: `merfish3d_wfacq`
- Defined in: `merfish3d_wfacq.gui`
- Purpose: attach the MERFISH dock to an existing pyMM main window

## Widget registration

The MERFISH dock is registered through `WidgetActionInfo` under the widget key:

- `merfish3d_wfacq.merfish_widget`

This keeps widget creation and docking under the standard `pymmcore-gui` registry instead of constructing a custom dock manually.

## Main widget class

### `merfish3d_wfacq.gui.MerfishFluidicsWidget`

Responsibilities:

- owns the MERFISH-specific Qt controls
- loads and previews input files
- reflects Micro-Manager-derived metadata in the UI
- emits REFRESH confirmation responses back to the engine
- validates UI state and dispatches acquisition preparation

The widget does not:

- build image frames
- execute image events
- write arrays to disk
