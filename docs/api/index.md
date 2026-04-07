# API Overview

`merfish3d-wfacq` adds a MERFISH-specific layer on top of `pymmcore-gui`, `pymmcore-plus`, and `ome-writers`.

The custom API surface is split by boundary:

- [Launch and GUI integration](launch.md)
- [Preparation and normalization](preparation.md)
- [Execution and custom actions](execution.md)
- [Writing and datastore setup](writing.md)

## Boundary summary

- `app.py` and `gui.py` attach the widget to `pymmcore-gui`.
- `ui_state.py`, `workflow.py`, and `dispatch.py` normalize widget state and build the prepared acquisition.
- `sequence.py` converts normalized metadata into a full `list[MDAEvent]`.
- `engine.py` subclasses `MDAEngine` and handles MERFISH `CustomAction`s.
- `sink.py` and `datastore.py` prepare the qi2lab datastore and stream frames through `ome-writers`.
