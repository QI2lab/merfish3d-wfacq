# merfish3d-wfacq

`merfish3d-wfacq` provides qi2lab 3D MERFISH widefield acquisition on top of `pymmcore-gui`, `pymmcore-plus`, and `ome-writers`.

## Documentation

- [Quickstart](quickstart.md)
- [Preparing input files](input-files.md)
- [MERFISH widget usage](widget.md)
- [API overview](api/index.md)
- [Launch and GUI integration](api/launch.md)
- [Preparation and normalization](api/preparation.md)
- [Execution and custom actions](api/execution.md)
- [Writing and datastore setup](api/writing.md)

## Local docs

Install the development dependencies and run Zensical locally:

```powershell
& 'C:\Users\dpshe\miniforge3\envs\merfish3d-wfacq\python.exe' -m pip install -e .[dev]
zensical preview
```

Build the static site into `site/`:

```powershell
zensical build
```

## Project links

- Source: <https://www.github.com/qi2lab/merfish3d-wfacq>
- Issue tracker: <https://www.github.com/qi2lab/merfish3d-wfacq/issues>
