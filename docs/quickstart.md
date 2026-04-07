# Quickstart

This quickstart assumes you already have a working Micro-Manager configuration and a Python environment with `merfish3d-wfacq` installed.

## Launch the application

```powershell
merfish3d-wfacq --mm-config C:\path\to\MMConfig.cfg
```

You can also start it without a config and load hardware later:

```powershell
merfish3d-wfacq
```

## Configure the upstream MDA widget

Before using the MERFISH dock:

- choose the active channel group and channel configs
- configure stage positions or Stage Explorer tiles
- configure the z plan
- choose the save destination in the MDA widget
- keep the MDA configuration consistent with the MERFISH files you plan to load

## Prepare the MERFISH files

Before loading files into the MERFISH widget, prepare:

- a fluidics CSV when the selected mode uses fluidics
- an experiment order file for imaging runs
- a codebook for imaging runs
- measured illumination profiles, or plan to enable `Use uniform / unknown illumination`

Use [Preparing input files](input-files.md) for the exact file formats and validation rules.

## Load MERFISH inputs

In the MERFISH widget:

- choose a run mode
- load the fluidics CSV when the mode uses fluidics
- load an experiment order file for imaging runs
- load the codebook for imaging runs
- load illumination profiles, or enable `Use uniform / unknown illumination`

## Check the contracts

Before running:

- the experiment order file must contain one column per active channel config name
- experiment order file column order does not matter
- exactly one experiment order file channel column must be `0` for every round, and that column is the fiducial channel
- iterative imaging requires at least one `RUN` step in the fluidics program
- iterative imaging requires experiment order file rounds to match the fluidics rounds that contain `RUN`
- the codebook must contain enough bit columns for the highest bit used in the experiment order file
- measured illumination profiles must match the number of active channels

## Run the acquisition

- use `Run fluidics` for fluidics-only programs
- use `Run acquisition` for full iterative MERFISH imaging
- use `Run single round imaging` for a one-round test
- use `Abort` to cancel a running acquisition

## Handle REFRESH steps

If the fluidics program includes `REFRESH`:

- perform the manual operator step
- click `Yes` in the dialog to continue
- click `No` to stop advancing the run

## Validate output

Imaging runs create a qi2lab datastore under the resolved experiment root:

```text
<experiment-root>/qi2labdatastore
```

The image arrays are streamed through `ome-writers`, and the resulting datastore can be validated with the qi2lab analysis stack.

