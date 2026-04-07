# Input Files

This page describes the three MERFISH input files that the widget validates before dispatch:

- the fluidics program CSV
- the experiment order file
- the codebook CSV

These rules match the current runtime contract in `merfish3d_wfacq.utils.data_io` and `merfish3d_wfacq.workflow`.

## Fluidics Program

The fluidics program must be a delimited text table with these required columns:

- `round`
- `source`
- `time`
- `pump`

Notes:

- column names are normalized to lowercase during load
- rows missing any required field are dropped
- `round` must be integer-like
- `time` and `pump` must be numeric
- `source` values are normalized to uppercase

Minimal example:

```csv
round,source,time,pump
1,B01,0.5,10
1,RUN,0,0
2,B02,0.5,10
2,RUN,0,0
```

Important behavior:

- every distinct `round` value is a fluidics round
- only rounds that contain `source = RUN` are treated as imaging rounds
- iterative imaging requires at least one `RUN` step
- `REFRESH` is allowed for manual operator intervention, but it does not count as an imaging trigger

Practical guidance:

- keep one `RUN` row for every round that should acquire images
- if a round should only perform fluidics and not image, omit `RUN` for that round
- for single-round imaging, the selected round must exist in the experiment order file, and if a fluidics program is loaded it must also contain a `RUN` step for that round

## Experiment Order File

The experiment order file must contain:

- one `round` column
- one column for every active MDA channel config name

Important behavior:

- channel names must match the active MDA channel config names exactly
- channel column order does not matter
- exactly one channel column must contain `0` for every round
- that all-zero column is the fiducial channel

Minimal example:

```csv
round,Cy5,FITC,Rhodamine
1,2,0,1
2,4,0,3
3,6,0,5
```

In that example:

- `FITC` is the fiducial channel because it is the unique all-zero column
- `Cy5` and `Rhodamine` are readout channels
- the column order is valid even though it does not match the MDA widget order

Validation rules:

- the file must contain at least one channel column after `round`
- rounds must be unique
- channel-bit values must be integers
- iterative imaging:
  - experiment order file rounds must exactly match the fluidics-program rounds that contain `RUN`
- single-round imaging:
  - the selected round must be present in the experiment order file
  - if a fluidics program is loaded, that round must also contain a `RUN` step

Practical guidance:

- use the exact channel config names shown in the MDA widget
- do not rename columns to generic labels like `readout 1`
- make the fiducial column the unique all-zero column

## Codebook

The codebook is loaded as a generic delimited table.

Minimum requirements:

- at least one identifier column
- at least one bit column
- at least one data row

Minimal example:

```csv
gene_id,bit01,bit02,bit03,bit04
GeneA,1,0,1,0
GeneB,0,1,0,1
```

Important behavior:

- the codebook loader does not require a specific identifier-column name
- however, the codebook must contain enough bit columns for the maximum bit used in the experiment order file

Example:

- if the experiment order file uses bit `16`, the codebook must contain at least 16 bit columns after the identifier column

## How The Three Files Relate

For iterative imaging:

- the fluidics program decides which rounds actually image by the presence of `RUN`
- the experiment order file must describe exactly those imaging rounds
- the codebook must cover the highest bit referenced by the experiment order file

For single-round imaging:

- the experiment order file must contain the selected round
- the codebook must cover the highest bit used in that round

For fluidics-only runs:

- only the fluidics program is required
- the experiment order file and the codebook are not used

## Common Failure Cases

- experiment order file column names do not exactly match the active MDA channel names
- more than one experiment order file column is all zero
- no experiment order file column is all zero
- iterative fluidics contains no `RUN` commands
- iterative experiment order file rounds do not match the fluidics `RUN` rounds
- the codebook does not contain enough bit columns for the experiment order

