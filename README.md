# merfish3d-wfacq

Control suite for the qi2lab 3D MERFISH widefield acquisition platform. This control suite generates data that can be directly processed  our [GPU 3D MERFISH analysis processing platform](https://github.com/QI2lab/merfish3d-analysis).

Documentation, including example input files, is available at [https://qi2lab.github.io/merfish3d-wfacq/](https://qi2lab.github.io/merfish3d-wfacq/)

## Associated preprint publication
[GPU-accelerated, self-optimizing processing for 3D multiplexed iterative RNA-FISH experiments](https://www.biorxiv.org/content/10.1101/2025.10.10.681751v2).

## Installation

Create a python 3.12 environment using your favorite package manager, e.g.
```
conda create -n merfish3d-wfacq python=3.12
```

Activate the environment.
```
conda activate merfish3d-wfacq
```

Next, clone the repository in your location of choice and enter the directory using
```
git clone https://github.com/QI2lab/merfish3d-wfacq
cd merfish3d-wfacq
``` 

and install using 
```
pip install .
```

For interactive editing use 
```
pip install -e .
``` 

## Overview

merfish3d-wfacq implements a custom GUI widget, acquisition engine, and file writing code to perform imaging spatial transcriptomics experiments. It is built on top of [Micro-Manager](https://github.com/micro-manager/micro-manager) and [pymmcore-plus](https://github.com/pymmcore-plus).


## Documentation

This repository is configured for Zensical documentation.

```powershell
python -m pip install -e .[dev]
zensical preview
```