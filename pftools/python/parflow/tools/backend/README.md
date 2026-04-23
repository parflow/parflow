# pfb_backend

## Introduction

`pfb_backend` is a `xarray` backend (or `engine`) that allows you to read [Parflow](https://github.com/parflow/parflow) files in `.pfb` format. It is based on Parflow’s [pftools/python](https://github.com/parflow/parflow/tree/master/pftools/python) code, but offers a different approach from the `pf_backend` provided natively by Parflow:
- Engine: `pf_ige`
- Based on pfidb for metadata
> `pf_backend`, on the other hand, uses pfmetadata. Perhaps the choice of metadata format used for `pf_ige` will be implemented in a future version.
- File discovery via directory scanning (`pf_backend` relies on the list contained in pfmetadata): this allows for processing isolated pfb files and easily managing files from different simulations.
- Automatic datetime calculation.
- Optimization of the pftools code: fixing issues related to dask, code refactoring, and reducing the number of reads.
- Handling of NaN values.

The library also includes [dask](https://www.dask.org/) wrappers for the [hydrology](https://github.com/parflow/parflow/blob/master/pftools/python/parflow/tools/hydrology.py) functions of pftools and other functions for processing with xarray in `pfb_backend.xarray_utils`. 

## Installation

### Basic

```bash
pip install git+https://gricad-gitlab.univ-grenoble-alpes.fr/phyrev/public/pfb_backend.git
```

or

```bash
pip install git+git@gricad-gitlab.univ-grenoble-alpes.fr:phyrev/public/pfb_backend.git
```

### Developer

- Clone
```bash
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/phyrev/public/pfb_backend.git
```

- Install

```bash
cd pfb_backend
pip install -e .
```

## Usage

Several examples of usage are presented in `examples/export.py` and `examples/export.ipynb`.

This backend can take three types of input: a directory, a pfidb file, or a pfb file (or a list of one type). 
It will then automatically search for the pfb files present in the case of a pfidb or a directory, or for the pfidb 
file in the case of a pfb. It will then generate an `xr.Dataset` with all the variables found.

> Regarding the chunksize: unless the file uses a subgrid, the entire file is read each time you want to 
> access data in it. The most efficient approach is therefore often to adapt the chunks parameter to the size of 
> your files which is the default behavior.

```python
import xarray as xr

xr.set_options(display_max_rows=50)
ds = xr.open_dataset("results_WA_2560_v0_y1_run6/WA_vd_KWE_RST.pfidb", engine="pf_ige", select_types=["input", "output"])

print(ds)
```

```commandline
<xarray.Dataset> Size: 6GB
Dimensions:                         (y: 1600, x: 2880, z: 11)
Coordinates:
  * y                               (y) float64 13kB -7.785e+05 ... 8.205e+05
  * x                               (x) float64 23kB -1.424e+06 ... 1.454e+06
  * z                               (z) float64 88B 100.0 30.0 10.0 ... 0.1 0.05
Data variables:
    input_Ksat                      (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_Sr                        (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_Ssat                      (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_alpha                     (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_manning_WA                (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    input_n                         (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_porosity                  (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    input_slopex                    (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    input_slopey                    (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    input_veg_map                   (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    dz_mult           (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    mannings          (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    mask              (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    perm_x            (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    perm_y            (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    perm_z            (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    porosity          (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
    slope_x           (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    slope_y           (y, x) float64 37MB dask.array<chunksize=(1600, 2880), meta=np.ndarray>
    specific_storage  (z, y, x) float64 406MB dask.array<chunksize=(7, 1152, 2078), meta=np.ndarray>
```

Arguments for this engine:
- filename_or_obj (`str | list[str]`): PFB file, PFIDB file, or directory containing PFB files (and possibly PFIDB files) or a list of one of these items.
- drop_variables (`Optional[str | list[str]]`): A variable or list of variables to exclude from being parsed from the dataset. This may be useful to drop variables with problems or inconsistent values.

Time specific arguments:
- start_date (`Optional[str | np.datetime64]`): start date of the simulation (timestep=0), used to calculate the date associated with each time step. If not specified, the default value is "2000-01-01".
- default_forcing_ts (`int`): Default value for forcing time step in s, default is 1800s (30min)
- default_clm_ts (`int`): Default value for clm timestep in s. The value used will be the one found in the pfidb file, if available, default is 10800s (3H).
- default_output_ts (`int`): Default value for output timestep in s. The value used will be the one found in the pfidb file, if available, default is 10800s (3H).
- time_label (`Literal["right", "left", "center"]`): Side of each interval to use for labeling for time axes, default is "right".

Variable specific arguments:
- select_types (`Optional[str | list[str]]`): allows you to choose the type of variables to be selected. There are currently five types of variables: “input”, “output”, “clm”, “temporal_output” and "forcing".
- select_variables (`Optional[str | list[str]]`): allows you to choose the variables to be selected. The engine will check if the given string is a substring of a given variable.
- select_sim (`Optional[str | list[str]]`): allows you to choose the simulation to be selected. The engine will check whether the given string id identical (case-insensitive) to the simulation name that has been associated with a variable.

Loading options (all True by default):
- z_first (`bool`): Whether the z axis is first in pfb files. If not, it will be last.
- replace_fill_value (`bool`): Automatically detects the fill value associated with each variable and replaces it with np.nan.
- compute_time (`bool`): Use the datetime computed from the time step as the label for the time axis. If not, use the time step.

## xarray_utils

The project also offers a set of functions for processing parflow data in xarray_utils (`import pfb_backend.xarray_utils`), some of which are simply wrapper for xarray of the pftools hydrology functions.

## Time, CLM_Time and Forc_Time

To manage the fact that forcing, CLM, and simulation data may not be on the same time scales, the backend uses time axes specific to each data type :
- `clm_time` for clm
- `forc_time` for forcing
- `time for` the simulation

## How it works:

### Xarray open_dataset

Xarray documentation "How to add a new backend": https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html#how-to-add-a-new-backend

Xarray offers an `open_dataset` method, which relies on the use of engines. These can be registered in the environment (hence the need to run `pip install`, as indicated in **Installation**).

In particular, in `pyproject.toml`, the following lines enable the installation of the backend (almost identical to those found in pftools `pyproject.toml`): 

```script
[project.entry-points.“xarray.backends”]
pf_ige = “pfb_backend.backend:ParflowBackendEntrypoint”
```

An engine is based on the creation of a class that inherits (child class) from the `BackendEntrypoint` class of xarray.

This class has a `guess_can_open method`, which xarray uses to determine which engine to use to open the data at the input of `open_dataset`. 
In our case, it accepts pfb files, pfidb files, or directories containing these files (or a list). Note that the pftools 
engine also accepts pfb files, so if we provide pfb files as input (rather than a directory or a pfidb), xarray may open 
them with this engine rather than this one if we do not specify otherwise manually in `open_dataset` with the argument `engine="pf_ige"`.

### Implementation

Now for what happens internally:

1 - File search

Retrieving the list of pfb and pfidb files:
- If pfb files are provided as input: the backend will search for the associated pfidb files.
- If directories or pfidb files are provided: the backend will search for the associated pfb and possibly pfidb files.

2 - Sorting files

File sorting is based on Handlers (an abstract parent class). Handler's child classes will be responsible for managing 
a type of pfb parflow data. There are currently five types:

- Static outputs

- Static inputs

- Temporal outputs

- CLM outputs

- Forcing data

Each handler will record the files it will manage and extract the information it needs from the file name. 
For temporal data, it groups files specific to the same variable together.

3 - Variable selection

We select the subsets of variables that interest us, which may have been specified with the 
arguments: `drop_variables`, `select_variables`, `select_types`, or `select_simulation` in `open_dataset`.

4 - Variable preparation

The next step is to prepare the `xr.DataArray` using Handlers. For each variable managed by the Handler, we will create 
an `xr.Variable` using dask. This part is based on the `ParflowBackendArray` class from pftools. The idea is to read the 
header of the pfb files and store the dimensions of the arrays as well as how to load the data into dask. This allows 
us to create arrays whose dimensions and data volume are known but which are not yet loaded, they will be loaded as 
needed. Once the xr.Variables have been created, we add various pieces of information to them, including 
coordinates: time, x, y, etc., and store everything in an `xr.DataArray`.

5 - Dataset preparation

This is the simplest step: we group all the `xr.DataArray` into an `xr.Dataset`, which we return.
