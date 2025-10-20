# PDI tools for ParFlow

## Yaml configuration file

The yaml file `conf.yml` contains the file architecture used by PDI to manage and output ParFlow data.
It describes the `vector` data structure used by ParFlow to store all physical parameters.
The whole structure is exposed to PDI.

If one runs ParFlow with PDI, the `conf.yml` mush be copied in the working directory.
This can be done easily by addind this at the beginning of the tcl file :

```tcl
file copy -force $env(PARFLOW_DIR)/../pdi/conf.yml ./
```

### Structure of the Yaml file

The Yaml file is a mirror of the general ParFlow data structure.

The main data strucutre is called `vector` and is described in the file `pfsimulator/vector.h`.
In the Yaml file, it corresponds to data `sparse_vector_data`.
The `vector` structure is composed of the following members:
- `Subvector    **subvectors`: list of subvectors, this structure is also described in `pfsimulator/vector.h`.
- `int data_size`: number of subvectors
- `int shmem_offset`: this variable is ignored using a memory displacement
- `Grid *grid`: the grids for the subvectors, this structure is described in `pfsimulator/grid.h`
- `SubgridArray  *data_space`: description of the subvector, `SubgridArray` is described in `pfsimulator/grid.h` and is the equivalent of `SubregionArray` described in `pfsimulator/region.h`.
- `int size`

The `subvector` structure described in `pfsimulator/vector.h` has the same name in the Yaml file.
This structure has the following members:
- `double  *data`
- `int allocated`
- `Subgrid *data_space`: this structure is describe in `pfsimulator/grid.h` and is the equivalent of `Subregion` described in `region.h`
- `int data_size`

The `Subregion` (`Subgrid`) and `SubregionArray` (`SubgridArray`) structures with the same names are described in `region.h`


### Output functions in ParFlow

The file that contains the calls to the PDI interface is `write_parflow_pdi.c`.
The way these functions are used is very similar to the other IO methods (PFB, Silo...).

The pdi parflow functions are called in th solver files:
- `pfsimulator/solver_impes.c`
- `pfsimulator/lb.c`
- `pfsimulator/richards.c`

## Installation and Usage

Detailed installation instructions, usage examples, and the Python script for comparing PDI files with ParFlow binary files are available in the main ParFlow [README](/https://github.com/parflow/parflow/tree/master/README-PDI.md).