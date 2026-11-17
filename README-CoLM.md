# ParFlow-CoLM Coupling

ParFlow-CoLM is a renewed coupling effort that connects ParFlow's
three-dimensional variably saturated groundwater and overland-flow
solver with the updated water and energy modules of CoLM. Following the
original ParFlow-CLM coupling framework, CoLM provides land-surface net
water fluxes to ParFlow as source/sink terms, while ParFlow
returns soil moisture and pressure head to CoLM. CoLM is the latest 2024
release of the Common Land Model, developed and maintained by the Common
Land Model team. This ParFlow-CoLM coupling was led by Chen Yang's group in
collaboration with the Common Land Model team. If you have any questions,
please contact yangch329@mail.sysu.edu.cn. The current ParFlow-CoLM effort
is described in:

[Yang, C., Sun, A., Zhang, S., Dai, Y., Kollet, S., & Maxwell, R. (2026). 20 years of trials and insights: bridging legacy and next generation in ParFlow and Land Surface Model Coupling. Geoscientific Model Development, 19(5), 1849-1866.](https://doi.org/10.5194/gmd-19-1849-2026)

## Enabling CoLM

CoLM is selected through the land surface model key:

```python
run.Solver.LSM = "CoLM"
```

or in TCL:

```tcl
pfset Solver.LSM CoLM
```

The coupled configuration requires ParFlow to be built with land surface
model support, currently enabled through the existing CLM build option:

```shell
cmake ../parflow -DPARFLOW_HAVE_CLM=ON ...
```

The example in `pfsimulator/colm/examples/colm_version` shows a compact
ParFlow-CoLM setup. It sets `Solver.LSM = "CoLM"` while continuing to use
the existing `Solver.CLM.*` ParFlow keys for shared land-model coupling
controls such as forcing, output, and root-zone settings.

Unlike ParFlow-CLM, for which the number of coupled soil layers is
configurable, **the current ParFlow-CoLM coupling supports exactly 10 coupled
soil layers**. This restriction applies to the soil column exchanged between
ParFlow and CoLM, not necessarily to the total number of layers in the
ParFlow computational grid. The example therefore sets
`Solver.CLM.RootZoneNZ = 10`, while using an additional deeper ParFlow layer
below the coupled soil column.

## CoLM Input Keys

CoLM and CLM use the same ParFlow input keys for land-model coupling. The
land surface model is selected with `Solver.LSM`, while forcing, output,
vegetation-water-stress, and root-zone controls for both models use the
existing `Solver.CLM.*` key namespace.

```python
<runname>.Solver.LSM = "CoLM"
<runname>.Solver.CLM.CLMFileDir = "clm_output_path"      # retained for CLM-key compatibility
<runname>.Solver.CLM.Print1dOut = False                  # retained for CLM-key compatibility
<runname>.Solver.CLM.CLMDumpInterval = 1

<runname>.Solver.CLM.MetForcing = "1D"
<runname>.Solver.CLM.MetFileName = "station0.txt"
<runname>.Solver.CLM.MetFilePath = "path/to/met/forcing/data/"
<runname>.Solver.CLM.MetFileNT = 24
<runname>.Solver.CLM.IstepStart = 1

<runname>.Solver.CLM.EvapBeta = "Linear"
<runname>.Solver.CLM.VegWaterStress = "Saturation"
<runname>.Solver.CLM.ResSat = 0.2
<runname>.Solver.CLM.WiltingPoint = 0.2
<runname>.Solver.CLM.FieldCapacity = 1.00
<runname>.Solver.CLM.IrrigationType = "none"             # retained for CLM-key compatibility

<runname>.Solver.CLM.RootZoneNZ = 10
<runname>.Solver.CLM.SoiLayer = 10                       # retained for CLM-key compatibility
<runname>.Solver.CLM.ReuseCount = 1
<runname>.Solver.CLM.WriteLogs = False                   # retained for CLM-key compatibility
<runname>.Solver.CLM.WriteLastRST = True                 # retained for CLM-key compatibility
<runname>.Solver.CLM.DailyRST = True                     # retained for CLM-key compatibility
<runname>.Solver.CLM.SingleFile = True
```

## Main CoLM Input Files

The example requires three CoLM input files:

- `CoLM_nlfile.nml`
- `CoLM_readin.dat`
- `snicar_par.dat`

The first two files contain case-specific run controls, patch metadata, and
soil properties. `snicar_par.dat` contains the SNICAR snow optics parameters.
This file normally does not need to be modified, but it must be present in the
run directory for CoLM to initialize and run. Additional files in the example
directory provide meteorological forcing, ParFlow terrain and subsurface
inputs, and run scripts.

### `CoLM_nlfile.nml`

`CoLM_nlfile.nml` is the CoLM run-control Fortran namelist read during
CoLM initialization. In the example, it contains the `nl_colm` namelist
and sets basic run-control options:

- `DEF_simulation_time%start_year`
- `DEF_simulation_time%start_month`
- `DEF_simulation_time%start_day`
- `DEF_simulation_time%start_sec`
- `DEF_WRST_FREQ`
- `DEF_hotstart`

For example:

```fortran
&nl_colm
   DEF_simulation_time%start_year   = 2010
   DEF_simulation_time%start_month  = 1
   DEF_simulation_time%start_day    = 1
   DEF_simulation_time%start_sec    = 0
   DEF_WRST_FREQ = 'DAILY'
   DEF_hotstart = .false.
/
```

The current coupling reads this file as `CoLM_nlfile.nml` from the run
directory.

### `CoLM_readin.dat`

`CoLM_readin.dat` is a formatted coupling-side table that provides CoLM
patch metadata and vertically resolved soil properties for each ParFlow
horizontal grid cell. The reader loops over the global ParFlow `x` and
`y` grid and reads the row associated with each cell.

The current reader expects each row in the following order:

1. `i`
2. `j`
3. `patchclass`
4. `patchlonr`
5. `patchlatr`
6. `int_soil_grav_l(1:8)`
7. `int_soil_sand_l(1:8)`
8. `int_soil_clay_l(1:8)`
9. `int_soil_oc_l(1:8)`
10. `int_soil_bd_l(1:8)`

`patchclass` identifies the CoLM land-cover or patch-type class, while
`patchlonr` and `patchlatr` provide the patch longitude and latitude.
The five vertically resolved soil-property groups are:

- `int_soil_bd_l` (`BD`): bulk density of fine earth, including its mineral
  and organic components, in `g/cm^3`.
- `int_soil_grav_l` (`GRAV`): gravel or coarse-fragment content as a
  percentage of soil volume.
- `int_soil_oc_l` (`SOC`): organic carbon content of fine earth as a
  percentage by weight.
- `int_soil_sand_l` (`SAND`): sand content of the mineral fine-earth
  fraction (50-2000 micrometers) as a percentage by weight.
- `int_soil_clay_l` (`CLAY`): clay content of the mineral fine-earth
  fraction (0-2 micrometers) as a percentage by weight.

The soil-property data used to prepare `CoLM_readin.dat` are derived from the
Global Soil Dataset for Earth System Modeling (GSDE).
GSDE provides these properties at eight depth intervals extending to about
2.3 m, corresponding to the eight values supplied for each property group in
the current coupling input.

Only eight soil-property levels are provided in `CoLM_readin.dat` because
the coupling maps these data onto the 10 CoLM soil layers by reusing the same
properties for layers 1 and 2, and again for layers 9 and 10. Thus, layers 1
and 2 share identical soil properties, layers 9 and 10 share identical soil
properties, and the remaining layers use the corresponding entries from the
eight-level input table.

### `snicar_par.dat`

`snicar_par.dat` provides the optical parameters used by the SNICAR snow
radiative-transfer scheme in CoLM. The supplied file should be used without
modification for normal ParFlow-CoLM simulations. Although users do not need
to configure its contents, the file is a required CoLM runtime input and must
be available as `snicar_par.dat` in the run directory.

## Example Directory

The example directory is:

```text
pfsimulator/colm/examples/colm_version
```

Important files include:

- `unname_test.py`: ParFlow Python input script for the coupled example.
- `CoLM_nlfile.nml`: CoLM namelist.
- `CoLM_readin.dat`: CoLM patch metadata and vertically resolved
  soil-property table.
- `station0.txt`: 1-D meteorological forcing used by
  `Solver.CLM.MetForcing = "1D"`.
- `snicar_par.dat`: required SNICAR snow optics parameter file; normally used
  without modification.
- `unname.slopex.pfb`, `unname.slopey.pfb`, `unname.manning.pfb`, and
  `unname.subsur.pfb`: ParFlow terrain, Manning's coefficient, and
  subsurface inputs used by the example.
