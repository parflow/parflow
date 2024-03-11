# ESMX ParFlow Test Application Instructions

## Build
Include ParFlow in ESMX Build configuration file
```
  ParFlow:
    fort_module: parflow_nuopc.mod
    libraries: parflow_nuopc
    build_args: -DPARFLOW_ENABLE_NUOPC=ON -DPARFLOW_AMPS_LAYER=mpi1 -DPARFLOW_HAVE_CLM=ON -DPARFLOW_ENABLE_HYPRE=ON -DPARFLOW_ENABLE_SILO=ON -DPARFLOW_ENABLE_NETCDF=ON
    test_dir: ParFlow/test/esmx
```

## Configuration
Edit esmxRun.config

### Component List
Provide list of component names to ESMX Driver.
```
ESMX_component_list: LND GWR
```

### ESMX Data Component
Configure the ESMX Data component
```
LND_model: ESMX_Data
```

| LND\_attributes | Value                    | Description                   |
| --------------- | ------------------------ | ----------------------------- |
| Verbosity       | max                      | Enable all generic output     |
| Diagnostic      | 0                        | No diagnostic output          |
| coordSys        | ESMF\_COORDSYS\_SPH\_DEG | Set coordinates using degrees |
| minx            | -98.426653               | Western longitude of LW       |
| maxx            | -97.718663               | Eastern longitude of LW       |
| miny            | 34.739932                | Southern latitude of LW       |
| maxy            | 35.031552                | Northern latitude of LW       |

| LND\_import\_fields                            | dim | min | max    |
| ---------------------------------------------- | --- | --- | ------ |
| soil\_moisture\_fraction\_layer\_1             | 2   | 0   | 1      |
| soil\_moisture\_fraction\_layer\_2             | 2   | 0   | 1      |
| soil\_moisture\_fraction\_layer\_3             | 2   | 0   | 1      |
| soil\_moisture\_fraction\_layer\_4             | 2   | 0   | 1      |
| liquid\_fraction\_of\_soil\_moisture\_layer\_1 | 2   | 0   | 1      |
| liquid\_fraction\_of\_soil\_moisture\_layer\_2 | 2   | 0   | 1      |
| liquid\_fraction\_of\_soil\_moisture\_layer\_3 | 2   | 0   | 1      |
| liquid\_fraction\_of\_soil\_moisture\_layer\_4 | 2   | 0   | 1      |
| ground\_water\_storage                         | 2   | 0   | 999999 |

| LND\_export\_fields          | dim | fill value         |
| ---------------------------- | --- | ------------------ |
| precip\_drip                 | 2   | 0.000024           |
| bare\_soil\_evaporation      | 2   | 0.000012           |
| total\_water\_flux\_layer\_1 | 2   | -0.000024000009596 |
| total\_water\_flux\_layer\_2 | 2   | -0.000000000004798 |
| total\_water\_flux\_layer\_3 | 2   | -0.000000000004798 |
| total\_water\_flux\_layer\_4 | 2   | -0.000000000004798 |

### ParFlow Component
Configure the ParFlow component
```
GWR_model: ParFlow
```

| HYD\_attributes      | Description                                   |
| -------------------- | --------------------------------------------- |
| Verbosity            | integer interpreted as a bit field            |
| Diagnostic           | integer interpreted as a bit field            |
| realize\_all\_import | true, false                                   |
| realize\_all\_export | true, false                                   |
| prep\_filename       | ParFlow preprocessor configuration file       |
| filename             | ParFlow configuration file                    |
| initialize\_import   | true, false                                   |
| initialize\_export   | true, false                                   |
| check\_import        | check import time stamp                       |
| coord\_type          | GRD\_COORD\_CARTESIAN or GRD\_COORD\_CLMVEGTF |
| coord\_filename      | file used when coordinates loaded from file   |
| output\_directory    | ParFlow cap output directory                  |


## Execution
```
mpirun -np 4 ./esmx
```

## Slurm Workload Manager
1. edit slurm\_template.sh
    - partition=\<partition\_names\>
    - account=\<account\>
    - constraint=\<list\>
    - qos=\<qos\>
    - setup environment as needed
```
sbatch slurm\_template.sh
```

## PBS Workload Manager
1. edit pbs\_template.sh
    - \-A \<account\>
    - \-q \<queue\>
    - setup environment as needed
```
qsub pbs\_template.sh
```

## Validation
Successful execution produces evaptrans, press, and satur files through 00003.

[Slurm Documentation](https://slurm.schedmd.com/documentation.html)

[PBS Pro Documentation](https://www.altair.com/pbs-works-documentation)
