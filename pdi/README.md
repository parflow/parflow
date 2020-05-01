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
