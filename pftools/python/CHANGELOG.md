# Changelog for Python-PFTools (pftools package)

## Unreleased [py_features branch](https://github.com/grapp1/parflow/tree/py_features):

- Domain builder to streamline run definition
	- Use mainstream action to automatically set parflow keys
	- A more natural problem definition that automatically sets ParFlow keys and values with appropriate defaults.

## v0.0.4 (released 2020-10-09):

- Validation can now skip valid keys to provide more compact feedback
	- Validation method has a new optional argument
	- Command line argument to skip valid keys in validation
- Using tables for subsurface properties
	- Using tables of various formats to setup a run with many subsurface / soils properties
	- Exporting table of subsurface properties from a run

## v0.0.3 (released 2020-09-17):
- Validating Parflow input keys
	- Before running parflow
	- Validating while parflow execution using command line arguments
- Dry run
	- Execute run while skipping parflow execution (CLI)
	- Export run settings (CLI)
- CLI
	- Convert TCL script to python
	- Sort pfidb for easy comparison
- Saving run/pfkeys settings to a file
	- pfidb/yaml/json
	- Using command line arguments without changing your run script
- Loading run settings from a file
	- pfidb/yaml/json
- Setting parflow keys from various inputs
	- pfkey/yaml/json/pfidb
- pfdist
	- Inside parflowio
	- Inside pftools
- Cloning run to set keys for another run
- Setting keys that may not exist in the main ParFlow library, or are in different formats (e.g., yaml hierarchy)
- Executing ParFlow
	- Optional validation
	- Changing working directory
	- CLI arguments
- File handling
	- Rooting paths relative to the run script location
	- Getting absolute path from a relative path
	- Getting absolute path based on an environment variable to automatically adapt to runtime environment
	- Copying and removing files/directories
	- Checking whether file/directory exists
	- Making and changing directories
- Solid files:
	- IO tools:
		- Creating a matrix of patches for a domain using images, PFB files, .asc, and .sa files.
		- Writing out matrix of patches to an .asc or .sa file.
	- Building solid files:
		- Building a solid file using a mask matrix as an input (using IO tools)
		- Specifying and/or changing numbers for patches on top, bottom, and sides
		- Writes out to a .pfsol file with the option of writing to a .vtk file for checking and visualization
		- Sets appropriate keys in ParFlow run (i.e., *geomItem*.InputType and *geomItem*.FileName) to reference solid file
