# Changelog for Python-PFTools (pftools package)

## v1.3.6 (Unreleased):

- Metadata keys and processing of CLM files
- Fix a bug on xarray indexing which require squeezing out multiple dimensions
- Add the ability to give keys to the pf.read_pfb function for subsetting
- Performance improvement with xarray by removing dask delayed call, which
  caused threadlocks. Lazy loading is implemented natively now with changes to
  the indexing methods.

## v1.0.0 (released 2020-11-12):

- Cleaned up code and API
- Added new selection methods to get attributes at arbitrary
locations. This includes `details()`, `doc()`, and `value()`.
- Selection methods (including `select()`) can use either dot
notation or slash notation now.
- Added ``from_definition`` class method to automatically create
a ``Run`` object from a file path.

## v0.0.6 (released 2020-10-28):

- Added ``set_name`` and ``get_name`` methods for ``Run`` object
- Changed validation printing to default to only print key/value
pairs with errors using ``verbose`` argument
- Updated code to comport with API change in parflowio v0.0.4
- Updated methods on ``Run`` object to revert to original directory after
executing the method
- Fixed bug for sorting .pfidb files

## v0.0.5 (released 2020-10-16):

- Domain builder to streamline run definition
	- Use mainstream action to automatically set parflow keys
	- A more natural problem definition that automatically sets ParFlow keys and values with appropriate defaults.
- Updates to subsurface table builder
    - Added more database options
- Added `pfidb_file` argument in `pfset` method to set keys from a pfidb file
- Added several keys to Python ParFlow library

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
