# ParFlow Release Notes 3.10.0
-----------------------------

ParFlow improvements and bug-fixes would not be possible without
contributions of the ParFlow community.  Thank you for all the great
contributions.

Note : Version 3.10.0 is a minor update to v3.9.0.  These release notes cover 
changes made in 3.10.0

## Overview of Changes

* Python dependency is now 3.6
* Python PFB reader/writer updated
* Bug fixes

## User Visible Changes

### Python: Python version dependency update

Python 3.6 or greater is now required for building and running ParFlow if Python is being used.

### Python: PFB reader/writer updated

Add simple and fast pure-python based readers and writers of PFB files, done by myself and Bill Hasling (@wh3248). This eliminates the need for the external ParflowIO dependency. Implemented a new backend for the xarray package that let's you open both .pfb files as well as .pfmetadata files directly into xarray datastructures. These are very useful for data wrangling and scientific analysis

Basic usage of the new functionality:

```
import parflow as pf

# Read a pfb file as numpy array:
x = pf.read_pfb('/path/to/file.pfb')

# Read a pfb file as an xarray dataset:
ds = xr.open_dataset('/path/to/file.pfb', name='example')

# Write a pfb file with distfile:
pf.write_pfb('/path/to/new_file.pfb', x, 
             p=p, q=q, r=r, dist=True)
```

### Python: SolidFileBuilder simplification

Support simple use case in SolidFileBuilder when all work can simply be delegated to pfmask-to-pfsol
Added a generate_asc_files (default False) argument to SolidFileBuilder.write 

### Python: Fixed reading of vegm array

Fixed indices so that the x index of the vegm_array correctly reflects the columns and y index reflects the rows. The _read_vegm function in PFTools was inconsistent with parflow-python xy indexing.

### Python : Python PFTools version updates

Updated Python PFTools dependency to current version 3.6.

## Bug Fixes

### Examples/Tests: Fix errors in LW_Test test case. 

LW_Test runs successfully and works in parallel.

### Increased input database maximum value size from 4097 to 65536.

The maximum input database value length was increased from 4097 to 65536. A bounds check is performed that emits a helpful error message when a database value is too big.

### Python: interface fixed issue where some keys failed to set when unless set in a particular order

1) Update some documentation for contributing to pf-keys
2) Fix a bugs found in pf-keys where some keys failed to set when unless set in a particular order
3) Add constraint for lists of names

This change lets us express that one list of names should be a subset of another list of names
Constraint Example

Values for PhaseSources.{phase_name}.GeomNames should be a subset of values from either GeomInput.{geom_input_name}.GeomNames or GeomInput.{geom_input_name}.GeomName. Setting the domain to EnumDomain like so expresses that constraint. A more detailed example can be seen in this test case.

## Internal/Developer Changes

### Core: Diverting ParFlow output to stream

Added new method for use when ParFlow is embedded in another application to control the file stream used for ParFlow logging messages. In the embedded case will be disabled by default unless redirected by the calling application.

Change required to meet IDEAS Watersheds best practices.

### Python: Add keys and generator for Simput

Added keys and generator to allow use Simput and applications based on Simput to write inputs for parflow with a graphical web interface.

### Core: Remove use of MPI_COMM_WORLD 

Enable use of a communicator other than MPI_COMM_WORLD for more general embedding.
Meet IDEAS Watersheds best practices policy.

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.
