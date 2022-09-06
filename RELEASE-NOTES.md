# ParFlow Release Notes 3.11.0
------------------------------

This release contains several bug fixes and minor feature updates.

ParFlow development and bug-fixes would not be possible without contributions of the ParFlow community.  Thank you for all the great contributions.

## Overview of Changes

* Improved reading of PFB file in Python PFTools
* ParFlow Documentation Update 
* PDF User Manual removed from the repository
* Initialization of evap_trans vector has been moved
* CUDA fixes
* OASIS array fix

## User Visible Changes

### Improved reading of PFB file in Python PFTools

Subgrid header information is read directly from the file to enable reading of files with edge data like the velocity files.

Fixes some cases where PFB files with different z-dimension shapes could not be merged together in xarray. Notably this happens for surface parameters which have shape (1, ny, nx) which really should be represented by xarray by squeezing out the z dimension. This now happens in xarray transparently. Loading files with the standard read_pfb or read_pfb_sequence will not auto-squeeze dimensions.

Perfomance of reading should be improved by using memmapped and only the first subgrid header is read when loading a sequence of PFB files.   Parallism should be better in the read.

The ability to give keys to the pf.read_pfb function for subsetting was added.
	
### ParFlow Documentation Update 

The User Manual is being transitioned to ReadTheDocs from the previous LaTex manual.   A first pass at the conversion of the ParFlow LaTeX manual to ReadTheDocs format. This new documentation format contains the selected sections from the ParFlow LaTeX manual along with Kitware's introduction to Python PFTools and resulting tutorials. Added new sections documenting the Python PFTools Hydrology module, the Data Accessor class, and updated the PFB reading/writing tutorial to use the updated PFTools functions instead of parflowio.
    
The original LaTeX files remain intact for now as this documentation conversion isn't fully complete.   Currently this version of the ReadTheDocs is not generating the KitWare version of the ParFlow keys documentation but as a longer-term task they can be re-integrated into the new manual.

### PDF User Manual removed from the repository

The PDF of the User Manual that was in the repository has been removed.  An online version of the users manual is available on [Read the Docks:Parflow Users Manual](https://parflow.readthedocs.io/en/latest/index.html).  A PDF version is available at [Parflow Users Manual PDF](https://parflow.readthedocs.io/_/downloads/en/latest/pdf/).
    
## Internal/Developer Changes

### Initialization of evap_trans has been moved
The Vector evap_trans was made part of the InstanceXtra structure, initialized is done in SetupRichards() and deallocated in TeardownRichards().

### CUDA 11.5 update

Starting from CUB 1.14, CUB_NS_QUALIFIER macro must be specified.  The CUB_NS_QUALIFIER macro in the same way it was added in the CUB (see https://github.com/NVIDIA/cub/blob/94a50bf20cc01f44863a524ba36e089fd80f342e/cub/util_namespace.cuh#L99-L109)

### CUDA Linux Repository Key Rotation
    
Updating NVidia CUDA repository keys due to rotation as documented [here](https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772)

## Bug Fixes


### Minor improvements/bugfix for python IO 

Fix a bug on xarray indexing which require squeezing out multiple dimensions. Lazy loading is implemented natively now with changes to the indexing methods.

### OASIS array fix

vshape should be a 1d array instead of a 2d array.  Its attributes are specified as [INTEGER, DIMENSION(2*id var nodims(1)), IN] based on the [OASIS3-MCT docs](https://gitlab.com/cerfacs/oasis3-mct/-/blob/OASIS3-MCT_3.1/doc/oasis3mct_UserGuide.pdf)


### Python pftools version parsing
    
Minor bugfix was needed in Python pftools for parsing versions.

## Known Issues

See https://github.com/parflow/parflow/issues for current bug/issue reports.
