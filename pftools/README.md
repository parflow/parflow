# ParFlow Utilities

## pfmask-to-pfsol

Utitility to build 3D PFSOL domain from 2D mask file(s).

### Usage

There are two modes of running.  In the first case a single mask file
is supplied, enabling a top surface to have multiple patches with the
sides/bottom labeled as one patch.  In the second mode, a mask file is
provided for each direction to enable labeling of each cell surface.
The values in the mask file are used to label the patches.

```shell
pfmask-to-pfsol 
	      --mask <top mask input filename> 
	      --vtk <VTK output filename> 
	      --pfsol <PFSOL output filename> 
	      --bottom-patch-label <bottom patch id> 
	      --side-patch-label <side_patch id>
```   

Creates a PFSOL file based on 2D mask input that defines the domain
and patches on the top surface.  The domain is extruded in Z to form a
3D domain.  The mask input file can be many of the standard file types
supported by ParFlow, such as ParFlow binary or simple ASCI. The ASC
file format is also supported.

The mask input must be 2D with number of points in Z = 1;

The bottom and side patches are labeled with the supplied patch id's.
The top mask file is used to label patches on the top surface based on
the values in the mask file.

```shell
pfmask-to-pfsol 
	--mask-top <top mask filename>
	--mask-bottom <bottom mask filename>
	--mask-left <left mask filename>
	--mask-right <right mask filename>
	--mask-front <front mask filename>
	--mask-back <back mask filename>
        --vtk <VTK output filename> --pfsol <PFSOL output filename>
```

Each of the mask values is used to label the external boundary patches
based on the value.  This enables the sides and bottom to have a more
complex patch labeling.   Faces are along each axis:

Top    +Z
Bottom -Z

Right  +X
Left   -X

Front  -Y
Back   +Y

Each mask file should be the same dimensions in X and Y and have
number of points in Z = 1.

Note that the axis alignment shown here is based on the standard PF
axis alignment used inside the code and in PF file formats.  Other
file formats, such as the ASC format, use different alignments.  In
the ASC format the Y axis is inverted from the PF file formats.

#### Setting Top/Bottom of domain

The mask file is typically coming from a 2D file so does not have any
depth information.  By default the top is 1000.0 and bottom is 0.0.  The
top and bottom can be set using the "--z-top" and "--z-bottom" flags:

```shell
pfmask-to-pfsol --z-top 200.0 --z-bottom 10.0 <other args>
```

#### Setting DX/DY

The simple PF file formats do not include DX/DY/DZ information, this
information may be set using the top, bottom, dx, and dy flags.  The
dx/dy flags have priority over information coming from from the input
file (such as the cellsize value from an ASC file).

```shell
pfmask-to-pfsol --z-top 200.0 --z-bottom 10.0 --dy 1.0 --dx 1.0 <other args>
```

### Mask ASC file format

The input mask is an ASC file format with the following format:

ncols        4
nrows        4
xllcorner    0.0
yllcorner    0.0
cellsize     1.0
NODATA_value  0.0
<ncols * nrows values>

A 0 value is outside the domain, any other value is inside the domain.

## pfsol-to-vtk

This utility is used to convert a PFSOL file to a VTK file for easier visualization.

### Usage

pfsol-to-vtk <PFSOL input filename> <VTK output filename>

## pfmaskdownsize

Utility to downsize a large mask file for creating smaller domains for testing.

A work in progress, currently hardcoded to extract the region : (0,0) - (3000,1000).

```shell
pfmaskdownsize <in mask file as ASC format> <out mask file as ASC format>
```






