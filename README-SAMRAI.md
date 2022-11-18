# Parflow with SAMRAI Notes

**WARNING!** The SAMRAI version of Parflow is still beta and there may
be issues with it.

The Paflow/SAMRAI version may use significantly less memory for some
problems due to a more flexible computation domain.  In Parflow the
computation domain is specified as a single parallelepiped in the TCL
input script.  Parflow/SAMRAI will also accept this input but in
addition supports specifying the computation domain as a set of
parallelepiped subgrids.  You may have more than one subgrid per
processor.  This enables the computation domain to more closely match
the active region of the problem.

Note for CLM and WRF runs only a single subgrid is allowed per
processor since CLM and WRF only decompose problems in X and Y.  Also
the subgrids will need to match the decomposition that WRF is using in
X and Y.


## Configuration

Building with SAMRAI is an optional specified in the configure process
using the '--with-samrai=<SAMRAI_DIR>' option.  Where <SAMRAI_DIR> is
the installed location of SAMRAI; should match the
'--prefix=<SAMRAI_DIR>' specified during the SAMRAI configure.

The same compilers should be used to build Parflow and SAMRAI to avoid
compilation issues.  The same HDF and other options in both the SAMRAI
and Parflow configures should be used to avoid library incompatibility
issues.

Parflow should be built with the C++ compiler when building for use
with SAMRAI.  This can be done by setting the CC variable used to
specify the compiler to Parflow.

So a minimal configure which includes SAMRAI would be:

```shell
CC=g++ ./configure --with-samrai=<SAMRAI_DIR>
```

If you fail to use a C++ compiler with the SAMRAI option Parflow will
not compile.

## Running Parflow with SAMRAI

You can run any existing input script with Parflow/SAMRAI; no changes
are necessary.  This should produce the same output as the non-SAMRAI
version.  However this offers no real advantage but is supported to
allow for backward compatibility.  If you want to run with a SAMRAI
grid you need to do two things differently.  First the compute domain
specification is more complicated.  Each subgrid must be specified.
Second you must use pfdistondomain instead of pfdist to distribute
input files.

### File distribution

All of the grid based input files to Parflow must be specified on the
SAMRAI grid.  The "pfdistondomain" command distributes a file onto a
domain in a similar way to pfdist worked for the old code.  Indicator
fields etc will need to be distributed using this utility.  Basically
if you used pfdist on something before, you must use pfdistondomain
with the SAMRAI version.

### Input file changes for running with a SAMRAI grid

This simplest example of specifying a SAMRAI grid is a single subgrid
run on a single processor:

```tcl
	pfset ProcessGrid.NumSubgrids 1
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	
	pfset ProcessGrid.0.NX 10
	pfset ProcessGrid.0.NY 10
	pfset ProcessGrid.0.NZ 8
```

NumSubgrids is the total number of subgrids in the specification
(across all processors).  For each subgrid you must specify the
processor (P) and the starting index and number of grid points along
each dimension.
 
To run the previous problem on 2 processors the input
might look like:

```tcl
	pfset ProcessGrid.NumSubgrids 2
	pfset ProcessGrid.0.P 0
	pfset ProcessGrid.0.IX 0
	pfset ProcessGrid.0.IY 0
	pfset ProcessGrid.0.IZ 0
	pfset ProcessGrid.0.NX 10
	pfset ProcessGrid.0.NY 5
	pfset ProcessGrid.0.NZ 8
	
	pfset ProcessGrid.1.P 1
	pfset ProcessGrid.1.IX 0
	pfset ProcessGrid.1.IY 5
	pfset ProcessGrid.1.IZ 0
	pfset ProcessGrid.1.NX 10
	pfset ProcessGrid.1.NY 5
	pfset ProcessGrid.1.NZ 8
```

Which specifies a split of the domain along the Y axis at Y=5.

See the test script "samrai.tcl" for some examples on different
processor topologies and using more than one subgrid per processor.

If you are manually building the grid for a CLM run you need to specify
subgrid extents to include overlap for the active region ghost layer.
Basically you need to make sure that the IZ and NZ values are such
that they cover the active domain for each processor plus the active
domain in neighboring processors ghost layers (IX-2 to NX+4).  The
overland flow calculation needs to have the information about the top
of the domain to correctly move water to/from neighboring subgrids and
communication is done only along subgrid boundaries so the subgrid
extents need to be high enough to communicate the information between
neighbors.

Manually building this grid is obviously less than ideal so some
automated support is provided to help build a computation grid that
follows the terrain.  Emphasis on "some"; we realize this is a
somewhat annoying procedure to have to do and hopefully we can automate
this in the future.

The automated approach first requires running Paflow using the
original single large computation domain approach for a single
time-step (can be really small).  Using the mask file from that is
created by this run one can use the "pfcomputedomain" and
"pfprintdomain" commands in pftools to write out a grid.  This grid
will use the processor topology you have specified to build a grid
such that each processor's subgrid covers only the extent of the
active region (which comes from the mask file).

A sample script compute_domain.tcl which does this is enclosed in test
directory. The important parts are shown below.

This assumes the mask file exists and is called "samrai.out.mask.pfb".

First the processor topology is specified.  The original large
computation domain is specified.  The mask file is then loaded.  The
top and bottom of the domain are computed from the mask.  These are
NX*NY arrays with values specifying the Z index of the top/bottom.
pfcomputedomain computes the subgrids that cover top to bottom on each
processor.  This grid specification is then saved to the
"samrai_grid.tcl" file.  You can use the TCL "source" command to
include this in your Parflow TCL input script (or cut and paste if you
prefer).

Note that if you change the processor topology you need to rerun this
script as the subgrids are processor dependent but you do not need to
recompute the mask file.

```tcl
set P  [lindex $argv 0]
set Q  [lindex $argv 1]
set R  [lindex $argv 2]

pfset Process.Topology.P $P
pfset Process.Topology.Q $Q   
pfset Process.Topology.R $R

set NumProcs [expr $P * $Q * $R]

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X                -10.0
pfset ComputationalGrid.Lower.Y                 10.0
pfset ComputationalGrid.Lower.Z                  1.0

pfset ComputationalGrid.DX                       8.8888888888888893
pfset ComputationalGrid.DY                      10.666666666666666
pfset ComputationalGrid.DZ                       1.0

pfset ComputationalGrid.NX                      10
pfset ComputationalGrid.NY                      10
pfset ComputationalGrid.NZ                       8

set mask [pfload samrai.out.mask.pfb]

set top [pfcomputetop $mask]
set bottom [pfcomputebottom $mask]

set domain [pfcomputedomain $top $bottom]

set out [pfprintdomain $domain]

set grid_file [open samrai_grid.tcl w]
puts $grid_file $out
close $grid_file
```