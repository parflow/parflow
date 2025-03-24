.. _Manipulating Data:

Manipulating Data: PFTools
==========================

Introduction to the ParFlow TCL commands (PFTCL) 
------------------------------------------------

Several tools for manipulating data are provided in PFTCL command set.
Tools can be accessed directly from the TCL shell or within a ParFlow
input script. In both cases you must first load the ParFlow package into
the TCL shell as follows:

.. container:: list

   ::

      #
      # To Import the ParFlow TCL package
      #
      lappend auto_path $env(PARFLOW_DIR)/bin
      package require parflow
      namespace import Parflow::*

In addition to these methods xpftools provides GUI access to most of
these features. However the simplest approach is generally to include
the tools commands within a tcl script. The following section lists all
of the available ParFlow TCL commands along with detailed instructions
for their use. §8.2 :ref:`PFTCL Commands` provides several examples of
pre and post processing using the tools. In addition, a list of tools
can be obtained by typing ``pfhelp`` into a TCL shell after importing 
ParFlow. Typing ``¿pfhelp¿`` followed by a command name will display a 
detailed description of the command in question.

.. _PFTCL Commands:

PFTCL Commands
--------------

The tables that follow `4.1 <#pftools1>`__, `4.2 <#pftools2>`__ and
`4.3 <#pftools3>`__ provide a list of ParFlow commands with short
descriptions grouped according to their function. The last two columns
in this table indicate what examples from §4.3 :ref:`common_pftcl`, if
any, the command is used in and whether the command is compatible with a
terrain following grid domain formulation.

.. container::
   :name: pftools1

   .. table:: List of PFTools commands by function.

      +----------------+----------------+--------------+----------------+
      | **Name**       | **Short        | **Examples** | **Compatible   |
      |                | Description**  |              | with TFG?**    |
      +================+================+==============+================+
      | pfhelp         | Get help for   |              | X              |
      |                | PF Tools       |              |                |
      +----------------+----------------+--------------+----------------+
      | Mathematical   |                |              |                |
      | Operations     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcellsum      | datasetx +     |              | X              |
      |                | datasety       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcelldiff     | datasetx -     |              | X              |
      |                | datasety       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcellmult     | datasetx \*    |              | X              |
      |                | datasety       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcelldiv      | datasetx /     |              | X              |
      |                | datasety       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcellsumconst | dataset +      |              | X              |
      |                | constant       |              |                |
      +----------------+----------------+--------------+----------------+
      | p              | dataset -      |              | X              |
      | fcelldiffconst | constant       |              |                |
      +----------------+----------------+--------------+----------------+
      | p              | dataset \*     |              | X              |
      | fcellmultconst | constant       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcelldivconst | dataset /      |              | X              |
      |                | constant       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsum          | Sum dataset    | 7, 9         | X              |
      +----------------+----------------+--------------+----------------+
      | pfdiffelt      | Element        |              | X              |
      |                | difference     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintdiff    | Print          |              | X              |
      |                | difference     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfmdiff        | Calculate area |              | X              |
      |                | where the      |              |                |
      |                | difference     |              |                |
      |                | between two    |              |                |
      |                | datasets is    |              |                |
      |                | less than a    |              |                |
      |                | threshold      |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintmdiff   | Print the      |              | X              |
      |                | locations with |              |                |
      |                | differences    |              |                |
      |                | greater than a |              |                |
      |                | minimum        |              |                |
      |                | threshold      |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsavediff     | Save the       |              | X              |
      |                | difference     |              |                |
      |                | between two    |              |                |
      |                | datasets       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfaxpy         | y=alpha*x+y    |              | X              |
      +----------------+----------------+--------------+----------------+
      | pfgetstats     | Calculate      |              | X              |
      |                | dataset        |              |                |
      |                | statistics     |              |                |
      |                | (min, max,     |              |                |
      |                | mean, var,     |              |                |
      |                | stdev)         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintstats   | Print          |              | X              |
      |                | formatted      |              |                |
      |                | statistics     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfstats        | Calculate and  |              | X              |
      |                | print dataset  |              |                |
      |                | statistics     |              |                |
      |                | (min, max,     |              |                |
      |                | mean, var,     |              |                |
      |                | stdev)         |              |                |
      +----------------+----------------+--------------+----------------+
      | Calculate      |                |              |                |
      | physical       |                |              |                |
      | parameters     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pfbfcvel       | Calculate      |              |                |
      |                | block face     |              |                |
      |                | centered       |              |                |
      |                | velocity       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcvel         | Calculate      |              |                |
      |                | Darcy velocity |              |                |
      +----------------+----------------+--------------+----------------+
      | pfvvel         | Calculate      |              |                |
      |                | Darcy velocity |              |                |
      |                | at cell        |              |                |
      |                | vertices       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfvmag         | Calculate      |              |                |
      |                | velocity       |              |                |
      |                | magnitude      |              |                |
      |                | given          |              |                |
      |                | components     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfflux         | Calculate      |              |                |
      |                | Darcy flux     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfhhead        | Calculate      | 2            |                |
      |                | hydraulic head |              |                |
      +----------------+----------------+--------------+----------------+
      | pfphead        | Calculate      |              |                |
      |                | pressure head  |              |                |
      |                | from hydraulic |              |                |
      |                | head           |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsattrans     | calculate      |              | X              |
      |                | saturated      |              |                |
      |                | transmissivity |              |                |
      +----------------+----------------+--------------+----------------+
      | pfupstreamarea | Calculate      |              | X              |
      |                | upstream area  |              |                |
      +----------------+----------------+--------------+----------------+
      | pfeff          | Calculate      |              | X              |
      | ectiverecharge | effective      |              |                |
      |                | recharge       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfw            | Calculate      |              | X              |
      | atertabledepth | water table    |              |                |
      |                | from           |              |                |
      |                | saturation     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfhydrostatic  | Calculate      |              |                |
      |                | hydrostatic    |              |                |
      |                | pressure field |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsub          | Calculate      | 7            | X              |
      | surfacestorage | total          |              |                |
      |                | sub-surface    |              |                |
      |                | storage        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfgwstorage    | Calculate      |              | X              |
      |                | saturated      |              |                |
      |                | subsurface     |              |                |
      |                | storage        |              |                |
      +----------------+----------------+--------------+----------------+
      | p              | Calculate      | 9            | X              |
      | fsurfacerunoff | total surface  |              |                |
      |                | runoff         |              |                |
      +----------------+----------------+--------------+----------------+
      | pf             | Calculate      | 8            | X              |
      | surfacestorage | total surface  |              |                |
      |                | storage        |              |                |
      +----------------+----------------+--------------+----------------+


.. container::
   :name: pftools2

   .. table:: List of PFTools commands by function (cont.).

      +----------------+----------------+--------------+----------------+
      | **Name**       | **Short        | **Examples** | **Compatible   |
      |                | Description**  |              | with TFG?**    |
      +================+================+==============+================+
      | DEM Operations |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pfslopex       | Calculate      | 5            | X              |
      |                | slopes in the  |              |                |
      |                | x-direction    |              |                |
      +----------------+----------------+--------------+----------------+
      | pfslopey       | Calculate      | 5            | X              |
      |                | slope in the   |              |                |
      |                | y-direction    |              |                |
      +----------------+----------------+--------------+----------------+
      | pfchildD8      | Calculate D8   |              | X              |
      |                | child          |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsegmentD8    | Calculate D8   |              | X              |
      |                | segment        |              |                |
      |                | lengths        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfslopeD8      | Calculate D8   |              | X              |
      |                | slopes         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfslopexD4     | Calculate D4   |              | X              |
      |                | slopes in the  |              |                |
      |                | x-direction    |              |                |
      +----------------+----------------+--------------+----------------+
      | pfslopeyD4     | Calculate D4   |              | X              |
      |                | slopes in the  |              |                |
      |                | y-direction    |              |                |
      +----------------+----------------+--------------+----------------+
      | pffillflats    | Fill DEM flats | 5            | X              |
      +----------------+----------------+--------------+----------------+
      | pfmovingavgdem | Fill dem sinks |              | X              |
      |                | with moving    |              |                |
      |                | average        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfpitfilldem   | Fill sinks in  | 5            | X              |
      |                | the dem using  |              |                |
      |                | iterative      |              |                |
      |                | pitfilling     |              |                |
      |                | routine        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfflintslawfit | Calculate      |              | X              |
      |                | Flint’s Law    |              |                |
      |                | parameters     |              |                |
      +----------------+----------------+--------------+----------------+
      | pfflintslaw    | Smooth DEM     |              | X              |
      |                | using Flints   |              |                |
      |                | Law            |              |                |
      +----------------+----------------+--------------+----------------+
      | pffl           | Smooth DEM     |              | X              |
      | intslawbybasin | using Flints   |              |                |
      |                | Law by basin   |              |                |
      +----------------+----------------+--------------+----------------+
      | Topmodel       |                |              |                |
      | functions      |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pftopodeficit  | Calculate      |              | X              |
      |                | TOPMODEL water |              |                |
      |                | deficit        |              |                |
      +----------------+----------------+--------------+----------------+
      | pftopoindex    | Calculate      |              | X              |
      |                | topographic    |              |                |
      |                | index          |              |                |
      +----------------+----------------+--------------+----------------+
      | pftopowt       | Calculate      |              | X              |
      |                | watertable     |              |                |
      |                | based on       |              |                |
      |                | topographic    |              |                |
      |                | index          |              |                |
      +----------------+----------------+--------------+----------------+
      | pftoporecharge | Calculate      |              | X              |
      |                | effective      |              |                |
      |                | recharge       |              |                |
      +----------------+----------------+--------------+----------------+
      | Domain         |                |              |                |
      | Operations     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | p              | Compute domain | 3            | X              |
      | fcomputedomain | mask           |              |                |
      +----------------+----------------+--------------+----------------+
      | pfcomputetop   | Compute domain | 3, 6, 8, 9   | X              |
      |                | top            |              |                |
      +----------------+----------------+--------------+----------------+
      | pfextracttop   | Extract domain | 6            | X              |
      |                | top            |              |                |
      +----------------+----------------+--------------+----------------+
      | p              | Compute domain | 3            | X              |
      | fcomputebottom | bottom         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsetgrid      | Set grid       | 5            | X              |
      +----------------+----------------+--------------+----------------+
      | pfgridtype     | Set grid type  |              | X              |
      +----------------+----------------+--------------+----------------+
      | pfgetgrid      | Return grid    |              | X              |
      |                | information    |              |                |
      +----------------+----------------+--------------+----------------+
      | pfgetelt       | Extract        | 10           | X              |
      |                | element from   |              |                |
      |                | domain         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfe            | Build 2D       |              | X              |
      | xtract2Ddomain | domain         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfenlargebox   | Compute        |              | X              |
      |                | expanded       |              |                |
      |                | dataset        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfgetsubbox    | Return subset  |              | X              |
      |                | of data        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintdomain  | Print domain   | 3            | X              |
      +----------------+----------------+--------------+----------------+
      | pfbuilddomain  | Build a        |              | X              |
      |                | subgrid array  |              |                |
      |                | from a ParFlow |              |                |
      |                | database       |              |                |
      +----------------+----------------+--------------+----------------+
      | Dataset        |                |              |                |
      | operations     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pflistdata     | Return dataset |              | X              |
      |                | names and      |              |                |
      |                | labels         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfgetlist      | Return dataset |              | X              |
      |                | descriptions   |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintlist    | Print list of  |              | X              |
      |                | datasets and   |              |                |
      |                | their labels   |              |                |
      +----------------+----------------+--------------+----------------+
      | pfnewlabel     | Change dataset |              | X              |
      |                | label          |              |                |
      +----------------+----------------+--------------+----------------+
      | pfnewdata      | Create new     |              | X              |
      |                | dataset        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintgrid    | Print grid     |              | X              |
      +----------------+----------------+--------------+----------------+
      | pfnewgrid      | Set grid for   |              | X              |
      |                | new dataset    |              |                |
      +----------------+----------------+--------------+----------------+
      | pfdelete       | Delete dataset |              | X              |
      +----------------+----------------+--------------+----------------+
      | pfreload       | Reload dataset |              | X              |
      +----------------+----------------+--------------+----------------+
      | pfreloadall    | Reload all     |              | X              |
      |                | current        |              |                |
      |                | datasets       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintdata    | Print all      |              | X              |
      |                | elements of a  |              |                |
      |                | dataset        |              |                |
      +----------------+----------------+--------------+----------------+
      | pfprintelt     | Print a single |              | X              |
      |                | element        |              |                |
      +----------------+----------------+--------------+----------------+


.. container::
   :name: pftools3

   .. table:: List of PFTools commands by function (cont.).

      +----------------+----------------+--------------+----------------+
      | **Name**       | **Short        | **Examples** | **Compatible   |
      |                | Description**  |              | with TFG?**    |
      +================+================+==============+================+
      | File           |                |              |                |
      | Operations     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pfload         | Load file      | All          | X              |
      +----------------+----------------+--------------+----------------+
      | pfloadsds      | Load           |              | X              |
      |                | Scientific     |              |                |
      |                | Data Set from  |              |                |
      |                | HDF file       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfdist         | Distribute     | 4            | X              |
      |                | files based on |              |                |
      |                | processor      |              |                |
      |                | topology       |              |                |
      +----------------+----------------+--------------+----------------+
      | pfdistondomain | Distribute     |              | X              |
      |                | files based on |              |                |
      |                | domain         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfundist       | Undistribute   |              | X              |
      |                | files          |              |                |
      +----------------+----------------+--------------+----------------+
      | pfsave         | Save dataset   | 1,2,5,6      | X              |
      +----------------+----------------+--------------+----------------+
      | pfsavesds      | Save dataset   |              | X              |
      |                | in an HDF      |              |                |
      |                | format         |              |                |
      +----------------+----------------+--------------+----------------+
      | pfvtksave      | Save dataset   | X            | X              |
      |                | in VTK format  |              |                |
      |                | using DEM      |              |                |
      +----------------+----------------+--------------+----------------+
      | pfwritedb      | Write the      |              | X              |
      |                | settings for a |              |                |
      |                | PF run to a    |              |                |
      |                | database       |              |                |
      +----------------+----------------+--------------+----------------+
      | Solid file     |                |              |                |
      | operations     |                |              |                |
      +----------------+----------------+--------------+----------------+
      | pfpatchysolid  | Build a solid  |              | X              |
      |                | file between   |              |                |
      |                | two complex    |              |                |
      |                | surfaces and   |              |                |
      |                | assign         |              |                |
      |                | user-defined   |              |                |
      |                | patches around |              |                |
      |                | the edges      |              |                |
      +----------------+----------------+--------------+----------------+
      | pfs            | Converts back  |              | X              |
      | olidfmtconvert | and forth      |              |                |
      |                | between ascii  |              |                |
      |                | and binary     |              |                |
      |                | formats for    |              |                |
      |                | solid files    |              |                |
      +----------------+----------------+--------------+----------------+


Detailed descriptions of every command are included below in
alphabetical order. Note that the required inputs are listed following
each command. Commands that perform operations on data sets will require
an identifier for each data set it takes as input. Inputs listed in
square brackets are optional and do not need to be provided.

.. container:: description

   ::

      pfaxpy alpha x y

   This command computes y = alpha*x+y where alpha is a scalar and x and
   y are identifiers representing data sets. No data set identifier is
   returned upon successful completion since data set y is overwritten.

   ::

      pfbfcvel conductivity phead

   This command computes the block face centered flow velocity at every
   grid cell. Conductivity and pressure head data sets are given as
   arguments. The output includes x, y, and z velocity components that
   are appended to the Tcl result.

   ::

      pfbuilddomain database

   This command builds a subgrid array given a ParFlow database that
   contains the domain parameters and the processor topology.

   ::

      pfcelldiff datasetx datasety mask

   This command computes cell-wise differences of two datasets
   (diff=datasetx-datasety). This is the difference at each individual
   cell, not over the domain. Datasets must have the same dimensions.

   ::

      pfcelldiffconst dataset constant mask

   This command subtracts a constant value from each (active) cell of
   dataset (dif=dataset - constant).

   ::

      pfcelldiv datasetx datasety mask

   This command computes the cell-wise quotient of datasetx and datasety
   (div = datasetx/datasety). This is the quotient at each individual
   cell. Datasets must have the same dimensions.

   ::

      pfcelldivconst dataset constant mask

   This command divides each (active) cell of dataset by a constant
   (div=dataset/constant).

   ::

      pfcellmult datasetx datasety mask

   This command computes the cell-wise product of datasetx and datasety
   (mult = datasetx \* datasety). This is the product at each individual
   cell. Datasets must have the same dimensions.

   ::

      pfcellmultconst dataset constant mask

   This command multiplies each (active) cell of dataset by a constant
   (mult=dataset \* constant).

   ::

      pfcellsum datasetp datasetq mask

   This command computes the cellwise sum of two datasets (i.e., the sum
   at each individual cell, not the sum over the domain). Datasets must
   have the same dimensions.

   ::

      pfcellsumconst dataset constant mask

   This command adds the value of constant to each (active) cell of
   dataset.

   ::

      pfchildD8 dem

   This command computes the unique D8 child for all cells. Child[i,j]
   is the elevation of the cell to which [i,j] drains (i.e. the
   elevation of [i,j]’s child). If [i,j] is a local minima the child
   elevation set the elevation of [i,j].

   ::

      pfcomputebottom mask

   This command computes the bottom of the domain based on the mask of
   active and inactive zones. The identifier of the data set created by
   this operation is returned upon successful completion.

   ::

      pfcomputedomain top bottom

   This command computes a domain based on the top and bottom data sets.
   The domain built will have a single subgrid per processor that covers
   the active data as defined by the top and bottom. This domain will
   more closely follow the topology of the terrain than the default
   single computation domain.

   A typical usage pattern for this is to start with a mask file (zeros
   in inactive cells and non-zero in active cells), create the top and
   bottom from the mask, compute the domain and then write out the
   domain. Refer to example number 3 in the following section.

   ::

      pfcomputetop mask

   This command computes the top of the domain based on the mask of
   active and inactive zones. This is the land-surface in ``clm`` 
   or overland flow simulations. The identifier of the data set created 
   by this operation is returned upon successful completion.

   ::

      pfcvel conductivity phead

   This command computes the Darcy velocity in cells for the
   conductivity data set represented by the identifier ‘conductivity’
   and the pressure head data set represented by the identifier ‘phead’.
   (note: This "cell" is not the same as the grid cells; its corners are
   defined by the grid vertices.) The identifier of the data set created
   by this operation is returned upon successful completion.

   ::

      pfdelete dataset

   This command deletes the data set represented by the identifier
   ‘dataset’. This command can be useful when working with multiple
   datasets / time series, such as those created when many timesteps of
   a file are loaded and processed. Deleting these datasets in between
   reads can help with tcl memory management.

   ::

      pfdiffelt datasetp datasetq i j k digits [zero]

   This command returns the difference of two corresponding coordinates
   from ‘datasetp’ and ‘datasetq’ if the number of digits in agreement
   (significant digits) differs by more than ‘digits’ significant digits
   and the difference is greater than the absolute zero given by ‘zero’.

   ::

      pfdist [options] filename 

   Distribute the file onto the virtual file system. This utility must
   be used to create files which ParFlow can use as input. ParFlow uses
   a virtual file system which allows each node of the parallel machine
   to read from the input file independently. The utility does the
   inverse of the pfundist command. If you are using a ParFlow binary
   file for input you should do a pfdist just before you do the pfrun.
   This command requires that the processor topology and computational
   grid be set in the input file so that it knows how to distribute the
   data. Note that the old syntax for pfdist required the NZ key be set
   to 1 to indicate a two dimensional file but this can now be specified
   manually when pfdist is called by using the optional argument -nz
   followed by the number of layers in the file to be distributed, then
   the filename. If the -nz argument is absent the NZ key is used by
   default for the processor topology.

   For example,

   .. container:: list

      ::

         pfdist -nz 1 slopex.pfb

   ::

      pfdistondomain filename domain

   Distribute the file onto the virtual file system based on the domain
   provided rather than the processor topology as used by pfdist. This
   is used by the SAMRAI version of which allows for a more complicated
   computation domain specification with different sized subgrids on
   each processor and allows for more than one subgrid per processor.
   Frequently this will be used with a domain created by the
   pfcomputedomain command.

   ::

      pfeffectiverecharge precip et slopex slopey dem

   This command computes the effective recharge at every grid cell based
   on total precipitation minus evapotranspiration (P-ET)in the upstream
   area. Effective recharge is consistent with TOPMODEL definition, NOT
   local P-ET. Inputs are total annual (or average annual) precipitation
   (precip) at each point, total annual (or average annual)
   evapotranspiration (ET) at each point, slope in the x direction,
   slope in the y direction and elevation.

   ::

      pfenlargebox dataset sx sy sz

   This command returns a new dataset which is enlarged to be of the new
   size indicated by sx, sy and sz. Expansion is done first in the z
   plane, then y plane and x plane.

   ::

      pfextract2Ddomain domain

   This command builds a 2D domain based off a 3D domain. This can be
   used for a pfdistondomain command for Parflow 2D data (such as slopes
   and soil indices).

   ::

      pfextracttop top data

   This command computes the top of the domain based on the top of the
   domain and another dataset. The identifier of the data set created by
   this operation is returned upon successful completion.

   ::

      pffillflats dem

   This command finds the flat regions in the DEM and eliminates them by
   bilinearly interpolating elevations across flat region.

   ::

      pfflintslaw dem c p

   This command smooths the digital elevation model dem according to
   Flints Law, with Flints Law parameters specified by c and p,
   respectively. Flints Law relates the slope magnitude at a given cell
   to its upstream contributing area: S = c*A**p. In this routine,
   elevations at local minima retain the same value as in the original
   dem. Elevations at all other cells are computed by applying Flints
   Law recursively up each drainage path, starting at its terminus (a
   local minimum) until a drainage divide is reached. Elevations are
   computed as:

   dem[i,j] = dem[child] + c*(A[i,j]**p)*ds[i,j]

   where child is the D8 child of [i,j] (i.e., the cell to which [i,j]
   drains according to the D8 method); ds[i,j] is the segment length
   between the [i,j] and its child; A[i,j] is the upstream contributing
   area of [i,j]; and c and p are constants.

   ::

      pfflintslawbybasin dem c0 p0 maxiter

   This command smooths the digital elevation model (dem) using the same
   approach as "pfflints law". However here the c and p parameters are
   fit for each basin separately. The Flint's Law parameters are
   calculated for the provided digital elevation model dem using the
   iterative Levenberg-Marquardt method of non-linear least squares
   minimization, as in "pfflintslawfit". The user must provide initial
   estimates of c0 and p0; results are not sensitive to these initial
   values. The user must also specify the maximum number of iterations
   as maxiter.

   ::

      pfflintslawfit dem c0 p0 maxiter

   This command fits Flint’s Law parameters c and p for the provided
   digital elevation model dem using the iterative Levenberg-Marquardt
   method of non-linear least squares minimization. The user must
   provide initial estimates of c0 and p0; results are not sensitive to
   these initial values. The user must also specify the maximum number
   of iterations as maxiter. Final values of c and p are printed to the
   screen, and a dataset containing smoothed elevation values is
   returned. Smoothed elevations are identical to running pfflintslaw
   for the final values of c and p. Note that dem must be a ParFlow
   dataset and must have the correct grid information – dx, dy, nx, and
   ny are used in parameter estimation and Flint’s Law calculations. If
   gridded elevation values are read in from a text file (e.g., using
   pfload’s simple ascii format), grid information must be specified
   using the pfsetgrid command.

   ::

      pfflux conductivity hhead

   This command computes the net Darcy flux at vertices for the
   conductivity data set ‘conductivity’ and the hydraulic head data set
   given by ‘hhead’. An identifier representing the flux computed will
   be returned upon successful completion.

   ::

      pfgetelt dataset i j k

   This command returns the value at element (i,j,k) in data set
   ‘dataset’. The i, j, and k above must range from 0 to (nx - 1), 0 to
   (ny - 1), and 0 to (nz - 1) respectively. The values nx, ny, and nz
   are the number of grid points along the x, y, and z axes
   respectively. The string ‘dataset’ is an identifier representing the
   data set whose element is to be retrieved.

   ::

      pfgetgrid dataset

   This command returns a description of the grid which serves as the
   domain of data set ‘dataset’. The format of the description is given
   below.

   -  ::

         (nx, ny, nz)

      The number of coordinates in each direction.

   -  ::

         (x, y, z)

      The origin of the grid.

   -  ::

         (dx, dy, dz)

      The distance between each coordinate in each direction.

   The above information is returned in the following Tcl list format:
   nx ny nz x y z dx dy dz

   ::

      pfgetlist dataset

   This command returns the name and description of a dataset if an
   argument is provided. If no argument is given, then all of the data
   set names followed by their descriptions is returned to the TCL
   interpreter. If an argument (dataset) is given, it should be the it
   should be the name of a loaded dataset.

   ::

      pfgetstats dataset

   This command calculates the following statistics for the data set
   represented by the identifier *dataset*: minimum, maximum, mean, sum,
   variance, and standard deviation.

   ::

      pfgetsubbox dataset il jl kl iu ju ku

   This command computes a new dataset with the subbox starting at il,
   jl, kl and going to iu, ju, ku.

   ::

      pfgridtype gridtype

   This command sets the grid type to either cell centered if ‘gridtype’
   is set to ‘cell’ or vertex centered if ‘gridtype’ is set to ‘vertex’.
   If no new value for ‘gridtype’ is given, then the current value of
   ‘gridtype’ is returned. The value of ‘gridtype’ will be returned upon
   successful completion of this command.

   ::

      pfgwstorage mask porosity pressure saturation specific_storage

   This command computes the sub-surface water storage (compressible and
   incompressible components) based on mask, porosity, saturation,
   storativity and pressure fields, similar to pfsubsurfacestorage, but
   only for the saturated cells.

   ::

      pfhelp [command]

   This command returns a list of pftools commands. If a command is
   provided it gives a detailed description of the command and the
   necessary inputs.

   ::

      pfhhead phead

   This command computes the hydraulic head from the pressure head
   represented by the identifier ‘phead’. An identifier for the
   hydraulic head computed is returned upon successful completion.

   ::

      pfhydrostatic wtdepth top mask

   Compute hydrostatic pressure field from water table depth

   ::

      pflistdata dataset

   This command returns a list of pairs if no argument is given. The
   first item in each pair will be an identifier representing the data
   set and the second item will be that data set’s label. If a data
   set’s identifier is given as an argument, then just that data set’s
   name and label will be returned.

   ::

      pfload [file format] filename

   Loads a dataset into memory so it can be manipulated using the other
   utilities. A file format may proceed the filename in order to
   indicate the file’s format. If no file type option is given, then the
   extension of the filename is used to determine the default file type.
   An identifier used to represent the data set will be returned upon
   successful completion.

   File type options include:

   -  ::

         pfb

      ParFlow binary format. Default file type for files with a ‘.pfb’
      extension.

   -  ::

         pfsb

      ParFlow scattered binary format. Default file type for files with
      a ‘.pfsb’ extension.

   -  ::

         sa

      ParFlow simple ASCII format. Default file type for files with a
      ‘.sa’ extension.

   -  ::

         sb

      ParFlow simple binary format. Default file type for files with a
      ‘.sb’ extension.

   -  ::

         silo

      Silo binary format. Default file type for files with a ‘.silo’
      extension.

   -  ::

         rsa

      ParFlow real scattered ASCII format. Default file type for files
      with a ‘.rsa’ extension

   ::

      pfloadsds filename dsnum

   This command is used to load Scientific Data Sets from HDF files. The
   SDS number ‘dsnum’ will be used to find the SDS you wish to load from
   the HDF file ‘filename’. The data set loaded into memory will be
   assigned an identifier which will be used to refer to the data set
   until it is deleted. This identifier will be returned upon successful
   completion of the command.

   ::

      pfmdiff datasetp datasetq digits [zero]

   If ‘digits’ is greater than or equal to zero, then this command
   computes the grid point at which the number of digits in agreement
   (significant digits) is fewest and differs by more than ‘digits’
   significant digits. If ‘digits’ is less than zero, then the point at
   which the number of digits in agreement (significant digits) is
   minimum is computed. Finally, the maximum absolute difference is
   computed. The above information is returned in a Tcl list of the
   following form: mi mj mk sd adiff

   Given the search criteria, (mi, mj, mk) is the coordinate where the
   minimum number of significant digits ‘sd’ was found and ‘adiff’ is
   the maximum absolute difference.

   ::

      pfmovingaveragedem dem wsize maxiter 

   This command fills sinks in the digital elevation model dem by a
   standard iterative moving-average routine. Sinks are identified as
   cells with zero slope in both x- and y-directions, or as local minima
   in elevation (i.e., all adjacent cells have higher elevations). At
   each iteration, a moving average is taken over a window of width
   wsize around each remaining sink; sinks are thus filled by averaging
   over neighboring cells. The procedure continues iteratively until all
   sinks are filled or the number of iterations reaches maxiter. For
   most applications, sinks should be filled prior to computing slopes
   (i.e., prior to executing pfslopex and pfslopey).

   ::

      pfnewdata {nx ny nz} {x y z} {dx dy dz} label

   This command creates a new data set whose dimension is described by
   the lists nx ny nz, x y z, and dx dy dz. The first list, describes
   the dimensions, the second indicates the origin, and the third gives
   the length intervals between each coordinate along each axis. The
   ‘label’ argument will be the label of the data set that gets created.
   This new data set that is created will have all of its data points
   set to zero automatically. An identifier for the new data set will be
   returned upon successful completion.

   ::

      pfnewgrid {nx ny nz} {x y z} {dx dy dz} label

   Create a new data set whose grid is described by passing three lists
   and a label as arguments. The first list will be the number of
   coordinates in the x, y, and z directions. The second list will
   describe the origin. The third contains the intervals between
   coordinates along each axis. The identifier of the data set created
   by this operation is returned upon successful completion.

   ::

      pfnewlabel dataset newlabel

   This command changes the label of the data set ‘dataset’ to
   ‘newlabel’.

   ::

      pfphead hhead

   This command computes the pressure head from the hydraulic head
   represented by the identifier ‘hhead’. An identifier for the pressure
   head is returned upon successful completion.

   ::

      pfpatchysolid -top topdata -bot botdata -msk emaskdata [optional args] 

   Creates a solid file with complex upper and lower surfaces from a top
   surface elevation dataset (topdata), a bottom elevation dataset
   (botdata), and an enhanced mask dataset (emaskdata) all of which must
   be passed as handles to 2-d datasets that share a common size and
   origin. The solid is built as the volume between the top and bottom
   surfaces using the mask to deactivate other regions. The “enhanced
   mask" used here is a gridded dataset containing integers where all
   active cells have values of one but inactive cells may be given a
   positive integer value that identifies a patch along the model edge
   or the values may be zero. Any mask cell with value 0 is omitted from
   the active domain and *is not* written to a patch. If an active cell
   is adjacent to a non-zero mask cell, the face between the active and
   inactive cell is assigned to the patch with the integer value of the
   adjacent inactive cell. Bottom and Top patches are always written for
   every active cell and the West, East, South, and North edges are
   written automatically anytime active cells touch the edges of the
   input dataset(s). Up to 30 user defined patches can be specified
   using arbitrary integer values that are *greater than* 1. Note that
   the -msk flag may be omitted and doing so will make every cell
   active.

   The -top and -bot flags, and -msk if it is used, MUST each be
   followed by the handle for the relevant dataset. Optional argument
   flag-name pairs include:

   -  -pfsol <file name>.pfsol (or -pfsolb <file name>.pfsolb)

   -  -vtk <file name>.vtk

   -  -sub

   where <file name> is replaced by the desired text string. The -pfsolb
   option creates a compact binary solid file; pfsolb cannot currently
   be read directly by ParFlow but it can be converted with
   *pfsolidfmtconvert* and full support is under development. If -pfsol
   (or -pfsolb) is not specified the default name "SolidFile.pfsol" will
   be used. If -vtk is omitted, no vtk file will be created. The vtk
   attributes will contain mean patch elevations and patch IDs from the
   enhanced mask. Edge patch IDs are shown as negative values in the
   vtk. The patchysolid tool also outputs the list of the patch names in
   the order they are written, which can be directly copied into a
   ParFlow TCL script for the list of patch names. The -sub option
   writes separate patches for each face (left,right,front,back), which
   are indicated in the output patch write order list.

   Assuming $Msk, $Top, and $Bot are valid dataset handles from pfload,
   two valid examples are:

   .. container:: list

      ::

         pfpatchysolid -msk $Msk -top $Top -bot $Bot -pfsol "MySolid.pfsol" -vtk "MySolid.vtk"
         pfpatchysolid -bot $Bot -top $Top -vtk "MySolid.vtk" -sub

   Note that all flag-name pairs may be specified in any order for this
   tool as long as the required argument immediately follows the flag.
   To use with a terrain following grid, you will need to subtract the
   surface elevations from the top and bottom datasets (this makes the
   top flat) then add back in the total thickness of your grid, which
   can be done using “pfcelldiff" and “pfcellsumconst".

   ::

      pfpitfilldem dem dpit maxiter 

   This command fills sinks in the digital elevation model dem by a
   standard iterative pit-filling routine. Sinks are identified as cells
   with zero slope in both x- and y-directions, or as local minima in
   elevation (i.e., all adjacent neighbors have higher elevations). At
   each iteration, the value dpit is added to all remaining sinks. The
   procedure continues iteratively until all sinks are filled or the
   number of iterations reaches maxiter. For most applications, sinks
   should be filled prior to computing slopes (i.e., prior to executing
   pfslopex and pfslopey).

   ::

      pfprintdata dataset

   This command executes ‘pfgetgrid’ and ‘pfgetelt’ in order to display
   all the elements in the data set represented by the identifier
   ‘dataset’.

   ::

      pfprintdiff datasetp datasetq digits [zero]

   This command executes ‘pfdiffelt’ and ‘pfmdiff’ to print differences
   to standard output. The differences are printed one per line along
   with the coordinates where they occur. The last two lines displayed
   will show the point at which there is a minimum number of significant
   digits in the difference as well as the maximum absolute difference.

   ::

      pfprintdomain domain

   This command creates a set of TCL commands that setup a domain as
   specified by the provided domain input which can be then be written
   to a file for inclusion in a Parflow input script. Note that this
   kind of domain is only supported by the SAMRAI version of Parflow.

   ::

      pfprintelt i j k dataset

   This command prints a single element from the provided dataset given
   an i, j, k location.

   ::

      pfprintgrid dataset

   This command executes pfgetgrid and formats its output before
   printing it on the screen. The triples (nx, ny, nz), (x, y, z), and
   (dx, dy, dz) are all printed on separate lines along with labels
   describing each.

   ::

      pfprintlist [dataset]

   This command executes pflistdata and formats the output of that
   command. The formatted output is then printed on the screen. The
   output consists of a list of data sets and their labels one per line
   if no argument was given or just one data set if an identifier was
   given.

   ::

      pfprintmdiff datasetp datasetq digits [zero]

   This command executes ‘pfmdiff’ and formats that command’s output
   before displaying it on the screen. Given the search criteria, a line
   displaying the point at which the difference has the least number of
   significant digits will be displayed. Another line displaying the
   maximum absolute difference will also be displayed.

   ::

      printstats dataset

   This command executes ‘pfstats’ and formats that command’s output
   before printing it on the screen. Each of the values mentioned in the
   description of ‘pfstats’ will be displayed along with a label.

   ::

      pfreload dataset

   This argument reloads a dataset. Only one arguments is required, the
   name of the dataset to reload.

   ::

      pfreloadall

   This command reloads all of the current datasets.

   ::

      pfsattrans mask perm

   Compute saturated transmissivity for all [i,j] as the sum of the
   permeability[i,j,k]*dz within a column [i,j]. Currently this routine
   uses dz from the input permeability so the dz in permeability must be
   correct. Also, it is assumed that dz is constant, so this command is
   not compatible with variable dz.

   ::

      pfsave dataset -filetype filename

   This command is used to save the data set given by the identifier
   ‘dataset’ to a file ‘filename’ of type ‘filetype’ in one of the
   ParFlow formats below.

   File type options include:

   -  pfb ParFlow binary format.

   -  sa ParFlow simple ASCII format.

   -  sb ParFlow simple binary format.

   -  silo Silo binary format.

   -  vis Vizamrai binary format.

   ::

      pfsavediff datasetp datasetq digits [zero] -file filename

   This command saves to a file the differences between the values of
   the data sets represented by ‘datasetp’ and ‘datasetq’ to file
   ‘filename’. The data points whose values differ in more than ‘digits’
   significant digits and whose differences are greater than ‘zero’ will
   be saved. Also, given the above criteria, the minimum number of
   digits in agreement (significant digits) will be saved.

   If ‘digits’ is less than zero, then only the minimum number of
   significant digits and the coordinate where the minimum was computed
   will be saved.

   In each of the above cases, the maximum absolute difference given the
   criteria will also be saved.

   ::

      pfsavesds dataset -filetype filename

   This command is used to save the data set represented by the
   identifier ‘dataset’ to the file ‘filename’ in the format given by
   ‘filetype’.

   The possible HDF formats are:

   -  -float32

   -  -float64

   -  -int8

   -  -uint8

   -  -int16

   -  -uint16

   -  -int32

   -  -uint32

   ::

      pfsegmentD8 dem

   This command computes the distance between the cell centers of every
   parent cell [i,j] and its child cell. Child cells are determined
   using the eight-point pour method (commonly referred to as the D8
   method) based on the digital elevation model dem. If [i,j] is a local
   minima the segment length is set to zero.

   ::

      pfsetgrid {nx ny nz} {x0 y0 z0} {dx dy dz} dataset

   This command replaces the grid information of dataset with the values
   provided.

   ::

      pfslopeD8 dem

   This command computes slopes according to the eight-point pour method
   (commonly referred to as the D8 method) based on the digital
   elevation model dem. Slopes are computed as the maximum downward
   gradient between a given cell and it’s lowest neighbor (adjacent or
   diagonal). Local minima are set to zero; where local minima occur on
   the edge of the domain, the 1st order upwind slope is used (i.e., the
   cell is assumed to drain out of the domain). Note that dem must be a
   ParFlow dataset and must have the correct grid information – dx and
   dy both used in slope calculations. If gridded elevation values are
   read in from a text file (e.g., using pfload’s simple ascii format),
   grid information must be specified using the pfsetgrid command. It
   should be noted that ParFlow uses slopex and slopey (NOT D8 slopes!)
   in runoff calculations.

   ::

      pfslopex dem

   This command computes slopes in the x-direction using 1st order
   upwind finite differences based on the digital elevation model dem.
   Slopes at local maxima (in x-direction) are calculated as the maximum
   downward gradient to an adjacent neighbor. Slopes at local minima (in
   x-direction) do not drain in the x-direction and are therefore set to
   zero. Note that dem must be a ParFlow dataset and must have the
   correct grid information – dx in particular is used in slope
   calculations. If gridded elevation values are read from a text file
   (e.g., using pfload’s simple ascii format), grid information must be
   specified using the pfsetgrid command.

   ::

      pfslopexD4 dem

   This command computes the slope in the x-direction for all [i,j]
   using a four point (D4) method. The slope is set to the maximum
   downward slope to the lowest adjacent neighbor. If [i,j] is a local
   minima the slope is set to zero (i.e. no drainage).

   ::

      pfslopey dem

   This command computes slopes in the y-direction using 1st order
   upwind finite differences based on the digital elevation model dem.
   Slopes at local maxima (in y-direction) are calculated as the maximum
   downward gradient to an adjacent neighbor. Slopes at local minima (in
   y-direction) do not drain in the y-direction and are therefore set to
   zero. Note that dem must be a ParFlow dataset and must have the
   correct grid information - dy in particular is used in slope
   calculations. If gridded elevation values are read in from a text
   file (e.g., using pfload’s simple ascii format), grid information
   must be specified using the pfsetgrid command.

   ::

      pfslopeyD4 dem

   This command computes the slope in the y-direction for all [i,j]
   using a four point (D4) method. The slope is set to the maximum
   downward slope to the lowest adjacent neighbor. If [i,j] is a local
   minima the slope is set to zero (i.e. no drainage).

   ::

      pfsolidfmtconvert filename1 filename2 

   This command converts solid files back and forth between the ascii
   .pfsol format and the binary .pfsolb format. The tool automatically
   detects the conversion mode based on the extensions of the input file
   names. The *filename1* is the name of source file and *filename2* is
   the target output file to be created or overwritten. Support to
   directly use a binary solid (.pfsolb) is under development but this
   allows a significant reduction in file sizes.

   For example, to convert from ascii to binary, then back to ascii:

   .. container:: list

      ::

         pfsolidfmtconvert "MySolid.pfsol" "MySolid.pfsolb"
         pfsolidfmtconvert "MySolid.pfsolb" "NewSolid.pfsol"

   ::

      pfstats dataset

   This command prints various statistics for the data set represented
   by the identifier ‘dataset’. The minimum, maximum, mean, sum,
   variance, and standard deviation are all computed. The above values
   are returned in a list of the following form: min max mean sum
   variance (standard deviation)

   ::

      pfsubsurfacestorage mask porosity pressure saturation specific_storage

   This command computes the sub-surface water storage (compressible and
   incompressible components) based on mask, porosity, saturation,
   storativity and pressure fields. The equations used to calculate this
   quantity are given in §5.9 :ref:`Water Balance`. The identifier of
   the data set created by this operation is returned upon successful
   completion.

   ::

      pfsum dataset

   This command computes the sum over the domain of the dataset.

   ::

      pfsurfacerunoff top slope_x slope_y  mannings pressure

   This command computes the surface water runoff (out of the domain)
   based on a computed top, pressure field, slopes and mannings
   roughness values. This is integrated along all domain boundaries and
   is calculated at any location that slopes at the edge of the domain
   point outward. This data is in units of :math:`[L^3 T^{-1}]` and the
   equations used to calculate this quantity are given in
   §5.9 :ref:`Water Balance`. The identifier of the data set created by
   this operation is returned upon successful completion.

   ::

      pfsurfacestorage top pressure

   This command computes the surface water storage (ponded water on top
   of the domain) based on a computed top and pressure field. The
   equations used to calculate this quantity are given in
   §5.9 :ref:`Water Balance`. The identifier of the data set created by
   this operation is returned upon successful completion.

   ::

      pftopodeficit profile m trans dem slopex slopey recharge ssat sres porosity mask

   Compute water deficit for all [i,j] based on TOPMODEL/topographic
   index. For more details on methods and assumptions refer to
   toposlopes.c in pftools.

   ::

      pftopoindex dem sx sy

   Compute topographic index for all [i,j]. Here topographic index is
   defined as the total upstream area divided by the contour length,
   divided by the local slope. For more details on methods and
   assumptions refer to toposlopes.c in pftools.

   ::

      pftoporecharge riverfile nriver  trans dem sx sy

   Compute effective recharge at all [i,j] over upstream area based on
   topmodel assumptions and given list of river points. Notes: See
   detailed notes in toposlopes.c regarding assumptions, methods, etc.
   Input Notes: nriver is an integer (number of river points) river is
   an array of integers [nriver][2] (list of river indices, ordered from
   outlet to headwaters) is a Databox of saturated transmissivity dem is
   a Databox of elevations at each cell sx is a Databox of slopes
   (x-dir) – lets you use processed slopes! sy is a Databox of slopes
   (y-dir) – lets you use processed slopes!

   ::

      pftopowt deficit porosity ssat sres mask top wtdepth

   Compute water depth from column water deficit for all [i,j] based on
   TOPMODEL/topographic index.

   ::

      pfundist filename, pfundist runname

   The command undistributes a ParFlow output file. ParFlow uses a
   distributed file system where each node can write to its own file.
   The pfundist command takes all of these individual files and
   collapses them into a single file.

   The arguments can be a runname or a filename. If a runname is given
   then all of the output files associated with that run are
   undistributed.

   Normally this is done after every pfrun command.

   ::

      pfupstreamarea slope_x slope_y

   This command computes the upstream area contributing to surface
   runoff at each cell based on the x and y slope values provided in
   datasets ``slope_x`` and ``slope_y``, respectively. Contributing 
   area is computed recursively for each cell; areas are not weighted 
   by slope direction. Areas are returned as the number of upstream 
   (contributing) cells; to compute actual area, simply multiply by 
   the cell area (dx*dy).

   ::

      pfvmag datasetx datasety datasetz

   This command computes the velocity magnitude when given three
   velocity components. The three parameters are identifiers which
   represent the x, y, and z components respectively. The identifier of
   the data set created by this operation is returned upon successful
   completion.

   ::

      pfvtksave dataset filetype filename [options]

   This command loads PFB or SILO output, reads a DEM from a file and
   generates a 3D VTK output field from that ParFlow output.

   The options: Any combination of these can be used and they can be
   specified in any order as long as the required elements immediately
   follow each option.

   -var specifies what the variable written to the dataset will be
   called. This is followed by a text string, like "Pressure" or
   "Saturation" to define the name of the data that will be written to
   the VTK. If this isn’t specified, you’ll get a property written to
   the file creatively called "Variable". This option is ignored if you
   are using -clmvtk since all its variables are predefined.

   -dem specifies that a DEM is to be used. The argument following -dem
   MUST be the handle of the dataset containing the elevations. If it
   cannot be found, the tool ignores it and reverts to non-dem mode. If
   the nx and ny dimensions of the grids don’t match, the tool will
   error out. This option shifts the layers so that the top of the
   domain coincides with the land surface defined by the DEM. Regardless
   of the actual number of layers in the DEM file, the tool only uses
   the elevations in the top layer of this dataset, meaning a 1-layer
   PFB can be used.

   -flt tells the tool to write the data as type float instead of
   double. Since the VTKs are really only used for visualization, this
   reduces the file size and speeds up plotting.

   -tfg causes the tool to override the specified dz in the dataset PFB
   and uses a user specified list of layer thicknesses instead. This is
   designed for terrain following grids and can only be used in
   conjunction with a DEM. The argument following the flag is a text
   string containing the number of layers and the dz list of actual
   layer thicknesses (not dz multipliers) for each layer from the bottom
   up such as: -tfg "5 200.0 1.0 0.7 0.2 0.1" Note that the quotation
   marks around the list are necessary.

   Example:

   .. container:: list

      ::

         file copy -force CLM_dem.cpfb CLM_dem.pfb

         set CLMdat [pfload -pfb clm.out.clm_output.00005.C.pfb]
         set Pdat [pfload -pfb clm.out.press.00005.pfb]
         set Perm [pfload -pfb clm.out.perm_x.pfb]
         set DEMdat [pfload -pfb CLM_dem.pfb]

         set dzlist "10 6.0 5.0 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5"

         pfvtksave $Pdat -vtk "CLM.out.Press.00005a.vtk" -var "Press"
         pfvtksave $Pdat -vtk "CLM.out.Press.00005b.vtk" -var "Press" -flt
         pfvtksave $Pdat -vtk "CLM.out.Press.00005c.vtk" -var "Press" -dem $DEMdat
         pfvtksave $Pdat -vtk "CLM.out.Press.00005d.vtk" -var "Press" -dem $DEMdat -flt
         pfvtksave $Pdat -vtk "CLM.out.Press.00005e.vtk" -var "Press" -dem $DEMdat -flt -tfg $dzlist
         pfvtksave $Perm -vtk "CLM.out.Perm.00005.vtk" -var "Perm" -flt -dem $DEMdat -tfg $dzlist

         pfvtksave $CLMdat -clmvtk "CLM.out.CLM.00005.vtk" -flt
         pfvtksave $CLMdat -clmvtk "CLM.out.CLM.00005.vtk" -flt -dem $DEMdat

         pfvtksave $DEMdat -vtk "CLM.out.Elev.00000.vtk" -flt -var "Elevation" -dem $DEMdat

   ::

      pfvvel conductivity phead

   This command computes the Darcy velocity in cells for the
   conductivity data set represented by the identifier ‘conductivity’
   and the pressure head data set represented by the identifier ‘phead’.
   The identifier of the data set created by this operation is returned
   upon successful completion.

   ::

      pfwatertabledepth top saturation 

   This command computes the water table depth (distance from top to
   first cell with saturation = 1). The identifier of the data set
   created by this operation is returned upon successful completion.

   ::

      pfwritedb runname

   This command writes the settings of parflow run to a pfidb database
   that can be used to run the model at a later time. In general this
   command is used in lieu of the pfrun command.

.. _common_pftcl:

Common examples using ParFlow TCL commands (PFTCL) 
--------------------------------------------------

This section contains some brief examples of how to use the pftools
commands (along with standard *TCL* commands) to postprocess data.

.. container:: enumerate

   1. Load a file as one format and write as another format.

   .. container:: list

      ::

         set press [pfload harvey_flow.out.press.pfb]
         pfsave $press -sa harvey_flow.out.sa

         #####################################################################
         # Also note that PFTCL automatically assigns
         #identifiers to each data set it stores. In this
         # example we load the pressure file and assign
         #it the identifier press. However if you
         #read in a file called foo.pfb into a TCL shell
         #with assigning your own identifier, you get
         #the following:

         #parflow> pfload foo.pfb
         #dataset0

         # In this example, the first line is typed in by the
         #user and the second line is printed out
         #by PFTCL. It indicates that the data read
         #from file foo.pfb is associated with the
         #identifier dataset0.

   2. Load pressure-head output from a file, convert to head-potential and write out as a new file.

   .. container:: list

      ::

         set press [pfload harvey_flow.out.press.pfb]
         set head [pfhhead $press]
         pfsave $head -pfb harvey_flow.head.pfb

   3. Build a SAMARI compatible domain decomposition based off of a mask file.

   .. container:: list

      ::

         #---------------------------------------------------------
         # This example script takes 3 command line arguments
         # for P,Q,R and then builds a SAMRAI compatible
         # domain decomposition based off of a mask file.
         #---------------------------------------------------------

         # Processor Topology
         set P [lindex $argv 0]
         set Q [lindex $argv 1]
         set R [lindex $argv 2]
         pfset Process.Topology.P $P
         pfset Process.Topology.Q $Q
         pfset Process.Topology.R $R

         # Computational Grid
         pfset ComputationalGrid.Lower.X -10.0
         pfset ComputationalGrid.Lower.Y 10.0
         pfset ComputationalGrid.Lower.Z 1.0

         pfset ComputationalGrid.DX 8.8888888888888893
         pfset ComputationalGrid.DY 10.666666666666666
         pfset ComputationalGrid.DZ 1.0

         pfset ComputationalGrid.NX 10
         pfset ComputationalGrid.NY 10
         pfset ComputationalGrid.NZ 8

         # Calculate top and bottom and build domain
         set mask [pfload samrai.out.mask.pfb]
         set top [pfcomputetop $mask]
         set bottom [pfcomputebottom $mask]

         set domain [pfcomputedomain $top $bottom]
         set out [pfprintdomain $domain]
         set grid\_file [open samrai_grid.tcl w]

         puts $grid_file $out
         close $grid_file

         #---------------------------------------------------------
         # The resulting TCL file samrai_grid.tcl may be read into
         # a Parflow input file using ¿¿source samrai_grid.tcl¿¿.
         #---------------------------------------------------------

   4. Distributing input files before running [dist example]

   .. container:: list

      ::

         #--------------------------------------------------------
         # A common problem for new ParFlow users is to
         # distribute slope files using
         # the 3-D computational grid that is
         # set at the begging of a run script.
         # This results in errors because slope
         # files are 2-D.
         # To avoid this problem the computational
         # grid should be reset before and after
         # distributing slope files. As follows:
         #---------------------------------------------------------

         #First set NZ to 1 and distribute the 2D slope files
         pfset ComputationalGrid.NX                40
         pfset ComputationalGrid.NY                40
         pfset ComputationalGrid.NZ                1
         pfdist slopex.pfb
         pfdist slopey.pfb

         #Reset NZ to the correct value and distribute any 3D inputs
         pfset ComputationalGrid.NX                40
         pfset ComputationalGrid.NY                40
         pfset ComputationalGrid.NZ                50
         pfdist IndicatorFile.pfb

   5. Calculate slopes from an elevation file

   .. container:: list

      ::

         #Read in DEM
         set dem [pfload -sa dem.txt]
         pfsetgrid {209 268 1} {0.0 0.0 0.0} {100 100 1.0} $dem

         # Fill flat areas (if any)
         set flatfill [pffillflats $dem]

         # Fill pits (if any)
         set  pitfill [pfpitfilldem $flatfill 0.01 10000]

         # Calculate Slopes
         set  slope_x [pfslopex $pitfill]
         set  slope_y [pfslopey $pitfill]

         # Write to output...
         pfsave $flatfill -silo klam.flatfill.silo
         pfsave $pitfill  -silo klam.pitfill.silo
         pfsave $slope_x  -pfb  klam.slope_x.pfb
         pfsave $slope_y  -pfb  klam.slope_y.pfb

   6. Calculate and output the *subsurface storage* in the domain at a point in time.

   .. container:: list

      ::

         set saturation [pfload runname.out.satur.00001.silo]
         set pressure [pfload runname.out.press.00001.silo]
         set specific_storage [pfload runname.out.specific_storage.silo]
         set porosity [pfload runname.out.porosity.silo]
         set mask [pfload runname.out.mask.silo]

         set subsurface_storage [pfsubsurfacestorage $mask $porosity \
         $pressure $saturation $specific_storage]
         set total_subsurface_storage [pfsum $subsurface_storage]
         puts [format "Subsurface storage\t\t\t\t : %.16e" $total_subsurface_storage]

   7. Calculate and output the *surface storage* in the domain at a point in time.

   .. container:: list

      ::

         set pressure [pfload runname.out.press.00001.silo]
         set mask [pfload runname.out.mask.silo]
         set top [pfcomputetop $mask]
         set surface_storage [pfsurfacestorage $top $pressure]
         set total_surface_storage [pfsum $surface_storage]
         puts [format "Surface storage\t\t\t\t : %.16e" $total_surface_storage]

   8. Calculate and output the runoff out of the *entire domain* over a timestep.

   .. container:: list

      ::

         set pressure [pfload runname.out.press.00001.silo]
         set slope_x [pfload runname.out.slope_x.silo]
         set slope_y [pfload runname.out.slope_y.silo]
         set mannings [pfload runname.out.mannings.silo]
         set mask [pfload runname.out.mask.silo]
         set top [pfcomputetop $mask]

         set surface_runoff [pfsurfacerunoff $top $slope_x $slope_y $mannings $pressure]
         set total_surface_runoff [expr [pfsum $surface_runoff] * [pfget TimeStep.Value]]
         puts [format "Surface runoff from pftools\t\t\t : %.16e" $total_surface_runoff]

   9. Calculate overland flow at a point using *Manning’s* equation

   .. container:: list

      ::

         #Set the location
         set Xloc 2
         set Yloc 2
         set Zloc 50  #This should be a z location on the surface of your domain

         #Set the grid dimension and Mannings roughness coefficient
         set dx  1000.0
         set n   0.000005

         #Get the slope at the point
         set slopex   [pfload runname.out.slope_x.pfb]
         set slopey   [pfload runname.out.slope_y.pfb]
         set sx1 [pfgetelt $slopex $Xloc $Yloc 0]
         set sy1 [pfgetelt $slopey $Xloc $Yloc 0]
         set S [expr ($sx**2+$sy**2)**0.5]

         #Get the pressure at the point
         set press [pfload runname.out.press.00001.pfb]
         set P [pfgetelt $press $Xloc $Yloc $Zloc]

         #If the pressure is less than zero set to zero
         if {$P < 0} { set P 0 }
         set QT [expr ($dx/$n)*($S**0.5)*($P**(5./3.))]
         puts $QT