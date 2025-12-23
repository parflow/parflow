.. _The ParFlow System:

The ParFlow System
==================

The ParFlow system is still evolving, but here we discuss how to define
the problem in :ref:`Defining the Problem`, how to run ParFlow in 
:ref:`Running Parflow`, and restart a simulation in :ref:`Restarting a Run`. We also cover options for visualizing the
results in :ref:`Visualizing Output` and summarize the contents of
a directory of test problems provided with ParFlow in :ref:`Test Directory`. Finally in :ref:`Tutorial` we walk
through two ParFlow input scripts in detail.

The reader is also referred to :ref:`Manipulating Data` for a
detailed listing of the of functions for manipulating ParFlow data.

.. _Defining the Problem:

Defining the Problem
--------------------

There are many ways to define a problem in ParFlow, here we summarize
the general approach for defining a domain
(:ref:`Defining a domain`) and simulating a real watershed
(:ref:`Defining a Real domain`).

The “main" ParFlow input file is one of the following: a ``.tcl`` TCL script, a ``.py`` Python script, or a ``.ipynb`` JuPyter notebook. 
This input script or notebook is used some special routines in PFTools to create 
a database which is used as the input for ParFlow.  This database has the extension ``.pfidb`` and is the database of keys that ParFlow needs to define a run. 
See :ref:`Main Input Files (.tcl, .py, .ipynb)` and :ref:`ParFlow Input Keys` for details on the format of 
these files. The input values into ParFlow 
are defined by a key/value pair and are listed in :ref:`ParFlow Input Keys`. For each key you provide the 
associated value using either the ``pfset`` command in TCL or associated it with a named run (we use *<runname>* in this manual to denote that) inside the input script.

Since the input file is a script or notebook you can use any feature of TCL or Python to
define the problem and to postprocess your run. This manual will make no effort to teach TCL or Python so
refer to one of the available manuals or the wealth of online content for more information. This is NOT
required, you can get along fine without understanding TCL or Python.

Looking at the example programs in the :ref:`Test Directory` and 
going through the annotated input scripts included in this 
manual (:ref:`Tutorial`) is one of the best ways to understand 
what a ParFlow input file looks like.

.. _Defining a domain:

Basic Domain Definition
~~~~~~~~~~~~~~~~~~~~~~~

ParFlow can handle complex geometries and defining the problem may
involve several steps. Users can specify simple box domains directly in
the input script. If a more complicated domain is required, the 
user may convert geometries into the ``.pfsol`` file format
(:ref:`ParFlow Solid Files (.pfsol)`) using the appropriate 
PFTools conversion utility (:ref:`Manipulating Data`). 
Alternatively, the topography can be specified using ``.pfb`` 
files of the slopes in the x and y directions.

Regardless of the approach the user must set the computational grid
within the ``.pfb`` script as follows:

.. container:: list

   ::

      #-----------------------------------------------------------------------------
      # Computational Grid
      #-----------------------------------------------------------------------------
      pfset ComputationalGrid.Lower.X                -10.0
      pfset ComputationalGrid.Lower.Y                 10.0
      pfset ComputationalGrid.Lower.Z                  1.0
       
      pfset ComputationalGrid.DX                       8.89
      pfset ComputationalGrid.DY                      10.67
      pfset ComputationalGrid.DZ                       1.0
       
      pfset ComputationalGrid.NX                      18
      pfset ComputationalGrid.NY                      15
      pfset ComputationalGrid.NZ                       8

The value is normally a single string, double, or integer. In some
cases, in particular for a list of names, you need to supply a space
separated sequence. This can be done using either a double quote or
braces.

.. container:: list

   ::

      pfset Geom.domain.Patches "left right front back bottom top"

      pfset Geom.domain.Patches {left right front back bottom top}

For commands longer than a single line, the TCL continuation character
can be used,

.. container:: list

   ::

      pfset Geom.domain.Patches "very_long_name_1 very_long_name_2 very_long_name_3 \
                                 very_long_name_4 very_long_name_5 very_long_name_6"

.. _Defining a Real domain:

Setting Up a Real Domain
~~~~~~~~~~~~~~~~~~~~~~~~

This section provides a brief outline of a sample workflow for setup
ParFlow ``CLM`` simulation of a real domain. Of course there are 
many ways to accomplish this and users are encouraged to develop 
a workflow that works for them.

This example assumes that you are running with ParFlow ``CLM`` and 
it uses slope files and an indicator file to define the topography 
and geologic units of the domain. An alternate approach would be 
to define geometries by building a ``.pfsol`` file (:ref:`ParFlow Solid Files (.pfsol)`) 
using the appropriate PFTools conversion utility (:ref:`Manipulating Data`).

The general approach is as follows:

.. container:: enumerate

   1. Gather input datasets to define the domain. First decide the
   resolution that you would like to simulate at. Then gather the
   following datasets at the appropriate resolution for your domain:

      a. Elevation (DEM)

      b. Soil data for the near surface layers

      c. Geologic maps for the deeper subsurface

      d. Land Cover

   2. Create consistent gridded layers that are all clipped to your domain
   and have the same number of grid cells

   3. Convert gridded files to ``.pfb`` (:ref:`ParFlow Binary Files (.pfb)`). 
   One way to accomplish this is by reformatting the gridded outputs to the 
   correct ParFlow ``.sa`` order (:ref:`ParFlow Simple ASCII Files (.sa and .sb)`) 
   and to convert the ``.sa`` file to ``.pfb`` using the conversion tools 
   (see :ref:`common_pftcl` Example 1). If you have an elevation file 
   in ``.pfb`` format, you may wish to preserve it as provenance for the slopes
   and for use in post-processing tools. You may point ParFlow to the elevation:

   .. container:: list

      ::

            pfset TopoSlopes.Elevation.FileName "elevation.pfb"

   4. Calculate slopes in the x and y directions from the elevation
   dataset. This can be done with the built in tools as shown in
   :ref:`common_pftcl` Example 5. In most cases some additional
   processing of the DEM will be required to ensure that the drainage
   patterns are correct. To check this you can run a “parking lot test"
   by setting the permeability of surface to almost zero and adding a
   flux to the top surface. If the results from this test don’t look
   right (i.e. your runoff patterns don’t match what you expect) you
   will need to go back and modify your DEM.

   5. Create an indicator file for the subsurface. The indicator file is a
   3D ``.pfb`` file with the same dimensions as your domain that has 
   an integer for every cell designating which unit it belongs to. 
   The units you define will correspond to the soil types and geologic 
   units from your input datasets.

   6. Determine the hydrologic properties for each of the subsurface units
   defined in the indicator file. You will need: Permeability, specific
   storage, porosity and van Genuchten parameters.

   7. At this point you are ready to run a ParFlow model without ``CLM`` and 
   if you don’t need to include the land surface model in your simulations 
   you can ignore the following steps. Either way, at this point it is 
   advisable to run a “spinup" simulation to initialize the water table. 
   There are several ways to approach this. One way is to start with the 
   water table at a constant depth and run for a long time with a constant 
   recharge forcing until the water table reaches a steady state. 
   There are some additional key for spinup runs that are provided 
   in :ref:`Spinup Options`.

   8. Convert land cover classifications to the IGBP [1]_ land cover
   classes that are used in CLM.

   -  1. Evergreen Needleleaf Forest

   -  2. Evergreen Broadleaf Forest

   -  3. Deciduous Needleleaf Forest

   -  4. Deciduous Broadleaf Forest

   -  5. Mixed Forests

   -  6. Closed Shrublands

   -  7. Open Shrublands

   -  8. Woody Savannas

   -  9. Savannas

   -  10. Grasslands

   -  11. Permanent Wetlands

   -  12. Croplands

   -  13. Urban and Built-Up

   -  14. Cropland/Natural Vegetation Mosaic

   -  15. Snow and Ice

   -  16. Barren or Sparsely Vegetated

   -  17. Water

   -  18. Wooded Tundra

   9. Create a ``CLM`` vegm file that designates the land cover fractions 
   for every cell (Refer to the ``clm input`` directory in the Washita 
   Example for an sample of what a ``vegm`` file should look like).

   10. Create a ``CLM`` driver file to set the parameters for the ``CLM`` 
   model (Refer to the ``clm input`` directory in the Washita Example 
   for a sample of a ``CLM`` driver file).

   11. Assemble meteorological forcing data for your domain. CLM uses
   Greenwich Mean Time (GMT), not local time. The year, date and hour
   (in GMT) that the forcing begins should match the values 
   in ``drv_clmin.dat``. ``CLM`` requires the following variables
   (also described in :ref:`Main Input Files (.tcl, .py, .ipynb)`):

   -  DSWR: Visible or short-wave radiation :math:`[W/m^2]`.

   -  DLWR: Long wave radiation :math:`[W/m^2]`

   -  APCP: Precipitation :math:`[mm/s]`

   -  Temp: Air Temperature :math:`[K]`

   -  UGRD: East-west wind speed :math:`[m/s]`

   -  VGRD: South-to-North wind speed :math:`[m/s]`

   -  Press: Atmospheric pressure :math:`[pa]`

   -  SPFH: Specific humidity :math:`[kg/kg]`

   If you choose to do spatially heterogeneous forcings you will need to
   generate separate files for each variable. The files should be
   formatted in the standard ParFlow format with the third (i.e. z
   dimension) as time. If you are doing hourly simulations it is
   standard practice to put 24 hours in one file, but you can decide how
   many time steps per file. For an example of heterogeneous forcing
   files refer to the ``NLDAS`` directory in the Washita Example).

   Alternatively, if you would like to force the model with spatially
   homogeneous forcings, then a single file can be provided where each
   variable is a column and rows designate time steps.

   12. Run your simulation!

.. [1]
    http://www.igbp.net

.. _Running ParFlow:

Running ParFlow
---------------

Once the problem input is defined, you need to add a few things to the
script to make it execute ParFlow. First you need to add the TCL or Python
commands to load the ParFlow command package.  We will cover TCL first, then Python below.

**TCL**

To set up and run ParFlow using PFTools in TCL, you need the following header lines.

.. container:: list

   ::

      #
      # Import the ParFlow TCL package
      #
      lappend auto_path $env(PARFLOW_DIR)/bin 
      package require parflow
      namespace import Parflow::*

This loads the ``pfset`` and other ParFlow commands into the TCL shell.

Since this is a script you need to actually run ParFlow. These are
normally the last lines of the input script.

.. container:: list

   ::

      #-----------------------------------------------------------------------------
      # Run and Unload the ParFlow output files
      #-----------------------------------------------------------------------------
      pfrun default_single
      pfundist default_single

The ``pfrun`` command runs ParFlow with the database as it exists 
at that point in the file. The argument is the name to give to the 
output files (which will normally be the same as the name of the 
script). Advanced users can set up multiple problems within the 
input script by using different output names.

The ``pfundist`` command takes the output files from the ParFlow 
run and undistributes them. ParFlow uses a virtual file system 
which allows files to be distributed across the processors. 
The ``pfundist`` takes these files and collapses them into a 
single file. On some machines if you don’t do the ``pfundist`` 
you will see many files after the run. Each of these contains 
the output from a single node; before attempting using them 
you should undistribute them.

Since the input file is a TCL script run it using the TCL shell or command interpreter:

.. container:: list

   ::

      tclsh runname.tcl

NOTE: Make sure you are using TCL 8.0 or later. The script will not work
with earlier releases.

**Python**

To run ParFlow via Python in either a Notebook or script you need to install PFTools. This makes the Python commands 
available within your environment.  To do this you can either
build ParFlow to include the building of PFTools in Python, or you can install the package
from PyPi.  This might look like:

.. container:: list

   ::

      pip install pftools 

At a minimum you need to import the ParFlow Python package and name your run.  There are a lot more tools
that bring substantial functionality that are discussed in other sections of this manual.

.. container:: list

   ::

      from parflow import Run
      from parflow.tools.fs import mkdir, get_absolute_path

      dsingle = Run("dsingle", __file__)
      #-----------------------------------------------------------------------------
      dsingle.FileVersion = 4

Then to build the key database and execute ParFlow you use the run command built into the Python PFTools structure. 

.. container:: list

   ::

      dsingle.run()

From the command line you would execute your Python script using the command interpreter.

.. container:: list

   ::

      python default_single.py 

A lot more detail, including several tutorials and examples, are given in the :ref:`Python` section of this manual.


One output file of particular interest is the ``<run name>.out.log`` file. 
This file contains information about the run such as number of 
processes used, convergence history of algorithms, timings and 
MFLOP rates. For Richards’ equation problems (including overland 
flow) the ``<run name>.out.kinsol.log`` file contains the nonlinear 
convergence information for each timestep. Additionally, 
the ``<run name>.out.txt`` contains all information routed 
to ``standard out`` of the machine you are running on and 
often contains error messages and other control information.

.. _Restarting a Run:

Restarting a Run
----------------

A ParFlow run may need to be restarted because either a system time
limit has been reached, ParFlow has been prematurely terminated or the
user specifically sets up a problem to run in segments. In order to
restart a run the user needs to know the conditions under which ParFlow
stopped. If ParFlow was prematurely terminated then the user must
examine the output files from the last “timed dump" to see if they are
complete. If not then those data files should be discarded and the
output files from the next to last “timed dump" will be used in the
restarting procedure. As an important note, if any set of “timed dump"
files are deleted remember to also delete corresponding lines in the
well output file or recombining the well output files from the
individual segments afterwards will be difficult. It is not necessary to
delete lines from the log file as you will only be noting information
from it. To summarize, make sure all the important output data files are
complete, accurate and consistent with each other.

Given a set of complete, consistent output files - to restart a run
follow this procedure :

#. Note the important information for restarting :

   -  Write down the dump sequence number for the last collection of
      “timed dump” data.

   -  Examine the log file to find out what real time that “timed dump"
      data was written out at and write it down.

#. Prepare input data files from output data files :

   -  Take the last pressure output file before the restart with the
      sequence number from above and format them for regular input using
      the keys detailed in 6.1.27 :ref:`Initial Conditions: Pressure`
      and possibly the ``pfdist`` utility in the input script.

#. Change the Main Input File 6.1 :ref:`Main Input Files (.tcl, .py, .ipynb)`:

   -  Edit the .tcl file (you may want to save the old one) and utilize
      the pressure initial condition input file option (as referenced
      above) to specify the input files you created above as initial
      conditions for concentrations.

#. Restart the run :

   -  Utilizing an editor recreate all the input parameters used in the
      run except for the following two items :

      -  Use the dump sequence number from step 1 as the start_count.

      -  Use the real time that the dump occurred at from step 1 as the
         start_time.

      -  To restart with ``CLM``, use the ``Solver.CLM.IstepStart`` 
         key described in :ref:`CLM Solver Parameters` with a 
         value equal to the dump sequence plus one. Make sure this 
         corresponds to changes to ``drv_clmin.dat``.

   

.. _Visualizing Output:

Visualizing Output
------------------

While ParFlow does not have any visualization capabilities built-in,
there are a number flexible, free options. Probably the best option is
to use *VisIt*. *VisIt* is a powerful, free, open-source, rendering
environment. It is multiplatform and may be downloaded directly 
from: `https://visit.llnl.gov/ <https://visit.llnl.gov/>`_. The most flexible 
option for using VisIt to view ParFlow output is to write files using 
the SILO format, which is available either as a direct output option 
(described in :ref:`Code Parameters`) or a conversion option 
using pftools. Many other output conversion options exist as described 
in :ref:`Manipulating Data` and this allows ParFlow output to 
be converted into formats used by almost all visualization software.

.. _Test Directory:

Directory of Test Cases
-----------------------

ParFlow comes with a directory containing a few simple input files for
use as templates in making new files and for use in testing the code.
These files sit in the ``/test`` directory described earlier. 
This section gives a brief description of the problems in this directory.

.. container:: description

   ``crater2D.tcl`` An example of a two-dimensional, variably-saturated 
   crater infiltration problem with time-varying boundary conditions. 
   It uses the solid file ``crater2D.pfsol``.

   ``default_richards.tcl`` The default variably-saturated Richards’ 
   Equation simulation test script.

   ``default_single.tcl`` The default parflow, single-processor, 
   fully-saturated test script.

   ``forsyth2.tcl`` An example two-dimensional, variably-saturated 
   infiltration problem with layers of different hydraulic properties. 
   It runs problem 2 in :cite:t:`FWP95` and uses the solid file ``fors2_hf.pfsol``.

   ``harvey.flow.tcl`` An example from :cite:t:`MWH07` for the Cape Cod bacterial 
   injection site. This example is a three-dimensional, fully-saturated 
   flow problem with spatially heterogeneous media (using a correlated, 
   random field approach). It also provides examples of how tcl/tk 
   scripts may be used in conjunction with ParFlow to loop iteratively 
   or to run other scripts or programs. It uses the input text 
   file ``stats4.txt``. This input script is fully detailed in :ref:`Tutorial`.

   ``default_overland.tcl`` An overland flow boundary condition 
   test and example script based loosely on the V-catchment 
   problem in :cite:t:`KM06`. There are options provided to expand this problem 
   into other overland flow-type, transient boundary-type problems 
   included in the file as well.

   ``LW_var_dz_spinup.tcl`` An example that uses the Little Washita 
   domain to demonstrate a steady-state spinup initialization using 
   P-E forcing. It also demonstrates the variable dz keys.

   ``LW_var_dz.tcl`` An example that uses the Little Washita domain 
   to demonstrate surface flow network development. It also uses the 
   variable dz keys.

   ``Evap_Trans_test.tcl`` An example that modifies the ``default_overland.tcl`` 
   to demonstrate steady-state external flux ``.pfb`` files.

   ``overland_flux.tcl`` An example that modifies the ``default_overland.tcl`` 
   to demonstrate transient external flux ``.pfb`` files.

   ``/clm/clm.tcl`` An example of how to use ParFlow coupled 
   to ``clm``. This directory also includes ``clm``-specific input. 
   Note: this problem will only run if ``–with-clm`` flag is used 
   during the configure and build process.

   ``water_balance_x.tcl`` and ``water_balance_y.tcl``. An overland 
   flow example script that uses the water-balance routines integrated 
   into ``pftools``. These two problems are based on simple overland 
   flow conditions with slopes primarily in the x or y-directions. 
   Note: this problem only will run if the Silo file capability 
   is used, that is a ``–with-silo=PATH`` flag is used during the 
   configure and build process.

   ``pfmg.tcl`` and ``pfmg_octree.tcl`` Tests of the external 
   Hypre preconditioner options. Note: this problem only will 
   run if the Hypre capability is used, that is a ``–with-hypre=PATH`` 
   flag is used during the configure and build process.

   ``test_x.tcl`` A test problem for the Richards’ solver that 
   compares output to an analytical solution.

   ``/washita/tcl_scripts/LW_Test.tcl`` A three day simulation 
   of the Little Washita domain using ParFlow ``CLM`` with 3D forcings.

.. _Tutorial:

Annotated Input Scripts
-----------------------

This section contains two annotated input scripts:

-  §3.6.1 :ref:`Harvey Flow Example` contains the harvey flow 
   example (``harvey.flow.tcl``) which is an idealized domain 
   with a heterogeneous subsurface. The example also demonstrates 
   how to generate multiple realizations of the subsurface and 
   add pumping wells.

-  §3.6.2 :ref:`Little Washita Example` contains the Little Washita
   example (``LW_Test.tcl``) which simulates a moderately sized 
   (41km by 41km) real domain using ParFlow ``CLM`` with 3D 
   meteorological forcings.

To run ParFlow, you use a script written in Tcl/TK. This script has a
lot of flexibility, as it is somewhere in between a program and a user
interface. The tcl script gives ParFlow the data it requires (or tells
ParFlow where to find or read in that data) and also tells ParFlow to
run.

To run the simulation:

#. Make any modifications to the tcl input script (and give a new name,
   if you want to)

#. Save the tcl script

#. For Linux/Unix/OSX: invoke the script from the command line using the
   tcl-shell, this looks like: ``>tclsh filename.tcl``

#. Wait patiently for the command prompt to return (Linux/Unix/OSX)
   indicating that ParFlow has finished. Intermediate files are written
   as the simulation runs, however there is no other indication that
   ParFlow is running.

To modify a tcl script, you right-click and select edit from the menu.
If you select open, you will run the script.

**Note:** The units for **K** (ım/d, usually) are critical to the entire
construction. These length and time units for **K** set the units for
all other variables (input or generated, throughout the entire
simulation) in the simulation. ParFlow can set to solve using hydraulic
conductivity by literally setting density, viscosity and gravity to one
(as is done in the script below). This means the pressure units are in
length (meters), so pressure is now so-called pressure-head.

.. _Harvey Flow Example:

Harvey Flow Example
~~~~~~~~~~~~~~~~~~~

This tutorial matches the ``harvey_flow.tcl`` file found in 
the ``/test`` directory. This example is directly from :cite:t:`MWH07`. 
This example demonstrates how to set up and run a fully saturated 
flow problem with heterogeneous hydraulic conductivity using the 
turning bands approach :cite:p:`TAG89`. Given statistical parameters describing 
the geology of your site, this script can be easily modified to 
make as many realizations of the subsurface as you like, each 
different and yet having the same statistical parameters, useful 
for a Monte Carlo simulation. This example is the basis for several 
fully-saturated ParFlow applications :cite:p:`Siirila12a,Siirila12b,SNSMM10,Atchley13a,Atchley13b,Cui14`.

When the script runs, it creates a new directory named ``/flow`` right 
in the directory where the tcl script is stored. ParFlow then puts all 
its output in ``/flow``. Of course, you can change the name and location 
of this output directory by modifying the tcl script that runs ParFlow.

Now for the tcl script:

::

   #
   # Import the ParFlow TCL package
   #

These first three lines are what link ParFlow and the tcl script, thus
allowing you to use a set of commands seen later, such as ``pfset``, etc.

::

   lappend auto_path $env(PARFLOW_DIR)/bin 
   package require parflow
   namespace import Parflow::*

   #-----------------------------------------------------------------------------
   # File input version number
   #-----------------------------------------------------------------------------
   pfset FileVersion 4

These next lines set the parallel process topology. The domain is
divided in *x*, *y* and *z* by ``P``, ``Q`` and ``R``. The total number 
of processors is ``P*Q*R`` (see :ref:`Computing Topology`).

::

   #----------------------------------------------------------------------------
   # Process Topology
   #----------------------------------------------------------------------------

   pfset Process.Topology.P     1
   pfset Process.Topology.Q     1
   pfset Process.Topology.R     1

Next we set up the computational grid (*see*
:ref:`Defining the Problem` and :ref:`Computational Grid`).

::

   #----------------------------------------------------------------------------
   # Computational Grid
   #----------------------------------------------------------------------------

Locate the origin in the domain.

::

   pfset ComputationalGrid.Lower.X    0.0
   pfset ComputationalGrid.Lower.Y    0.0
   pfset ComputationalGrid.Lower.Z    0.0

Define the size of the domain grid block. Length units, same as those on
hydraulic conductivity.

::

   pfset ComputationalGrid.DX     0.34
   pfset ComputationalGrid.DY     0.34
   pfset ComputationalGrid.DZ     0.038

Define the number of grid blocks in the domain.

::

   pfset ComputationalGrid.NX      50
   pfset ComputationalGrid.NY      30
   pfset ComputationalGrid.NZ      100

This next piece is comparable to a pre-declaration of variables. These
will be areas in our domain geometry. The regions themselves will be
defined later. You must always have one that is the name of your entire
domain. If you want subsections within your domain, you may declare
these as well. For Cape Cod, we have the entire domain, and also the 2
(upper and lower) permeability zones in the aquifer.

::

   #----------------------------------------------------------------------------
   # The Names of the GeomInputs
   #----------------------------------------------------------------------------
   pfset GeomInput.Names "domain_input upper_aquifer_input lower_aquifer_input"

Now you characterize your domain that you just pre-declared to be a ``box`` 
(see :ref:`Geometries`), and you also give it a name, ``domain``.

::

   #----------------------------------------------------------------------------
   # Domain Geometry Input
   #----------------------------------------------------------------------------
   pfset GeomInput.domain_input.InputType       Box
   pfset GeomInput.domain_input.GeomName   domain

Here, you set the limits in space for your entire domain. The span from ``Lower.X`` 
to ``Upper.X`` will be equal to the product of ``ComputationalGrid.DX`` 
times ``ComputationalGrid.NX``. Same for Y and Z (i.e. the number of grid elements 
times size of the grid element has to equal the size of the grid in each dimension). 
The ``Patches`` key assigns names to the outside edges, because the domain is the 
limit of the problem in space.

::

   #----------------------------------------------------------------------------
   # Domain Geometry
   #----------------------------------------------------------------------------
   pfset Geom.domain.Lower.X       0.0 
   pfset Geom.domain.Lower.Y       0.0
   pfset Geom.domain.Lower.Z       0.0

   pfset Geom.domain.Upper.X       17.0
   pfset Geom.domain.Upper.Y       10.2
   pfset Geom.domain.Upper.Z       3.8

   pfset Geom.domain.Patches "left right front back bottom top"

Just like domain geometry, you also set the limits in space for the
individual components (upper and lower, as defined in the Names of
GeomInputs pre-declaration). There are no patches for these geometries
as they are internal to the domain.

::

   #----------------------------------------------------------------------------
   # Upper Aquifer Geometry Input
   #----------------------------------------------------------------------------
   pfset GeomInput.upper_aquifer_input.InputType        Box
   pfset GeomInput.upper_aquifer_input.GeomName   upper_aquifer

   #----------------------------------------------------------------------------
   # Upper Aquifer Geometry
   #----------------------------------------------------------------------------
   pfset Geom.upper_aquifer.Lower.X                        0.0 
   pfset Geom.upper_aquifer.Lower.Y                        0.0
   pfset Geom.upper_aquifer.Lower.Z                        1.5

   pfset Geom.upper_aquifer.Upper.X                        17.0
   pfset Geom.upper_aquifer.Upper.Y                        10.2
   pfset Geom.upper_aquifer.Upper.Z                        1.5

   #----------------------------------------------------------------------------
   # Lower Aquifer Geometry Input
   #----------------------------------------------------------------------------
   pfset GeomInput.lower_aquifer_input.InputType       Box
   pfset GeomInput.lower_aquifer_input.GeomName   lower_aquifer

   #----------------------------------------------------------------------------
   # Lower Aquifer Geometry
   #----------------------------------------------------------------------------
   pfset Geom.lower_aquifer.Lower.X      0.0 
   pfset Geom.lower_aquifer.Lower.Y      0.0
   pfset Geom.lower_aquifer.Lower.Z      0.0

   pfset Geom.lower_aquifer.Upper.X     17.0
   pfset Geom.lower_aquifer.Upper.Y     10.2
   pfset Geom.lower_aquifer.Upper.Z      1.5

Now you add permeability data to the domain sections defined above
(:ref:`Permeability`). You can reassign values simply by
re-stating them – there is no need to comment out or delete the previous
version – the final statement is the only one that counts.

::

   #----------------------------------------------------------------------------
   # Perm
   #----------------------------------------------------------------------------

Name the permeability regions you will describe.

::

   pfset Geom.Perm.Names "upper_aquifer lower_aquifer"

You can set, for example homogeneous, constant permeability, or you can
generate a random field that meets your statistical requirements. To
define a constant permeability for the entire domain:

::

   #pfset Geom.domain.Perm.Type     Constant
   #pfset Geom.domain.Perm.Value    4.0

However, for Cape Cod, we did not want a constant permeability field, so
we instead generated a random permeability field meeting our statistical
parameters for each the upper and lower zones. Third from the bottom is
the ``Seed``. This is a random starting point to generate the K field. 
Pick any large ODD number. First we do something tricky with Tcl/TK. 
We use the native commands within tcl to open a text file and read in 
locally set variables. Note we use set here and not pfset. One is a native 
tcl command, the other a ParFlow-specific command. For this problem, we 
are linking the parameter estimation code, PEST to ParFlow. PEST writes 
out the ascii file ``stats4.txt`` (also located in the ``/test`` directory) 
as the result of a calibration run. Since we are not coupled to PEST in this 
example, we just read in the file and use the values to assign statistical properties.

::

   # we open a file, in this case from PEST to set upper and lower # kg and sigma
   #
   set fileId [open stats4.txt r 0600]
   set kgu [gets $fileId]
   set varu [gets $fileId]
   set kgl [gets $fileId]
   set varl [gets $fileId]
   close $fileId

Now we set the heterogeneous parameters for the Upper and Lower aquifers
(*see* :ref:`Permeability`). Note the special section at the
very end of this block where we reset the geometric mean and standard
deviation to our values we read in from a file. **Note:** ParFlow uses
*Standard Deviation* not *Variance*.

::

   pfset Geom.upper_aquifer.Perm.Type "TurnBands"
   pfset Geom.upper_aquifer.Perm.LambdaX  3.60
   pfset Geom.upper_aquifer.Perm.LambdaY  3.60
   pfset Geom.upper_aquifer.Perm.LambdaZ  0.19
   pfset Geom.upper_aquifer.Perm.GeomMean  112.00

   pfset Geom.upper_aquifer.Perm.Sigma   1.0
   pfset Geom.upper_aquifer.Perm.Sigma   0.48989794
   pfset Geom.upper_aquifer.Perm.NumLines 150
   pfset Geom.upper_aquifer.Perm.RZeta  5.0
   pfset Geom.upper_aquifer.Perm.KMax  100.0
   pfset Geom.upper_aquifer.Perm.DelK  0.2
   pfset Geom.upper_aquifer.Perm.Seed  33333
   pfset Geom.upper_aquifer.Perm.LogNormal Log
   pfset Geom.upper_aquifer.Perm.StratType Bottom
   pfset Geom.lower_aquifer.Perm.Type "TurnBands"
   pfset Geom.lower_aquifer.Perm.LambdaX  3.60
   pfset Geom.lower_aquifer.Perm.LambdaY  3.60
   pfset Geom.lower_aquifer.Perm.LambdaZ  0.19

   pfset Geom.lower_aquifer.Perm.GeomMean  77.0
   pfset Geom.lower_aquifer.Perm.Sigma   1.0
   pfset Geom.lower_aquifer.Perm.Sigma   0.48989794
   pfset Geom.lower_aquifer.Perm.NumLines 150
   pfset Geom.lower_aquifer.Perm.RZeta  5.0
   pfset Geom.lower_aquifer.Perm.KMax  100.0
   pfset Geom.lower_aquifer.Perm.DelK  0.2
   pfset Geom.lower_aquifer.Perm.Seed  33333
   pfset Geom.lower_aquifer.Perm.LogNormal Log
   pfset Geom.lower_aquifer.Perm.StratType Bottom

   #pfset lower aqu and upper aq stats to pest/read in values

   pfset Geom.upper_aquifer.Perm.GeomMean  $kgu
   pfset Geom.upper_aquifer.Perm.Sigma  $varu

   pfset Geom.lower_aquifer.Perm.GeomMean  $kgl
   pfset Geom.lower_aquifer.Perm.Sigma  $varl

The following section allows you to specify the permeability tensor. In
the case below, permeability is symmetric in all directions (x, y, and
z) and therefore each is set to 1.0.

::

   pfset Perm.TensorType               TensorByGeom

   pfset Geom.Perm.TensorByGeom.Names  "domain"

   pfset Geom.domain.Perm.TensorValX  1.0
   pfset Geom.domain.Perm.TensorValY  1.0
   pfset Geom.domain.Perm.TensorValZ  1.0

Next we set the specific storage, though this is not used in the
IMPES/steady-state calculation.

::

   #----------------------------------------------------------------------------
   # Specific Storage
   #----------------------------------------------------------------------------
   # specific storage does not figure into the impes (fully sat) 
   # case but we still need a key for it

   pfset SpecificStorage.Type            Constant
   pfset SpecificStorage.GeomNames       ""
   pfset Geom.domain.SpecificStorage.Value 1.0e-4

ParFlow has the capability to deal with a multiphase system, but we only
have one (water) at Cape Cod. As we stated earlier, we set density and
viscosity artificially (and later gravity) both to 1.0. Again, this is
merely a trick to solve for hydraulic conductivity and pressure head. If
you were to set density and viscosity to their true values, the code
would calculate **k** (permeability). By using the *normalized* values
instead, you effectively embed the conversion of **k** to **K**
(hydraulic conductivity). So this way, we get hydraulic conductivity,
which is what we want for this problem.

::

   #----------------------------------------------------------------------------
   # Phases
   #----------------------------------------------------------------------------

   pfset Phase.Names "water"

   pfset Phase.water.Density.Type	Constant
   pfset Phase.water.Density.Value	1.0

   pfset Phase.water.Viscosity.Type	Constant
   pfset Phase.water.Viscosity.Value	1.0

We will not use the ParFlow grid based transport scheme. We will then
leave contaminants blank because we will use a different code to model
(virus, tracer) contamination.

::

   #----------------------------------------------------------------------------
   # Contaminants
   #----------------------------------------------------------------------------
   pfset Contaminants.Names			""

As with density and viscosity, gravity is normalized here. If we used
the true value (in the *[L]* and *[T]* units of hydraulic conductivity)
the code would be calculating permeability. Instead, we normalize so
that the code calculates hydraulic conductivity.

::

   #----------------------------------------------------------------------------
   # Gravity
   #----------------------------------------------------------------------------

   pfset Gravity				1.0

   #----------------------------------------------------------------------------
   # Setup timing info
   #----------------------------------------------------------------------------

This basic time unit of 1.0 is used for transient boundary and well
conditions. We are not using those features in this example.

::

   pfset TimingInfo.BaseUnit		1.0

Cape Cod is a steady state problem, so these timing features are again
unused, but need to be included.

::

   pfset TimingInfo.StartCount   -1
   pfset TimingInfo.StartTime     0.0
   pfset TimingInfo.StopTime     0.0

Set the ``dump interval`` to -1 to report info at the end of every 
calculation, which in this case is only when steady state has been 
reached.

::

   pfset TimingInfo.DumpInterval	       -1

Next, we assign the porosity (*see* §6.1.12 :ref:`Porosity`). For the
Cape Cod, the porosity is 0.39.

::

   #----------------------------------------------------------------------------
   # Porosity
   #----------------------------------------------------------------------------

   pfset Geom.Porosity.GeomNames          domain

   pfset Geom.domain.Porosity.Type    Constant
   pfset Geom.domain.Porosity.Value   0.390

Having defined the geometry of our problem before and named it ``domain``, we 
are now ready to report/upload that problem, which we do here.

::

   #----------------------------------------------------------------------------
   # Domain
   #----------------------------------------------------------------------------
   pfset Domain.GeomName domain

Mobility between phases is set to 1.0 because we only have one phase
(water).

::

   #----------------------------------------------------------------------------
   # Mobility
   #----------------------------------------------------------------------------
   pfset Phase.water.Mobility.Type        Constant
   pfset Phase.water.Mobility.Value       1.0

Again, ParFlow has more capabilities than we are using here in the Cape
Cod example. For this example, we handle monitoring wells in a separate
code as we assume they do not remove a significant amount of water from
the domain. Note that since there are no well names listed here, ParFlow
assumes we have no wells. If we had pumping wells, we would have to
include them here, because they would affect the head distribution
throughout our domain. See below for an example of how to include
pumping wells in this script.

::

   #----------------------------------------------------------------------------
   # Wells
   #----------------------------------------------------------------------------
   pfset Wells.Names ""

You can give certain periods of time names if you want to (ie.
Pre-injection, post-injection, etc). Here, however we do not have
multiple time intervals and are simulating in steady state, so time
cycle keys are simple. We have only one time cycle and it’s constant for
the duration of the simulation. We accomplish this by giving it a repeat
value of -1, which repeats indefinitely. The length of the cycle is the
length specified below (an integer) multiplied by the base unit value we
specified earlier.

::

   #----------------------------------------------------------------------------
   # Time Cycles
   #----------------------------------------------------------------------------
   pfset Cycle.Names constant
   pfset Cycle.constant.Names		"alltime"
   pfset Cycle.constant.alltime.Length	 1
   pfset Cycle.constant.Repeat		-1

Now, we assign Boundary Conditions for each face (each of the Patches in
the domain defined before). Recall the previously stated Patches and
associate them with the boundary conditions that follow.

::

   pfset BCPressure.PatchNames "left right front back bottom top"

These are Dirichlet BCs (i.e. constant head over cell so the pressure
head is set to hydrostatic– *see* :ref:`Boundary Conditions: Pressure`). There is no time
dependence, so use the ``constant`` time cycle we defined 
previously. ``RefGeom`` links this to the established domain geometry 
and tells ParFlow what to use for a datum when calculating hydrostatic 
head conditions.

::

   pfset Patch.left.BCPressure.Type          DirEquilRefPatch
   pfset Patch.left.BCPressure.Cycle        "constant"
   pfset Patch.left.BCPressure.RefGeom domain

Reference the current (left) patch to the bottom to define the line of
intersection between the two.

::

   pfset Patch.left.BCPressure.RefPatch  bottom

Set the head permanently to 10.0m. Pressure-head will of course vary top
to bottom because of hydrostatics, but head potential will be constant.

::

   pfset Patch.left.BCPressure.alltime.Value  10.0

Repeat the declarations for the rest of the faces of the domain. The
left to right (*X*) dimension is aligned with the hydraulic gradient.
The difference between the values assigned to right and left divided by
the length of the domain corresponds to the correct hydraulic gradient.

::

   pfset Patch.right.BCPressure.Type               DirEquilRefPatch
   pfset Patch.right.BCPressure.Cycle              "constant"
   pfset Patch.right.BCPressure.RefGeom       domain
   pfset Patch.right.BCPressure.RefPatch       bottom
   pfset Patch.right.BCPressure.alltime.Value 9.97501

   pfset Patch.front.BCPressure.Type                FluxConst
   pfset Patch.front.BCPressure.Cycle              "constant"
   pfset Patch.front.BCPressure.alltime.Value 0.0

   pfset Patch.back.BCPressure.Type               FluxConst
   pfset Patch.back.BCPressure.Cycle              "constant"
   pfset Patch.back.BCPressure.alltime.Value 0.0

   pfset Patch.bottom.BCPressure.Type               FluxConst
   pfset Patch.bottom.BCPressure.Cycle              "constant"
   pfset Patch.bottom.BCPressure.alltime.Value 0.0

   pfset Patch.top.BCPressure.Type			FluxConst
   pfset Patch.top.BCPressure.Cycle			"constant"
   pfset Patch.top.BCPressure.alltime.Value		0.0

Next we define topographic slopes and Mannings *n* values. These are not
used, since we do not solve for overland flow. However, the keys still
need to appear in the input script.

::

   #---------------------------------------------------------
   # Topo slopes in x-direction
   #---------------------------------------------------------
   # topo slopes do not figure into the impes (fully sat) case but we still
   # need keys for them

   pfset TopoSlopesX.Type "Constant"
   pfset TopoSlopesX.GeomNames ""

   pfset TopoSlopesX.Geom.domain.Value 0.0

   #---------------------------------------------------------
   # Topo slopes in y-direction
   #---------------------------------------------------------

   pfset TopoSlopesY.Type "Constant"
   pfset TopoSlopesY.GeomNames ""

   pfset TopoSlopesY.Geom.domain.Value 0.0

   # You may also indicate an elevation file used to derive the slopes.
   # This is optional but can be useful when post-processing terrain-
   # following grids:
   pfset TopoSlopes.Elevation.FileName "elevation.pfb"

   #---------------------------------------------------------
   # Mannings coefficient 
   #---------------------------------------------------------
   # mannings roughnesses do not figure into the impes (fully sat) case but we still
   # need a key for them

   pfset Mannings.Type "Constant"
   pfset Mannings.GeomNames ""
   pfset Mannings.Geom.domain.Value 0.

Phase sources allows you to add sources other than wells and boundaries,
but we do not have any so this key is constant, 0.0 over entire domain.

::

   #----------------------------------------------------------------------------
   # Phase sources:
   #----------------------------------------------------------------------------

   pfset PhaseSources.water.Type                         Constant
   pfset PhaseSources.water.GeomNames                    domain
   pfset PhaseSources.water.Geom.domain.Value        0.0

Next we define solver parameters for **IMPES**. Since this is the
default solver, we do not need a solver key.

::

   #---------------------------------------------------------
   #  Solver Impes  
   #---------------------------------------------------------

We allow up to 50 iterations of the linear solver before it quits or
converges.

::

   pfset Solver.MaxIter 50

The solution must be accurate to this level

::

   pfset Solver.AbsTol  1E-10

We drop significant digits beyond E-15

::

   pfset Solver.Drop   1E-15

   #--------------------------------------------------------
   # Run and Unload the ParFlow output files
   #---------------------------------------------------------

Here you set the number of realizations again using a local tcl
variable. We have set only one run but by setting the ``n_runs`` 
variable to something else we can run more than one realization 
of hydraulic conductivity.

::

   # this script is setup to run 100 realizations, for testing we just run one
   ###set n_runs 100
   set n_runs 1

Here is where you tell ParFlow where to put the output. In this case, it
is a directory called flow. Then you cd (change directory) into that new
directory. If you wanted to put an entire path rather than just a name,
you would have more control over where your output file goes. For
example, you would put ``file mkdir “/cape_cod/revised_statistics/flow"`` 
and then change into that directory.

::

   file mkdir "flow"
   cd "flow"

Now we loop through the realizations, again using tcl. ``k`` is the integer 
counter that is incremented for each realization. When you use a variable 
(rather than define it), you precede it with ``$``. The hanging character ``{`` 
opens the do loop for ``k``.

::

   #
   #  Loop through runs
   #
   for {set k 1} {$k <= $n_runs} {incr k 1} {

The following expressions sets the variable ``seed`` equal to the expression 
in brackets, which increments with each turn of the do loop and each seed 
will produce a different random field of K. You set upper and lower aquifer, 
because in the Cape Cod site, these are the two subsets of the domain. 
Note the seed starts at a different point to allow for different random 
field generation for the upper and lower zones.

::

   #
   # set the random seed to be different for every run
   #
   pfset Geom.upper_aquifer.Perm.Seed  [ expr 33333+2*$k ] 
   pfset Geom.lower_aquifer.Perm.Seed  [ expr 31313+2*$k ]

The following command runs ParFlow and gives you a suite of output files
for each realization. The file names will 
begin ``harvey_flow.1.xxxxx``, ``harvey_flow.2.xxxx``, etc up to as 
many realizations as you run. The .xxxxx part will designate 
x, y, and z permeability, etc. Recall that in this case, since we normalized 
gravity, viscosity, and density, remember that we are really getting hydraulic 
conductivity.

::

   pfrun harvey_flow.$k

This command removes a large number of superfluous dummy files or
un-distributes parallel files back into a single file. If you compile
with the ``–with-amps-sequential-io`` option then a single ParFlow 
file is written with corresponding ``XXXX.dist`` files and 
the ``pfundist`` command just removes these ``.dist`` files 
(though you don’t really need to remove them if you don’t want to).

::

   pfundist harvey_flow.$k

The following commands take advantage of PFTools (*see*
:ref:`PFTCL Commands`) and load pressure head output of the
/parflow model into a pressure matrix.

::

   # we use pf tools to convert from pressure to head
   # we could do a number of other things here like copy files to different
   # format
   set press [pfload harvey_flow.$k.out.press.pfb]

The next command takes the pressures that were just loaded and converts
it to head and loads them into a head matrix tcl variable.

::

   set head [pfhhead $press]

Finally, the head matrix is saved as a ParFlow binary file (.pfb) and
the k do loop is closed by the ``}`` character. Then we move up to the
root directory when we are finished

::

    pfsave $head -pfb harvey_flow.$k.head.pfb
   }

   cd ".."

Once you have modified the tcl input script (if necessary) and run
ParFlow, you will have as many realizations of your subsurface as you
specified. Each of these realizations will be used as input for a
particle or streamline calculation in the future. We can see below, that
since we have a tcl script as input, we can do a lot of different
operations, for example, we might run a particle tracking transport code
simulation using the results of the ParFlow runs. This actually
corresponds to the example presented in the ``SLIM`` user’s manual.

::

   # this could run other tcl scripts now an example is below
   #puts stdout "running SLIM"
   #source bromide_trans.sm.tcl

We can add options to this script. For example if we wanted to add a
pumping well these additions are described below.

Adding a Pumping Well
~~~~~~~~~~~~~~~~~~~~~

Let us change the input problem by adding a pumping well:

.. container:: enumerate

   1. Add the following lines to the input file near where the existing
   well information is in the input file. You need to replace the
   “Wells.Names” line with the one included here to get both wells
   activated (this value lists the names of the wells):

   .. container:: list

      ::

         pfset Wells.Names {new_well}

         pfset Wells.new_well.InputType                Recirc

         pfset Wells.new_well.Cycle		    constant

         pfset Wells.new_well.ExtractionType	    Flux
         pfset Wells.new_well.InjectionType            Flux

         pfset Wells.new_well.X			    10.0 
         pfset Wells.new_well.Y			    10.0
         pfset Wells.new_well.ExtractionZLower	     0.5
         pfset Wells.new_well.ExtractionZUpper	     0.5
         pfset Wells.new_well.InjectionZLower	     0.2
         pfset Wells.new_well.InjectionZUpper	     0.2

         pfset Wells.new_well.ExtractionMethod	    Standard
         pfset Wells.new_well.InjectionMethod          Standard

         pfset Wells.new_well.alltime.Extraction.Flux.water.Value        	     0.50
         pfset Wells.new_well.alltime.Injection.Flux.water.Value		     0.75

For more information on defining the problem, see
:ref:`Defining the Problem`.

We could also visualize the results of the ParFlow simulations, using
*VisIt*. For example, we can turn on *SILO* file output which allows
these files to be directly read and visualized. We would do this by
adding the following ``pfset`` commands, I usually add them to t
he solver section:

.. container:: list

   ::

      pfset Solver.WriteSiloSubsurfData True
      pfset Solver.WriteSiloPressure True
      pfset Solver.WriteSiloSaturation True

You can then directly open the file ``harvey_flow.#.out.perm_x.silo`` 
(where ``#`` is the realization number). The resulting image will 
be the hydraulic conductivity field of your domain, showing the 
variation in x-permeability in 3-D space. You can also generate 
representations of head or pressure (or y or z permeability) 
throughout your domain using ParFlow output files. See the section 
on visualization for more details.

.. _Little Washita Example:

Little Washita Example
~~~~~~~~~~~~~~~~~~~~~~

This tutorial matches the ``LW_Test.tcl`` file found in 
the ``/test/washita/tcl_scripts`` directory and corresponds to :cite:t:`Condon14a,Condon14b`. 
This script runs the Little Washita domain for three days using 
ParFlow ``CLM`` with 3D forcings. The domain is setup using terrain 
following grid (:ref:`TFG`) and subsurface geologes are 
specified using a ``.pfb`` indicator file. Input files were 
generated using the workflow detailed in :ref:`Defining a Real domain`.

Now for the tcl script:

::

   #
   # Import the ParFlow TCL package
   #

These first three lines are what link ParFlow and the tcl script, thus
allowing you to use a set of commands seen later, such as ``pfset``, etc.

::

   lappend auto_path $env(PARFLOW_DIR)/bin 
   package require parflow
   namespace import Parflow::*

   #-----------------------------------------------------------------------------
   # File input version number
   #-----------------------------------------------------------------------------
   pfset FileVersion 4

These next lines set the parallel process topology. The domain is
divided in *x*, *y* and *z* by ``P``, ``Q`` and ``R``. The total 
number of processors is ``P*Q*R`` (see :ref:`Computing Topology`).

::

   #----------------------------------------------------------------------------
   # Process Topology
   #----------------------------------------------------------------------------

   pfset Process.Topology.P     1
   pfset Process.Topology.Q     1
   pfset Process.Topology.R     1

Before we really get started make a directory for our outputs and copy
all of the required input files into the run directory. These files will
be described in detail later as they get used.

::

   #-----------------------------------------------------------------------------
   # Make a directory for the simulation and copy inputs into it
   #-----------------------------------------------------------------------------
   exec mkdir "Outputs"
   cd "./Outputs"

   # ParFlow Inputs
   file copy -force "../../parflow_input/LW.slopex.pfb" .
   file copy -force "../../parflow_input/LW.slopey.pfb" .
   file copy -force "../../parflow_input/IndicatorFile_Gleeson.50z.pfb"   .
   file copy -force "../../parflow_input/press.init.pfb"  .

   #CLM Inputs
   file copy -force "../../clm_input/drv_clmin.dat" .
   file copy -force "../../clm_input/drv_vegp.dat"  .
   file copy -force "../../clm_input/drv_vegm.alluv.dat"  . 

   puts "Files Copied"

Next we set up the computational grid (*see*
:ref:`Defining the Problem` and
:ref:`Computational Grid`).

::

   #----------------------------------------------------------------------------
   # Computational Grid
   #----------------------------------------------------------------------------

Locate the origin in the domain.

::

   pfset ComputationalGrid.Lower.X    0.0
   pfset ComputationalGrid.Lower.Y    0.0
   pfset ComputationalGrid.Lower.Z    0.0

Define the size of the domain grid block. Length units, same as those on
hydraulic conductivity.

::

   pfset ComputationalGrid.DX     1000.0
   pfset ComputationalGrid.DY     1000.0
   pfset ComputationalGrid.DZ     2.0

Define the number of grid blocks in the domain.

::

   pfset ComputationalGrid.NX      41
   pfset ComputationalGrid.NY      41
   pfset ComputationalGrid.NZ      50

This next piece is comparable to a pre-declaration of variables. These
will be areas in our domain geometry. The regions themselves will be
defined later. You must always have one that is the name of your entire
domain. If you want subsections within your domain, you may declare
these as well. Here we define two geometries one is the domain and one
is for the indicator file (which will also span the entire domain).

::

   #-----------------------------------------------------------------------------
   # The Names of the GeomInputs
   #-----------------------------------------------------------------------------
   pfset GeomInput.Names                     "box_input indi_input"

Now you characterize the domain that you just pre-declared 
to be a ``box`` (see :ref:`Geometries`), and you also 
give it a name, ``domain``.

::

   #-----------------------------------------------------------------------------
   # Domain Geometry Input
   #-----------------------------------------------------------------------------
   pfset GeomInput.box_input.InputType      Box
   pfset GeomInput.box_input.GeomName      domain

Here, you set the limits in space for your entire domain. The span 
from ``Lower.X`` to ``Upper.X`` will be equal to the product 
of ``ComputationalGrid.DX`` times ``ComputationalGrid.NX``. 
Same for Y and Z (i.e. the number of grid elements times size 
of the grid element has to equal the size of the grid in each 
dimension). The ``Patches`` key assigns names to the outside 
edges, because the domain is the limit of the problem in space.

::

   #-----------------------------------------------------------------------------
   # Domain Geometry 
   #-----------------------------------------------------------------------------
   pfset Geom.domain.Lower.X                        0.0
   pfset Geom.domain.Lower.Y                        0.0
   pfset Geom.domain.Lower.Z                        0.0
    
   pfset Geom.domain.Upper.X                        41000.0
   pfset Geom.domain.Upper.Y                        41000.0
   pfset Geom.domain.Upper.Z                          100.0

   pfset Geom.domain.Patches             "x-lower x-upper y-lower y-upper z-lower z-upper"

Now we setup the indicator file. As noted above, the indicator file has
integer values for every grid cell in the domain designating what
geologic unit it belongs to. The ``GeomNames`` list should include 
a name for every unit in your indicator file. In this example we 
have thirteen soil units and eight geologic units. The ``FileName`` points 
to the indicator file that ParFlow will read. Recall that this file 
into the run directory at the start of the script.

::

   #-----------------------------------------------------------------------------
   # Indicator Geometry Input
   #-----------------------------------------------------------------------------
   pfset GeomInput.indi_input.InputType      IndicatorField
   pfset GeomInput.indi_input.GeomNames      "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
   pfset Geom.indi_input.FileName            "IndicatorFile_Gleeson.50z.pfb"

For every name in the ``GeomNames`` list we define the corresponding 
value in the indicator file. For example, here we are saying that 
our first soil unit (``s1``) is represented by the number “1" in 
the indicator file, while the first geologic unit (``g1``) is 
represented by the number “21". Note that the integers used in the 
indicator file do not need to be consecutive.

::

   pfset GeomInput.s1.Value                1
   pfset GeomInput.s2.Value                2
   pfset GeomInput.s3.Value                3
   pfset GeomInput.s4.Value                4
   pfset GeomInput.s5.Value                5
   pfset GeomInput.s6.Value                6
   pfset GeomInput.s7.Value                7
   pfset GeomInput.s8.Value                8
   pfset GeomInput.s9.Value                9
   pfset GeomInput.s10.Value               10
   pfset GeomInput.s11.Value               11
   pfset GeomInput.s12.Value               12
   pfset GeomInput.s13.Value               13
   pfset GeomInput.g1.Value                21
   pfset GeomInput.g2.Value                22
   pfset GeomInput.g3.Value                23
   pfset GeomInput.g4.Value                24
   pfset GeomInput.g5.Value                25
   pfset GeomInput.g6.Value                26
   pfset GeomInput.g7.Value                27
   pfset GeomInput.g8.Value                28

Now you add permeability data to the domain sections defined above
(:ref:`Permeability`). You can reassign values simply by
re-stating them – there is no need to comment out or delete the previous
version – the final statement is the only one that counts. Also, note
that you do not need to assign permeability values to all of the
geometries names. Any geometry that is not assigned its own permeability
value will take the ``domain`` value. However, every geometry listed 
in ``Porosity.GeomNames`` must have values assigned.

::

   #-----------------------------------------------------------------------------
   # Permeability (values in m/hr)
   #-----------------------------------------------------------------------------
   pfset Geom.Perm.Names                     "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8"

   pfset Geom.domain.Perm.Type           Constant
   pfset Geom.domain.Perm.Value          0.2

   pfset Geom.s1.Perm.Type               Constant
   pfset Geom.s1.Perm.Value              0.269022595

   pfset Geom.s2.Perm.Type               Constant
   pfset Geom.s2.Perm.Value              0.043630356

   pfset Geom.s3.Perm.Type               Constant
   pfset Geom.s3.Perm.Value              0.015841225

   pfset Geom.s4.Perm.Type               Constant
   pfset Geom.s4.Perm.Value              0.007582087

   pfset Geom.s5.Perm.Type               Constant
   pfset Geom.s5.Perm.Value              0.01818816

   pfset Geom.s6.Perm.Type               Constant
   pfset Geom.s6.Perm.Value              0.005009435

   pfset Geom.s7.Perm.Type               Constant
   pfset Geom.s7.Perm.Value              0.005492736

   pfset Geom.s8.Perm.Type               Constant
   pfset Geom.s8.Perm.Value              0.004675077

   pfset Geom.s9.Perm.Type               Constant
   pfset Geom.s9.Perm.Value              0.003386794

   pfset Geom.g2.Perm.Type               Constant
   pfset Geom.g2.Perm.Value              0.025

   pfset Geom.g3.Perm.Type               Constant
   pfset Geom.g3.Perm.Value              0.059

   pfset Geom.g6.Perm.Type               Constant
   pfset Geom.g6.Perm.Value              0.2

   pfset Geom.g8.Perm.Type              Constant
   pfset Geom.g8.Perm.Value             0.68

The following section allows you to specify the permeability tensor. In
the case below, permeability is symmetric in all directions (x, y, and
z) and therefore each is set to 1.0. Also note that we just specify this
once for the whole domain because we want isotropic permeability
everywhere. You can specify different tensors for different units by
repeating these lines with different ``Geom.Names``.

::

   pfset Perm.TensorType                     TensorByGeom
   pfset Geom.Perm.TensorByGeom.Names        "domain"
   pfset Geom.domain.Perm.TensorValX         1.0d0
   pfset Geom.domain.Perm.TensorValY         1.0d0
   pfset Geom.domain.Perm.TensorValZ         1.0d0

Next we set the specific storage. Here again we specify one value for
the whole domain but these lines can be easily repeated to set different
values for different units.

::

   #-----------------------------------------------------------------------------
   # Specific Storage
   #-----------------------------------------------------------------------------
   pfset SpecificStorage.Type                Constant
   pfset SpecificStorage.GeomNames           "domain"
   pfset Geom.domain.SpecificStorage.Value   1.0e-5

ParFlow has the capability to deal with a multiphase system, but we only
have one (water) in this example. As we stated earlier, we set density
and viscosity artificially (and later gravity) both to 1.0. Again, this
is merely a trick to solve for hydraulic conductivity and pressure head.
If you were to set density and viscosity to their true values, the code
would calculate **k** (permeability). By using the *normalized* values
instead, you effectively embed the conversion of **k** to **K**
(hydraulic conductivity). So this way, we get hydraulic conductivity,
which is what we want for this problem.

::

   #-----------------------------------------------------------------------------
   # Phases
   #-----------------------------------------------------------------------------
   pfset Phase.Names                         "water"

   pfset Phase.water.Density.Type            Constant
   pfset Phase.water.Density.Value           1.0

   pfset Phase.water.Viscosity.Type          Constant
   pfset Phase.water.Viscosity.Value         1.0

This example does not include the ParFlow grid based transport scheme.
Therefore we leave contaminants blank.

::

   #-----------------------------------------------------------------------------
   # Contaminants
   #-----------------------------------------------------------------------------
   pfset Contaminants.Names                  ""

As with density and viscosity, gravity is normalized here. If we used
the true value (in the *[L]* and *[T]* units of hydraulic conductivity)
the code would be calculating permeability. Instead, we normalize so
that the code calculates hydraulic conductivity.

::

   #-----------------------------------------------------------------------------
   # Gravity
   #-----------------------------------------------------------------------------
   pfset Gravity                             1.0

Next we set up the timing for our simulation.

::

   #-----------------------------------------------------------------------------
   # Timing (time units is set by units of permeability)
   #-----------------------------------------------------------------------------

This specifies the base unit of time for all time values entered. All
time should be expressed as multiples of this value. To keep things
simple here we set it to 1. Because we expressed our permeability in
units of m/hr in this example this means that our basin unit of time is
1hr.

::

   pfset TimingInfo.BaseUnit                 1.0

This key specifies the time step number that will be associated with the
first advection cycle of the transient problem. Because we are starting
from scratch we set this to 0. If we were restarting a run we would set
this to the last time step of your previous simulation. Refer to
§3.3 :ref:`Restarting a Run` for additional instructions on restarting
a run.

::

   pfset TimingInfo.StartCount               0.0

``StartTime`` and ``StopTime`` specify the start and stop times 
for the simulation. These values should correspond with the 
forcing files you are using.

::

   pfset TimingInfo.StartTime                0.0
   pfset TimingInfo.StopTime                 72.0

This key specifies the timing interval at which ParFlow time dependent
outputs will be written. Here we have a base unit of 1hr so a dump
interval of 24 means that we are writing daily outputs. Note that this
key only controls the ParFlow output interval and not the interval that
``CLM`` outputs will be written out at.

::

   pfset TimingInfo.DumpInterval             24.0

Here we set the time step value. For this example we use a constant time
step of 1hr.

::

   pfset TimeStep.Type                       Constant
   pfset TimeStep.Value                      1.0

Next, we assign the porosity (*see* §6.1.12 :ref:`Porosity`). As with
the permeability we assign different values for different indicator
geometries. Here we assign values for all of our soil units but not for
the geologic units, they will default to the domain value of 0.4. Note
that every geometry listed in ``Porosity.GeomNames`` must have values assigned.

::

   #-----------------------------------------------------------------------------
   # Porosity
   #-----------------------------------------------------------------------------
   pfset Geom.Porosity.GeomNames             "domain s1 s2 s3 s4 s5 s6 s7 s8 s9"

   pfset Geom.domain.Porosity.Type          Constant
   pfset Geom.domain.Porosity.Value         0.4

   pfset Geom.s1.Porosity.Type    Constant
   pfset Geom.s1.Porosity.Value   0.375

   pfset Geom.s2.Porosity.Type    Constant
   pfset Geom.s2.Porosity.Value   0.39

   pfset Geom.s3.Porosity.Type    Constant
   pfset Geom.s3.Porosity.Value   0.387

   pfset Geom.s4.Porosity.Type    Constant
   pfset Geom.s4.Porosity.Value   0.439

   pfset Geom.s5.Porosity.Type    Constant
   pfset Geom.s5.Porosity.Value   0.489

   pfset Geom.s6.Porosity.Type    Constant
   pfset Geom.s6.Porosity.Value   0.399

   pfset Geom.s7.Porosity.Type    Constant
   pfset Geom.s7.Porosity.Value   0.384

   pfset Geom.s8.Porosity.Type            Constant
   pfset Geom.s8.Porosity.Value           0.482

   pfset Geom.s9.Porosity.Type            Constant
   pfset Geom.s9.Porosity.Value           0.442

Having defined the geometry of our problem before and named it ``domain``, 
we are now ready to report/upload that problem, which we do here.

::

   #-----------------------------------------------------------------------------
   # Domain
   #-----------------------------------------------------------------------------
   pfset Domain.GeomName                     "domain"

Mobility between phases is set to 1.0 because we only have one phase
(water).

::

   #----------------------------------------------------------------------------
   # Mobility
   #----------------------------------------------------------------------------
   pfset Phase.water.Mobility.Type        Constant
   pfset Phase.water.Mobility.Value       1.0

Again, ParFlow has more capabilities than we are using here in this
example. Note that since there are no well names listed here, ParFlow
assumes we have no wells. If we had pumping wells, we would have to
include them here, because they would affect the head distribution
throughout our domain. See :ref:`Harvey Flow Example` for an
example of how to include pumping wells in this script.

::

   #-----------------------------------------------------------------------------
   # Wells
   #-----------------------------------------------------------------------------
   pfset Wells.Names                         ""

You can give certain periods of time names if you want. For example if
you aren’t running with ``CLM`` and you would like to have periods 
with rain and periods without. Here, however we have only one time 
cycle because ``CLM`` will handle the variable forcings. Therefore, 
we specify one time cycle and it’s constant for the duration of the 
simulation. We accomplish this by giving it a repeat value of -1, 
which repeats indefinitely. The length of the cycle is the length 
specified below (an integer) multiplied by the base unit value we 
specified earlier.

::

   #-----------------------------------------------------------------------------
   # Time Cycles
   #-----------------------------------------------------------------------------
   pfset Cycle.Names                         "constant"
   pfset Cycle.constant.Names                "alltime"
   pfset Cycle.constant.alltime.Length        1
   pfset Cycle.constant.Repeat               -1

Now, we assign Boundary Conditions for each face (each of the Patches in
the domain defined before). Recall the previously stated Patches and
associate them with the boundary conditions that follow.

::

   #-----------------------------------------------------------------------------
   # Boundary Conditions
   #-----------------------------------------------------------------------------
   pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

The bottom and sides of our domain are all set to no-flow (i.e. constant
flux of 0) boundaries.

::

   pfset Patch.x-lower.BCPressure.Type		      FluxConst
   pfset Patch.x-lower.BCPressure.Cycle		      "constant"
   pfset Patch.x-lower.BCPressure.alltime.Value	      0.0

   pfset Patch.y-lower.BCPressure.Type		      FluxConst
   pfset Patch.y-lower.BCPressure.Cycle		      "constant"
   pfset Patch.y-lower.BCPressure.alltime.Value	      0.0

   pfset Patch.z-lower.BCPressure.Type		      FluxConst
   pfset Patch.z-lower.BCPressure.Cycle		      "constant"
   pfset Patch.z-lower.BCPressure.alltime.Value	      0.0

   pfset Patch.x-upper.BCPressure.Type		      FluxConst
   pfset Patch.x-upper.BCPressure.Cycle		      "constant"
   pfset Patch.x-upper.BCPressure.alltime.Value	      0.0

   pfset Patch.y-upper.BCPressure.Type		      FluxConst
   pfset Patch.y-upper.BCPressure.Cycle		      "constant"
   pfset Patch.y-upper.BCPressure.alltime.Value	      0.0

The top is set to an ``OverlandFLow`` boundary to turn on the 
fully-coupled overland flow routing.

::

   pfset Patch.z-upper.BCPressure.Type		      OverlandFlow
   pfset Patch.z-upper.BCPressure.Cycle		      "constant"
   pfset Patch.z-upper.BCPressure.alltime.Value	      0.0

Next we define topographic slopes and values. These slope values were
derived from a digital elevation model of the domain following the
workflow outlined in :ref:`Defining a Real domain`. In this
example we read the slope files in from ``.pfb`` files that were 
copied into the run directory at the start of this script.

::

   #-----------------------------------------------------------------------------
   # Topo slopes in x-direction
   #-----------------------------------------------------------------------------
   pfset TopoSlopesX.Type                                "PFBFile"
   pfset TopoSlopesX.GeomNames                           "domain"
   pfset TopoSlopesX.FileName                            "LW.slopex.pfb"

   #-----------------------------------------------------------------------------
   # Topo slopes in y-direction
   #-----------------------------------------------------------------------------
   pfset TopoSlopesY.Type                                "PFBFile"
   pfset TopoSlopesY.GeomNames                           "domain"
   pfset TopoSlopesY.FileName                            "LW.slopey.pfb"

And now we define the Mannings *n*, again just one value for the whole
domain in this example.

::

   #-----------------------------------------------------------------------------
   # Mannings coefficient
   #-----------------------------------------------------------------------------
   pfset Mannings.Type                                   "Constant"
   pfset Mannings.GeomNames                              "domain"
   pfset Mannings.Geom.domain.Value                      5.52e-6

Following the same approach as we did for ``Porosity`` we define 
the relative permeability inputs that will be used for Richards’ 
equation implementation (:ref:`Richards RelPerm`). Here we 
use ``VanGenuchten`` parameters. Note that every geometry 
listed in ``Porosity.GeomNames`` must have values assigned.

::

   #-----------------------------------------------------------------------------
   # Relative Permeability
   #-----------------------------------------------------------------------------
   pfset Phase.RelPerm.Type                  VanGenuchten
   pfset Phase.RelPerm.GeomNames             "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

   pfset Geom.domain.RelPerm.Alpha           3.5
   pfset Geom.domain.RelPerm.N               2.0

   pfset Geom.s1.RelPerm.Alpha        3.548
   pfset Geom.s1.RelPerm.N            4.162

   pfset Geom.s2.RelPerm.Alpha        3.467
   pfset Geom.s2.RelPerm.N            2.738

   pfset Geom.s3.RelPerm.Alpha        2.692
   pfset Geom.s3.RelPerm.N            2.445

   pfset Geom.s4.RelPerm.Alpha        0.501
   pfset Geom.s4.RelPerm.N            2.659

   pfset Geom.s5.RelPerm.Alpha        0.661
   pfset Geom.s5.RelPerm.N            2.659

   pfset Geom.s6.RelPerm.Alpha        1.122
   pfset Geom.s6.RelPerm.N            2.479

   pfset Geom.s7.RelPerm.Alpha        2.089
   pfset Geom.s7.RelPerm.N            2.318

   pfset Geom.s8.RelPerm.Alpha        0.832
   pfset Geom.s8.RelPerm.N            2.514

   pfset Geom.s9.RelPerm.Alpha        1.585
   pfset Geom.s9.RelPerm.N            2.413

Next we do the same thing for saturation (:ref:`Saturation`)
again using the ``VanGenuchten`` parameters Note that every geometry listed 
in ``Porosity.GeomNames`` must have values assigned.

::

   #-----------------------------------------------------------------------------
   # Saturation
   #-----------------------------------------------------------------------------
   pfset Phase.Saturation.Type               VanGenuchten
   pfset Phase.Saturation.GeomNames          "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 "

   pfset Geom.domain.Saturation.Alpha        3.5
   pfset Geom.domain.Saturation.N            2.
   pfset Geom.domain.Saturation.SRes         0.2
   pfset Geom.domain.Saturation.SSat         1.0

   pfset Geom.s1.Saturation.Alpha        3.548
   pfset Geom.s1.Saturation.N            4.162
   pfset Geom.s1.Saturation.SRes         0.000001
   pfset Geom.s1.Saturation.SSat         1.0

   pfset Geom.s2.Saturation.Alpha        3.467
   pfset Geom.s2.Saturation.N            2.738
   pfset Geom.s2.Saturation.SRes         0.000001
   pfset Geom.s2.Saturation.SSat         1.0

   pfset Geom.s3.Saturation.Alpha        2.692
   pfset Geom.s3.Saturation.N            2.445
   pfset Geom.s3.Saturation.SRes         0.000001
   pfset Geom.s3.Saturation.SSat         1.0

   pfset Geom.s4.Saturation.Alpha        0.501
   pfset Geom.s4.Saturation.N            2.659
   pfset Geom.s4.Saturation.SRes         0.000001
   pfset Geom.s4.Saturation.SSat         1.0

   pfset Geom.s5.Saturation.Alpha        0.661
   pfset Geom.s5.Saturation.N            2.659
   pfset Geom.s5.Saturation.SRes         0.000001
   pfset Geom.s5.Saturation.SSat         1.0

   pfset Geom.s6.Saturation.Alpha        1.122
   pfset Geom.s6.Saturation.N            2.479
   pfset Geom.s6.Saturation.SRes         0.000001
   pfset Geom.s6.Saturation.SSat         1.0

   pfset Geom.s7.Saturation.Alpha        2.089
   pfset Geom.s7.Saturation.N            2.318
   pfset Geom.s7.Saturation.SRes         0.000001
   pfset Geom.s7.Saturation.SSat         1.0

   pfset Geom.s8.Saturation.Alpha        0.832
   pfset Geom.s8.Saturation.N            2.514
   pfset Geom.s8.Saturation.SRes         0.000001
   pfset Geom.s8.Saturation.SSat         1.0

   pfset Geom.s9.Saturation.Alpha        1.585
   pfset Geom.s9.Saturation.N            2.413
   pfset Geom.s9.Saturation.SRes         0.000001
   pfset Geom.s9.Saturation.SSat         1.0

Phase sources allows you to add sources other than wells and boundaries,
but we do not have any so this key is constant, 0.0 over entire domain.

::

   #-----------------------------------------------------------------------------
   # Phase sources:
   #-----------------------------------------------------------------------------
   pfset PhaseSources.water.Type                         "Constant"
   pfset PhaseSources.water.GeomNames                    "domain"
   pfset PhaseSources.water.Geom.domain.Value            0.0

In this example we are using ParFlow ``CLM`` so we must provide some parameters 
for ``CLM`` (:ref:`CLM Solver Parameters`). Note 
that ``CLM`` will also require some additional inputs outside of the tcl script. 
Refer to ``/washita/clm_input/`` for examples of the ``CLM``, ``vegm`` 
and ``driver`` files. These inputs are also discussed briefly in :ref:`Defining a Real domain`.

::

   #----------------------------------------------------------------
   # CLM Settings:
   # ------------------------------------------------------------

First we specify that we will be using ``CLM`` as the land 
surface model and provide the name of a directory that outputs 
will be written to. For this example we do not need outputs 
for each processor or a binary output directory. Finally we 
set the dump interval to 1, indicating that we will be writing 
outputs for every time step. Note that this does not have to 
match the dump interval for ParFlow outputs. Recall that 
earlier we set the ParFlow dump interval to 24.

::

   pfset Solver.LSM                                      CLM
   pfset Solver.CLM.CLMFileDir                           "clm_output/"
   pfset Solver.CLM.Print1dOut                           False
   pfset Solver.BinaryOutDir                             False
   pfset Solver.CLM.CLMDumpInterval                      1

Next we specify the details of the meteorological forcing files 
that ``CLM`` will read. First we provide the name of the files 
and the directory they can be found in. Next we specify that 
we are using ``3D`` forcing files meaning that we have spatially 
distributed forcing with multiple time steps in every file. 
Therefore we must also specify the number of times steps 
(``MetFileNT``) in every file, in this case 24. Finally, 
we specify the initial value for the CLM counter.

::

   pfset Solver.CLM.MetFileName                          "NLDAS"
   pfset Solver.CLM.MetFilePath                          "../../NLDAS/"
   pfset Solver.CLM.MetForcing                           3D
   pfset Solver.CLM.MetFileNT                            24
   pfset Solver.CLM.IstepStart                           1

This last set of ``CLM`` parameters refers to the physical 
properties of the system. Refer to :ref:`CLM Solver Parameters` for details.

::

   pfset Solver.CLM.EvapBeta                             Linear
   pfset Solver.CLM.VegWaterStress                       Saturation
   pfset Solver.CLM.ResSat                               0.1
   pfset Solver.CLM.WiltingPoint                         0.12
   pfset Solver.CLM.FieldCapacity                        0.98
   pfset Solver.CLM.IrrigationType                       none

Next we set the initial conditions for the domain. In this example we
are using a pressure ``.pfb`` file that was obtained by spinning up 
the model in the workflow outlined in :ref:`Defining a Real domain`. 
Alternatively, the water table can be set to a constant value by 
changing the ``ICPressure.Type``. Again, the input file that is 
referenced here was was copied into the run directory at the top 
of this script.

::

   #---------------------------------------------------------
   # Initial conditions: water pressure
   #---------------------------------------------------------
   pfset ICPressure.Type                                 PFBFile
   pfset ICPressure.GeomNames                            domain
   pfset Geom.domain.ICPressure.RefPatch                   z-upper
   pfset Geom.domain.ICPressure.FileName                   press.init.pfb

Now we specify what outputs we would like written. In this example we
specify that we would like to write out ``CLM`` variables as well 
as ``Pressure`` and ``Saturation``. However, there are many options 
for this and you should change these options according to what type 
of analysis you will be performing on your results. A complete list 
of print options is provided in :ref:`Code Parameters`.

::

   #----------------------------------------------------------------
   # Outputs
   # ------------------------------------------------------------
   #Writing output (all pfb):
   pfset Solver.PrintSubsurfData                         False
   pfset Solver.PrintPressure                            True
   pfset Solver.PrintSaturation                          True
   pfset Solver.PrintMask                                True

   pfset Solver.WriteCLMBinary                           False
   pfset Solver.PrintCLM                                 True
   pfset Solver.WriteSiloSpecificStorage                 False
   pfset Solver.WriteSiloMannings                        False
   pfset Solver.WriteSiloMask                            False
   pfset Solver.WriteSiloSlopes                          False
   pfset Solver.WriteSiloSubsurfData                     False
   pfset Solver.WriteSiloPressure                        False
   pfset Solver.WriteSiloSaturation                      False
   pfset Solver.WriteSiloEvapTrans                       False
   pfset Solver.WriteSiloEvapTransSum                    False
   pfset Solver.WriteSiloOverlandSum                     False
   pfset Solver.WriteSiloCLM                             False

Next we specify the solver settings for the ParFlow
(:ref:`RE Solver Parameters`). First we turn 
on solver Richards and the terrain following grid. We turn off 
variable dz.

::

   #-----------------------------------------------------------------------------
   # Set solver parameters
   #-----------------------------------------------------------------------------
   # ParFlow Solution
   pfset Solver                                          Richards
   pfset Solver.TerrainFollowingGrid                     True
   pfset Solver.Nonlinear.VariableDz                     False

We then set the max solver settings and linear and nonlinear convergence
tolerance settings. The linear system will be solved to a norm of
:math:`10^{-8}` and the nonlinear system will be solved to less than
:math:`10^{-6}`. Of note in latter key block is the EtaChoice and that
we use the analytical Jacobian (*UseJacobian* = **True**). We are
using the *FullJacobian* preconditioner, which is a more robust approach
but is more expensive.

::

   pfset Solver.MaxIter                                  25000
   pfset Solver.Drop                                     1E-20
   pfset Solver.AbsTol                                   1E-8
   pfset Solver.MaxConvergenceFailures                   8
   pfset Solver.Nonlinear.MaxIter                        80
   pfset Solver.Nonlinear.ResidualTol                    1e-6

   pfset Solver.Nonlinear.EtaChoice                         EtaConstant
   pfset Solver.Nonlinear.EtaValue                          0.001
   pfset Solver.Nonlinear.UseJacobian                       True 
   pfset Solver.Nonlinear.DerivativeEpsilon                 1e-16
   pfset Solver.Nonlinear.StepTol				 			1e-30
   pfset Solver.Nonlinear.Globalization                     LineSearch
   pfset Solver.Linear.KrylovDimension                      70
   pfset Solver.Linear.MaxRestarts                           2

   pfset Solver.Linear.Preconditioner                       PFMG
   pfset Solver.Linear.Preconditioner.PCMatrixType     FullJacobian

This key is just for testing the Richards’ formulation, so we are not
using it.

::

   #-----------------------------------------------------------------------------
   # Exact solution specification for error calculations
   #-----------------------------------------------------------------------------
   pfset KnownSolution                                   NoKnownSolution

Next we distribute all the inputs as described by the keys in
:ref:`PFTCL Commands`. Note the slopes are 2D files, while the
rest of the ParFlow inputs are 3D so we need to alter the NZ accordingly
following example 4 in :ref:`common_pftcl`.

::

   #-----------------------------------------------------------------------------
   # Distribute inputs
   #-----------------------------------------------------------------------------
   pfset ComputationalGrid.NX                41 
   pfset ComputationalGrid.NY                41 
   pfset ComputationalGrid.NZ                1
   pfdist LW.slopex.pfb
   pfdist LW.slopey.pfb

   pfset ComputationalGrid.NX                41 
   pfset ComputationalGrid.NY                41 
   pfset ComputationalGrid.NZ                50 
   pfdist IndicatorFile_Gleeson.50z.pfb
   pfdist press.init.pfb

Now we run the simulation. Note that we use a tcl variable to set the
run name.

::

   #-----------------------------------------------------------------------------
   # Run Simulation
   #-----------------------------------------------------------------------------
   set runname "LW"
   puts $runname
   pfrun    $runname

All that is left is to undistribute files.

::

   #-----------------------------------------------------------------------------
   # Undistribute Files
   #-----------------------------------------------------------------------------
   pfundist $runname
   pfundist press.init.pfb
   pfundist LW.slopex.pfb
   pfundist LW.slopey.pfb
   pfundist IndicatorFile_Gleeson.50z.pfb

   puts "ParFlow run Complete"