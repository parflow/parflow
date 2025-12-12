.. _ParFlow Input Keys:

ParFlow Input Keys
==================

The basic idea behind ParFlow input is a simple database of keys. The database
contains entries which have a key and a value associated with that key. When ParFlow runs, it queries the database you
have created by key names to get the values you have specified.

The commands ``pfset`` in TCL or ``<runname>.Key=`` in Python are used to create the database entries. 
A simple ParFlow input script contains a long list of these commands that set key values.  Note that the ``<runname>`` is the name a user gives to their run, and is a unique identifier to organize the key database and to anchor the files ParFlow writes.

It should be noted that the keys are “dynamic” in that many are built up
from values of other keys. For example if you have two wells named
*northwell* and *southwell* then you will have to set some keys which
specify the parameters for each well. The keys are built up in a simple
sort of hierarchy.

The following sections contain a description of all of the keys used by
ParFlow. For an example of input files you can look at the ``test`` subdirectory 
of the ParFlow distribution. Looking over some examples should give you 
a good feel for how the file scripts are put together.

Each key entry has the form:

*type* **KeyName** default value Description

The “type” is one of integer, double, string, list. Integer and double
are IEEE numbers. String is a text string (for example, a filename).
Strings can contain spaces if you use the proper TCL syntax (i.e. using
double quotes). These types are standard TCL types. Lists are strings
but they indicate the names of a series of items. For example you might
need to specify the names of the geometries. You would do this using
space separated names (what we are calling a list) “layer1 layer2
layer3”.

The descriptions that follow are organized into functional areas. An
example for each database entry is given.

Note that units used for each physical quantity specified in the input
file must be consistent with units used for all other quantities. The
exact units used can be any consistent set as ParFlow does not assume
any specific set of units. However, it is up to the user to make sure
all specifications are indeed consistent.

.. _Input File Format Number:

Input File Format Number
~~~~~~~~~~~~~~~~~~~~~~~~

*integer* **FileVersion** no default This gives the value of the input
file version number that this file fits.

.. container:: list

   ::

      pfset FileVersion 4           ## TCL syntax

      <runname>.FileVersion = 4     ## Python syntax

As development of the ParFlow code continues, the input file format will
vary. We have thus included an input file format number as a way of
verifying that the correct format type is being used. The user can check
in the ``parflow/config/file_versions.h`` file to verify that the format 
number specified in the input file matches the defined value 
of  ``PFIN_VERSION``.

.. _Computing Topology:

Computing Topology
~~~~~~~~~~~~~~~~~~

This section describes how processors are assigned in order to solve the
domain in parallel. “P” allocates the number of processes to the
grid-cells in x. “Q” allocates the number of processes to the grid-cells
in y. “R” allocates the number of processes to the grid-cells in z.
Please note “R” should always be 1 if you are running with Solver
Richards :cite:p:`Jones-Woodward01` unless you’re running a
totally saturated domain (solver IMPES).

*integer* **Process.Topology.P** no default This assigns the process
splits in the *x* direction.

.. container:: list

   ::

      pfset Process.Topology.P        2   ## TCL syntax

      <runname>.Process.Topology.P = 2    ## Python syntax

*integer* **Process.Topology.Q** no default This assigns the process
splits in the *y* direction.

.. container:: list

   ::

      pfset Process.Topology.Q       1   ## TCL syntax

      <runname>.Process.Topology.Q = 1   ## Python syntax

*integer* **Process.Topology.R** no default This assigns the process
splits in the *z* direction.

.. container:: list

   ::

      pfset Process.Topology.R       1   ## TCL syntax

      <runname>.Process.Topology.R = 1   ## Python syntax

In addition, you can assign the computing topology when you initiate
your parflow script using tcl. You must include the topology allocation
when using tclsh and the parflow script.

Example Usage (in TCL):

::

   [from Terminal] tclsh default_single.tcl 2 1 1

   [At the top of default_single.tcl you must include the following]
   set NP  [lindex $argv 0]
   set NQ  [lindex $argv 1]

   pfset Process.Topology.P        $NP
   pfset Process.Topology.Q        $NQ
   pfset Process.Topology.R        1 

.. _Computational Grid:

Computational Grid
~~~~~~~~~~~~~~~~~~

The computational grid is briefly described in
:ref:`Defining the Problem`. The computational grid keys set the
bottom left corner of the domain to a specific point in space. If using
a ``.pfsol`` file, the bottom left corner location of the ``.pfsol`` file must
be the points designated in the computational grid. The user can also
assign the *x*, *y* and *z* location to correspond to a specific
coordinate system (i.e. UTM).

*double* **ComputationalGrid.Lower.X** no default This assigns the lower
*x* coordinate location for the computational grid.

.. container:: list

   ::

      pfset   ComputationalGrid.Lower.X  0.0       ## TCL syntax

      <runname>.ComputationalGrid.Lower.X = 0.0    ## Python syntax

*double* **ComputationalGrid.Lower.Y** no default This assigns the lower
*y* coordinate location for the computational grid.

.. container:: list

   ::

      pfset   ComputationalGrid.Lower.Y  0.0       ## TCL syntax

      <runname>.ComputationalGrid.Lower.Y = 0.0    ## Python syntax

*double* **ComputationalGrid.Lower.Z** no default This assigns the lower
*z* coordinate location for the computational grid.

.. container:: list

   ::

      pfset   ComputationalGrid.Lower.Z  0.0       ## TCL syntax

      <runname>.ComputationalGrid.Lower.Z = 0.0    ## Python syntax

*integer* **ComputationalGrid.NX** no default This assigns the number of
grid cells in the *x* direction for the computational grid.

.. container:: list

   ::
 
      pfset  ComputationalGrid.NX  10        ## TCL syntax

     <runname>.ComputationalGrid.NX = 10     ## Python syntax

*integer* **ComputationalGrid.NY** no default This assigns the number of
grid cells in the *y* direction for the computational grid.

.. container:: list

   ::

      pfset  ComputationalGrid.NY  10        ## TCL syntax

      <runname>.ComputationalGrid.NY = 10    ## Python syntax

*integer* **ComputationalGrid.NZ** no default This assigns the number of
grid cells in the *z* direction for the computational grid.

.. container:: list

   ::

      pfset  ComputationalGrid.NZ  10        ## TCL syntax

      <runname>.ComputationalGrid.NZ = 10    ## Python syntax

*real* **ComputationalGrid.DX** no default This defines the size of grid
cells in the *x* direction. Units are *L* and are defined by the units
of the hydraulic conductivity used in the problem.

.. container:: list

   ::

      pfset  ComputationalGrid.DX  10.0      ## TCL syntax

      <runname>.ComputationalGrid.DX = 10.0  ## Python syntax

*real* **ComputationalGrid.DY** no default This defines the size of grid
cells in the *y* direction. Units are *L* and are defined by the units
of the hydraulic conductivity used in the problem.

.. container:: list

   ::

      pfset  ComputationalGrid.DY  10.0         ## TCL syntax

      <runname>.ComputationalGrid.DY = 10.0     ## Python syntax

*real* **ComputationalGrid.DZ** no default This defines the size of grid
cells in the *z* direction. Units are *L* and are defined by the units
of the hydraulic conductivity used in the problem.

.. container:: list

   ::

      pfset  ComputationalGrid.DZ  1.0       ## TCL syntax

      <runname>.ComputationalGrid.DZ = 1.0   ## Python syntax

Example Usage (TCL):

::

   #---------------------------------------------------------
   # Computational Grid
   #---------------------------------------------------------
   pfset ComputationalGrid.Lower.X	-10.0
   pfset ComputationalGrid.Lower.Y     10.0
   pfset ComputationalGrid.Lower.Z	1.0

   pfset ComputationalGrid.NX		18
   pfset ComputationalGrid.NY		18
   pfset ComputationalGrid.NZ		8

   pfset ComputationalGrid.DX		8.0
   pfset ComputationalGrid.DY		10.0
   pfset ComputationalGrid.DZ		1.0

Example Usage (Python):

::

   #---------------------------------------------------------
   # Computational Grid
   #---------------------------------------------------------
   
   <runname>.ComputationalGrid.Lower.X	= -10.0
   <runname>.ComputationalGrid.Lower.Y = 10.0
   <runname>.ComputationalGrid.Lower.Z	= 1.0

   <runname>.ComputationalGrid.NX	= 18
   <runname>.ComputationalGrid.NY	= 18
   <runname>.ComputationalGrid.NZ	= 8

   <runname>.ComputationalGrid.DX   = 8.0
   <runname>.ComputationalGrid.DY	= 10.0
   <runname>.ComputationalGrid.DZ	= 1.0

*string* **UseClustering** True Run a clustering algorithm to create
boxes in index space for iteration. By default an octree representation
is used for iteration, this may result in iterating over many nodes in
the octree. The **UseClustering** key will run a clustering algorithm to
build a set of boxes for iteration.

This does not always have a significant impact on performance and the
clustering algorithm can be expensive to compute. For small problems and
short running problems clustering is not recommended. Long running
problems may or may not see a benefit. The result varies significantly
based on the geometries in the problem.

The Berger-Rigoutsos algorithm is currently used for clustering.

::

   pfset UseClustering False         ## TCL syntax

   <runname>.UseClustering = False     ## Python syntax

.. _Geometries:

Geometries
~~~~~~~~~~

Here we define all “geometrical” information needed by ParFlow. For
example, the domain (and patches on the domain where boundary conditions
are to be imposed), lithology or hydrostratigraphic units, faults,
initial plume shapes, and so on, are considered geometries.

This input section is a little confusing. Two items are being specified,
geometry inputs and geometries. A geometry input is a type of geometry
input (for example a box or an input file). A geometry input can contain
more than one geometry. A geometry input of type Box has a single
geometry (the square box defined by the extants of the two points). A
SolidFile input type can contain several geometries.

*list* **GeomInput.Names** no default This is a list of the geometry
input names which define the containers for all of the geometries
defined for this problem.

.. container:: list

   ::

      pfset GeomInput.Names    "solidinput indinput boxinput"     ## TCL syntax

      <runname>.GeomInput.Names = "solidinput indinput boxinput"  ## Python syntax

*string* **GeomInput.\ *geom_input_name*.InputType** no default This
defines the input type for the geometry input with *geom_input_name*.
This key must be one of: **SolidFile, IndicatorField**, **Box**.

.. container:: list

   ::
 
      pfset GeomInput.solidinput.InputType  "SolidFile"        ## TCL syntax

      <runname>.GeomInput.solidinput.InputType  = "SolidFile"  ## Python syntax

*list* **GeomInput.\ *geom_input_name*.GeomNames** no default This is a
list of the names of the geometries defined by the geometry input. For a
geometry input type of Box, the list should contain a single geometry
name. For the SolidFile geometry type this should contain a list with
the same number of gemetries as were defined using GMS. The order of
geometries in the SolidFile should match the names. For IndicatorField
types you need to specify the value in the input field which matches the
name using GeomInput.\ *geom_input_name*.Value.

.. container:: list

   ::

      pfset GeomInput.solidinput.GeomNames "domain bottomlayer \
                                            middlelayer toplayer"  ## TCL syntax
      
      <runname>.GeomInput.solidinput.GeomNames = "domain bottomlayer middlelayer toplayer"  ## Python syntax

*string* **GeomInput.\ *geom_input_name*.Filename** no default For
IndicatorField and SolidFile geometry inputs this key specifies the
input filename which contains the field or solid information.

.. container:: list

   ::

      pfset GeomInput.solidinput.FileName   "ocwd.pfsol"       ## TCL syntax

      <runname>.GeomInput.solidinput.FileName = "ocwd.pfsol"   ## Python syntax

*integer* **GeomInput.\ *geometry_input_name*.Value** no default For
IndicatorField geometry inputs you need to specify the mapping between
values in the input file and the geometry names. The named geometry will
be defined wherever the input file is equal to the specified value.

.. container:: list

   ::

      pfset GeomInput.sourceregion.Value   11      ## TCL syntax

      <runname>.GeomInput.sourceregion.Value = 11  ## Python syntax

For box geometries you need to specify the location of the box. This is
done by defining two corners of the the box.

*double* **Geom.\ *box_geom_name*.Lower.X** no default This gives the
lower X real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Lower.X   -1.0         ## TCL syntax

      <runname>.Geom.background.Lower.X = -1.0     ## Python syntax

*double* **Geom.\ *box_geom_name*.Lower.Y** no default This gives the
lower Y real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Lower.Y   -1.0         ## TCL syntax

      <runname>.Geom.background.Lower.Y = -1.0     ## Python syntax

*double* **Geom.\ *box_geom_name*.Lower.Z** no default This gives the
lower Z real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Lower.Z   -1.0         ## TCL syntax

      <runname>.Geom.background.Lower.Z = -1.0     ## Python syntax

*double* **Geom.\ *box_geom_name*.Upper.X** no default This gives the
upper X real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Upper.X   151.0        ## TCL syntax

      <runname>.Geom.background.Upper.X = 151.0    ## Python syntax

*double* **Geom.\ *box_geom_name*.Upper.Y** no default This gives the
upper Y real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Upper.Y   171.0        ## TCL syntax

      <runname>.Geom.background.Upper.Y = 171.0    ## Python syntax

*double* **Geom.\ *box_geom_name*.Upper.Z** no default This gives the
upper Z real space coordinate value of the previously specified box
geometry of name *box_geom_name*.

.. container:: list

   ::

      pfset Geom.background.Upper.Z   11.0         ## TCL syntax

      <runname>.Geom.background.Upper.Z = 11.0     ## Python syntax

*list* **Geom.\ *geom_name*.Patches** no default Patches are defined on
the surfaces of geometries. Currently you can only define patches on Box
geometries and on the the first geometry in a SolidFile. For a Box the
order is fixed (left right front back bottom top) but you can name the
sides anything you want.

For SolidFiles the order is printed by the conversion routine that
converts GMS to SolidFile format.

.. container:: list

   ::

      pfset Geom.background.Patches   "left right front back bottom top"      ## TCL syntax

      <runname>.Geom.background.Patches = "left right front back bottom top"  ## Python syntax   

Here is an example geometry input section which has three geometry
inputs (TCL).

.. container:: list

   ::

      #---------------------------------------------------------
      # The Names of the GeomInputs
      #---------------------------------------------------------
      pfset GeomInput.Names 			"solidinput indinput boxinput"
      #
      # For a solid file geometry input type you need to specify the names
      # of the gemetries and the filename
      #

      pfset GeomInput.solidinput.InputType	"SolidFile"

      # The names of the geometries contained in the solid file. Order is
      # important and defines the mapping. First geometry gets the first name. 
      pfset GeomInput.solidinput.GeomNames	"domain"
      #
      # Filename that contains the geometry
      #

      pfset GeomInput.solidinput.FileName 	"ocwd.pfsol"

      #
      # An indicator field is a 3D field of values. 
      # The values within the field can be mapped 
      # to ParFlow geometries. Indicator fields must match the
      # computation grid exactly!
      #

      pfset GeomInput.indinput.InputType     "IndicatorField"
      pfset GeomInput.indinput.GeomNames    	"sourceregion concenregion"
      pfset GeomInput.indinput.FileName		"ocwd.pfb"

      #
      # Within the indicator.pfb file, assign the values to each GeomNames
      # 
      pfset GeomInput.sourceregion.Value 	11
      pfset GeomInput.concenregion.Value 	12

      #
      # A box is just a box defined by two points.
      #

      pfset GeomInput.boxinput.InputType	"Box"
      pfset GeomInput.boxinput.GeomName   "background"
      pfset Geom.background.Lower.X 		-1.0
      pfset Geom.background.Lower.Y 		-1.0
      pfset Geom.background.Lower.Z 		-1.0
      pfset Geom.background.Upper.X 		151.0
      pfset Geom.background.Upper.Y 		171.0
      pfset Geom.background.Upper.Z 		11.0

      #
      # The patch order is fixed in the .pfsol file, but you 
      # can call the patch name anything you 
      # want (i.e. left right front back bottom top)
      #

      pfset Geom.domain.Patches           "z-upper x-lower y-lower \
                                            	x-upper y-upper z-lower"

.. _Reservoirs:

Reservoirs
~~~~~~~~~~
Here we define reservoirs for the model. Currently reservoirs have only been tested on domains
where the top of domain lies at the top of the grid. This applies to all box domains and some 
terrain following grid domains. The format for this section of input
is:

*string* **Reservoirs.Names** no default This key specifies the names of the
reservoirs for which input data will be given.

.. container:: list

   ::

      Reservoirs.Names "reservoir_1 reservoir_2 reservoir_3"

*double* **Reservoirs.\ *reservoir_name*.Release_X** no default This key specifies 
the x location of where the reservoir releases water. This cell will always be placed
on the domain surface.

*double* **Reservoirs.\ *reservoir_name*.Release_Y** no default This key specifies 
the y location of where the reservoir releases water. This cell will always be placed
on the domain surface.

*double* **Reservoirs.\ *reservoir_name*.Intake_X** no default This key specifies 
the x location of where the reservoir intakes water. This cell will always be placed
on the domain surface.

*double* **Reservoirs.\ *reservoir_name*.Intake_Y** no default This key specifies 
the y location of where the reservoir intakes water. This cell will always be placed
on the domain surface.

.. This value is set as an int because bools do not work with the table reader right now
*int* **Reservoirs.\ *reservoir_name*.Has_Secondary_Intake_Cell** no default This key specifies if 
the reservoir has a secondary intake cell, with 0 evaluating to false and 1 evaluating to true. This
cell will always be placed on the domain surface.

*double* **Reservoirs.\ *reservoir_name*.Secondary_Intake_X** no default This optional key 
specifies the x location of where the reservoir's secondary intake cell intakes water. This 
cell will always be placed on the domain surface. This key is only used when the reservoir has
a secondary intake cell, in which case it is required.

*double* **Reservoirs.\ *reservoir_name*.Secondary_Intake_Y** no default This optional key 
specifies the y location of where the reservoir's secondary intake cell intakes water. This 
cell will always be placed on the domain surface. This key is only used when the reservoir has
a secondary intake cell, in which case it is required.

*double* **Reservoirs.\ *reservoir_name*.Min_Release_Storage** no default This key specifies 
the storage amount below which the reservoir will stop releasing water. Has units [L\ :sup:`3`].

*double* **Reservoirs.\ *reservoir_name*.Max_Storage** no default This key specifies a reservoirs 
maximum storage. If storage rises above this value, a reservoir will release extra water if necessary
to get back down to this amount by the next timestep. Has units [L\ :sup:`3`]

*double* **Reservoirs.\ *reservoir_name*.Storage** no default This key specifies the amount of water 
stored in the reservoir as a volume. Has same length units as the problem domain i.e. if domain is 
sized in meters this will be in m\ :sup:`3`.

*double* **Reservoirs.\ *reservoir_name*.Release_Rate** no default [Type: double] This key specifies 
the rate in volume/time [L\ :sup:`3` \ :sup:`-1`] that the reservoir release water. The amount of time over which 
this amount is released is independent of solver timestep size.

Overland_Flow_Solver

*string* **Reservoirs.Overland_Flow_Solver** no default This key specifies which overland flow 
condition is used in the domain so that the slopes aroundthe reservoirs can be adjusted properly. 
Supported Options are **OverlandFlow** and **OverlandKinematic**.

.. _Timing Information:

Timing Information
~~~~~~~~~~~~~~~~~~

The data given in the timing section describe all the “temporal”
information needed by ParFlow. The data items are used to describe time
units for later sections, sequence iterations in time, indicate actual
starting and stopping values and give instructions on when data is
printed out.

*double* **TimingInfo.BaseUnit** no default This key is used to indicate
the base unit of time for entering time values. All time should be
expressed as a multiple of this value. This should be set to the
smallest interval of time to be used in the problem. For example, a base
unit of “1” means that all times will be integer valued. A base unit of
“0.5” would allow integers and fractions of 0.5 to be used for time
input values.

The rationale behind this restriction is to allow time to be discretized
on some interval to enable integer arithmetic to be used when
computing/comparing times. This avoids the problems associated with real
value comparisons which can lead to events occurring at different
timesteps on different architectures or compilers.

This value is also used when describing “time cycling data” in,
currently, the well and boundary condition sections. The lengths of the
cycles in those sections will be integer multiples of this value,
therefore it needs to be the smallest divisor which produces an integral
result for every “real time” cycle interval length needed.

.. container:: list

   ::

      pfset TimingInfo.BaseUnit      1.0     ## TCL syntax

      <runname>.TimingInfo.BaseUnit = 1.0    ## Python syntax

*integer* **TimingInfo.StartCount** no default This key is used to
indicate the time step number that will be associated with the first
advection cycle in a transient problem. The value **-1** indicates that
advection is not to be done. The value **0** indicates that advection
should begin with the given initial conditions. 

.. container:: list

   ::

      pfset TimingInfo.StartCount    0       ## TCL syntax

      <runname>.TimingInfo.StartCount = 0    ## Python syntax

*double* **TimingInfo.StartTime** no default This key is used to
indicate the starting time for the simulation.

.. container:: list

   ::

      pfset TimingInfo.StartTime     0.0     ## TCL syntax

      <runname>.TimingInfo.StartTime = 0.0   ## Python syntax

*double* **TimingInfo.StopTime** no default This key is used to indicate
the stopping time for the simulation.

.. container:: list

   ::

      pfset TimingInfo.StopTime      100.0      ## TCL syntax

      <runname>.TimingInfo.StopTime = 100.0     ## Python syntax

*double* **TimingInfo.DumpInterval** no default This key is the real
time interval at which time-dependent output should be written. A value
of **0** will produce undefined behavior. If the value is negative,
output will be dumped out every :math:`n` time steps, where :math:`n` is
the absolute value of the integer part of the value.

.. container:: list

   ::

      pfset TimingInfo.DumpInterval  10.0       ## TCL syntax

      <runname>.TimingInfo.DumpInterval = 10.0  ## Python syntax

*integer* **TimingInfo.DumpIntervalExecutionTimeLimit** 0 This key is
used to indicate a wall clock time to halt the execution of a run. At
the end of each dump interval the time remaining in the batch job is
compared with the user supplied value, if remaining time is less than or
equal to the supplied value the execution is halted. Typically used when
running on batch systems with time limits to force a clean shutdown near
the end of the batch job. Time units is seconds, a value of **0** (the
default) disables the check.

Currently only supported on SLURM based systems, “–with-slurm” must be
specified at configure time to enable.

.. container:: list

   ::

      pfset TimingInfo.DumpIntervalExecutionTimeLimit 360         ## TCL syntax

      <runname>.TimingInfo.DumpIntervalExecutionTimeLimit = 360   ## Python syntax

For *Richards’ equation cases only* input is collected for time step
selection. Input for this section is given as follows:

*list* **TimeStep.Type** no default This key must be one of:
**Constant** or **Growth**. The value **Constant** defines a constant
time step. The value **Growth** defines a time step that starts as
:math:`dt_0` and is defined for other steps as
:math:`dt^{new} = \gamma dt^{old}` such that :math:`dt^{new} \leq 
dt_{max}` and :math:`dt^{new} \geq dt_{min}`.

.. container:: list

   ::

      pfset TimeStep.Type      "Constant"      ## TCL syntax

      <runname>.TimeStep.Type = "Constant"   ## Python syntax

*double* **TimeStep.Value** no default This key is used only if a
constant time step is selected and indicates the value of the time step
for all steps taken.

.. container:: list

   ::

      pfset TimeStep.Value      0.001     ## TCL syntax

      <runanme>.TimeStep.Value = 0.001    ## Python syntax

*double* **TimeStep.InitialStep** no default This key specifies the
initial time step :math:`dt_0` if the **Growth** type time step is
selected.

.. container:: list

   ::

      pfset TimeStep.InitialStep    0.001       ## TCL syntax

      <runname>.TimeStep.InitialStep = 0.001    ## Python syntax

*double* **TimeStep.GrowthFactor** no default This key specifies the
growth factor :math:`\gamma` by which a time step will be multiplied to
get the new time step when the **Growth** type time step is selected.

.. container:: list

   ::

      pfset TimeStep.GrowthFactor      1.5      ## TCL syntax

      <runname>.TimeStep.GrowthFactor = 1.5     ## Python syntax

*double* **TimeStep.MaxStep** no default This key specifies the maximum
time step allowed, :math:`dt_{max}`, when the **Growth** type time step
is selected.

.. container:: list

   ::

      pfset TimeStep.MaxStep      86400      ## TCL syntax

      <runname>.TimeStep.MaxStep = 86400     ## Python syntax

*double* **TimeStep.MinStep** no default This key specifies the minimum
time step allowed, :math:`dt_{min}`, when the **Growth** type time step
is selected.

.. container:: list

   ::

      pfset TimeStep.MinStep      1.0e-3     ## TCL syntax

      <runname>.TimeStep.MinStep = 1.0e-3    ## Python syntax

Here is a detailed example of how timing keys might be used in a
simulation.

.. container:: list

   ::

      ## TCL example

      #-----------------------------------------------------------------------------
      # Setup timing info [hr]
      # 8760 hours in a year. Dumping files every 24 hours. Hourly timestep
      #-----------------------------------------------------------------------------
      pfset TimingInfo.BaseUnit		   1.0
      pfset TimingInfo.StartCount		0
      pfset TimingInfo.StartTime		   0.0
      pfset TimingInfo.StopTime		   8760.0
      pfset TimingInfo.DumpInterval 	-24

      ## Timing constant example
      pfset TimeStep.Type			      "Constant"
      pfset TimeStep.Value			      1.0

      ## Timing growth example
      pfset TimeStep.Type			      "Growth"
      pfset TimeStep.InitialStep		   0.0001
      pfset TimeStep.GrowthFactor		1.4
      pfset TimeStep.MaxStep			   1.0
      pfset TimeStep.MinStep			   0.0001


      ## Python Example

      #-----------------------------------------------------------------------------
      # Setup timing info [hr]
      # 8760 hours in a year. Dumping files every 24 hours. Hourly timestep
      #-----------------------------------------------------------------------------
      <runname>.TimingInfo.BaseUnit = 1.0
      <runname>.TimingInfo.StartCount = 0
      <runname>.TimingInfo.StartTime = 0.0
      <runname>.TimingInfo.StopTime = 8760.0
      <runname>.TimingInfo.DumpInterval = -24

      ## Timing constant example
      <runname>.TimeStep.Type	= "Constant"
      <runname>.TimeStep.Value = 1.0

      ## Timing growth example
      <runname>.TimeStep.Type	= "Growth"
      <runname>.TimeStep.InitialStep = 0.0001
      <runname>.TimeStep.GrowthFactor = 1.4
      <runname>.TimeStep.MaxStep	= 1.0
      <runname>.TimeStep.MinStep	= 0.0001

.. _Time Cycles:

Time Cycles
~~~~~~~~~~~

The data given in the time cycle section describes how time intervals are
created and named to be used for time-dependent boundary and well
information needed by ParFlow. All the time cycles are synched to the
**TimingInfo.BaseUnit** key described above and are *integer
multipliers* of that value.

*list* **Cycle.Names** no default This key is used to specify the named
time cycles to be used in a simulation. It is a list of names and each
name defines a time cycle and the number of items determines the total
number of time cycles specified. Each named cycle is described using a
number of keys defined below.

.. container:: list

   ::

      pfset Cycle.Names "constant onoff"        ## TCL syntax

      <runname>.Cycle.Names = "constant onoff"  ## Python syntax

*list* **Cycle.\ *cycle_name*.Names** no default This key is used to
specify the named time intervals for each cycle. It is a list of names
and each name defines a time interval when a specific boundary condition
is applied and the number of items determines the total number of
intervals in that time cycle.

.. container:: list

   ::

      pfset Cycle.onoff.Names "on off"          ## TCL syntax

      <runname>.Cycle.onoff.Names = "on off"    ## Python syntax

*integer* **Cycle.\ *cycle_name.interval_name*.Length** no default This
key is used to specify the length of a named time intervals. It is an
*integer multiplier* of the value set for the **TimingInfo.BaseUnit**
key described above. The total length of a given time cycle is the sum
of all the intervals multiplied by the base unit.

.. container:: list

   ::

      pfset Cycle.onoff.on.Length       10     ## TCL syntax

      <runname>.Cycle.onoff.on.Length = 10     ## Python syntax

*integer* **Cycle.\ *cycle_name*.Repeat** no default This key is used to
specify the how many times a named time interval repeats. A positive
value specifies a number of repeat cycles a value of -1 specifies that
the cycle repeat for the entire simulation.

.. container:: list

   ::

      pfset Cycle.onoff.Repeat       -1

      <runname>.Cycle.onoff.Repeat = -1

Here is a detailed example of how time cycles might be used in a
simulation.

.. container:: list

   ::

      ## TCL example

      #-----------------------------------------------------------------------------
      # Time Cycles
      #-----------------------------------------------------------------------------
      pfset Cycle.Names                      "constant rainrec"
      pfset Cycle.constant.Names             "alltime"
      pfset Cycle.constant.alltime.Length    8760
      pfset Cycle.constant.Repeat            -1

      # Creating a rain and recession period for the rest of year
      pfset Cycle.rainrec.Names              "rain rec"
      pfset Cycle.rainrec.rain.Length	      10
      pfset Cycle.rainrec.rec.Length	      8750
      pfset Cycle.rainrec.Repeat             -1

      ## Python example

      #-----------------------------------------------------------------------------
      # Time Cycles
      #-----------------------------------------------------------------------------
      <runname>.Cycle.Names = "constant rainrec"
      <runname>.Cycle.constant.Names = "alltime"
      <runname>.Cycle.constant.alltime.Length = 8760
      <runname>.Cycle.constant.Repeat = -1

      # Creating a rain and recession period for the rest of year
      <runname>.Cycle.rainrec.Names	= "rain rec"
      <runname>.Cycle.rainrec.rain.Length	= 10
      <runname>.Cycle.rainrec.rec.Length = 8750
      <runname>.Cycle.rainrec.Repeat = -1

.. _Domain:

Domain
~~~~~~

The domain may be represented by any of the solid types in :ref:`Geometries` above that allow the definition of surface
patches. These surface patches are used to define boundary conditions in :ref:`Boundary Conditions: Pressure` and :ref:`Boundary Conditions: Saturation` below. Subsequently, it
is required that the union (or combination) of the defined surface
patches equal the entire domain surface. NOTE: This requirement is NOT
checked in the code.

*string* **Domain.GeomName** no default This key specifies which of the
named geometries is the problem domain.

.. container:: list

   ::

      pfset Domain.GeomName    "domain"        ## TCL syntax

      <runname>.Domain.GeomName = "domain"   ## Python syntax

.. _Phases and Contaminants:

Phases and Contaminants
~~~~~~~~~~~~~~~~~~~~~~~

*list* **Phase.Names** no default This specifies the names of phases to
be modeled. Currently only 1 or 2 phases may be modeled.

.. container:: list

   ::

      pfset Phase.Names    "water"        ## TCL syntax

      <runname>.Phase.Names = "water"     ## Python syntax

*list* **Contaminants.Names** no default This specifies the names of
contaminants to be advected.

.. container:: list

   ::

      pfset Contaminants.Names   "tce"       ## TCL syntax

      <runname>.Contaminants.Names = "tce"   ## Python syntax

.. _Gravity, Phase Density and Phase Viscosity:

Gravity, Phase Density and Phase Viscosity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*double* **Gravity** no default Specifies the gravity constant to be
used.  

.. container:: list

   ::

      pfset Gravity	1.0         ## TCL syntax

      <runname>.Gravity	= 1.0    ## Python syntax

*string* **Phase.\ *phase_name*.Density.Type** no default This key
specifies whether density will be a constant value or if it will be
given by an equation of state of the form :math:`(rd)exp(cP)`, where
:math:`P` is pressure, :math:`rd` is the density at atmospheric
pressure, and :math:`c` is the phase compressibility constant. This key
must be either **Constant** or **EquationOfState**.

.. container:: list

   ::

      pfset Phase.water.Density.Type	 "Constant"       ## TCL syntax

      <runname>.Phase.water.Density.Type = "Constant"    ## Python syntax

*double* **Phase.\ *phase_name*.Density.Value** no default This
specifies the value of density if this phase was specified to have a
constant density value for the phase *phase_name*.

.. container:: list

   ::

      pfset Phase.water.Density.Value   1.0        ## TCL syntax

     <runname>.Phase.water.Density.Value = 1.0     ## Python syntax

*double* **Phase.\ *phase_name*.Density.ReferenceDensity** no default
This key specifies the reference density if an equation of state density
function is specified for the phase *phase_name*.

.. container:: list

   ::

      pfset Phase.water.Density.ReferenceDensity   1.0      ## TCL syntax

      <runname>.Phase.water.Density.ReferenceDensity = 1.0  ## Python syntax

*double* **Phase.\ *phase_name*.Density.CompressibilityConstant** no
default This key specifies the phase compressibility constant if an
equation of state density function is specified for the phase
*phase|-name*.

.. container:: list

   ::

      pfset Phase.water.Density.CompressibilityConstant   1.0        ## TCL syntax

      <runname>.Phase.water.Density.CompressibilityConstant = 1.0    ## Python syntax

*string* **Phase.\ *phase_name*.Viscosity.Type** Constant This key
specifies whether viscosity will be a constant value. Currently, the
only choice for this key is **Constant**.

.. container:: list

   ::

      pfset Phase.water.Viscosity.Type   "Constant"         ## TCL syntax

      <runname>.Phase.water.Viscosity.Type = "Constant"     ## Python syntax

*double* **Phase.\ *phase_name*.Viscosity.Value** no default This
specifies the value of viscosity if this phase was specified to have a
constant viscosity value.

.. container:: list

   ::

      pfset Phase.water.Viscosity.Value    1.0     ## TCL syntax

      <runname>.Phase.water.Viscosity.Value = 1.0  ## Python syntax

.. _Chemical Reactions:

Chemical Reactions
~~~~~~~~~~~~~~~~~~

*double* **Contaminants.\ *contaminant_name*.Degradation.Value** no
default This key specifies the half-life decay rate of the named
contaminant, *contaminant_name*. At present only first order decay
reactions are implemented and it is assumed that one contaminant cannot
decay into another.

.. container:: list

   ::

      pfset Contaminants.tce.Degradation.Value        0.0      ## TCL syntax

      <runname>.Contaminants.tce.Degradation.Value  = 0.0      ## Python syntax

.. _Permeability:

Permeability
~~~~~~~~~~~~

In this section, permeability property values are assigned to grid
points within geometries (specified in :ref:`Geometries` above)
using one of the methods described below. Permeabilities are assumed to
be a diagonal tensor with entries given as,

.. math::

   \left( 
   \begin{array}{ccc}
   k_x({\bf x}) & 0 & 0 \\
   0 & k_y({\bf x}) & 0 \\
   0 & 0 & k_z({\bf x}) 
   \end{array} \right) 
   K({\bf x}),

where :math:`K({\bf x})` is the permeability field given below.
Specification of the tensor entries (:math:`k_x, k_y` and :math:`k_z`)
will be given at the end of this section.

The random field routines (*turning bands* and *pgs*) can use
conditioning data if the user so desires. It is not necessary to use
conditioning as ParFlow automatically defaults to not use conditioning
data, but if conditioning is desired, the following key should be set:

*string* **Perm.Conditioning.FileName** “NA” This key specifies the name
of the file that contains the conditioning data. The default string
**NA** indicates that conditioning data is not applicable.

.. container:: list

   ::

      pfset Perm.Conditioning.FileName   "well_cond.txt"       ## TCL syntax

      <runname>.Perm.Conditioning.FileName = "well_cond.txt"   ## Python syntax

The file that contains the conditioning data is a simple ascii file
containing points and values. The format is:

.. container:: list

   ::

      nlines
      x1 y1 z1 value1
      x2 y2 z2 value2
      .  .  .    .
      .  .  .    .
      .  .  .    .
      xn yn zn valuen

The value of *nlines* is just the number of lines to follow in the file,
which is equal to the number of data points.

The variables *xi,yi,zi* are the real space coordinates (in the units
used for the given parflow run) of a point at which a fixed permeability
value is to be assigned. The variable *valuei* is the actual
permeability value that is known.

Note that the coordinates are not related to the grid in any way.
Conditioning does not require that fixed values be on a grid. The PGS
algorithm will map the given value to the closest grid point and that
will be fixed. This is done for speed reasons. The conditioned turning
bands algorithm does not do this; conditioning is done for every grid
point using the given conditioning data at the location given. Mapping
to grid points for that algorithm does not give any speedup, so there is
no need to do it.

NOTE: The given values should be the actual measured values - adjustment
in the conditioning for the lognormal distribution that is assumed is
taken care of in the algorithms.

The general format for the permeability input is as follows:

*list* **Geom.Perm.Names** no default This key specifies all of the
geometries to which a permeability field will be assigned. These
geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset GeomInput.Names   "background domain concen_region"      ## TCL syntax

      <runname>.GeomInput.Names = "background domain concen_region"  ## Python syntax

*string* **Geom.geometry_name.Perm.Type** no default This key specifies
which method is to be used to assign permeability data to the named
geometry, *geometry_name*. It must be either **Constant**,
**TurnBands**, **ParGuass**, or **PFBFile**. The **Constant** value
indicates that a constant is to be assigned to all grid cells within a
geometry. The **TurnBand** value indicates that Tompson’s Turning Bands
method is to be used to assign permeability data to all grid cells
within a geometry :cite:p:`TAG89`. The **ParGauss** value
indicates that a Parallel Gaussian Simulator method is to be used to
assign permeability data to all grid cells within a geometry. The
**PFBFile** value indicates that premeabilities are to be read from a 
ParFlow 3D binary file. Both the Turning Bands and Parallel Gaussian
Simulators generate a random field with correlation lengths in the
:math:`3` spatial directions given by :math:`\lambda_x`,
:math:`\lambda_y`, and :math:`\lambda_z` with the geometric mean of the
log normal field given by :math:`\mu` and the standard deviation of the
normal field given by :math:`\sigma`. In generating the field both of
these methods can be made to stratify the data, that is follow the top
or bottom surface. The generated field can also be made so that the data
is normal or log normal, with or without bounds truncation. Turning
Bands uses a line process, the number of lines used and the resolution
of the process can be changed as well as the maximum normalized
frequency :math:`K_{\rm max}` and the normalized frequency increment
:math:`\delta K`. The Parallel Gaussian Simulator uses a search
neighborhood, the number of simulated points and the number of
conditioning points can be changed.

.. container:: list

   ::

      pfset Geom.background.Perm.Type   "Constant"       ## TCL syntax

      <runname>.Geom.background.Perm.Type = "Constant"   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.Value** no default This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset Geom.domain.Perm.Value   1.0        ## TCL syntax

      <runname>.Geom.domain.Perm.Value = 1.0    ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.LambdaX** no default This key
specifies the x correlation length, :math:`\lambda_x`, of the field
generated for the named geometry, *geometry_name*, if either the Turning
Bands or Parallel Gaussian Simulator are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.LambdaX   200.0       ## TCL syntax

      <runname>.Geom.domain.Perm.LambdaX = 200.0   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.LambdaY** no default This key
specifies the y correlation length, :math:`\lambda_y`, of the field
generated for the named geometry, *geometry_name*, if either the Turning
Bands or Parallel Gaussian Simulator are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.LambdaY   200.0       ## TCL syntax

      <runname>.Geom.domain.Perm.LambdaY = 200.0   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.LambdaZ** no default This key
specifies the z correlation length, :math:`\lambda_z`, of the field
generated for the named geometry, *geometry_name*, if either the Turning
Bands or Parallel Gaussian Simulator are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.LambdaZ   10.0        ## TCL syntax

      <runname>.Geom.domain.Perm.LambdaZ = 10.0    ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.GeomMean** no default This key
specifies the geometric mean, :math:`\mu`, of the log normal field
generated for the named geometry, *geometry_name*, if either the Turning
Bands or Parallel Gaussian Simulator are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.GeomMean   4.56       ## TCL syntax

      <runname>.Geom.domain.Perm.GeomMean = 4.56   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.Sigma** no default This key
specifies the standard deviation, :math:`\sigma`, of the normal field
generated for the named geometry, *geometry_name*, if either the Turning
Bands or Parallel Gaussian Simulator are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.Sigma   2.08       ## TCL syntax

      <runname>.Geom.domain.Perm.Sigma = 2.08   ## Python syntax

*integer* **Geom.\ *geometry_name*.Perm.Seed** 1 This key specifies the
initial seed for the random number generator used to generate the field
for the named geometry, *geometry_name*, if either the Turning Bands or
Parallel Gaussian Simulator are chosen. This number must be positive.

.. container:: list

   ::

      pfset Geom.domain.Perm.Seed   1        ## TCL syntax

      <runname>.Geom.domain.Perm.Seed = 1    ## Python syntax

*integer* **Geom.\ *geometry_name*.Perm.NumLines** 100 This key
specifies the number of lines to be used in the Turning Bands algorithm
for the named geometry, *geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.NumLines   100        ## TCL syntax

      <runname>.Geom.domain.Perm.NumLines = 100    ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.RZeta** 5.0 This key specifies
the resolution of the line processes, in terms of the minimum grid
spacing, to be used in the Turning Bands algorithm for the named
geometry, *geometry_name*. Large values imply high resolution.

.. container:: list

   ::

      pfset Geom.domain.Perm.RZeta   5.0        ## TCL syntax

      <runname>.Geom.domain.Perm.RZeta = 5.0    ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.KMax** 100.0 This key specifies
the the maximum normalized frequency, :math:`K_{\rm max}`, to be used in
the Turning Bands algorithm for the named geometry, *geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.KMax   100.0       ## TCL syntax

      <runname>.Geom.domain.Perm.KMax = 100.0   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.DelK** 0.2 This key specifies the
normalized frequency increment, :math:`\delta K`, to be used in the
Turning Bands algorithm for the named geometry, *geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.DelK   0.2         ## TCL syntax

      <runname>.Geom.domain.Perm.DelK = 0.2     ## Python syntax

*integer* **Geom.\ *geometry_name*.Perm.MaxNPts** no default This key
sets limits on the number of simulated points in the search neighborhood
to be used in the Parallel Gaussian Simulator for the named geometry,
*geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.MaxNPts   5        ## TCL syntax

      <runname>.Geom.domain.Perm.MaxNPts = 5    ## Python syntax

*integer* **Geom.\ *geometry_name*.Perm.MaxCpts** no default This key
sets limits on the number of external conditioning points in the search
neighborhood to be used in the Parallel Gaussian Simulator for the named
geometry, *geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.MaxCpts   200      ## TCL syntax

      <runname>.Geom.domain.Perm.MaxCpts = 200  ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.LogNormal** "LogTruncated" The
key specifies when a normal, log normal, truncated normal or truncated
log normal field is to be generated by the method for the named
geometry, *geometry_name*. This value must be one of **Normal**,
**Log**, **NormalTruncated** or **LogTruncate** and can be used with
either Turning Bands or the Parallel Gaussian Simulator.

.. container:: list

   ::

      pfset Geom.domain.Perm.LogNormal   "LogTruncated"        ## TCL syntax

      <runname>.Geom.domain.Perm.LogNormal = "LogTruncated"    ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.StratType** "Bottom" This key
specifies the stratification of the permeability field generated by the
method for the named geometry, *geometry_name*. The value must be one of
**Horizontal**, **Bottom** or **Top** and can be used with either the
Turning Bands or the Parallel Gaussian Simulator.

.. container:: list

   ::

      pfset Geom.domain.Perm.StratType  "Bottom"         ## TCL syntax

      <runname>.Geom.domain.Perm.StratType = "Bottom"    ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.LowCutoff** no default This key
specifies the low cutoff value for truncating the generated field for
the named geometry, *geometry_name*, when either the NormalTruncated or
LogTruncated values are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.LowCutoff   0.0       ## TCL syntax

      <runname>.Geom.domain.Perm.LowCutoff = 0.0   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.HighCutoff** no default This key
specifies the high cutoff value for truncating the generated field for
the named geometry, *geometry_name*, when either the NormalTruncated or
LogTruncated values are chosen.

.. container:: list

   ::

      pfset Geom.domain.Perm.HighCutoff   100.0       ## TCL syntax

      <runname>.Geom.domain.Perm.HighCutoff = 100.0   ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.FileName** no default This key
specifies that permeability values for the specified geometry,
*geometry_name*, are given according to a user-supplied description in
the “ParFlow binary” file whose filename is given as the value. For a
description of the ParFlow Binary file format, see
:ref:`ParFlow Binary Files (.pfb)`. The ParFlow binary file
associated with the named geometry must contain a collection of
permeability values corresponding in a one-to-one manner to the entire
computational grid. That is to say, when the contents of the file are
read into the simulator, a complete permeability description for the
entire domain is supplied. Only those values associated with
computational cells residing within the geometry (as it is represented
on the computational grid) will be copied into data structures used
during the course of a simulation. Thus, the values associated with
cells outside of the geounit are irrelevant. For clarity, consider a
couple of different scenarios. For example, the user may create a file
for each geometry such that appropriate permeability values are given
for the geometry and “garbage" values (e.g., some flag value) are given
for the rest of the computational domain. In this case, a separate
binary file is specified for each geometry. Alternatively, one may place
all values representing the permeability field on the union of the
geometries into a single binary file. Note that the permeability values
must be represented in precisely the same configuration as the
computational grid. Then, the same file could be specified for each
geounit in the input file. Or, the computational domain could be
described as a single geouint (in the ParFlow input file) in which case
the permeability values would be read in only once.

.. container:: list

   ::

      pfset Geom.domain.Perm.FileName "domain_perm.pfb"        ## TCL syntax

      <runname>.Geom.domain.Perm.FileName = "domain_perm.pfb"  ## Python syntax

*string* **Perm.TensorType** no default This key specifies whether the
permeability tensor entries :math:`k_x, k_y` and :math:`k_z` will be
specified as three constants within a set of regions covering the domain
or whether the entries will be specified cell-wise by files. The choices
for this key are **TensorByGeom** and **TensorByFile**.

.. container:: list

   ::

      pfset Perm.TensorType     "TensorByGeom"     ## TCL syntax

      <runname>.Perm.TensorType = "TensorByGeom"   ## Python syntax

*string* **Geom.Perm.TensorByGeom.Names** no default This key specifies
all of the geometries to which permeability tensor entries will be
assigned. These geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset Geom.Perm.TensorByGeom.Names   "background domain"       ## TCL syntax

      <runname>.Geom.Perm.TensorByGeom.Names = "background domain"   ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.TensorValX** no default This key
specifies the value of :math:`k_x` for the geometry given by
*geometry_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorValX   1.0         ## TCL syntax

      <runname>.Geom.domain.Perm.TensorValX = 1.0     ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.TensorValY** no default This key
specifies the value of :math:`k_y` for the geometry given by
*geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorValY   1.0         ## TCL syntax

      <runname>.Geom.domain.Perm.TensorValY = 1.0     ## Python syntax

*double* **Geom.\ *geometry_name*.Perm.TensorValZ** no default This key
specifies the value of :math:`k_z` for the geometry given by
*geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorValZ   1.0      ## TCL syntax

      <runname>.Geom.domain.Perm.TensorValZ = 1.0  ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.TensorFileX** no default This key
specifies that :math:`k_x` values for the specified geometry,
*geometry_name*, are given according to a user-supplied description in
a ParFlow 3D binary file whose filename is given as the value. The only
choice for the value of *geometry_name* is “domain”.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorFileX   "perm_x.pfb"         ## TCL syntax

      <runname>.Geom.domain.Perm.TensorByFileX = "perm_x.pfb"    ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.TensorFileY** no default This key
specifies that :math:`k_y` values for the specified geometry,
*geometry_name*, are given according to a user-supplied description in
a ParFlow 3D binary file whose filename is given as the value. The only
choice for the value of *geometry_name* is “domain”.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorFileY   "perm_y.pfb"         ## TCL syntax

      <runname>.Geom.domain.Perm.TensorByFileY = "perm_y.pfb"     ## Python syntax

*string* **Geom.\ *geometry_name*.Perm.TensorFileZ** no default This key
specifies that :math:`k_z` values for the specified geometry,
*geometry_name*, are given according to a user-supplied description in
a ParFlow 3D binary file whose filename is given as the value. The only
choice for the value of *geometry_name* is “domain”.

.. container:: list

   ::

      pfset Geom.domain.Perm.TensorFileZ   "perm_z.pfb"         ## TCL syntax

      <runname>.Geom.domain.Perm.TensorByFileZ = "perm_z.pfb"     ## Python syntax

.. _Porosity:

Porosity
~~~~~~~~

Here, porosity values are assigned within geounits (specified in
:ref:`Geometries` above) using one of the methods described
below.

The format for this section of input is:

*list* **Geom.Porosity.GeomNames** no default This key specifies all of
the geometries on which a porosity will be assigned. These geometries
must cover the entire computational domain.

.. container:: list

   ::

      pfset Geom.Porosity.GeomNames   "background"          ## TCL syntax

      <runname>.Geom.Porosity.GeomNames = "background"      ## Python syntax

*string* **Geom.\ *geometry_name*.Porosity.Type** no default This key
specifies which method is to be used to assign porosity data to the
named geometry, *geometry_name*. The choices for this key are **Constant**
and **PFBFile**. **Constant** indicates that a constant is to be assigned to all
grid cells within a geometry. The **PFBFile** value indicates that porosity values
are to be read from a ParFlow 3D binary file.

.. container:: list

   ::

      pfset Geom.background.Porosity.Type   "Constant"         ## TCL syntax

      <runname>.Geom.background.Porosity.Type = "Constant"     ## Python syntax

*double* **Geom.\ *geometry_name*.Porosity.Value** no default This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to **Constant**.

.. container:: list

   ::

      pfset Geom.domain.Porosity.Value   1.0       ## TCL syntax

      <runname>.Geom.domain.Porosity.Value = 1.0   ## Python syntax

*string* **Geom.\ *geometry_name*.Porosity.FileName** no default This key
specifies that porosity values for the specified geometry,
*geometry_name*, are given according to a user-supplied description in
a ParFlow 3D binary file whose filename is given as the value.

.. container:: list

   ::

      pfset Geom.domain.Porosity.FileName   "porosity.pfb"         ## TCL syntax

      <runname>.Geom.domain.Porosity.FileName = "porosity.pfb"     ## Python syntax

.. _Specific Storage:

Specific Storage
~~~~~~~~~~~~~~~~

Here, specific storage (:math:`S_s` in Equation
:eq:`richard`) values are assigned within geounits
(specified in :ref:`Geometries` above) using one of the methods
described below.

The format for this section of input is:

*list* **Specific Storage.GeomNames** no default This key specifies all
of the geometries on which a different specific storage value will be
assigned. These geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset SpecificStorage.GeomNames       "domain"     ## TCL syntax

      <runname>.SpecificStorage.GeomNames = "domain"     ## Python syntax

*string* **SpecificStorage.Type** no default This key specifies which
method is to be used to assign specific storage data. The only choice
currently available is **Constant** which indicates that a constant is
to be assigned to all grid cells within a geometry.

.. container:: list

   ::

      pfset SpecificStorage.Type        "Constant"       ## TCL syntax

      <runname>.SpecificStorage.Type = "Constant"        ## Python syntax

*double* **Geom.\ *geometry_name*.SpecificStorage.Value** no default
This key specifies the value assigned to all points in the named
geometry, *geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset Geom.domain.SpecificStorage.Value 1.0e-4        ## TCL syntax

      <runname>.Geom.domain.SpecificStorage.Value = 1.0e-4  ## Python syntax

.. _dZ Multipliers:

dZMultipliers
~~~~~~~~~~~~~

Here, dZ multipliers (:math:`\delta Z * m`) values are assigned within
geounits (specified in :ref:`Geometries` above) using one of the
methods described below.

The format for this section of input is:

*string* **Solver.Nonlinear.VariableDz** False This key specifies
whether dZ multipliers are to be used, the default is False. The default
indicates a false or non-active variable dz and each layer thickness is
1.0 [L].

.. container:: list

   ::

      pfset Solver.Nonlinear.VariableDz     True      ## TCL syntax

      <runnname>.Solver.Nonlinear.VariableDz = True   ## Python syntax

*list* **dzScale.GeomNames** no default This key specifies which problem
domain is being applied a variable dz subsurface. These geometries must
cover the entire computational domain.

.. container:: list

   ::

      pfset dzScale.GeomNames "domain"          ## TCL syntax

      <runname>.dzScale.GeomNames = "domain"    ## Python syntax

*string* **dzScale.Type** no default This key specifies which method is
to be used to assign variable vertical grid spacing. The choices
currently available are **Constant** which indicates that a constant is
to be assigned to all grid cells within a geometry, **nzList** which
assigns all layers of a given model to a list value, and **PFBFile**
which reads in values from a distributed ParFlow 3D binary file.

.. container:: list

   ::

      pfset dzScale.Type       "Constant"       ## TCL syntax

      <runname>.dzScale.Type = "Constant"       ## Python syntax

*list* **Specific dzScale.GeomNames** no default This key specifies all
of the geometries on which a different dz scaling value will be
assigned. These geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset dzScale.GeomNames       "domain"    ## TCL syntax

      <runname>.dzScale.GeomNames = "domain"    ## Python syntax

*double* **Geom.\ *geometry_name*.dzScale.Value** no default This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset Geom.domain.dzScale.Value 1.0          ## TCL syntax

      <runname>.Geom.domain.dzScale.Value = 1.0    ## Python syntax

*string* **Geom.\ *geometry_name*.dzScale.FileName** no default This key
specifies file to be read in for variable dz values for the given
geometry, *geometry_name*, if the type was set to **PFBFile**.

.. container:: list

   ::

      pfset Geom.domain.dzScale.FileName       "vardz.pfb"       ## TCL syntax 

      <runname>.Geom.domain.dzScale.FileName = "vardz.pfb"       ## Python syntax

*integer* **dzScale.nzListNumber** no default This key indicates the
number of layers with variable dz in the subsurface. This value is the
same as the *ComputationalGrid.NZ* key.

.. container:: list

   ::

      pfset dzScale.nzListNumber  10         ## TCL syntax

      <runname>.dzScale.nzListNumber = 10    ## Python syntax

*double* **Cell.\ *nzListNumber*.dzScale.Value** no default This key
assigns the thickness of each layer defined by nzListNumber. ParFlow
assigns the layers from the bottom-up (i.e. the bottom of the domain is
layer 0, the top is layer NZ-1). The total domain depth
(*Geom.domain.Upper.Z*) does not change with variable dz. The layer
thickness is calculated by *ComputationalGrid.DZ \*dZScale*. *Note that* in Python a number is not an allowed character for a variable.
Thus we proceed the layer number with an underscore "_" as shown in the example below.

.. container:: list

   ::

      pfset Cell.0.dzScale.Value 1.0         ## TCL syntax 

      <runname>.Cell._0.dzScale.Value = 1.0  ## Python syntax

Example Usage (TCL):

.. container:: list

   ::


      #--------------------------------------------
      # Variable dz Assignments
      #------------------------------------------
      # Set VariableDz to be true
      # Indicate number of layers (nzlistnumber), which is the same as nz
      # (1) There is nz*dz = total depth to allocate,  
      # (2) Each layer’s thickness is dz*dzScale, and
      # (3) Assign the layer thickness from the bottom up.
      # In this example nz = 5; dz = 10; total depth 40;
      # Layers 	Thickness [m]
      # 0 		15 			Bottom layer
      # 1		15
      # 2		5
      # 3		4.5			
      # 4 		0.5			Top layer
      pfset Solver.Nonlinear.VariableDz     True
      pfset dzScale.GeomNames            "domain"
      pfset dzScale.Type            "nzList"
      pfset dzScale.nzListNumber       5
      pfset Cell.0.dzScale.Value 1.5
      pfset Cell.1.dzScale.Value 1.5
      pfset Cell.2.dzScale.Value 0.5
      pfset Cell.3.dzScale.Value 0.45
      pfset Cell.4.dzScale.Value 0.05

Example Usage (Python):

.. container:: list

   ::


      #--------------------------------------------
      # Variable dz Assignments
      #------------------------------------------
      # Set VariableDz to be true
      # Indicate number of layers (nzlistnumber), which is the same as nz
      # (1) There is nz*dz = total depth to allocate,  
      # (2) Each layer’s thickness is dz*dzScale, and
      # (3) Assign the layer thickness from the bottom up.
      # In this example nz = 5; dz = 10; total depth 40;
      # Layers 	Thickness [m]
      # 0 		15 			Bottom layer
      # 1		15
      # 2		5
      # 3		4.5			
      # 4 		0.5			Top layer
      <runname>.Solver.Nonlinear.VariableDz = True
      <runname>.dzScale.GeomNames = "domain"
      <runname>.dzScale.Type = "nzList"
      <runname>.dzScale.nzListNumber = 5
      <runname>.Cell._0.dzScale.Value = 1.5
      <runname>.Cell._1.dzScale.Value = 1.5
      <runname>.Cell._2.dzScale.Value = 0.5
      <runname>.Cell._3.dzScale.Value = 0.45
      <runname>.Cell._4.dzScale.Value = 0.05

.. _Flow Barrier Keys:

Flow Barriers
~~~~~~~~~~~~~

Here, the values for Flow Barriers described in :ref:`FB` can be
input. These are only available with Solver **Richards** and can be
specified in X, Y or Z directions independently using ParFlow binary files. These
barriers are applied at the cell face at the location :math:`i+1/2`. That
is a value of :math:`FB_x` specified at :math:`i` will be applied to the
cell face at :math:`i+1/2` or between cells :math:`i` and
:math:`i+1`. The same goes for :math:`FB_y` (:math:`j+1/2`) and
:math:`FB_z` (:math:`k+1/2`). The flow barrier values are unitless and
multiply the flux equation as shown in :eq:`qFBx`.

The format for this section of input is:

*string* **Solver.Nonlinear.FlowBarrierX** False This key specifies
whether Flow Barriers are to be used in the X direction, the default is
False. The default indicates a false or :math:`FB_x` value of one [-]
everywhere in the domain.

::

   pfset Solver.Nonlinear.FlowBarrierX       True     ## TCL syntax

   <runname>.Solver.Nonlinear.FlowBarrierX = True     ## Python syntax

*string* **Solver.Nonlinear.FlowBarrierY** False This key specifies
whether Flow Barriers are to be used in the Y direction, the default is
False. The default indicates a false or :math:`FB_y` value of one [-]
everywhere in the domain.

::

   pfset Solver.Nonlinear.FlowBarrierY       True       ## TCL syntax

   <runname>.Solver.Nonlinear.FlowBarrierY = True       ## Python syntax

*string* **Solver.Nonlinear.FlowBarrierZ** False This key specifies
whether Flow Barriers are to be used in the Z direction, the default is
False. The default indicates a false or :math:`FB_z` value of one [-]
everywhere in the domain.

::

   pfset Solver.Nonlinear.FlowBarrierZ       True     ## TCL syntax

   <runname>.Solver.Nonlinear.FlowBarrierZ = True     ## Python syntax

*string* **FBx.Type** no default This key specifies which method is to
be used to assign flow barriers in X. The only choice currently
available is **PFBFile** which reads in values from a distributed ParFlow binary
file.

::

   pfset FBx.Type       "PFBFile"      ## TCL syntax

   <runname>.FBx.Type = "PFBFile"      ## Python syntax

*string* **FBy.Type** no default This key specifies which method is to
be used to assign flow barriers in Y. The only choice currently
available is **PFBFile** which reads in values from a distributed pfb
file.

::

   pfset FBy.Type       "PFBFile"    ## TCL syntax

   <runname>.FBy.Type = "PFBFile"    ## Python syntax

*string* **FBz.Type** no default This key specifies which method is to
be used to assign flow barriers in Z. The only choice currently
available is **PFBFile** which reads in values from a distributed ParFlow binary 
file.

::

   pfset FBz.Type       "PFBFile"    ## TCL syntax

   <runname>.FBz.Type = "PFBFile"    ## Python syntax

The Flow Barrier values may be read in from a ParFlow binary file over the entire
domain. This is done as follows:

*string* **Geom.domain.FBx.FileName** no default This key specifies file
to be read in for the X flow barrier values for the domain, if the type
was set to **PFBFile**.

::

   pfset Geom.domain.FBx.FileName       "Flow_Barrier_X.pfb"      ## TCL syntax

   <runname>.Geom.domain.FBx.FileName = "Flow_Barrier_X.pfb"      ## Python syntax

*string* **Geom.domain.FBy.FileName** no default This key specifies file
to be read in for the Y flow barrier values for the domain, if the type
was set to **PFBFile**.

::

   pfset Geom.domain.FBy.FileName      "Flow_Barrier_Y.pfb"     ## TCL syntax

   <runname>.Geom.domain.FBy.FileName = "Flow_Barrier_Y.pfb"    ## Python syntax

*string* **Geom.domain.FBz.FileName** no default This key specifies file
to be read in for the Z flow barrier values for the domain, if the type
was set to **PFBFile**.

::

   pfset Geom.domain.FBz.FileName  "Flow_Barrier_Z.pfb"        ## TCL syntax

   <runname>.Geom.domain.FBz.FileName = "Flow_Barrier_Z.pfb"   ## Python syntax


.. _Manning's Roughness Values:

Manning’s Roughness Values
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, Manning’s roughness values (:math:`n` in Equations
:eq:`manningsx` and :eq:`manningsy`) are assigned to the upper boundary
of the domain using one of the methods described below.

The format for this section of input is:

*list* **Mannings.GeomNames** no default This key specifies all of the
geometries on which a different Mannings roughness value will be
assigned. Mannings values may be assigned by **PFBFile** or as
**Constant** by geometry. These geometries must cover the entire upper
surface of the computational domain.

.. container:: list

   ::

      pfset Mannings.GeomNames       "domain"    ## TCL syntax

      <runname>.Mannings.GeomNames = "domain"    ## Python syntax

*string* **Mannings.Type** no default This key specifies which method is
to be used to assign Mannings roughness data. The choices currently
available are **Constant** which indicates that a constant is to be
assigned to all grid cells within a geometry and **PFBFile** which
indicates that all values are read in from a distributed, grid-based
ParFlow 2D binary file.

.. container:: list

   ::

      pfset Mannings.Type     "Constant"     ## TCL syntax

      <runname>.Mannings.Type = "Constant"   ## Python syntax

*double* **Mannings.Geom.\ *geometry_name*.Value** no default This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset Mannings.Geom.domain.Value 5.52e-6        ## TCL syntax

      <runname>.Mannings.Geom.domain.Value = 5.52e-6  ## Python syntax

*double* **Mannings.FileName** no default This key specifies the value
assigned to all points be read in from a ParFlow 2D binary file.

.. container:: list

   ::

      pfset Mannings.FileName "roughness.pfb"         ## TCL syntax

      <runname>.Mannings.FileName = "roughness.pfb"   ## Python syntax

Complete example of setting Mannings roughness :math:`n` values by
geometry:

.. container:: list

   ::
    
    ## TCL example
    pfset Mannings.Type "Constant"
    pfset Mannings.GeomNames "domain"
    pfset Mannings.Geom.domain.Value 5.52e-6


    ## Python example
    <runname>.Mannings.Type = "Constant"
    <runname>.Mannings.GeomNames = "domain"
    <runname>.Mannings.Geom.domain.Value = 5.52e-6

.. _Topographical Slopes:

Topographical Slopes
~~~~~~~~~~~~~~~~~~~~

Here, topographical slope values (:math:`S_{f,x}` and :math:`S_{f,y}` in
Equations :eq:`manningsx` and :eq:`manningsy`) are assigned to the upper boundary
of the domain using one of the methods described below. Note that due to
the negative sign in these equations :math:`S_{f,x}` and :math:`S_{f,y}`
take a sign in the direction *opposite* of the direction of the slope.
That is, negative slopes point "downhill" and positive slopes "uphill".

The format for this section of input is:

*list* **ToposlopesX.GeomNames** no default This key specifies all of
the geometries on which a different :math:`x` topographic slope values
will be assigned. Topographic slopes may be assigned by **PFBFile** or
as **Constant** by geometry. These geometries must cover the entire
upper surface of the computational domain.

.. container:: list

   ::

      pfset ToposlopesX.GeomNames       "domain"      ## TCL syntax

      <runname>.ToposlopesX.GeomNames = "domain"      ## Python syntax

*list* **ToposlopesY.GeomNames** no default This key specifies all of
the geometries on which a different :math:`y` topographic slope values
will be assigned. Topographic slopes may be assigned by **PFBFile** or
as **Constant** by geometry. These geometries must cover the entire
upper surface of the computational domain.

.. container:: list

   ::

      pfset ToposlopesY.GeomNames       "domain"      ## TCL syntax

      <runname>.ToposlopesY.GeomNames = "domain"      ## Python syntax

*string* **ToposlopesX.Type** no default This key specifies which method
is to be used to assign topographic slopes. The choices currently
available are **Constant** which indicates that a constant is to be
assigned to all grid cells within a geometry and **PFBFile** which
indicates that all values are read in from a distributed, grid-based
ParFlow 2D binary file.

.. container:: list

   ::

      pfset ToposlopesX.Type "Constant"         ## TCL syntax

      <runname>.ToposlopesX.Type = "Constant"   ## Python syntax

*double* **ToposlopesX.Geom.\ *geometry_name*.Value** no default This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset ToposlopeX.Geom.domain.Value       0.001     ## TCL syntax

      <runname>.ToposlopeX.Geom.domain.Value = 0.001     ## Python syntax

*double* **ToposlopesX.FileName** no default This key specifies the
value assigned to all points be read in from a ParFlow 2D binary file.

.. container:: list

   ::

      pfset TopoSlopesX.FileName       "lw.1km.slope_x.pfb"    ## TCL syntax

      <runname>.TopoSlopesX.FileName = "lw.1km.slope_x.pfb"    ## Python syntax

*double* **ToposlopesY.FileName** no default This key specifies the
value assigned to all points be read in from a ParFlow 2D binary file.

.. container:: list

   ::

      pfset TopoSlopesY.FileName       "lw.1km.slope_y.pfb"    ## TCL syntax

      <runname>.TopoSlopesY.FileName = "lw.1km.slope_y.pfb"    ## Python syntax

Example of setting :math:`x` and :math:`y` slopes by geometry:

.. container:: list

   ::

      pfset TopoSlopesX.Type "Constant"
      pfset TopoSlopesX.GeomNames "domain"
      pfset TopoSlopesX.Geom.domain.Value 0.001

      pfset TopoSlopesY.Type "Constant"
      pfset TopoSlopesY.GeomNames "domain"
      pfset TopoSlopesY.Geom.domain.Value -0.001

Example of setting :math:`x` and :math:`y` slopes by file:

.. container:: list

   ::

      pfset TopoSlopesX.Type "PFBFile"
      pfset TopoSlopesX.GeomNames "domain"
      pfset TopoSlopesX.FileName "lw.1km.slope_x.pfb"

      pfset TopoSlopesY.Type "PFBFile"
      pfset TopoSlopesY.GeomNames "domain"
      pfset TopoSlopesY.FileName "lw.1km.slope_y.pfb"


.. _Channelwidths:

Channelwidths
~~~~~~~~~~~~~
These keys are in development. They have been added to the pftools
Python interface and can be set to read and print channewidth values,
but have not yet been integrated with overland flow.

Here, channel width values are assigned to the upper boundary
of the domain using one of the methods described below.

The format for this section of input is:

*list* **Solver.Nonlinear.ChannelWidthExistX** False This key specifies
whether a channelwidthX input is provided.

*list* **Solver.Nonlinear.ChannelWidthExistY** False This key specifies
whether a channelwidthY input is provided.

*list* **ChannelWidthX.GeomNames** no default This key specifies all of
the geometries on which a different ChannelWidthX values
will be assigned. ChannelWidthX may be assigned by **PFBFile** or **NCFile**
as **Constant** by geometry. These geometries must cover the entire
upper surface of the computational domain.

.. container:: list

   ::

      pfset ChannelWidthX.GeomNames       "domain"      ## TCL syntax

      <runname>.ChannelWidthX.GeomNames = "domain"      ## Python syntax

*list* **ChannelWidthY.GeomNames** no default This key specifies all of
the geometries on which a different ChannelWidthY values
will be assigned. ChannelWidthX may be assigned by **PFBFile** or **NCFile**
as **Constant** by geometry. These geometries must cover the entire
upper surface of the computational domain.

.. container:: list

   ::

      pfset ChannelWidthY.GeomNames       "domain"      ## TCL syntax

      <runname>.ChannelWidthY.GeomNames = "domain"      ## Python syntax


*string* **ChannelWidthX.Type** Constant This key specifies which method
is to be used to assign ChannelWidthX. The choices currently
available are **Constant** which indicates that a constant is to be
assigned to all grid cells within a geometry, **PFBFile** which
indicates that all values are read in from a distributed, grid-based
ParFlow 2D binary file and **NCFile** which indicates that all values
are read in from a netcdf file.

.. container:: list

   ::

      pfset ChannelWidthX.Type "Constant"         ## TCL syntax

      <runname>.ChannelWidthX.Type = "Constant"   ## Python syntax

*double* **ChannelWidthX.Geom.\ *geometry_name*.Value** 0.0 This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset ChannelWidthX.Geom.domain.Value       100     ## TCL syntax

      <runname>.ChannelWidthX.Geom.domain.Value = 100     ## Python syntax

*double* **ChannelWidthX.FileName** no default This key specifies the
value assigned to all points be read in from a ParFlow 2D binary file or
a netcdf file.

.. container:: list

   ::

      pfset ChannelWidthX.FileName       "channel_x.pfb"    ## TCL syntax

      <runname>.ChannelWidthX.FileName = "channel_x.pfb"    ## Python syntax

*string* **ChannelWidthY.Type** Constant This key specifies which method
is to be used to assign ChannelWidthY. The choices currently
available are **Constant** which indicates that a constant is to be
assigned to all grid cells within a geometry, **PFBFile** which
indicates that all values are read in from a distributed, grid-based
ParFlow 2D binary file and **NCFile** which indicates that all values
are read in from a netcdf file.

.. container:: list

   ::

      pfset ChannelWidthY.Type "Constant"         ## TCL syntax

      <runname>.ChannelWidthY.Type = "Constant"   ## Python syntax

*double* **ChannelWidthY.Geom.\ *geometry_name*.Value** 0.0 This key
specifies the value assigned to all points in the named geometry,
*geometry_name*, if the type was set to constant.

.. container:: list

   ::

      pfset ChannelWidthY.Geom.domain.Value       100     ## TCL syntax

      <runname>.ChannelWidthY.Geom.domain.Value = 100     ## Python syntax

*double* **ChannelWidthY.FileName** no default This key specifies the
value assigned to all points be read in from a ParFlow 2D binary file or
a netcdf file.

.. container:: list

   ::

      pfset ChannelWidthY.FileName       "channel_y.pfb"    ## TCL syntax

      <runname>.ChannelWidthY.FileName = "channel_y.pfb"    ## Python syntax


Example of setting :math:`x` and :math:`y` channelwidths by geometry:

.. container:: list

   ::

      pfset ChannelWidthX.Type "Constant"
      pfset ChannelWidthX.GeomNames "domain"
      pfset ChannelWidthX.Geom.domain.Value 100

      pfset ChannelWidthY.Type "Constant"
      pfset ChannelWidthY.GeomNames "domain"
      pfset ChannelWidthY.Geom.domain.Value 100

Example of setting :math:`x` and :math:`y` channelwidths by file:

.. container:: list

   ::

      pfset ChannelWidthX.Type "PFBFile"
      pfset ChannelWidthX.GeomNames "domain"
      pfset ChannelWidthX.FileName "channel_x.pfb"

      pfset ChannelWidthY.Type "PFBFile"
      pfset ChannelWidthY.GeomNames "domain"
      pfset ChannelWidthY.FileName "channel_y.pfb"


.. _Retardation:

Retardation
~~~~~~~~~~~

Here, retardation values are assigned for contaminants within geounits
(specified in `Geometries` above) using one of the
functions described below. The format for this section of input is:

*list* **Geom.Retardation.GeomNames** no default This key specifies all
of the geometries to which the contaminants will have a retardation
function applied.

.. container:: list

   ::

      pfset GeomInput.Names       "background"     ## TCL syntax

      <runname>.GeomInput.Names = "background"     ## Python syntax

*string*
**Geom.\ *geometry_name*.\ *contaminant_name*.Retardation.Type** no
default This key specifies which function is to be used to compute the
retardation for the named contaminant, *contaminant_name*, in the named
geometry, *geometry_name*. The only choice currently available is
**Linear** which indicates that a simple linear retardation function is
to be used to compute the retardation.

.. container:: list

   ::

      pfset Geom.background.tce.Retardation.Type   "Linear"       ## TCL syntax

      <runname>.Geom.background.tce.Retardation.Type = "Linear"   ## Python syntax

*double*
**Geom.\ *geometry_name*.\ *contaminant_name*.Retardation.Value** no
default This key specifies the distribution coefficient for the linear
function used to compute the retardation of the named contaminant,
*contaminant_name*, in the named geometry, *geometry_name*. The value
should be scaled by the density of the material in the geometry.

.. container:: list

   ::

      pfset Geom.domain.Retardation.Value   0.2          ## TCL syntax

      <runname>.Geom.domain.Retardation.Value = 0.2      ## Python syntax

Full Multiphase Mobilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we define phase mobilities by specifying the relative permeability
function. Input is specified differently depending on what problem is
being specified. For full multi-phase problems, the following input keys
are used. See the next section for the correct Richards’ equation input
format.

*string* **Phase.\ *phase_name*.Mobility.Type** no default This key
specifies whether the mobility for *phase_name* will be a given constant
or a polynomial of the form, :math:`(S - S_0)^{a}`, where :math:`S` is
saturation, :math:`S_0` is irreducible saturation, and :math:`a` is some
exponent. The possibilities for this key are **Constant** and
**Polynomial**.

.. container:: list

   ::

      pfset Phase.water.Mobility.Type   "Constant"       ## TCL syntax

      <runname>.Phase.water.Mobility.Type = "Constant"   ## Python syntax

*double* **Phase.\ *phase_name*.Mobility.Value** no default This key
specifies the constant mobility value for phase *phase_name*.

.. container:: list

   ::

      pfset Phase.water.Mobility.Value   1.0       ## TCL syntax

      <runname>.Phase.water.Mobility.Value = 1.0   ## Python syntax

*double* **Phase.\ *phase_name*.Mobility.Exponent** 2.0 This key
specifies the exponent used in a polynomial representation of the
relative permeability. Currently, only a value of :math:`2.0` is allowed
for this key.

.. container:: list

   ::

      pfset Phase.water.Mobility.Exponent   2.0          ## TCL syntax

      <runname>.Phase.water.Mobility.Exponent = 2.0      ## Python syntax

*double* **Phase.\ *phase_name*.Mobility.IrreducibleSaturation** 0.0
This key specifies the irreducible saturation used in a polynomial
representation of the relative permeability. Currently, only a value of
0.0 is allowed for this key.

.. container:: list

   ::

      pfset Phase.water.Mobility.IrreducibleSaturation   0.0      ## TCL syntax

      <runname>.Phase.water.Mobility.IrreducibleSaturation = 0.0  ## Python syntax

.. _Richards RelPerm:

Richards’ Equation Relative Permeabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following keys are used to describe relative permeability input for
the Richards’ equation implementation. They will be ignored if a full
two-phase formulation is used.

*string* **Phase.RelPerm.Type** no default This key specifies the type
of relative permeability function that will be used on all specified
geometries. Note that only one type of relative permeability may be used
for the entire problem. However, parameters may be different for that
type in different geometries. For instance, if the problem consists of
three geometries, then **VanGenuchten** may be specified with three
different sets of parameters for the three different geometries.
However, once **VanGenuchten** is specified, one geometry cannot later
be specified to have **Data** as its relative permeability. The possible
values for this key are **Constant, VanGenuchten, Haverkamp, Data,** and
**Polynomial**.

.. container:: list

   ::

      pfset Phase.RelPerm.Type   "Constant"        ## TCL syntax

      <runname>.Phase.RelPerm.Type = "Constant"    ## Python syntax

The various possible functions are defined as follows. The **Constant**
specification means that the relative permeability will be constant on
the specified geounit. The **VanGenuchten** specification means that the
relative permeability will be given as a Van Genuchten function
:cite:p:`VanGenuchten80` with the form,

.. math::

   \begin{aligned}
   k_r(p) = \frac{(1 - \frac{(\alpha p)^{n-1}}{(1 + (\alpha p)^n)^m})^2}
   {(1 + (\alpha p)^n)^{m/2}},\end{aligned}

where :math:`\alpha` and :math:`n` are soil parameters and
:math:`m = 1 - 1/n`, on each region. The **Haverkamp** specification
means that the relative permeability will be given in the following form
:cite:p:`Haverkamp-Vauclin81`,

.. math::

   \begin{aligned}
   k_r(p) = \frac{A}{A + p^{\gamma}},\end{aligned}

where :math:`A` and :math:`\gamma` are soil parameters, on each region.
The **Data** specification is currently unsupported but will later mean
that data points for the relative permeability curve will be given and
ParFlow will set up the proper interpolation coefficients to get values
between the given data points. The **Polynomial** specification defines
a polynomial relative permeability function for each region of the form,

.. math::

   \begin{aligned}
   k_r(p) = \sum_{i=0}^{degree} c_ip^i.\end{aligned}

*list* **Phase.RelPerm.GeomNames** no default This key specifies the
geometries on which relative permeability will be given. The union of
these geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset Phase.RelPerm.Geonames   "domain"      ## TCL syntax

      <runname>.Phase.RelPerm.Geonames = "domain"  ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.Value** no default This key
specifies the constant relative permeability value on the specified
geometry.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.Value    0.5       ## TCL syntax

      <runname>.Geom.domain.RelPerm.Value = 0.5    ## Python syntax

*integer* **Phase.RelPerm.VanGenuchten.File** 0 This key specifies
whether soil parameters for the VanGenuchten function are specified in a
ParFlow 3D binary file or by region. The options are either 0 for specification by
region, or 1 for specification in a file. Note that either all
parameters are specified in files (each has their own input file) or
none are specified by files. Parameters specified by files are:
:math:`\alpha` and N.

.. container:: list

   ::

      pfset Phase.RelPerm.VanGenuchten.File   1       ## TCL syntax

      <runname>.Phase.RelPerm.VanGenuchten.File = 1   ## Python syntax

*string* **Geom.\ *geom_name*.RelPerm.Alpha.Filename** no default This
key specifies a ParFlow binary filename containing the alpha parameters for the
VanGenuchten function cell-by-cell. The ONLY option for *geom_name* is
“domain”.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.Alpha.Filename   "alphas.pfb"        ## TCL syntax

      <runname>.Geom.domain.RelPerm.Alpha.Filename = "alphas.pfb"    ## Python syntax

*string* **Geom.\ *geom_name*.RelPerm.N.Filename** no default This key
specifies a ParFlow binary filename containing the N parameters for the
VanGenuchten function cell-by-cell. The ONLY option for *geom_name* is
“domain”.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.N.Filename   "Ns.pfb"       ## TCL syntax

      <runname>.Geom.domain.RelPerm.N.Filename = "Ns.pfb"   ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.Alpha** no default This key
specifies the :math:`\alpha` parameter for the Van Genuchten function
specified on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.Alpha  0.005          ## TCL syntax

      <runname>.Geom.domain.RelPerm.Alpha = 0.005     ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.N** no default This key specifies
the :math:`N` parameter for the Van Genuchten function specified on
*geom_name*.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.N   2.0         ## TCL syntax

      <runname>.Geom.domain.RelPerm.N = 2.0     ## Python syntax

*int* **Geom.\ *geom_name*.RelPerm.NumSamplePoints** 0 This key
specifies the number of sample points for a spline base interpolation
table for the Van Genuchten function specified on *geom_name*. If this
number is 0 (the default) then the function is evaluated directly. Using
the interpolation table is faster but is less accurate.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.NumSamplePoints  20000         ## TCL syntax

      <runname>.Geom.domain.RelPerm.NumSamplePoints = 20000    ## Python syntax

*int* **Geom.\ *geom_name*.RelPerm.MinPressureHead** no default This key
specifies the lower value for a spline base interpolation table for the
Van Genuchten function specified on *geom_name*. The upper value of the
range is 0. This value is used only when the table lookup method is used
(*NumSamplePoints* is greater than 0).

.. container:: list

   ::

      pfset Geom.domain.RelPerm.MinPressureHead -300        ## TCL syntax

      <runname>.Geom.domain.RelPerm.MinPressureHead = -300  ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.A** no default This key specifies
the :math:`A` parameter for the Haverkamp relative permeability on
*geom_name*.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.A  1.0          ## TCL syntax

      <runname>.Geom.domain.RelPerm.A = 1.0     ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.Gamma** no default This key
specifies the the :math:`\gamma` parameter for the Haverkamp relative
permeability on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.Gamma  1.0         ## TCL syntax

      <runname>.Geom.domain.RelPerm.Gamma = 1.0    ## Python syntax

*integer* **Geom.\ *geom_name*.RelPerm.Degree** no default This key
specifies the degree of the polynomial for the Polynomial relative
permeability given on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.RelPerm.Degree  1       ## TCL syntax

      <runname>.Geom.domain.RelPerm.Degree = 1  ## Python syntax

*double* **Geom.\ *geom_name*.RelPerm.Coeff.\ *coeff_number*** no
default This key specifies the *coeff_number*\ th coefficient of the
Polynomial relative permeability given on *geom_name*.

.. container:: list

   ::
      
      ## TCL syntax
      pfset Geom.domain.RelPerm.Coeff.0  0.5
      pfset Geom.domain.RelPerm.Coeff.1  1.0
      
      ## Python syntax
      <runname>.Geom.domain.RelPerm.Coeff.0 = 0.5
      <runname>.Geom.domain.RelPerm.Coeff.1 = 1.0


NOTE: For all these cases, if only one region is to be used (the
domain), the background region should NOT be set as that single region.
Using the background will prevent the upstream weighting from being
correct near Dirichlet boundaries.

.. _Phase Sources:

Phase Sources
~~~~~~~~~~~~~

The following keys are used to specify phase source terms. The units of
the source term are :math:`1/T`. So, for example, to specify a region
with constant flux rate of :math:`L^3/T`, one must be careful to convert
this rate to the proper units by dividing by the volume of the enclosing
region. For *Richards’ equation* input, the source term must be given as
a flux multiplied by density.

*string* **PhaseSources.\ *phase_name*.Type** no default This key
specifies the type of source to use for phase *phase_name*. Possible
values for this key are **Constant** and **PredefinedFunction**.
**Constant** type phase sources specify a constant phase source value
for a given set of regions. **PredefinedFunction** type phase sources
use a preset function (choices are listed below) to specify the source.
Note that the **PredefinedFunction** type can only be used to set a
single source over the entire domain and not separate sources over
different regions.

.. container:: list

   ::

      pfset PhaseSources.water.Type   "Constant"      ## TCL syntax

      <runname>.PhaseSources.water.Type = "Constant"  ## Python syntax

*list* **PhaseSources.\ *phase_name*.GeomNames** no default This key
specifies the names of the geometries on which source terms will be
specified. This is used only for **Constant** type phase sources.
Regions listed later “overlay” regions listed earlier.

.. container:: list

   ::

      pfset PhaseSources.water.GeomNames   "bottomlayer middlelayer toplayer"       ## TCL syntax

      <runname>.PhaseSources.water.GeomNames = "bottomlayer middlelayer toplayer"   ## Python syntax


*double* **PhaseSources.\ *phase_name*.Geom.\ *geom_name*.Value** no
default This key specifies the value of a constant source term applied
to phase *phase \_name* on geometry *geom_name*.

.. container:: list

   ::

      pfset PhaseSources.water.Geom.toplayer.Value   1.0       ## TCL syntax

      <runname>.PhaseSources.water.Geom.toplayer.Value = 1.0   ## Python syntax

*string* **PhaseSources.\ *phase_name*.PredefinedFunction** no default
This key specifies which of the predefined functions will be used for
the source. Possible values for this key are **X, XPlusYPlusZ,
X3Y2PlusSinXYPlus1,** and **XYZTPlus1PermTensor**.

.. container:: list

   ::

      pfset PhaseSources.water.PredefinedFunction   "XPlusYPlusZ"       ## TCL syntax

      <runname>.PhaseSources.water.PredefinedFunction = "XPlusYPlusZ"   ## Python syntax


The choices for this key correspond to sources as follows:

**X**: 
   :math:`{\rm source}\; = 0.0`

**XPlusYPlusX**: 
   :math:`{\rm source}\; = 0.0`

**X3Y2PlusSinXYPlus1**:
   | :math:`{\rm source}\; = -(3x^2 y^2 + y\cos(xy))^2 - (2x^3 y + x\cos(xy))^2 
     - (x^3 y^2 + \sin(xy) + 1) (6x y^2 + 2x^3 -(x^2 +y^2) \sin(xy))`
   | This function type specifies that the source applied over the
     entire domain is as noted above. This corresponds to
     :math:`p=x^{3}y^{2}+\sin(xy)+1` in the problem
     :math:`-\nabla\cdot (p\nabla p)=f`.

**X3Y4PlusX2PlusSinXYCosYPlus1**:
   | :math:`{\rm source}\; = -(3x^22 y^4 + 2x + y\cos(xy)\cos(y))^2 
     - (4x^3 y^3 + x\cos(xy)\cos(y) - \sin(xy)\sin(y))^2 
     - (x^3 y^4 + x^2 + \sin(xy)\cos(y) + 1)
     (6xy^4 + 2 - (x^2 + y^2 + 1)\sin(xy)\cos(y) 
     + 12x^3 y^2 - 2x\cos(xy)\sin(y))`
   | This function type specifies that the source applied over the
     entire domain is as noted above. This corresponds to
     :math:`p=x^{3}y^{4}+x^{2}+\sin (xy)\cos(y) +1` in the problem
     :math:`-\nabla\cdot (p\nabla p)=f`.

**XYZTPlus1**: 
   | :math:`{\rm source}\; = xyz - t^2 (x^2 y^2 +x^2 z^2 +y^2 z^2)`
   | This function type specifies that the source applied over the
     entire domain is as noted above. This corresponds to
     :math:`p = xyzt + 1` in the problem
     :math:`\frac{\partial p}{\partial t}-\nabla\cdot (p\nabla p)=f`.

**XYZTPlus1PermTensor**: 
   | :math:`{\rm source}\; = xyz - t^2 (x^2 y^2 3 + x^2 z^2 2 + y^2 z^2)`
   | This function type specifies that the source applied over the
     entire domain is as noted above. This corresponds to
     :math:`p = xyzt + 1` in the problem
     :math:`\frac{\partial p}{\partial t}-\nabla\cdot (Kp\nabla p)=f`,
     where :math:`K = diag(1 \;\; 2 \;\; 3)`.

.. _Capillary Pressures:

Capillary Pressures
~~~~~~~~~~~~~~~~~~~

Here we define capillary pressure. Note: this section needs to be
defined *only* for multi-phase flow and should not be defined for single
phase and Richards’ equation cases. The format for this section of input
is:

*string* **CapPressure.\ *phase_name*.Type** "Constant" This key
specifies the capillary pressure between phase :math:`0` and the named
phase, *phase_name*. The only choice available is **Constant** which
indicates that a constant capillary pressure exists between the phases.

.. container:: list

   ::

      pfset CapPressure.water.Type   "Constant"        ## TCL syntax

      <runname>.CapPressure.water.Type = "Constant"    ## Python syntax

*list* **CapPressure.\ *phase_name*.GeomNames** no default This key
specifies the geometries that capillary pressures will be computed for
in the named phase, *phase_name*. Regions listed later “overlay” regions
listed earlier. Any geometries not listed will be assigned :math:`0.0`
capillary pressure by ParFlow.

.. container:: list

   ::

      pfset CapPressure.water.GeomNames   "domain"       ## TCL syntax

      <runname>.CapPressure.water.GeomNames = "domain"   ## Python syntax


*double* **Geom.\ *geometry_name*.CapPressure.\ *phase_name*.Value** 0.0
This key specifies the value of the capillary pressure in the named
geometry, *geometry_name*, for the named phase, *phase_name*.

.. container:: list

   ::

      pfset Geom.domain.CapPressure.water.Value   0.0       ## TCL syntax

      <runname>.Geom.domain.CapPressure.water.Value = 0.0   ## Python syntax

*Important note*: the code currently works only for capillary pressure
equal zero.

.. _Saturation:

Saturation
~~~~~~~~~~

This section is *only* relevant to the Richards’ equation cases. All
keys relating to this section will be ignored for other cases. The
following keys are used to define the saturation-pressure curve.

*string* **Phase.Saturation.Type** no default This key specifies the
type of saturation function that will be used on all specified
geometries. Note that only one type of saturation may be used for the
entire problem. However, parameters may be different for that type in
different geometries. For instance, if the problem consists of three
geometries, then **VanGenuchten** may be specified with three different
sets of parameters for the three different geometries. However, once
**VanGenuchten** is specified, one geometry cannot later be specified to
have **Data** as its saturation. The possible values for this key are
**Constant, VanGenuchten, Haverkamp, Data, Polynomial** and **PFBFile**.

.. container:: list

   ::

      pfset Phase.Saturation.Type   "Constant"         ## TCL syntax

      <runname>.Phase.Saturation.Type = "Constant"     ## Python syntax


The various possible functions are defined as follows. The **Constant**
specification means that the saturation will be constant on the
specified geounit. The **VanGenuchten** specification means that the
saturation will be given as a Van Genuchten function
:cite:p:`VanGenuchten80` with the form,

.. math::

   \begin{aligned}
   s(p) = \frac{s_{sat} - s_{res}}{(1 + (\alpha p)^n)^m} + s_{res},\end{aligned}

where :math:`s_{sat}` is the saturation at saturated conditions,
:math:`s_{res}` is the residual saturation, and :math:`\alpha` and
:math:`n` are soil parameters with :math:`m = 1 - 1/n`, on each region.
The **Haverkamp** specification means that the saturation will be given
in the following form :cite:p:`Haverkamp-Vauclin81`,

.. math::

   \begin{aligned}
   s(p) = \frac{\A(s_{sat} - s_{res})}{A + p^{\gamma}} + s_{res},\end{aligned}

where :math:`A` and :math:`\gamma` are soil parameters, on each region.
The **Data** specification is currently unsupported but will later mean
that data points for the saturation curve will be given and ParFlow will
set up the proper interpolation coefficients to get values between the
given data points. The **Polynomial** specification defines a polynomial
saturation function for each region of the form,

.. math::

   \begin{aligned}
   s(p) = \sum_{i=0}^{degree} c_ip^i.\end{aligned}

The **PFBFile** specification means that the saturation will be taken as
a spatially varying but constant in pressure function given by data in a
ParFlow 3D binary file.

*list* **Phase.Saturation.GeomNames** no default This key specifies the
geometries on which saturation will be given. The union of these
geometries must cover the entire computational domain.

.. container:: list

   ::

      pfset Phase.Saturation.Geonames   "domain"         ## TCL syntax

      <runname>.Phase.Saturation.Geonames = "domain"     ## Python syntax


*double* **Geom.\ *geom_name*.Saturation.Value** no default This key
specifies the constant saturation value on the *geom_name* region.

.. container:: list

   ::

      pfset Geom.domain.Saturation.Value    0.5       ## TCL syntax

      <runname>.Geom.domain.Saturation.Value = 0.5    ## Python syntax


*integer* **Phase.Saturation.VanGenuchten.File** 0 This key specifies
whether soil parameters for the VanGenuchten function are specified in a
ParFlow 3D binary file or by region. The options are either 0 for specification by
region, or 1 for specification in a file. Note that either all
parameters are specified in files (each has their own input file) or
none are specified by files. Parameters specified by files are
:math:`\alpha`, N, SRes, and SSat.

.. container:: list

   ::

      pfset Phase.Saturation.VanGenuchten.File   1       ## TCL syntax

      <runname>.Phase.Saturation.VanGenuchten.File = 1   ## Python syntax


*string* **Geom.\ *geom_name*.Saturation.Alpha.Filename** no default
This key specifies a ParFlow binary filename containing the alpha parameters for
the VanGenuchten function cell-by-cell. The ONLY option for *geom_name*
is “domain”.

.. container:: list

   ::

      pfset Geom.domain.Saturation.Filename   "alphas.pfb"     ## TCL syntax

      <runname.Geom.domain.Saturation.Filename = "alphas.pfb"  ## Python syntax


*string* **Geom.\ *geom_name*.Saturation.N.Filename** no default This
key specifies a ParFlow binary filename containing the N parameters for the
VanGenuchten function cell-by-cell. The ONLY option for *geom_name* is
“domain”.

.. container:: list

   ::

      pfset Geom.domain.Saturation.N.Filename   "Ns.pfb"    ## TCL syntax

      pfset Geom.domain.Saturation.N.Filename = "Ns.pfb"    ## Python syntax

*string* **Geom.\ *geom_name*.Saturation.SRes.Filename** no default This
key specifies a ParFlow binary filename containing the SRes parameters for the
VanGenuchten function cell-by-cell. The ONLY option for *geom_name* is
“domain”.

.. container:: list

   ::

      pfset Geom.domain.Saturation.SRes.Filename   "SRess.pfb"          ## TCL syntax

      <runname>.Geom.domain.Saturation.SRes.Filename = "SRess.pfb"      ## Python syntax


*string* **Geom.\ *geom_name*.Saturation.SSat.Filename** no default This
key specifies a ParFlow binary filename containing the SSat parameters for the
VanGenuchten function cell-by-cell. The ONLY option for *geom_name* is
“domain”.

.. container:: list

   ::

      pfset Geom.domain.Saturation.SSat.Filename   "SSats.pfb"       ## TCL syntax

      <runname>.Geom.domain.Saturation.SSat.Filename = "SSats.pfb"   ## Python syntax


*double* **Geom.\ *geom_name*.Saturation.Alpha** no default This key
specifies the :math:`\alpha` parameter for the Van Genuchten function
specified on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.Alpha  0.005          ## TCL syntax

      <runname>.Geom.domain.Saturation.Alpha = 0.005     ## Python syntax

*double* **Geom.\ *geom_name*.Saturation.N** no default This key
specifies the :math:`N` parameter for the Van Genuchten function
specified on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.N   2.0         ## TCL syntax

      <runname>.Geom.domain.Saturation.N = 2.0     ## Python syntax

Note that if both a Van Genuchten saturation and relative permeability
are specified, then the soil parameters should be the same for each in
order to have a consistent problem.

*double* **Geom.\ *geom_name*.Saturation.SRes** no default This key
specifies the residual saturation on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.SRes   0.0         ## TCL syntax

      <runname>.Geom.domain.Saturation.SRes = 0.0     ## Python syntax

*double* **Geom.\ *geom_name*.Saturation.SSat** no default This key
specifies the saturation at saturated conditions on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.SSat   1.0         ## TCL syntax

      <runname>.Geom.domain.Saturation.SSat = 1.0     ## Python syntax

*double* **Geom.\ *geom_name*.Saturation.A** no default This key
specifies the :math:`A` parameter for the Haverkamp saturation on
*geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.A   1.0         ## TCL syntax

      <runname>.Geom.domain.Saturation.A = 1.0     ## Python syntax

*double* **Geom.\ *geom_name*.Saturation.Gamma** no default This key
specifies the the :math:`\gamma` parameter for the Haverkamp saturation
on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.Gamma   1.0        ## TCL syntax

      <runname>.Geom.domain.Saturation.Gamma = 1.0    ## Python syntax

*integer* **Geom.\ *geom_name*.Saturation.Degree** no default This key
specifies the degree of the polynomial for the Polynomial saturation
given on *geom_name*.

.. container:: list

   ::

      pfset Geom.domain.Saturation.Degree   1      ## TCL syntax

      <runname>.Geom.domain.Saturation.Degree = 1  ## Python syntax

*double* **Geom.\ *geom_name*.Saturation.Coeff.\ *coeff_number*** no
default This key specifies the *coeff_number*\ th coefficient of the
Polynomial saturation given on *geom_name*.

.. container:: list

   ::

      ## TCL syntax
      pfset Geom.domain.Saturation.Coeff.0   0.5
      pfset Geom.domain.Saturation.Coeff.1   1.0

      ## Python syntax
      <runname>.Geom.domain.Saturation.Coeff.0 = 0.5
      <runname>.Geom.domain.Saturation.Coeff.1 = 1.0


*string* **Geom.\ *geom_name*.Saturation.FileName** no default This key
specifies the name of the file containing saturation values for the
domain. It is assumed that *geom_name* is “domain” for this key.

.. container:: list

   ::

      pfset Geom.domain.Saturation.FileName  "domain_sats.pfb"       ## TCL syntax

      <runname>.Geom.domain.Saturation.FileName = "domain_sats.pfb"  ## Python syntax


.. _Internal Boundary Conditions:

Internal Boundary Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section, we define internal Dirichlet boundary conditions by
setting the pressure at points in the domain. The format for this
section of input is:

*string* **InternalBC.Names** no default This key specifies the names
for the internal boundary conditions. At each named point,
:math:`{\rm x}`, :math:`{\rm y}` and :math:`{\rm z}` will specify the
coordinate locations and :math:`{\rm h}` will specify the hydraulic head
value of the condition. This real location is “snapped” to the nearest
gridpoint in ParFlow.

NOTE: Currently, ParFlow assumes that internal boundary conditions and
pressure wells are separated by at least one cell from any external
boundary. The user should be careful of this when defining the input
file and grid.

.. container:: list

   ::

      pfset InternalBC.Names   "fixedvalue"        ## TCL syntax

      <runname>.InternalBC.Names = "fixedvalue"    ## Python syntax

*double* **InternalBC.\ *internal_bc_name*.X** no default This key
specifies the x-coordinate, :math:`{\rm x}`, of the named,
*internal_bc_name*, condition.

.. container:: list

   ::

      pfset InternalBC.fixedheadvalue.X   40.0        ## TCL syntax

      <runname>.InternalBC.fixedheadvalue.X = 40.0    ## Python syntax

*double* **InternalBC.\ *internal_bc_name*.Y** no default This key
specifies the y-coordinate, :math:`{\rm y}`, of the named,
*internal_bc_name*, condition.

.. container:: list

   ::

      pfset InternalBC.fixedheadvalue.Y   65.2        ## TCL syntax

      <runname>.InternalBC.fixedheadvalue.Y = 65.2    ## Python syntax

*double* **InternalBC.\ *internal_bc_name*.Z** no default This key
specifies the z-coordinate, :math:`{\rm z}`, of the named,
*internal_bc_name*, condition.

.. container:: list

   ::

      pfset InternalBC.fixedheadvalue.Z   12.1        ## TCL syntax

      <runname>.InternalBC.fixedheadvalue.Z = 12.1    ## Python syntax

*double* **InternalBC.\ *internal_bc_name*.Value** no default This key
specifies the value of the named, *internal_bc_name*, condition.

.. container:: list

   ::

      pfset InternalBC.fixedheadvalue.Value   100.0         ## TCL syntax

      <runname>.InternalBC.fixedheadvalue.Value = 100.0     ## Python syntax

.. _`Boundary Conditions: Pressure`:

Boundary Conditions: Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we define the pressure boundary conditions. The Dirichlet
conditions below are hydrostatic conditions, and it is assumed that at
each phase interface the pressure is constant. *It is also assumed here
that all phases are distributed within the domain at all times such that
the lighter phases are vertically higher than the heavier phases.*

Boundary condition input is associated with domain patches (see :ref:`Domain`). Note that different patches may have different
types of boundary conditions on them.

*list* **BCPressure.PatchNames** no default This key specifies the names
of patches on which pressure boundary conditions will be specified. Note
that these must all be patches on the external boundary of the domain
and these patches must “cover” that external boundary.

.. container:: list

   ::

      pfset BCPressure.PatchNames    "left right front back top bottom"

*string* **Patch.\ *patch_name*.BCPressure.Type** no default This key
specifies the type of boundary condition data given for patch
*patch_name*. Possible values for this key are **DirEquilRefPatch,
DirEquilPLinear, FluxConst, FluxVolumetric, PressureFile, FluxFile,
OverlandFow, OverlandFlowPFB, SeepageFace, OverlandKinematic,
OverlandDiffusive** and **ExactSolution**. The choice
**DirEquilRefPatch** specifies that the pressure on the specified patch
will be in hydrostatic equilibrium with a constant reference pressure
given on a reference patch. The choice **DirEquilPLinear** specifies
that the pressure on the specified patch will be in hydrostatic
equilibrium with pressure given along a piecewise line at elevation
:math:`z=0`. The choice **FluxConst** defines a constant normal flux
boundary condition through the domain patch. This flux must be specified
in units of :math:`[L]/[T]`. For *Richards’ equation*, fluxes must be
specified as a mass flux and given as the above flux multiplied by the
density. Thus, this choice of input type for a Richards’ equation
problem has units of :math:`([L]/[T])([M]/[L]^3)`. The choice
**FluxVolumetric** defines a volumetric flux boundary condition through
the domain patch. The units should be consistent with all other user
input for the problem. For *Richards’ equation* fluxes must be specified
as a mass flux and given as the above flux multiplied by the density.
The choice **PressureFile** defines a hydraulic head boundary condition
that is read from a properly distributed ParFlow binary file. Only the values
needed for the patch are used. The choice **FluxFile** defines a flux
boundary condition that is read form a properly distributed ParFlow binary file
defined on a grid consistent with the pressure field grid. Only the
values needed for the patch are used. The choices **OverlandFlow** and
**OverlandFlowPFB** both turn on fully-coupled overland flow routing as
described in :cite:t:`KM06` and :ref:`Overland Flow`. The key **OverlandFlow**
corresponds to a **Value** key with a positive or negative value, to
indicate uniform fluxes (such as rainfall or evapotranspiration) over
the entire domain while the key **OverlandFlowPFB** allows a ParFlow 2D binary file to
contain grid-based, spatially-variable fluxes. The **OverlandKinematic**
and **OverlandDiffusive** both turn on a kinematic and diffusive wave
overland flow routing boundary that solve Maning's equation in
:ref:`Overland Flow` and do the upwinding internally
(i.e. assuming that the user provides cell face slopes, as opposed to
the traditional cell centered slopes). The key **SeepageFace** simulates
a boundary that allows flow to exit but keeps the surface pressure at
zero. Consider a sign flip in top boundary condition values (i.e., outgoing
fluxes are positive and incoming fluxes are negative). The choice
**ExactSolution** specifies that an exact known
solution is to be applied as a Dirichlet boundary condition on the
respective patch. Note that this does not change according to any cycle.
Instead, time dependence is handled by evaluating at the time the
boundary condition value is desired. The solution is specified by using
a predefined function (choices are described below). NOTE: These last
six types of boundary condition input is for *Richards’ equation cases
only!*

.. container:: list

   ::

      pfset Patch.top.BCPressure.Type  DirEquilRefPatch

*string* **Patch.\ *patch_name*.BCPressure.Cycle** no default This key
specifies the time cycle to which boundary condition data for patch
*patch_name* corresponds.

.. container:: list

   ::

      pfset Patch.top.BCPressure.Cycle   Constant

*string* **Patch.\ *patch_name*.BCPressure.RefGeom** no default This key
specifies the name of the solid on which the reference patch for the
**DirEquilRefPatch** boundary condition data is given. Care should be
taken to make sure the correct solid is specified in cases of layered
domains.

.. container:: list

   ::

      pfset Patch.top.BCPressure.RefGeom   "domain"

*string* **Patch.\ *patch_name*.BCPressure.RefPatch** no default This
key specifies the reference patch on which the **DirEquilRefPatch**
boundary condition data is given. This patch must be on the reference
solid specified by the Patch.\ *patch_name*.BCPressure.RefGeom key.

.. container:: list

   ::

      pfset Patch.top.BCPressure.RefPatch    "bottom"

*bool* **Patch.\ *patch_name*.BCPressure.Seepage** False When set to
True for an OverlandKinematic pressure boundary condition, this patch
will be treated as a seepage patch for the overland kinematic wave
formulation.

Example of mixed boundary conditions: 

::
   CONUS2.Geom.domain.Patches = "ocean land top lake sink bottom"
   CONUS2.BCPressure.PatchNames ="ocean land top lake sink bottom"

In this example, the top gets assigned to the **OverlandKinematic** BC
and the sink and lake will be treated as a seepage patch for the
overland kinematic wave formulation
 
::
   
   CONUS2.Patch.top.BCPressure.Type = 'OverlandKinematic'
   CONUS2.Patch.lake.BCPressure.Type = 'OverlandKinematic'
   CONUS2.Patch.lake.BCPressure.Seepage = True
   CONUS2.Patch.sink.BCPressure.Type = 'OverlandKinematic'
   CONUS2.Patch.sink.BCPressure.Seepage = True

.. container:: list

   ::

      pfset Patch.top.BCPressure.Seepage    "True"
      
*double* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.Value** no
default This key specifies the reference pressure value for the
**DirEquilRefPatch** boundary condition or the constant flux value for
the **FluxConst** boundary condition, or the constant volumetric flux
for the **FluxVolumetric** boundary condition.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.Value  -14.0

*double*
**Patch.\ *patch_name*.BCPressure.\ *interval_name*.\ *phase_name*.IntValue**
no default Note that the reference conditions for types
**DirEquilPLinear** and **DirEquilRefPatch** boundary conditions are for
phase 0 *only*. This key specifies the constant pressure value along the
interface with phase *phase_name* for cases with two phases present.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.water.IntValue   -13.0

*double* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.XLower** no
default This key specifies the lower :math:`x` coordinate of a line in
the xy-plane.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.XLower  0.0

*double* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.YLower** no
default This key specifies the lower :math:`y` coordinate of a line in
the xy-plane.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.YLower  0.0

*double* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.XUpper** no
default This key specifies the upper :math:`x` coordinate of a line in
the xy-plane.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.XUpper  1.0

*double* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.YUpper** no
default This key specifies the upper :math:`y` coordinate of a line in
the xy-plane.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.YUpper  1.0

*integer*
**Patch.\ *patch_name*.BCPressure.\ *interval_name*.NumPoints** no
default This key specifies the number of points on which pressure data
is given along the line used in the type **DirEquilPLinear** boundary
conditions.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.NumPoints   2

*double*
**Patch.\ *patch_name*.BCPressure.\ *interval_name*.\ *point_number*.Location**
no default This key specifies a number between 0 and 1 which represents
the location of a point on the line on which data is given for type
**DirEquilPLinear** boundary conditions. Here 0 corresponds to the lower
end of the line, and 1 corresponds to the upper end.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.0.Location   0.0

*double*
**Patch.\ *patch_name*.BCPressure.\ *interval_name*.\ *point_number*.Value**
no default This key specifies the pressure value for phase 0 at point
number *point_number* and :math:`z=0` for type **DirEquilPLinear**
boundary conditions. All pressure values on the patch are determined by
first projecting the boundary condition coordinate onto the line, then
linearly interpolating between the neighboring point pressure values on
the line.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.0.Value   14.0

*string* **Patch.\ *patch_name*.BCPressure.\ *interval_name*.FileName**
no default This key specifies the name of a properly distributed ParFlow binary file 
that contains boundary data to be read for types PressureFile and FluxFile. 
For flux data, the data must be defined over a grid consistent with the 
pressure field. In both cases, only the values needed for the patch will 
be used. The rest of the data is ignored.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.FileName   "ocwd_bc.pfb"

*string*
**Patch.\ *patch_name*.BCPressure.\ *interval_name*.PredefinedFunction**
no default This key specifies the predefined function that will be used
to specify Dirichlet boundary conditions on patch *patch_name*. Note
that this does not change according to any cycle. Instead, time
dependence is handled by evaluating at the time the boundary condition
value is desired. Choices for this key include **X, XPlusYPlusZ,
X3Y2PlusSinXYPlus1, X3Y4PlusX2PlusSinXYCosYPlus1, XYZTPlus1** and
**XYZTPlus1PermTensor**.

.. container:: list

   ::

      pfset Patch.top.BCPressure.alltime.PredefinedFunction  "XPlusYPlusZ"

The choices for this key correspond to pressures as follows.

**X**: 
   :math:`p = x`

**XPlusYPlusZ**: 
   :math:`p = x + y + z`

**X3Y2PlusSinXYPlus1**: 
   :math:`p = x^3 y^2 + \sin(xy) + 1`

**X3Y4PlusX2PlusSinXYCosYPlus1**: 
   :math:`p = x^3 y^4 + x^2 + \sin(xy)\cos y + 1`

**XYZTPlus1**: 
   :math:`p = xyzt + 1`

**XYZTPlus1PermTensor**: 
   :math:`p = xyzt + 1`

.. _`Boundary Conditions: Saturation`:

Boundary Conditions: Saturation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: this section needs to be defined *only* for multi-phase flow and
should *not* be defined for the single phase and Richards’ equation
cases.

Here we define the boundary conditions for the saturations. Boundary
condition input is associated with domain patches (see :ref:`Domain`). Note that different patches may have different
types of boundary conditions on them.

*list* **BCSaturation.PatchNames** no default This key specifies the
names of patches on which saturation boundary conditions will be
specified. Note that these must all be patches on the external boundary
of the domain and these patches must “cover” that external boundary.

.. container:: list

   ::

      pfset BCSaturation.PatchNames    "left right front back top bottom"

*string* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.Type** no
default This key specifies the type of boundary condition data given for
the given phase, *phase_name*, on the given patch *patch_name*. Possible
values for this key are **DirConstant**, **ConstantWTHeight** and
**PLinearWTHeight**. The choice **DirConstant** specifies that the
saturation is constant on the whole patch. The choice
**ConstantWTHeight** specifies a constant height of the water-table on
the whole patch. The choice **PLinearWTHeight** specifies that the
height of the water-table on the patch will be given by a piecewise
linear function.

Note: the types **ConstantWTHeight** and **PLinearWTHeight** assume we
are running a 2-phase problem where phase 0 is the water phase.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.Type  "ConstantWTHeight"

*double* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.Value** no
default This key specifies either the constant saturation value if
**DirConstant** is selected or the constant water-table height if
**ConstantWTHeight** is selected.

.. container:: list

   ::

      pfset Patch.top.BCSaturation.air.Value 1.0

*double* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.XLower** no
default This key specifies the lower :math:`x` coordinate of a line in
the xy-plane if type **PLinearWTHeight** boundary conditions are
specified.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.XLower -10.0

*double* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.YLower** no
default This key specifies the lower :math:`y` coordinate of a line in
the xy-plane if type **PLinearWTHeight** boundary conditions are
specified.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.YLower 5.0

*double* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.XUpper** no
default This key specifies the upper :math:`x` coordinate of a line in
the xy-plane if type **PLinearWTHeight** boundary conditions are
specified.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.XUpper  125.0

*double* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.YUpper** no
default This key specifies the upper :math:`y` coordinate of a line in
the xy-plane if type **PLinearWTHeight** boundary conditions are
specified.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.YUpper  82.0

*integer* **Patch.\ *patch_name*.BCSaturation.\ *phase_name*.NumPoints**
no default This key specifies the number of points on which saturation
data is given along the line used for type **DirEquilPLinear** boundary
conditions.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.NumPoints 2

*double*
**Patch.\ *patch_name*.BCSaturation.\ *phase_name*.\ *point_number*.Location**
no default This key specifies a number between 0 and 1 which represents
the location of a point on the line for which data is given in type
**DirEquilPLinear** boundary conditions. The line is parameterized so
that 0 corresponds to the lower end of the line, and 1 corresponds to
the upper end.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.0.Location 0.333

*double*
**Patch.\ *patch_name*.BCSaturation.\ *phase_name*.\ *point_number*.Value**
no default This key specifies the water-table height for the given point
if type **DirEquilPLinear** boundary conditions are selected. All
saturation values on the patch are determined by first projecting the
water-table height value onto the line, then linearly interpolating
between the neighboring water-table height values onto the line.

.. container:: list

   ::

      pfset Patch.left.BCSaturation.water.0.Value  4.5

.. _`Initial Conditions: Phase Saturations`:

Initial Conditions: Phase Saturations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: this section needs to be defined *only* for multi-phase flow and
should *not* be defined for single phase and Richards’ equation cases.

Here we define initial phase saturation conditions. The format for this
section of input is:

*string* **ICSaturation.\ *phase_name*.Type** no default This key
specifies the type of initial condition that will be applied to
different geometries for given phase, *phase_name*. The only key
currently available is **Constant**. The choice **Constant** will apply
constants values within geometries for the phase.

.. container:: list

   ::

      ICSaturation.water.Type Constant

*string* **ICSaturation.\ *phase_name*.GeomNames** no default This key
specifies the geometries on which an initial condition will be given if
the type is set to **Constant**.

Note that geometries listed later “overlay” geometries listed earlier.

.. container:: list

   ::

      ICSaturation.water.GeomNames "domain"

*double* **Geom.\ *geom_input_name*.ICSaturation.\ *phase_name*.Value**
no default This key specifies the initial condition value assigned to
all points in the named geometry, *geom_input_name*, if the type was set
to **Constant**.

.. container:: list

   ::

      Geom.domain.ICSaturation.water.Value 1.0

.. _`Initial Conditions: Pressure`:

Initial Conditions: Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The keys in this section are used to specify pressure initial conditions
for Richards’ equation cases *only*. These keys will be ignored if any
other case is run.

*string* **ICPressure.Type** no default This key specifies the type of
initial condition given. The choices for this key are **Constant,
HydroStaticDepth, HydroStaticPatch** and **PFBFile**. The choice
**Constant** specifies that the initial pressure will be constant over
the regions given. The choice **HydroStaticDepth** specifies that the
initial pressure within a region will be in hydrostatic equilibrium with
a given pressure specified at a given depth. The choice
**HydroStaticPatch** specifies that the initial pressure within a region
will be in hydrostatic equilibrium with a given pressure on a specified
patch. Note that all regions must have the same type of initial data -
different regions cannot have different types of initial data. However,
the parameters for the type may be different. The **PFBFile**
specification means that the initial pressure will be taken as a
spatially varying function given by data in a ParFlow 3D binary file.

.. container:: list

   ::

      pfset ICPressure.Type   "Constant"

*list* **ICPressure.GeomNames** no default This key specifies the
geometry names on which the initial pressure data will be given. These
geometries must comprise the entire domain. Note that conditions for
regions that overlap other regions will have unpredictable results. The
regions given must be disjoint.

.. container:: list

   ::

      pfset ICPressure.GeomNames   "toplayer middlelayer bottomlayer"

*double* **Geom.\ *geom_name*.ICPressure.Value** no default This key
specifies the initial pressure value for type **Constant** initial
pressures and the reference pressure value for types
**HydroStaticDepth** and **HydroStaticPatch**.

.. container:: list

   ::

      pfset Geom.toplayer.ICPressure.Value  -734.0

*double* **Geom.\ *geom_name*.ICPressure.RefElevation** no default This
key specifies the reference elevation on which the reference pressure is
given for type **HydroStaticDepth** initial pressures.

.. container:: list

   ::

      pfset Geom.toplayer.ICPressure.RefElevation  0.0

*double* **Geom.\ *geom_name*.ICPressure.RefGeom** no default This key
specifies the geometry on which the reference patch resides for type
**HydroStaticPatch** initial pressures.

.. container:: list

   ::

      pfset Geom.toplayer.ICPressure.RefGeom   "bottomlayer"

*double* **Geom.\ *geom_name*.ICPressure.RefPatch** no default This key
specifies the patch on which the reference pressure is given for type
**HydorStaticPatch** initial pressures.

.. container:: list

   ::

      pfset Geom.toplayer.ICPressure.RefPatch   "bottom"

*string* **Geom.\ *geom_name*.ICPressure.FileName** no default This key
specifies the name of the file containing pressure values for the
domain. It is assumed that *geom_name* is “domain” for this key.

.. container:: list

   ::

      pfset Geom.domain.ICPressure.FileName  "ic_pressure.pfb"

Example Script:

.. container:: list

   ::


      #---------------------------------------------------------
      # Initial conditions: water pressure [m]
      #---------------------------------------------------------
      # Using a patch is great when you are not using a box domain
      # If using a box domain HydroStaticDepth is fine
      # If your RefPatch is z-lower (bottom of domain), the pressure is positive.
      # If your RefPatch is z-upper (top of domain), the pressure is negative.
      ### Set water table to be at the bottom of the domain, the top layer is initially dry
      pfset ICPressure.Type				      "HydroStaticPatch"
      pfset ICPressure.GeomNames		         "domain"
      pfset Geom.domain.ICPressure.Value	   2.2

      pfset Geom.domain.ICPressure.RefGeom	"domain"
      pfset Geom.domain.ICPressure.RefPatch	z-lower

      ### Using a .pfb to initialize
      pfset ICPressure.Type                  "PFBFile"
      pfset ICPressure.GeomNames		         "domain"
      pfset Geom.domain.ICPressure.FileName	"press.00090.pfb"

      pfset Geom.domain.ICPressure.RefGeom	"domain"
      pfset Geom.domain.ICPressure.RefPatch	"z-upper"

.. _`Initial Conditions: Phase Concentrations`:

Initial Conditions: Phase Concentrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we define initial concentration conditions for contaminants. The
format for this section of input is:

*string* **PhaseConcen.\ *phase_name*.\ *contaminant_name*.Type** no
default This key specifies the type of initial condition that will be
applied to different geometries for given phase, *phase_name*, and the
given contaminant, *contaminant_name*. The choices for this key are
**Constant** or **PFBFile**. The choice **Constant** will apply
constants values to different geometries. The choice **PFBFile** will
read values from a ParFlow 3D binary file (see
:ref:`ParFlow Binary Files (.pfb)`).

.. container:: list

   ::

      PhaseConcen.water.tce.Type "Constant"

*string* **PhaseConcen.\ *phase_name*.GeomNames** no default This key
specifies the geometries on which an initial condition will be given, if
the type was set to **Constant**.

Note that geometries listed later “overlay” geometries listed earlier.

.. container:: list

   ::

      PhaseConcen.water.GeomNames "ic_concen_region"

*double*
**PhaseConcen.\ *phase_name*.\ *contaminant_name*.\ *geom_input_name*.Value**
no default This key specifies the initial condition value assigned to
all points in the named geometry, *geom_input_name*, if the type was set
to **Constant**.

.. container:: list

   ::

      PhaseConcen.water.tce.ic_concen_region.Value 0.001

*string* **PhaseConcen.\ *phase_name*.\ *contaminant_name*.FileName** no
default This key specifies the name of the ParFlow 3D binary file which
contains the initial condition values if the type was set to
**PFBFile**.

.. container:: list

   ::

      PhaseConcen.water.tce.FileName "initial_concen_tce.pfb"

.. _ExactSolution:

Known Exact Solution
~~~~~~~~~~~~~~~~~~~~

For *Richards equation cases only* we allow specification of an exact
solution to be used for testing the code. Only types that have been
coded and predefined are allowed. Note that if this is speccified as
something other than no known solution, corresponding boundary
conditions and phase sources should also be specified.

*string* **KnownSolution** no default This specifies the predefined
function that will be used as the known solution. Possible choices for
this key are **NoKnownSolution, Constant, X, XPlusYPlusZ,
X3Y2PlusSinXYPlus1, X3Y4PlusX2PlusSinXYCosYPlus1, XYZTPlus1** and
**XYZTPlus1PermTensor**.

.. container:: list

   ::

      pfset KnownSolution  "XPlusYPlusZ"

Choices for this key correspond to solutions as follows.

**NoKnownSolution**: 
   No solution is known for this problem.

**Constant**: 
   :math:`p = {\rm constant}`

**X**: 
   :math:`p = x`

**XPlusYPlusZ**: 
   :math:`p = x + y + z`

**X3Y2PlusSinXYPlus1**: 
   :math:`p = x^3 y^2 + sin(xy) + 1`

**X3Y4PlusX2PlusSinXYCosYPlus1**: 
   :math:`p = x^3 y^4 + x^2 + \sin(xy)\cos y + 1`

**XYZTPlus1**: 
   :math:`p = xyzt + 1`

**XYZTPlus1PermTensor**: 
   :math:`p = xyzt + 1`

*double* **KnownSolution.Value** no default This key specifies the
constant value of the known solution for type **Constant** known
solutions.

.. container:: list

   ::

      pfset KnownSolution.Value  1.0

Only for known solution test cases will information on the
:math:`L^2`-norm of the pressure error be printed.

.. _Wells:

Wells
~~~~~

Here we define wells for the model. The format for this section of input
is:

*string* **Wells.Names** no default This key specifies the names of the
wells for which input data will be given.

.. container:: list

   ::

      Wells.Names "test_well inj_well ext_well"

*string* **Wells.\ *well_name*.InputType** no default This key specifies
the type of well to be defined for the given well, *well_name*. This key
can be either **Vertical** or **Recirc**. The value **Vertical**
indicates that this is a single segmented well whose action will be
specified by the user. The value **Recirc** indicates that this is a
dual segmented, recirculating, well with one segment being an extraction
well and another being an injection well. The extraction well filters
out a specified fraction of each contaminant and recirculates the
remainder to the injection well where the diluted fluid is injected back
in. The phase saturations at the extraction well are passed without
modification to the injection well.

Note with the recirculating well, several input options are not needed
as the extraction well will provide these values to the injection well.

.. container:: list

   ::

      Wells.test_well.InputType "Vertical"

*string* **Wells.\ *well_name*.Action** no default This key specifies
the pumping action of the well. This key can be either **Injection** or
**Extraction**. A value of **Injection** indicates that this is an
injection well. A value of **Extraction** indicates that this is an
extraction well.

.. container:: list

   ::

      Wells.test_well.Action "Injection"

*double* **Wells.\ *well_name*.Type** no default This key specifies the
mechanism by which the well works (how ParFlow works with the well data)
if the input type key is set to **Vectical**. This key can be either
**Pressure** or **Flux**. A value of **Pressure** indicates that the
data provided for the well is in terms of hydrostatic pressure and
ParFlow will ensure that the computed pressure field satisfies this
condition in the computational cells which define the well. A value of
**Flux** indicates that the data provided is in terms of volumetric flux
rates and ParFlow will ensure that the flux field satisfies this
condition in the computational cells which define the well.

.. container:: list

   ::

      Wells.test_well.Type "Flux"

*string* **Wells.\ *well_name*.ExtractionType** no default This key
specifies the mechanism by which the extraction well works (how ParFlow
works with the well data) if the input type key is set to **Recirc**.
This key can be either **Pressure** or **Flux**. A value of **Pressure**
indicates that the data provided for the well is in terms of hydrostatic
pressure and ParFlow will ensure that the computed pressure field
satisfies this condition in the computational cells which define the
well. A value of **Flux** indicates that the data provided is in terms
of volumetric flux rates and ParFlow will ensure that the flux field
satisfies this condition in the computational cells which define the
well.

.. container:: list

   ::

      Wells.ext_well.ExtractionType "Pressure"

*string* **Wells.\ *well_name*.InjectionType** no default This key
specifies the mechanism by which the injection well works (how ParFlow
works with the well data) if the input type key is set to **Recirc**.
This key can be either **Pressure** or **Flux**. A value of **Pressure**
indicates that the data provided for the well is in terms of hydrostatic
pressure and ParFlow will ensure that the computed pressure field
satisfies this condition in the computational cells which define the
well. A value of **Flux** indicates that the data provided is in terms
of volumetric flux rates and ParFlow will ensure that the flux field
satisfies this condition in the computational cells which define the
well.

.. container:: list

   ::

      Wells.inj_well.InjectionType "Flux"

*double* **Wells.\ *well_name*.X** no default This key specifies the x
location of the vectical well if the input type is set to **Vectical**
or of both the extraction and injection wells if the input type is set
to **Recirc**.

.. container:: list

   ::

      Wells.test_well.X 20.0

*double* **Wells.\ *well_name*.Y** no default This key specifies the y
location of the vectical well if the input type is set to **Vectical**
or of both the extraction and injection wells if the input type is set
to **Recirc**.

.. container:: list

   ::

      Wells.test_well.Y 36.5

*double* **Wells.\ *well_name*.ZUpper** no default This key specifies
the z location of the upper extent of a vectical well if the input type
is set to **Vectical**.

.. container:: list

   ::

      Wells.test_well.ZUpper 8.0

*double* **Wells.\ *well_name*.ExtractionZUpper** no default This key
specifies the z location of the upper extent of a extraction well if the
input type is set to **Recirc**.

.. container:: list

   ::

      Wells.ext_well.ExtractionZUpper 3.0

*double* **Wells.\ *well_name*.InjectionZUpper** no default This key
specifies the z location of the upper extent of a injection well if the
input type is set to **Recirc**.

.. container:: list

   ::

      Wells.inj_well.InjectionZUpper 6.0

*double* **Wells.\ *well_name*.ZLower** no default This key specifies
the z location of the lower extent of a vectical well if the input type
is set to **Vectical**.

.. container:: list

   ::

      Wells.test_well.ZLower 2.0

*double* **Wells.\ *well_name*.ExtractionZLower** no default This key
specifies the z location of the lower extent of a extraction well if the
input type is set to **Recirc**.

.. container:: list

   ::

      Wells.ext_well.ExtractionZLower 1.0

*double* **Wells.\ *well_name*.InjectionZLower** no default This key
specifies the z location of the lower extent of a injection well if the
input type is set to **Recirc**.

.. container:: list

   ::

      Wells.inj_well.InjectionZLower 4.0

*string* **Wells.\ *well_name*.Method** no default This key specifies a
method by which pressure or flux for a vertical well will be weighted
before assignment to computational cells. This key can only be
**Standard** if the type key is set to **Pressure**; or this key can be
either **Standard**, **Weighted** or **Patterned** if the type key is
set to **Flux**. A value of **Standard** indicates that the pressure or
flux data will be used as is. A value of **Weighted** indicates that the
flux data is to be weighted by the cells permeability divided by the sum
of all cell permeabilities which define the well. The value of
**Patterned** is not implemented.

.. container:: list

   ::

      Wells.test_well.Method "Weighted"

*string* **Wells.\ *well_name*.ExtractionMethod** no default This key
specifies a method by which pressure or flux for an extraction well will
be weighted before assignment to computational cells. This key can only
be **Standard** if the type key is set to **Pressure**; or this key can
be either **Standard**, **Weighted** or **Patterned** if the type key is
set to **Flux**. A value of **Standard** indicates that the pressure or
flux data will be used as is. A value of **Weighted** indicates that the
flux data is to be weighted by the cells permeability divided by the sum
of all cell permeabilities which define the well. The value of
**Patterned** is not implemented.

.. container:: list

   ::

      Wells.ext_well.ExtractionMethod "Standard"

*string* **Wells.\ *well_name*.InjectionMethod** no default This key
specifies a method by which pressure or flux for an injection well will
be weighted before assignment to computational cells. This key can only
be **Standard** if the type key is set to **Pressure**; or this key can
be either **Standard**, **Weighted** or **Patterned** if the type key is
set to **Flux**. A value of **Standard** indicates that the pressure or
flux data will be used as is. A value of **Weighted** indicates that the
flux data is to be weighted by the cells permeability divided by the sum
of all cell permeabilities which define the well. The value of
**Patterned** is not implemented.

.. container:: list

   ::

      Wells.inj_well.InjectionMethod "Standard"

*string* **Wells.\ *well_name*.Cycle** no default This key specifies the
time cycles to which data for the well *well_name* corresponds.

.. container:: list

   ::

      Wells.test_well.Cycle "all_time"

*double* **Wells.\ *well_name*.\ *interval_name*.Pressure.Value** no
default This key specifies the hydrostatic pressure value for a vectical
well if the type key is set to **Pressure**.

Note This value gives the pressure of the primary phase (water) at
:math:`z=0`. The other phase pressures (if any) are computed from the
physical relationships that exist between the phases.

.. container:: list

   ::

      Wells.test_well.all_time.Pressure.Value 6.0

*double*
**Wells.\ *well_name*.\ *interval_name*.Extraction.Pressure.Value** no
default This key specifies the hydrostatic pressure value for an
extraction well if the extraction type key is set to **Pressure**.

Note This value gives the pressure of the primary phase (water) at
:math:`z=0`. The other phase pressures (if any) are computed from the
physical relationships that exist between the phases.

.. container:: list

   ::

      Wells.ext_well.all_time.Extraction.Pressure.Value 4.5

*double*
**Wells.\ *well_name*.\ *interval_name*.Injection.Pressure.Value** no
default This key specifies the hydrostatic pressure value for an
injection well if the injection type key is set to **Pressure**.

Note This value gives the pressure of the primary phase (water) at
:math:`z=0`. The other phase pressures (if any) are computed from the
physical relationships that exist between the phases.

.. container:: list

   ::

      Wells.inj_well.all_time.Injection.Pressure.Value 10.2

*double*
**Wells.\ *well_name*.\ *interval_name*.Flux.\ *phase_name*.Value** no
default This key specifies the volumetric flux for a vectical well if
the type key is set to **Flux**.

Note only a positive number should be entered, ParFlow assigns the
correct sign based on the chosen action for the well.

.. container:: list

   ::

      Wells.test_well.all_time.Flux.water.Value 250.0

*double*
**Wells.\ *well_name*.\ *interval_name*.Extraction.Flux.\ *phase_name*.Value**
no default This key specifies the volumetric flux for an extraction well
if the extraction type key is set to **Flux**.

Note only a positive number should be entered, ParFlow assigns the
correct sign based on the chosen action for the well.

.. container:: list

   ::

      Wells.ext_well.all_time.Extraction.Flux.water.Value 125.0

*double*
**Wells.\ *well_name*.\ *interval_name*.Injection.Flux.\ *phase_name*.Value**
no default This key specifies the volumetric flux for an injection well
if the injection type key is set to **Flux**.

Note only a positive number should be entered, ParFlow assigns the
correct sign based on the chosen action for the well.

.. container:: list

   ::

      Wells.inj_well.all_time.Injection.Flux.water.Value 80.0

*double*
**Wells.\ *well_name*.\ *interval_name*.Saturation.\ *phase_name*.Value**
no default This key specifies the saturation value of a vertical well.

.. container:: list

   ::

      Wells.test_well.all_time.Saturation.water.Value 1.0

*double*
**Wells.\ *well_name*.\ *interval_name*.Concentration.\ *phase_name*.\ *contaminant_name*.Value**
no default This key specifies the contaminant value of a vertical well.

.. container:: list

   ::

      Wells.test_well.all_time.Concentration.water.tce.Value 0.0005

*double*
**Wells.\ *well_name*.\ *interval_name*.Injection.Concentration.\ *phase_name*.\ *contaminant_name*.Fraction**
no default This key specifies the fraction of the extracted contaminant
which gets resupplied to the injection well.

.. container:: list

   ::

      Wells.inj_well.all_time.Injection.Concentration.water.tce.Fraction 0.01

Multiple wells assigned to one grid location can occur in several
instances. The current actions taken by the code are as follows:

-  If multiple pressure wells are assigned to one grid cell, the code
   retains only the last set of overlapping well values entered.

-  If multiple flux wells are assigned to one grid cell, the code sums
   the contributions of all overlapping wells to get one effective well
   flux.

-  If multiple pressure and flux wells are assigned to one grid cell,
   the code retains the last set of overlapping hydrostatic pressure
   values entered and sums all the overlapping flux well values to get
   an effective pressure/flux well value.

.. _Code Parameters:

Code Parameters
~~~~~~~~~~~~~~~

In addition to input keys related to the physics capabilities and
modeling specifics there are some key values used by various algorithms
and general control flags for ParFlow. These are described next :

*string* **Solver.Linear** PCG This key specifies the linear solver used
for solver **IMPES**. Choices for this key are **MGSemi, PPCG, PCG** and
**CGHS**. The choice **MGSemi** is an algebraic mulitgrid linear solver
(not a preconditioned conjugate gradient) which may be less robust than
**PCG** as described in :cite:t:`Ashby-Falgout90`. The choice
**PPCG** is a preconditioned conjugate gradient solver. The choice
**PCG** is a conjugate gradient solver with a multigrid preconditioner.
The choice **CGHS** is a conjugate gradient solver.

.. container:: list

   ::

      pfset Solver.Linear   "MGSemi"         ## TCL syntax

      <runname>.Solver.Linear = "MGSemi"     ## Python syntax

*integer* **Solver.SadvectOrder** 2 This key controls the order of the
explicit method used in advancing the saturations. This value can be
either 1 for a standard upwind first order or 2 for a second order
Godunov method.

.. container:: list

   ::

      pfset Solver.SadvectOrder 1         ## TCL syntax

      <runname>.Solver.SadvectOrder = 1   ## Python syntax

*integer* **Solver.AdvectOrder** 2 This key controls the order of the
explicit method used in advancing the concentrations. This value can be
either 1 for a standard upwind first order or 2 for a second order
Godunov method.

.. container:: list

   ::

      pfset Solver.AdvectOrder 2          ## TCL syntax

      <runname>.Solver.AdvectOrder = 2    ## Python syntax

*double* **Solver.CFL** 0.7 This key gives the value of the weight put
on the computed CFL limit before computing a global timestep value.
Values greater than 1 are not suggested and in fact because this is an
approximation, values slightly less than 1 can also produce
instabilities.

.. container:: list

   ::

      pfset Solver.CFL 0.7          ## TCL syntax

      <runname>.Solver.CFL = 0.7    ## Python syntax

*integer* **Solver.MaxIter** 1000000 This key gives the maximum number
of iterations that will be allowed for time-stepping. This is to prevent
a run-away simulation.

.. container:: list

   ::

      pfset Solver.MaxIter 100         ## TCL syntax

      <runname>.Solver.MaxIter = 100   ## Python syntax

*double* **Solver.RelTol** 1.0 This value gives the relative tolerance
for the linear solve algorithm.

.. container:: list

   ::

      pfset Solver.RelTol 1.0          ## TCL syntax

      <runname>.Solver.RelTol = 1.0    ## Python syntax

*double* **Solver.AbsTol** 1E-9 This value gives the absolute tolerance
for the linear solve algorithm.

.. container:: list

   ::

      pfset Solver.AbsTol 1E-8         ## TCL syntax

      <runname>.Solver.AbsTol = 1E-8   ## Python syntax

*double* **Solver.Drop** 1E-8 This key gives a clipping value for data
written to PFSB files. Data values greater than the negative of this
value and less than the value itself are treated as zero and not written
to PFSB files.

.. container:: list

   ::

      pfset Solver.Drop 1E-6           ## TCL syntax

      <runname>.Solver.Drop = 1E-6     ## Python syntax

*double* **Solver.OverlandDiffusive.Epsilon** 1E-5 This key provides a
minimum value for the :math:`\bar{S_{f}}` used in the
**OverlandDiffusive** boundary condition.

::

   pfset Solver.OverlandDiffusive.Epsilon 1E-7           ## TCL syntax

   <runname>.Solver.OverlandDiffusive.Epsilon = 1E-7     ## Python syntax

*double* **Solver.OverlandKinematic.Epsilon** 1E-5 This key provides a
minimum value for the :math:`\bar{S_{f}}` used in the
**OverlandKinematic** boundary condition.

::

      pfset Solver.OverlandKinematic.Epsilon 1E-7           ## TCL syntax

      <runname>.Solver.OverlandKinematic.Epsilon = 1E-7     ## Python syntax


*string* **Solver.PrintInitialConditions** True This key is used to
      turn on printing of the initial conditions.  This includes the
      pressure, saturation, slopes, etc.  By default the initial
      conditions output is generated before the first time
      advancement; when doing a restart this leads to a duplication
      of files on each restart.  Setting this key to False will
      prevent the duplication.

      Note setting this key to False overrides the other individual
      output flags that are enabled.

.. container:: list

   ::

      pfset Solver.PrintInitialConditions False           ## TCL syntax

      <runname>.Solver.PrintInitalConditions = False     ## Python syntax


*string* **Solver.PrintSubsurf** True This key is used to turn on
printing of the subsurface data, Permeability and Porosity. The data is
printed after it is generated and before the main time stepping loop -
only once during the run. The data is written as a ParFlow binary file.

.. container:: list

   ::

      pfset Solver.PrintSubsurf False           ## TCL syntax

      <runname>.Solver.PrintSubsurf = False     ## Python syntax

*string* **Solver.PrintChannelWidth** True This key is used to turn on
printing of the channelwidth data, ChannelWidthX and ChannelWidthY. The data
is printed before the main time stepping loop - only once during the run.
The data is written as two ParFlow binary files.

.. container:: list

   ::

      pfset Solver.PrintChannelWidth False           ## TCL syntax

      <runname>.Solver.PrintChannelWidth = False     ## Python syntax      

*string* **Solver.PrintPressure** True This key is used to turn on
printing of the pressure data. The printing of the data is controlled by
values in the timing information section. The data is written as a PFB
file.

.. container:: list

   ::

      pfset Solver.PrintPressure False          ## TCL syntax

      <runname>.Solver.PrintPressure = False    ## Python syntax

*string* **Solver.PrintVelocities** False This key is used to turn on
printing of the x, y, and z velocity (Darcy flux) data. The printing of
the data is controlled by values in the timing information section. The
x, y, and z data are written to separate ParFlow binary files. The dimensions of
these files are slightly different than most PF data, with the dimension
of interest representing interfaces, and the other two dimensions
representing cells. E.g. the x-velocity PFB has dimensions [NX+1, NY,
NZ]. This key produces files in the format of
``<runname>.out.phase<x||y||z>.00.0000.pfb`` when using ParFlow’s saturated
solver and ``<runname>.out.vel<x||y||z>.00000.pfb`` when using the Richards
equation solver.

::

   pfset Solver.PrintVelocities True         ## TCL syntax

   <runname>.Solver.PrintVelocities = True   ## Python syntax

*string* **Solver.PrintSaturation** True This key is used to turn on
printing of the saturation data. The printing of the data is controlled
by values in the timing information section. The data is written as a
ParFlow binary file.

.. container:: list

   ::

      pfset Solver.PrintSaturation False        ## TCL syntax

      <runname>.Solver.PrintSaturation = False  ## Python syntax

*string* **Solver.PrintConcentration** True This key is used to turn on
printing of the concentration data. The printing of the data is
controlled by values in the timing information section. The data is
written as a PFSB file.

.. container:: list

   ::

      pfset Solver.PrintConcentration False           ## TCL syntax

      <runname>.Solver.PrintConcentration = False     ## Python syntax


*string* **Solver.PrintTop** False This key is used to turn on printing
of the top of domain data.  'TopZIndex' is a NX * NY file with the Z
index of the top of the domain. 'TopPatch' is the Patch index for the
top of the domain.  A value of -1 indicates an (i,j) column does not
intersect the domain.  The data is written as a ParFlow binary file.

.. container:: list

   ::

      pfset Solver.PrintTop False                    ## TCL syntax

      <runname>.Solver.PrintTop = False              ## Python syntax

*string* **Solver.PrintBottom** False This key is used to turn on printing
of the bottom of domain data.  'BottomZIndex' is a NX * NY file with the Z
index of the top of the domain.  A value of -1 indicates an (i,j) column does 
not intersect the domain.The data is written as a ParFlow binary file.

.. container:: list

   ::

      pfset Solver.PrintBottom False                 ## TCL syntax

      <runname>.Solver.PrintBottom = False           ## Python syntax

*string* **Solver.PrintWells** True This key is used to turn on
collection and printing of the well data. The data is collected at
intervals given by values in the timing information section. Printing
occurs at the end of the run when all collected data is written.

.. container:: list

   ::

      pfset Solver.PrintWells False          ## TCL syntax

      <runname>.Solver.PrintWells = False    ## Python syntax

*string* **Solver.PrintReservoirs** True This key is used to turn on
collection and printing of the reservoir data. The data is collected at
intervals given by values in the timing information section. Printing
occurs at the end of the run when all collected data is written.

.. container:: list

   ::

      pfset Solver.PrintReservoirs False          ## TCL syntax

      <runname>.Solver.PrintReservoirs = False    ## Python syntax

*string* **Solver.PrintLSMSink** False This key is used to turn on
printing of the flux array passed from ``CLM`` to ParFlow. 
Printing occurs at each **DumpInterval** time.

.. container:: list

   ::

      pfset Solver.PrintLSMSink True            ## TCL syntax

      <runname>.Solver.PrintLSMSink = True      ## Python syntax

*string* **Solver.WriteSiloSubsurfData** False This key is used to
specify printing of the subsurface data, Permeability and Porosity in
silo binary file format. The data is printed after it is generated and
before the main time stepping loop - only once during the run. This data
may be read in by VisIT and other visualization packages.

.. container:: list

   ::

      pfset Solver.WriteSiloSubsurfData True          ## TCL syntax

      <runname>.Solver.WriteSiloSubsurfData = True    ## Python syntax

*string* **Solver.WriteSiloPressure** False This key is used to specify
printing of the saturation data in silo binary format. The printing of
the data is controlled by values in the timing information section. This
data may be read in by VisIT and other visualization packages.

.. container:: list

   ::

      pfset Solver.WriteSiloPressure True          ## TCL syntax

      <runname>.Solver.WriteSiloPressure = True    ## Python syntax

*string* **Solver.WriteSiloSaturation** False This key is used to
specify printing of the saturation data using silo binary format. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloSaturation True        ## TCL syntax

      <runname>.Solver.WriteSiloSaturation = True  ## Python syntax

*string* **Solver.WriteSiloConcentration** False This key is used to
specify printing of the concentration data in silo binary format. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloConcentration True           ## TCL syntax

      <runname>.Solver.WriteSiloConcentration = True     ## Python syntax

*string* **Solver.WriteSiloVelocities** False This key is used to
specify printing of the x, y and z velocity data in silo binary format.
The printing of the data is controlled by values in the timing
information section.

.. container:: list

   ::

      pfset Solver.WriteSiloVelocities True           ## TCL syntax

      <runname>.Solver.WriteSiloVelocities = True     ## Python syntax

*string* **Solver.WriteSiloSlopes** False This key is used to specify
printing of the x and y slope data using silo binary format. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloSlopes  True        ## TCL syntax

      <runname>.Solver.WriteSiloSlopes = True   ## Python syntax

*string* **Solver.WriteSiloMannings** False This key is used to specify
printing of the Manning’s roughness data in silo binary format. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloMannings True          ## TCL syntax

      <runname>.Solver.WriteSiloMannings = True    ## Python syntax

*string* **Solver.WriteSiloSpecificStorage** False This key is used to
specify printing of the specific storage data in silo binary format. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloSpecificStorage True         ## TCL syntax

      <runname>.Solver.WriteSiloSpecificStorage = True   ## Python syntax

*string* **Solver.WriteSiloMask** False This key is used to specify
printing of the mask data using silo binary format. The mask contains
values equal to one for active cells and zero for inactive cells. The
printing of the data is controlled by values in the timing information
section.

.. container:: list

   ::

      pfset Solver.WriteSiloMask  True          ## TCL syntax

      <runname>.Solver.WriteSiloMask = True     ## Python syntax

*string* **Solver.WriteSiloEvapTrans** False This key is used to specify
printing of the evaporation and rainfall flux data using silo binary
format. This data comes from either ``clm`` or from external calls to 
ParFlow such as WRF. This data is in units of :math:`[L^3 T^{-1}]`. The printing 
of the data is controlled by values in the timing information section.

.. container:: list

   ::

      pfset Solver.WriteSiloEvapTrans  True        ## TCL syntax

      <runname>.Solver.WriteSiloEvapTrans = True   ## Python syntax

*string* **Solver.WriteSiloEvapTransSum** False This key is used to
specify printing of the evaporation and rainfall flux data using silo
binary format as a running, cumulative amount. This data comes from
either ``clm`` or from external calls to ParFlow such as WRF. This 
data is in units of :math:`[L^3]`. The printing of the data is controlled by 
values in the timing information section.

.. container:: list

   ::

      pfset Solver.WriteSiloEvapTransSum  True           ## TCL syntax

      <runname>.Solver.WriteSiloEvapTransSum = True      ## Python syntax

*string* **Solver.WriteSiloOverlandSum** False This key is used to
specify calculation and printing of the total overland outflow from the
domain using silo binary format as a running cumulative amount. This is
integrated along all domain boundaries and is calculated any location
that slopes at the edge of the domain point outward. This data is in
units of :math:`[L^3]`. The printing of the data is controlled by values
in the timing information section.

.. container:: list

   ::

      pfset Solver.WriteSiloOverlandSum  True            ## TCL syntax

      <runname>.Solver.WriteSiloOverlandSum = True       ## Python syntax

*string* **Solver.WriteSiloTop** False Key used to control writing of
two Silo files for the top of the domain. 'TopZIndex' is a NX * NY
file with the Z index of the top of the domain. 'TopPatch' is the
Patch index for the top of the domain.  A value of -1 indicates an
(i,j) column does not intersect the domain.

.. container:: list

   ::

      pfset Solver.WriteSiloTop True                  ## TCL syntax

      <runname>.Solver.WriteSiloTop = True            ## Python syntax

*string* **Solver.WriteSiloBottom** False Key used to control writing of
one Silo file for the bottom of the domain.  'BottomZIndex' is a NX * NY
file with the Z index of the bottom of the domain. A value of -1 indicates
an (i,j) column does not intersect the domain.

.. container:: list

   ::

      pfset Solver.WriteSiloBottom True               ## TCL syntax

      <runname>.Solver.WriteSiloBottom = True         ## Python syntax

*string* **Solver.WritePDISubsurfData** False This key is used to specify exposing of
      the subsurface data, Permeability and Porosity to PDI library. The data is
      exposed after it is generated and before the main time stepping loop - only
      once during the run. The data is subsequently managed by the PDI
      plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDISubsurfData  True            ## TCL syntax

      <runname>.Solver.WritePDISubsurfData = True       ## Python syntax

*string* **Solver.WritePDIMannings** False This key is used to specify exposing of
      Manning’s roughness data to PDI library. The data exposure is controlled
      by values in the timing information section and is subsequently managed
      by the PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIMannings  True            ## TCL syntax

      <runname>.Solver.WritePDIMannings = True       ## Python syntax

*string* **Solver.WritePDISlopes** False This key is used to turn on exposure of x
      and y slope data to PDI library. The data exposure is controlled by values
      in the timing information section and is subsequently managed by the PDI
      plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDISlopes  True            ## TCL syntax

      <runname>.Solver.WritePDISlopes = True       ## Python syntax

*string* **Solver.WritePDIPressure** False This key is used to specify exposure of
      pressure data to PDI library. The data exposure is controlled by values
      in the timing information section and is subsequently managed by the PDI
      plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIPressure  True            ## TCL syntax

      <runname>.Solver.WritePDIPressure = True       ## Python syntax

*string* **Solver.WritePDISpecificStorage** False This key is used to specify exposure
      of specific storage data to PDI library. The data exposure is controlled
      by values in the timing information section and is subsequently managed by
      the PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDISpecificStorage  True            ## TCL syntax

      <runname>.Solver.WritePDISpecificStorage = True       ## Python syntax

*string* **Solver.WritePDIVelocities** False This key is used to turn on exposure of
      x,y,and z velocity data to PDI library. The data exposure is controlled by
      values in the timing information section and is subsequently managed by the
      PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIVelocities  True            ## TCL syntax

      <runname>.Solver.WritePDIVelocities = True       ## Python syntax

*string* **Solver.WritePDISaturation** False This key is used to specify exposre of
      the saturation data to PDI library. The data exposure is controlled by
      values in the timing information section and is subsequently managed by
      the PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDISaturation  True            ## TCL syntax

      <runname>.Solver.WritePDISaturation = True       ## Python syntax

*string* **Solver.WritePDIMask** False This key is used to specify exposure of mask
      data to PDI library. The mask contains values equal to one for active
      cells and zero for inactive cells. The data exposure is controlled by
      values in the timing information section and is subsequently managed by
      the PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIMask  True            ## TCL syntax

      <runname>.Solver.WritePDIMask = True       ## Python syntax

*string* **Solver.WritePDIDZMultiplier** False This key is used to specifiy the exposrue
      of DZ multipliers to PDI library.

.. container:: list

   ::

      pfset Solver.WritePDIDZMultiplier  True            ## TCL syntax

      <runname>.Solver.WritePDIDZMultiplier = True       ## Python syntax

*string* **Solver.WritePDIEvapTransSum** False This key is used to specify exposure
      of evaporation and rainfall flux data to PDI libraary, cumulative amount.
      This data comes from either clm or from external calls to ParFlow such as
      WRF. This data is in units of :math:`[L3]`. The data exposure is controlled
      by values in the timing information section and is subsequently managed by
      the PDI plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIEvapTransSum  True            ## TCL syntax

      <runname>.Solver.WritePDIEvapTransSum = True       ## Python syntax

*string* **Solver.WritePDIEvapTrans** False This key is used to specify exposure
      of the evaporation and rainfall flux data to PDI library. This data comes
      from either clm or from external calls to ParFlow such as WRF. This data
      is in units of :math:`[L3T-1]`. The data exposure is controlled by values
      in the timing information section and is subsequently managed by the PDI
      plugin according to the specification tree defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIEvapTrans  True            ## TCL syntax

      <runname>.Solver.WritePDIEvapTrans = True       ## Python syntax

*string* **Solver.WritePDIOverlandSum** False This key is used to specify
      calculation and exposrue of the total overland outflow from the domain
      PDI library as a running cumulative amount. This is integrated along all
      domain boundaries and is calculated any location that slopes at the edge
      of the domain point outward. This data is in units of :math:`[L^3]`. The
      data exposure is controlled by values in the timing information section and
      is subsequently managed by the PDI plugin according to the specification tree
      defined in conf.yaml.

.. container:: list

   ::

      pfset Solver.WritePDIOverlandSum  True            ## TCL syntax

      <runname>.Solver.WritePDIOverlandSum = True       ## Python syntax    

*string* **Solver.WritePDIOverlandBCFlux** False This key is used to specify the
      expousre of overland bc flux to PDI library.

.. container:: list

   ::

      pfset Solver.WritePDIOverlandBCFlux	True        ## TCL syntax
      <runname>.Solver.WritePDIOverlandBCFlux = True  ## Python syntax

*string* **Solver.WritePDIWells** False This key is used to specify the
      expousre of wells data to PDI library.

.. container:: list

   ::

      pfset Solver.WritePDIWells	True        ## TCL syntax
      <runname>.Solver.WritePDIWells = True  ## Python syntax

*string* **Solver.WritePDIConcentration** False This key is used to specify the
      exposure of concentration data to PDI library. The data exposure is
      controlled by values in the timing information section.

.. container:: list

   ::

      pfset Solver.WritePDIConcentration	True        ## TCL syntax
      <runname>.Solver.WritePDIConcentration = True  ## Python syntax

*string* **Solver.TerrainFollowingGrid** False This key specifies that a
terrain-following coordinate transform is used for solver Richards. This
key sets x and y subsurface slopes to be the same as the Topographic
slopes (a value of False sets these subsurface slopes to zero). These
slopes are used in the Darcy fluxes to add a density, gravity -dependent
term. This key will not change the output files (that is the output is
still orthogonal) or the geometries (they will still follow the
computational grid)– these two things are both to do items. This key
only changes solver Richards, not solver Impes.

.. container:: list

   ::

      pfset Solver.TerrainFollowingGrid  True         ## TCL syntax

      <runname>.Solver.TerrainFollowingGrid = True    ## Python syntax

*string* **Solver.TerrainFollowingGrid.SlopeUpwindFormulation** Original
This key specifies optional modifications to the terrain following grid
formulation described in :ref:`TFG`. Choices for
this key are **Original, Upwind, UpwindSine**. **Original** is the
original TFG formulation documented in :cite:p:`M13`.
The **Original** option calculates the :math:`\theta_x` and
:math:`\theta_y` for a cell face as the average of the two adjacent cell
slopes (i.e. assuming a cell centered slope calculation). The **Upwind**
option uses the the :math:`\theta_x` and :math:`\theta_y` of a cell
directly without averaging (i.e. assuming a face centered slope
calculation). The **UpwindSine** is the same as the **Upwind** option
but it also removes the Sine term from the TFG Darcy Formulation (in :ref:`TFG`).
Note the **UpwindSine** option is for experimental purposes only and
should not be used in standard simulations. Also note that the choice of
**upwind** or **Original** formulation should consistent with the
choice of overland flow boundary condition if overland flow is being
used. The **upwind** and **UpwindSine** are consistent with
**OverlandDiffusive** and **OverlandKinematic** while **Original** is
consistent with **OverlandFow**

::

   pfset Solver.TerrainFollowingGrid.SlopeUpwindFormulation   "Upwind"        ## TCL syntax

   <runname>.Solver.TerrainFollowingGrid.SlopeUpwindFormulation = "Upwind"    ## Python syntax


   


.. _SILO Options:

SILO Options
~~~~~~~~~~~~

The following keys are used to control how SILO writes data. SILO allows
writing to PDB and HDF5 file formats. SILO also allows data compression
to be used, which can save signicant amounts of disk space for some
problems.

*string* **SILO.Filetype** PDB This key is used to specify the SILO
filetype. Allowed values are PDB and HDF5. Note that you must have
configured SILO with HDF5 in order to use that option.

.. container:: list

   ::

      pfset SILO.Filetype  "PDB"       ## TCL syntax

      <runname>.SILO.Filetype = "PDB"  ## Python syntax

*string* **SILO.CompressionOptions** This key is used to specify the
SILO compression options. See the SILO manual for the DB_SetCompression
command for information on available options. NOTE: the options
available are highly dependent on the configure options when building
SILO.

.. container:: list

   ::

      pfset SILO.CompressionOptions  "METHOD=GZIP"          ## TCL syntax

      <runname>.SILO.CompressionOptions = "METHOD=GZIP"     ## Python syntax

.. _RE Solver Parameters:

Richards’ Equation Solver Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following keys are used to specify various parameters used by the
linear and nonlinear solvers in the Richards’ equation implementation.
For information about these solvers, see :cite:t:`Woodward98`
and :cite:t:`Ashby-Falgout90`.

*double* **Solver.Nonlinear.ResidualTol** 1e-7 This key specifies the
tolerance that measures how much the relative reduction in the nonlinear
residual should be before nonlinear iterations stop. The magnitude of
the residual is measured with the :math:`l^1` (max) norm.

.. container:: list

   ::

      pfset Solver.Nonlinear.ResidualTol   1e-4          ## TCL syntax

      <runname>.Solver.Nonlinear.ResidualTol = 1e-4      ## Python syntax

*double* **Solver.Nonlinear.StepTol** 1e-7 This key specifies the
tolerance that measures how small the difference between two consecutive
nonlinear steps can be before nonlinear iterations stop.

.. container:: list

   ::

      pfset Solver.Nonlinear.StepTol   1e-4        ## TCL syntax

      <runname>.Solver.Nonlinear.StepTol = 1e-4    ## Python syntax

*integer* **Solver.Nonlinear.MaxIter** 15 This key specifies the maximum
number of nonlinear iterations allowed before iterations stop with a
convergence failure.

.. container:: list

   ::

      pfset Solver.Nonlinear.MaxIter   50       ## TCL syntax

      <runname>.Solver.Nonlinear.MaxIter = 50   ## Python syntax

*integer* **Solver.Linear.KrylovDimension** 10 This key specifies the
maximum number of vectors to be used in setting up the Krylov subspace
in the GMRES iterative solver. These vectors are of problem size and it
should be noted that large increases in this parameter can limit problem
sizes. However, increasing this parameter can sometimes help nonlinear
solver convergence.

.. container:: list

   ::

      pfset Solver.Linear.KrylovDimension   15        ## TCL syntax

      <runname>.Solver.Linear.KrylovDimension = 15    ## Python syntax

*integer* **Solver.Linear.MaxRestarts** 0 This key specifies the number
of restarts allowed to the GMRES solver. Restarts start the development
of the Krylov subspace over using the current iterate as the initial
iterate for the next pass.

.. container:: list

   ::

      pfset Solver.Linear.MaxRestarts   2       ## TCL syntax

      <runname>.Solver.Linear.MaxRestarts = 2   ## Python syntax

*integer* **Solver.MaxConvergenceFailures** 3 This key gives the maximum
number of convergence failures allowed. Each convergence failure cuts
the timestep in half and the solver tries to advance the solution with
the reduced timestep.

The default value is 3.

Note that setting this value to a value greater than 9 may result in
errors in how time cycles are calculated. Time is discretized in terms
of the base time unit and if the solver begins to take very small
timesteps :math:`smaller than base time unit 1000` the values based
on time cycles will be change at slightly incorrect times. If the
problem is failing converge so poorly that a large number of restarts
are required, consider setting the timestep to a smaller value.

.. container:: list

   ::

      pfset Solver.MaxConvergenceFailures 4           ## TCL syntax

      <runname>.Solver.MaxConvergenceFailures = 4     ## Python syntax

*string* **Solver.Nonlinear.PrintFlag** HighVerbosity This key specifies
the amount of informational data that is printed to the ``*.out.kinsol.log`` 
file. Choices for this key are **NoVerbosity**, **LowVerbosity**, **NormalVerbosity** 
and **HighVerbosity**. The choice **NoVerbosity** prints no statistics about the 
nonlinear convergence process. The choice **LowVerbosity** outputs the nonlinear 
iteration count, the scaled norm of the nonlinear function, and the number of 
function calls. The choice **NormalVerbosity** prints the same as for **LowVerbosity** 
and also the global strategy statistics. The choice **HighVerbosity** prints the 
same as for **NormalVerbosity** with the addition of further Krylov iteration 
statistics.

.. container:: list

   ::

      pfset Solver.Nonlinear.PrintFlag   "NormalVerbosity"        ## TCL syntax

      <runname>.Solver.Nonlinear.PrintFlag = "NormalVerbosity"    ## Python syntax

*string* **Solver.Nonlinear.EtaChoice** Walker2 This key specifies how
the linear system tolerance will be selected. The linear system is
solved until a relative residual reduction of :math:`\eta` is achieved.
Linear residual norms are measured in the :math:`l^2` norm. Choices for
this key include **EtaConstant, Walker1** and **Walker2**. If the choice
**EtaConstant** is specified, then :math:`\eta` will be taken as
constant. The choices **Walker1** and **Walker2** specify choices for
:math:`\eta` developed by Eisenstat and Walker :cite:p:`EW96`.
The choice **Walker1** specifies that :math:`\eta` will be given by
:math:`| \|F(u^k)\| - \|F(u^{k-1}) + J(u^{k-1})*p \|  |  / \|F(u^{k-1})\|`.
The choice **Walker2** specifies that :math:`\eta` will be given by
:math:`\gamma \|F(u^k)\| / \|F(u^{k-1})\|^{\alpha}`. For both of the
last two choices, :math:`\eta` is never allowed to be less than 1e-4.

.. container:: list

   ::

      pfset Solver.Nonlinear.EtaChoice   "EtaConstant"         ## TCL syntax

      <runname>.Solver.Nonlinear.EtaChoice = "EtaConstant"     ## Python syntax

*double* **Solver.Nonlinear.EtaValue** 1e-4 This key specifies the
constant value of :math:`\eta` for the EtaChoice key **EtaConstant**.

.. container:: list

   ::

      pfset Solver.Nonlinear.EtaValue   1e-7          ## TCL syntax

      <runname>.Solver.Nonlinear.EtaValue = 1e-7      ## Python syntax

*double* **Solver.Nonlinear.EtaAlpha** 2.0 This key specifies the value
of :math:`\alpha` for the case of EtaChoice being **Walker2**.

.. container:: list

   ::

      pfset Solver.Nonlinear.EtaAlpha   1.0        ## TCL syntax

      <runname>.Solver.Nonlinear.EtaAlpha = 1.0    ## Python syntax

*double* **Solver.Nonlinear.EtaGamma** 0.9 This key specifies the value
of :math:`\gamma` for the case of EtaChoice being **Walker2**.

.. container:: list

   ::

      pfset Solver.Nonlinear.EtaGamma   0.7        ## TCL syntax

      <runname>.Solver.Nonlinear.EtaGamma = 0.7    ## Python syntax

*string* **Solver.Nonlinear.UseJacobian** False This key specifies
whether the Jacobian will be used in matrix-vector products or whether a
matrix-free version of the code will run. Choices for this key are
**False** and **True**. Using the Jacobian will most likely decrease the
number of nonlinear iterations but require more memory to run.

.. container:: list

   ::

      pfset Solver.Nonlinear.UseJacobian   True          ## TCL syntax

      <runname>.Solver.Nonlinear.UseJacobian = True      ## Python syntax

*double* **Solver.Nonlinear.DerivativeEpsilon** 1e-7 This key specifies
the value of :math:`\epsilon` used in approximating the action of the
Jacobian on a vector with approximate directional derivatives of the
nonlinear function. This parameter is only used when the UseJacobian key
is **False**.

.. container:: list

   ::

      pfset Solver.Nonlinear.DerivativeEpsilon   1e-8       ## TCL syntax

      <runname>.Solver.Nonlinear.DerivativeEpsilon = 1e-8   ## Python syntax

*string* **Solver.Nonlinear.Globalization** LineSearch This key
specifies the type of global strategy to use. Possible choices for this
key are **InexactNewton** and **LineSearch**. The choice
**InexactNewton** specifies no global strategy, and the choice
**LineSearch** specifies that a line search strategy should be used
where the nonlinear step can be lengthened or decreased to satisfy
certain criteria.

.. container:: list

   ::

      pfset Solver.Nonlinear.Globalization   "LineSearch"         ## TCL syntax

      <runname>.Solver.Nonlinear.Globalization = "LineSearch"     ## Python syntax

*string* **Solver.Linear.Preconditioner** MGSemi This key specifies
which preconditioner to use. Currently, the three choices are **NoPC,
MGSemi, PFMG, PFMGOctree** and **SMG**. The choice **NoPC** specifies
that no preconditioner should be used. The choice **MGSemi** specifies a
semi-coarsening multigrid algorithm which uses a point relaxation
method. The choice **SMG** specifies a semi-coarsening multigrid
algorithm which uses plane relaxations. This method is more robust than
**MGSemi**, but generally requires more memory and compute time. The
choice **PFMGOctree** can be more efficient for problems with large
numbers of inactive cells.

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner   "MGSemi"         ## TCL syntax

      <runname>.Solver.Linear.Preconditioner = "MGSemi"     ## Python syntax

*string* **Solver.Linear.Preconditioner.SymmetricMat** Symmetric This
key specifies whether the preconditioning matrix is symmetric. Choices
for this key are **Symmetric** and **Nonsymmetric**. The choice
**Symmetric** specifies that the symmetric part of the Jacobian will be
used as the preconditioning matrix. The choice **Nonsymmetric**
specifies that the full Jacobian will be used as the preconditioning
matrix. NOTE: ONLY **Symmetric** CAN BE USED IF MGSemi IS THE SPECIFIED
PRECONDITIONER!

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner.SymmetricMat     "Symmetric"      ## TCL syntax

      <runname>.Solver.Linear.Preconditioner.SymmetricMat = "Symmetric"    ## Python syntax

*integer* **Solver.Linear.Preconditioner.\ *precond_method*.MaxIter** 1
This key specifies the maximum number of iterations to take in solving
the preconditioner system with *precond_method* solver.

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner.SMG.MaxIter    2         ## TCL syntax

      <runname>.Solver.Linear.Preconditioner.SMG.MaxIter = 2      ## Python syntax

*integer* **Solver.Linear.Preconditioner.SMG.NumPreRelax** 1 This key
specifies the number of relaxations to take before coarsening in the
specified preconditioner method. Note that this key is only relevant to
the SMG multigrid preconditioner.

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner.SMG.NumPreRelax    2        ## TCL syntax

      <runname>.Solver.Linear.Preconditioner.SMG.NumPreRelax = 2     ## Python syntax

*integer* **Solver.Linear.Preconditioner.SMG.NumPostRelax** 1 This key
specifies the number of relaxations to take after coarsening in the
specified preconditioner method. Note that this key is only relevant to
the SMG multigrid preconditioner.

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner.SMG.NumPostRelax    0       ## TCL syntax

      <runname>.Solver.Linear.Preconditioner.SMG.NumPostRelax = 0    ## Python syntax

*string* **Solver.Linear.Preconditioner.PFMG.RAPType** NonGalerkin For
the PFMG solver, this key specifies the *Hypre* RAP type. Valid values
are **Galerkin** or **NonGalerkin**

.. container:: list

   ::

      pfset Solver.Linear.Preconditioner.PFMG.RAPType    "Galerkin"     ## TCL syntax

      <runname>.Solver.Linear.Preconditioner.PFMG.RAPType = "Galerkin"  ## Python syntax


*logical* **Solver.ResetSurfacePressure** False This key changes any surface pressure greater than a threshold value to 
another value in between solver timesteps. It works differently than the Spinup keys and is intended to 
help with slope errors and issues and provides some diagnostic information.  The threshold keys are specified below.

.. container:: list

   ::

      pfset Solver.ResetSurfacePressure        True        ## TCL syntax
      <runname>.Solver.ResetSurfacePressure  = "True"    ## Python syntax

*double* **Solver.ResetSurfacePressure.ThresholdPressure** 0.0 This key specifies a threshold value used in the **ResetSurfacePressure** key above.

.. container:: list

   ::

      pfset Solver.ResetSurfacePressure.ThresholdPressure        10.0        ## TCL syntax
      <runname>.Solver.ResetSurfacePressure.ThresholdPressure  = 10.0    ## Python syntax

*double* **Solver.ResetSurfacePressure.ResetPressure** 0.0 This key specifies a reset value used in the **ResetSurfacePressure** key above.

.. container:: list

   ::

      pfset Solver.ResetSurfacePressure.ResetPressure        0.0        ## TCL syntax
      <runname>.Solver.ResetSurfacePressure.ResetPressure  = 0.0    ## Python syntax


*logical* **Solver.SurfacePredictor** False This key activates a routine that uses the evap trans flux, Darcy flux, and available water storage in a surface cell to predict whether an unsaturated cell will pond during the next timestep. The pressure values are set with the key below.
.. container:: list

   ::

      pfset Solver.SurfacePredictor        True        ## TCL syntax
      <runname>.Solver.SurfacePredictor  = "True"    ## Python syntax

*double* **Solver.SurfacePredictor.PressureValue** 0.00001 This key specifies a surface pressure if the **SurfacePredictor** key above is True and ponded conditions are predicted at a surface cell.  A negative value allows the surface predictor algorithm to estimate the new surface pressure based on surrounding fluxes.

.. container:: list

   ::

      pfset Solver.SurfacePredictor.PressureValue        0.001        ## TCL syntax
      <runname>.Solver.SurfacePredictor.PressureValue  = 0.001    ## Python syntax

*logical* **Solver.SurfacePredictor.PrintValues** False This key specifies if the **SurfacePredictor** values are printed.

.. container:: list

   ::

      pfset Solver.SurfacePredictor.PrintValues        True        ## TCL syntax
      <runname>.Solver.SurfacePredictor.PrintValue  = "True"    ## Python syntax


*logical* **Solver.EvapTransFile** False This key specifies specifies
that the Flux terms for Richards’ equation are read in from a ParFlow 3D binary
file. This file has [T^-1] units corresponding to the flux value(s) divided by
the layer thickness **DZ**. Note this key is for a steady-state flux
and should not be used in conjunction with the transient key below.

.. container:: list

   ::

      pfset Solver.EvapTransFile    True        ## TCL syntax

      <runname>.Solver.EvapTransFile = True     ## Python syntax

*logical* **Solver.EvapTransFileTransient** False This key specifies
specifies that the Flux terms for Richards’ equation are read in from a
series of ParFlow 3D binary files. Each file has :math:`[T^-1]` units
corresponding to the flux value(s) divided by the layer thickness **DZ**.
Note this key should not be used with the key above, only one of these
keys should be set to ``True`` at a time, not both.

.. container:: list

   ::

      pfset Solver.EvapTransFileTransient    True        ## TCL syntax

      <runname>.Solver.EvapTransFileTransient = True     ## Python syntax

*string* **Solver.EvapTrans.FileName** no default This key specifies
specifies filename for the distributed ParFlow 3D binary file that contains the 
flux values for Richards’ equation. This file has :math:`[T^-1]` units 
corresponding to the flux value(s) divided by the layer thickness **DZ**. 
For the steady-state option (*Solver.EvapTransFile*=**True**) this key 
should be the complete filename. For the transient option 
(*Solver.EvapTransFileTransient*=**True**) then the filename is a header and 
ParFlow will load one file per timestep, with the form ``filename.00000.pfb``.
EvapTrans values are considered as sources or sinks in Richards' equation, so
they have no conflicts with boundary conditions. Consequently, sign flip is not
required (i.e., incoming flluxes are positive and outgoing fluxes are negative).

.. container:: list

   ::

      pfset Solver.EvapTrans.FileName   "evap.trans.test.pfb"        ## TCL syntax

      <runname>.Solver.EvapTrans.FileName = "evap.trans.test.pfb"    ## Python syntax

*string* **Solver.LSM** none This key specifies whether a land surface
model, such as ``CLM``, will be called each solver timestep. Choices 
for this key include **none** and **CLM**. Note that ``CLM`` must be compiled 
and linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.LSM "CLM"           ## TCL syntax

      <runname>.Solver.LSM = "CLM"     ## Python syntax

.. _Spinup Options:

Spinup Options
~~~~~~~~~~~~~~

These keys allow for *reduced or dampened physics* during model spinup
or initialization. They are **only** intended for these initialization
periods, **not** for regular runtime.

*integer* **OverlandFlowSpinUp** 0 This key specifies that a
*simplified* form of the overland flow boundary condition (Equation
:eq:`overland_bc`) be used in place of the full
equation. This formulation *removes lateral flow* and drives and ponded
water pressures to zero. While this can be helpful in spinning up the
subsurface, this is no longer coupled subsurface-surface flow. If set to
zero (the default) this key behaves normally.

.. container:: list

   ::

      pfset OverlandFlowSpinUp   1        ## TCL syntax
      <runname>.OverlandFlowSpinUp = 1    ## Python syntax

*double* **OverlandFlowSpinUpDampP1** 0.0 This key sets :math:`P_1` and
provides exponential dampening to the pressure relationship in the
overland flow equation by adding the following term:
:math:`P_2*exp(\psi*P_1)`

.. container:: list

   ::

      pfset OverlandSpinupDampP1  10.0       ## TCL syntax
      <runname>.OverlandSpinupDampP1 = 10.0  ## Python syntax

*double* **OverlandFlowSpinUpDampP2** 0.0 This key sets :math:`P_2` and
provides exponential dampening to the pressure relationship in the
overland flow equation adding the following term:
:math:`P_2*exp(\psi*P_1)`

.. container:: list

   ::

      pfset OverlandSpinupDampP2  0.1        ## TCL syntax
      <runname>.OverlandSpinupDampP2 = 0.1   ## Python syntax


*logical* **Solver.SpinUp** False This key removes surface pressure in between solver timesteps.
It works differently than the Spinup keys above as the pressure will build up, then all pressures greater than
zero will be reset to zero.

.. container:: list

   ::

      pfset Solver.SpinUp   True        ## TCL syntax
      <runname>.Solver.SpinUp = "True"    ## Python syntax
      
.. _CLM Solver Parameters:

CLM Solver Parameters
~~~~~~~~~~~~~~~~~~~~~

*string* **Solver.CLM.Print1dOut** False This key specifies 
whether the ``CLM`` one dimensional (averaged over each processor) 
output file is written or not. Choices for this key include True and 
False. Note that ``CLM`` must be compiled and linked at runtime 
for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.Print1dOut   False       ## TCL syntax
      <runname>.Solver.CLM.Print1dOut = False   ## Python syntax

*integer* **Solver.CLM.IstepStart** 1 This key specifies the value of
the counter, *istep* in ``CLM``. This key primarily determines the start 
of the output counter for ``CLM``. It is used to restart a run by setting 
the key to the ending step of the previous run plus one. Note 
that ``CLM`` must be compiled and linked at runtime for this option to 
be active.

.. container:: list

   ::

      pfset Solver.CLM.IstepStart     8761      ## TCL syntax
      <runname>.Solver.CLM.IstepStart = 8761    ## Python syntax   

*String* **Solver.CLM.MetForcing** no default This key specifies defines
whether 1D (uniform over the domain), 2D (spatially distributed) or 3D
(spatially distributed with multiple timesteps per ``.pfb`` forcing file) 
forcing data is used. Choices for this key are **1D**, **2D** and **3D**. This key 
has no default so the user *must* set it to 1D, 2D or 3D. Failure to set 
this key will cause ``CLM`` to still be run but with unpredictable values 
causing ``CLM`` to eventually crash. 1D meteorological forcing files 
are text files with single columns for each variable and each timestep 
per row, while 2D forcing files are distributed ParFlow binary files, one 
for each variable and timestep. File names are specified in the 
**Solver.CLM.MetFileName** variable below. Note that ``CLM`` must be compiled 
and linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.MetForcing   "2D"        ## TCL syntax
      <runname>.Solver.CLM.MetForcing = "2D"    ## Python syntax

*String* **Solver.CLM.MetFileName** no default This key specifies
defines the file name for 1D, 2D or 3D forcing data. 1D meteorological
forcing files are text files with single columns for each variable and
each timestep per row, while 2D and 3D forcing files are distributed
ParFlow binary files, one for each variable and timestep (2D) or one for
each variable and *multiple* timesteps (3D). Behavior of this key is
different for 1D and 2D and 3D cases, as specified by the
**Solver.CLM.MetForcing** key above. For 1D cases, it is the *FULL FILE
NAME*. Note that in this configuration, this forcing file is **not**
distributed, the user does not provide copies such 
as ``narr.1hr.txt.0``, ``narr.1hr.txt.1`` for each processor. ParFlow only 
needs the single original file (*e.g.* ``narr.1hr.txt``). For 2D cases, this key 
is the BASE FILE NAME for the 2D forcing files, currently set to NLDAS, 
with individual files determined as follows ``NLDAS.<variable>.<time step>.pfb``. 
Where the ``<variable>`` is the forcing variable and ``<timestep>`` is the 
integer file counter corresponding to istep above. Forcing is needed 
for following variables:

**DSWR**: 
   Downward Visible or Short-Wave radiation :math:`[W/m^2]`.

**DLWR**: 
   Downward Infa-Red or Long-Wave radiation :math:`[W/m^2]`

**APCP**: 
   Precipitation rate :math:`[mm/s]`

**Temp**: 
   Air temperature :math:`[K]`

**UGRD**: 
   West-to-East or U-component of wind :math:`[m/s]`

**VGRD**: 
   South-to-North or V-component of wind :math:`[m/s]`

**Press**: 
   Atmospheric Pressure :math:`[pa]`

**SPFH**: 
   Water-vapor specific humidity :math:`[kg/kg]` 

Note that ``CLM`` must be compiled and linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.MetFileName     "narr.1hr.txt"    ## TCL syntax
      <runname>.Solver.CLM.MetFileName = "narr.1hr.txt"  ## Python syntax

*String* **Solver.CLM.MetFilePath** no default This key specifies
defines the location of 1D, 2D or 3D forcing data. For 1D cases, this is
the path to a single forcing file (*e.g.* ``narr.1hr.txt``). For 2D and 
3D cases, this is the path to the directory containing all forcing files. 
Note that ``CLM`` must be compiled and linked at runtime for this 
option to be active.

.. container:: list

   ::

      pfset Solver.CLM.MetFilePath "path/to/met/forcing/data/"          ## TCL syntax
      <runname>.Solver.CLM.MetFilePath = "path/to/met/forcing/data/"    ## Python syntax

*integer* **Solver.CLM.MetFileNT** no default This key specifies the
number of timesteps per file for 3D forcing data.

.. container:: list

   ::

      pfset Solver.CLM.MetFileNT	24          ## TCL syntax
      <runname>.Solver.CLM.MetFileNT = 24    ## Python syntax

*string* **Solver.CLM.ForceVegetation** False This key specifies whether
vegetation should be forced in ``CLM``. Currently this option only works 
for 1D and 3D forcings, as specified by the key ``Solver.CLM.MetForcing``. 
Choices for this key include **True** and **False**. Forced vegetation variables 
are :

**LAI**: 
   Leaf Area Index :math:`[-]`

**SAI**: 
   Stem Area Index :math:`[-]`

**Z0M**: 
   Aerodynamic roughness length :math:`[m]`

**DISPLA**: 
   Displacement height :math:`[m]` 

In the case of 1D meteorological forcings, ``CLM`` requires four files 
for vegetation time series and one vegetation map. The four files should 
be named respectively ``lai.dat``, ``sai.dat``, ``z0m.dat``, ``displa.dat``. 
They are ASCII files and contain 18 time-series columns (one per IGBP 
vegetation class, and each timestep per row). The vegetation map should 
be a properly distributed 2D ParFlow binary file (``.pfb``) which contains 
vegetation indices (from 1 to 18). The vegetation map filename is ``veg_map.pfb``. 
ParFlow uses the vegetation map to pass to ``CLM`` a 2D map for each 
vegetation variable at each time step. In the case of 3D meteorological 
forcings, ParFlow expects four distincts properly distributed ParFlow binary 
file (``.pfb``), the third dimension being the timesteps. The files should 
be named ``LAI.pfb``, ``SAI.pfb``, ``Z0M.pfb``, ``DISPLA.pfb``. No 
vegetation map is needed in this case.

.. container:: list

   ::

      pfset Solver.CLM.ForceVegetation  True       ## TCL syntax
      <runname>.Solver.CLM.ForceVegetation = True  ## Python syntax

*string* **Solver.WriteSiloCLM** False This key specifies whether the ``CLM`` 
writes two dimensional binary output files to a silo binary format. This data 
may be read in by VisIT and other visualization packages. Note that ``CLM`` 
and silo must be compiled and linked at runtime for this option to be active. 
These files are all written according to the standard format used for all ParFlow 
variables, using the *runname*, and *istep*. Variables are either two-dimensional 
or over the number of ``CLM`` layers (default of ten).

.. container:: list

   ::

      pfset Solver.WriteSiloCLM True         ## TCL syntax
      <runname>.Solver.WriteSiloCLM = True   ## Python syntax

The output variables are:

.. container:: description

   ``eflx_lh_tot`` for latent heat flux total :math:`[W/m^2]` using the silo variable *LatentHeat*;

   ``eflx_lwrad_out`` for outgoing long-wave radiation :math:`[W/m^2]` using the silo variable *LongWave*;

   ``eflx_sh_tot`` for sensible heat flux total :math:`[W/m^2]` using the silo variable *SensibleHeat*;

   ``eflx_soil_grnd`` for ground heat flux :math:`[W/m^2]` using the silo variable *GroundHeat*;

   ``qflx_evap_tot`` for total evaporation :math:`[mm/s]` using the silo variable *EvaporationTotal*;

   ``qflx_evap_grnd`` for ground evaporation without condensation :math:`[mm/s]` using the silo 
   variable *EvaporationGroundNoSublimation*;

   ``qflx_evap_soi`` for soil evaporation :math:`[mm/s]` using the silo variable *EvaporationGround*;

   ``qflx_evap_veg`` for vegetation evaporation :math:`[mm/s]` using the silo variable *EvaporationCanopy*;

   ``qflx_tran_veg`` for vegetation transpiration :math:`[mm/s]` using the silo variable *Transpiration*;

   ``qflx_infl`` for soil infiltration :math:`[mm/s]` using the silo variable *Infiltration*;

   ``swe_out`` for snow water equivalent :math:`[mm]` using the silo variable *SWE*;

   ``t_grnd`` for ground surface temperature :math:`[K]` using the silo variable *TemperatureGround*; and

   ``t_soil`` for soil temperature over all layers :math:`[K]` using the silo variable *TemperatureSoil*.

*string* **Solver.PrintCLM** False This key specifies whether the ``CLM`` writes two dimensional 
binary output files to a ``PFB`` binary format. Note that ``CLM`` must be compiled and linked 
at runtime for this option to be active. These files are all written according to the 
standard format used for all ParFlow variables, using the *runname*, and *istep*. Variables 
are either two-dimensional or over the number of ``CLM`` layers (default of ten).

.. container:: list

   ::

      pfset Solver.PrintCLM True          ## TCL syntax
      <runname>.Solver.PrintCLM = True    ## Python syntax 

The output variables are:

.. container:: description

   ``eflx_lh_tot`` for latent heat flux total :math:`[W/m^2]` using the silo variable *LatentHeat*;

   ``eflx_lwrad_out`` for outgoing long-wave radiation :math:`[W/m^2]` using the silo variable *LongWave*;

   ``eflx_sh_tot`` for sensible heat flux total :math:`[W/m^2]` using the silo variable *SensibleHeat*;

   ``eflx_soil_grnd`` for ground heat flux :math:`[W/m^2]` using the silo variable *GroundHeat*;

   ``qflx_evap_tot`` for total evaporation :math:`[mm/s]` using the silo variable *EvaporationTotal*;

   ``qflx_evap_grnd`` for ground evaporation without sublimation :math:`[mm/s]` using the silo 
   variable *EvaporationGroundNoSublimation*;

   ``qflx_evap_soi`` for soil evaporation :math:`[mm/s]` using the silo variable *EvaporationGround*;

   ``qflx_evap_veg`` for vegetation evaporation :math:`[mm/s]` using the silo variable *EvaporationCanopy*;

   ``qflx_tran_veg`` for vegetation transpiration :math:`[mm/s]` using the silo variable *Transpiration*;

   ``qflx_infl`` for soil infiltration :math:`[mm/s]` using the silo variable *Infiltration*;

   ``swe_out`` for snow water equivalent :math:`[mm]` using the silo variable *SWE*;

   ``t_grnd`` for ground surface temperature :math:`[K]` using the silo variable *TemperatureGround*; and

   ``t_soil`` for soil temperature over all layers :math:`[K]` using the silo variable *TemperatureSoil*.

*string* **Solver.WriteCLMBinary** True This key specifies whether the ``CLM`` writes two dimensional 
binary output files in a generic binary format. Note that ``CLM`` must be compiled and linked at 
runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.WriteCLMBinary False         ## TCL syntax
      <runname>.Solver.WriteCLMBinary = False   ## Python syntax

*string* **Solver.CLM.BinaryOutDir** True This key specifies whether the ``CLM`` writes 
each set of two dimensional binary output files to a corresponding directory. These 
directories my be created before ParFlow is run (using the tcl script, for example). 
Choices for this key include **True** and **False**. Note that ``CLM`` must be compiled and 
linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.BinaryOutDir True        ## TCL syntax
      <runname>.Solver.CLM.BinaryOutDir = True  ## Python syntax

These directories are:

.. container:: description

   ``/qflx_top_soil`` for soil flux;

   ``/qflx_infl`` for infiltration;

   ``/qflx_evap_grnd`` for ground evaporation;

   ``/eflx_soil_grnd`` for ground heat flux;

   ``/qflx_evap_veg`` for vegetation evaporation;

   ``/eflx_sh_tot`` for sensible heat flux;

   ``/eflx_lh_tot`` for latent heat flux;

   ``/qflx_evap_tot`` for total evaporation;

   ``/t_grnd`` for ground surface temperature;

   ``/qflx_evap_soi`` for soil evaporation;

   ``/qflx_tran_veg`` for vegetation transpiration;

   ``/eflx_lwrad_out`` for outgoing long-wave radiation;

   ``/swe_out`` for snow water equivalent; and

   ``/diag_out`` for diagnostics.

*string* **Solver.CLM.CLMFileDir** no default This key specifies what
directory all output from the ``CLM`` is written to. This key may be 
set to ``"./"`` or ``""`` to write output to the ParFlow run directory. 
This directory must be created before ParFlow is run. Note 
that ``CLM`` must be compiled and linked at runtime for this option 
to be active.

.. container:: list

   ::

      pfset Solver.CLM.CLMFileDir "CLM_Output/"          ## TCL syntax
      <runname>.Solver.CLM.CLMFileDir = "CLM_Output/"    ## Python syntax

*integer* **Solver.CLM.CLMDumpInterval** 1 This key specifies how often
output from the ``CLM`` is written. This key is the real
time interval at which time-dependent output should be written. A value
of **0** will produce undefined behavior. If the value is negative,
output will be dumped out every :math:`n` time steps, where :math:`n` is
the absolute value of the integer part of the value.  Note that ``CLM`` must be compiled and linked 
at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.CLMDumpInterval 2           ## TCL syntax
      <runname>.Solver.CLM.CLMDumpInterval = 2     ## Python syntax

*string* **Solver.CLM.EvapBeta** Linear This key specifies the form of
the bare soil evaporation :math:`\beta` parameter in ``CLM``. The 
valid types for this key are **None**, **Linear**, **Cosine**.

**None**: 
   No beta formulation, :math:`\beta=1`.

**Linear**: 
   :math:`\beta=\frac{\phi S-\phi S_{res}}{\phi-\phi S_{res}}`

**Cosine**: 
   :math:`\beta=\frac{1}{2}(1-\cos(\frac{(\phi -\phi S_{res})}{(\phi S-\phi S_{res})}\pi)`

Note that :math:`S_{res}` is specified by the key ``Solver.CLM.ResSat`` below, 
that :math:`\beta` is limited between zero and one and also that ``CLM`` must 
be compiled and linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.EvapBeta "Linear"           ## TCL syntax
      <runname>.Solver.CLM.EvapBeta = "Linear"     ## Python syntax

*double* **Solver.CLM.ResSat** 0.1 This key specifies the residual
saturation for the :math:`\beta` function in ``CLM`` specified above. 
Note that ``CLM`` must be compiled and linked at runtime for this 
option to be active.

.. container:: list

   ::

      pfset Solver.CLM.ResSat  0.15          ## TCL syntax
      <runname>.Solver.CLM.ResSat = 0.15     ## Python syntax 

*string* **Solver.CLM.VegWaterStress** Saturation This key specifies the
form of the plant water stress function :math:`\beta_t` parameter in ``CLM``. 
The valid types for this key are **None**, **Saturation**, **Pressure**.

**None**: 
   No transpiration water stress formulation, :math:`\beta_t=1`.

**Saturation**: 
   :math:`\beta_t=\frac{\phi S -\phi S_{wp}}{\phi S_{fc}-\phi S_{wp}}`

**Pressure**: 
   :math:`\beta_t=\frac{P - P_{wp}}{P_{fc}-P_{wp}}`

Note that the wilting point, :math:`S_{wp}` or :math:`p_{wp}`, is
specified by the key ``Solver.CLM.WiltingPoint`` below, that the 
field capacity, :math:`S_{fc}` or :math:`p_{fc}`, is specified by the 
key ``Solver.CLM.FieldCapacity`` below, that :math:`\beta_t` is limited 
between zero and one and also that ``CLM`` must be compiled and 
linked at runtime for this option to be active.

.. container:: list

   ::

      pfset Solver.CLM.VegWaterStress  "Pressure"        ## TCL syntax
      <runname>.Solver.CLM.VegWaterStress = "Pressure"   ## Python syntax

*double* **Solver.CLM.WiltingPoint** 0.1 This key specifies the wilting
point for the :math:`\beta_t` function in ``CLM`` specified above. Note 
that the units for this function are pressure :math:`[m]` for a **Pressure** 
formulation and saturation :math:`[-]` for a **Saturation** formulation. Note 
that ``CLM`` must be compiled and linked at runtime for this option 
to be active.

.. container:: list

   ::

      pfset Solver.CLM.WiltingPoint  0.15       ## TCL syntax
      <runname>.Solver.CLM.WiltingPoint = 0.15  ## Python syntax

*double* **Solver.CLM.FieldCapacity** 1.0 This key specifies the field
capacity for the :math:`\beta_t` function in ``CLM`` specified above. 
Note that the units for this function are pressure :math:`[m]` for a **Pressure** 
formulation and saturation :math:`[-]` for a **Saturation** formulation. Note 
that ``CLM`` must be compiled and linked at runtime for this option 
to be active.

.. container:: list

   ::

      pfset Solver.CLM.FieldCapacity  0.95         ## TCL syntax
      <runname>.Solver.CLM.FieldCapacity = 0.95    ## Python syntax

*string* **Solver.CLM.IrrigationType** none This key specifies the form
of the irrigation in ``CLM``. The valid types for this key are **none**, 
**Spray**, **Drip**, **Instant**.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationType "Drip"      ## TCL syntax
      <runname>.Solver.CLM.IrrigationType "Drip"  ## Python syntax

*string* **Solver.CLM.IrrigationCycle** Constant This key specifies the
cycle of the irrigation in ``CLM``. The valid types for this key are 
**Constant**, **Deficit**. Note only **Constant** is currently implemented. Constant 
cycle applies irrigation each day from IrrigationStartTime to 
IrrigationStopTime in hours of the day (24-hour time) in GMT.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationCycle "Constant"        ## TCL syntax
      <runname>.Solver.CLM.IrrigationCycle = "Constant"  ## Python syntax

*double* **Solver.CLM.IrrigationRate** no default This key specifies the
rate of the irrigation in ``CLM`` in :math:`[mm/s]`.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationRate 10.          ## TCL syntax
      <runname>.Solver.CLM.IrrigationRate = 10.    ## Python syntax 

*double* **Solver.CLM.IrrigationStartTime** no default This key
specifies the start time of the irrigation in ``CLM`` GMT.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationStartTime 8.0          ## TCL syntax
      <runname>.Solver.CLM.IrrigationStartTime = 8.0    ## Python syntax

*double* **Solver.CLM.IrrigationStopTime** no default This key specifies
the stop time of the irrigation in ``CLM`` GMT.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationStopTime 12.0        ## TCL syntax
      <runname>.Solver.CLM.IrrigationStopTime = 12.0  ## Python syntax

*double* **Solver.CLM.IrrigationThreshold** 0.5 This key specifies the
threshold value for the irrigation in ``CLM``.

.. container:: list

   ::

      pfset Solver.CLM.IrrigationThreshold 0.2          ## TCL syntax
      <runname>.Solver.CLM.IrrigationThreshold = 0.2    ## Python syntax

*integer* **Solver.CLM.ReuseCount** 1 How many times to reuse a ``CLM`` 
atmospheric forcing file input. For example timestep=1, reuse =1 is 
normal behavior but reuse=2 and timestep=0.5 subdivides the time step 
using the same ``CLM`` input for both halves instead of needing two files. 
This is particularly useful for large, distributed runs when the user 
wants to run ParFlow at a smaller timestep than the ``CLM`` 
forcing. Forcing files will be reused and total fluxes adjusted 
accordingly without needing duplicate files.

.. container:: list

   ::

      pfset Solver.CLM.ReuseCount      5     ## TCL syntax
      <runname>.Solver.CLM.ReuseCount = 5    ## Python syntax

*string* **Solver.CLM.WriteLogs** True When **False**, this disables
writing of the CLM output log files for each processor. For example, in
the clm.tcl test case, if this flag is added **False**,
``washita.output.txt.p`` and ``washita.para.out.dat.p`` (were *p* is the
processor #) are not created, assuming *washita* is the run name.

.. container:: list

   ::

      pfset Solver.CLM.WriteLogs    False       ## TCL syntax
      <runname>.Solver.CLM.WriteLogs = False    ## Python syntax

*string* **Solver.CLM.WriteLastRST** False Controls whether CLM restart
files are sequentially written or whether a single file *restart file
name*.00000.\ *p* is overwritten each time the restart file is output,
where *p* is the processor number. If "True" only one file is
written/overwritten and if "False" outputs are written more frequently.
Compatible with DailyRST and ReuseCount; for the latter, outputs are
written every n steps where n is the value of ReuseCount.

.. container:: list

   ::

      pfset Solver.CLM.WriteLastRST   True      ## TCL syntax
      <runname>.Solver.CLM.WriteLastRST = True  ## Python syntax 

*string* **Solver.CLM.DailyRST** True Controls whether CLM writes daily
restart files (default) or at every time step when set to False; outputs
are numbered according to the istep from ParFlow. If **ReuseCount=n**,
with n greater than 1, the output will be written every n steps (i.e. it
still writes hourly restart files if your time step is 0.5 or 0.25,
etc...). Fully compatible with **WriteLastRST=False** so that each daily
output is overwritten to time 00000 in *restart file name*.00000.p where
*p* is the processor number.

.. container:: list

   ::

      pfset Solver.CLM.DailyRST    False     ## TCL syntax
      <runname>.Solver.CLM.DailyRST = False  ## Python syntax

*string* **Solver.CLM.SingleFile** False Controls whether ParFlow writes
all ``CLM`` output variables as a single file per time step. When "True", 
this combines the output of all the CLM output variables into a special 
multi-layer PFB with the file extension ".C.pfb". The first 13 layers 
correspond to the 2-D CLM outputs and the remaining layers are the soil 
temperatures in each layer. For example, a model with 4 soil layers will 
create a SingleFile CLM output with 17 layers at each time step. The file 
pseudo code is given below in :ref:`ParFlow Binary Files (.c.pfb)` and 
the variables and units are as specified in the multiple ``PFB`` 
and ``SILO`` formats as above.

.. container:: list

   ::

      pfset Solver.CLM.SingleFile   True        ## TCL syntax
      <runname>.Solver.CLM.SingleFile = True    ## Python syntax

*integer* **Solver.CLM.RootZoneNZ** 10 This key sets the number of soil
layers the ParFlow expects from ``CLM``. It will allocate and format all 
the arrays for passing variables to and from ``CLM`` accordingly. 
This value now sets the CLM number as well so recompilation is
not required anymore. Most likely the key ``Solver.CLM.SoiLayer``, 
described below, will also need to be changed.

.. container:: list

   ::

      pfset Solver.CLM.RootZoneNZ      4     ## TCL syntax
      <runname>.Solver.CLM.RootZoneNZ = 4    ## Python syntax

*integer* **Solver.CLM.RZWaterStress** 0 This key sets the distribution
of transpiration over the root zone and changes the behavior of plant water limitations.
As discussed in :cite:p:`Ferguson2016` water stress approaches result in
different cut-offs for transpiration ().  Two options are currently
implemented both use the beta-type water stress defined above.  Option 0
(default) will limit transpiration when the top soil layer drops below
wilting point, option 1 limits each layer independently.

.. container:: list

   ::

      pfset Solver.CLM.RZWaterStress 1          ## TCL syntax
      <runname>.Solver.CLM.RZWaterStress = 1    ## Python syntax
      
*integer* **Solver.CLM.SoiLayer** 7 This key sets the soil layer, and
thus the soil depth, that ``CLM`` uses for the seasonal temperature 
adjustment for all leaf and stem area indices.

.. container:: list

   ::

      pfset Solver.CLM.SoiLayer      4    ## TCL syntax
      <runname>.Solver.CLM.SoiLayer = 4   ## Python syntax

*string* **Solver.CLM.UseSlopeAspect** False This key specifies whether
or not allows for the inclusion of slopes when determining solar zenith
angles. Note that must be compiled and linked at runtime for this option
to be active.

::

   pfset Solver.CLM.UseSlopeAspect True         ## TCL syntax
   <runname>.Solver.CLM.UseSlopeAspect = True   ## Python syntax


.. _ParFlow NetCDF4 Parallel I/O:

ParFlow NetCDF4 Parallel I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NetCDF4 parallel I/O is being implemented in ParFlow. As of now only
output capability is implemented. Input functionality will be added in
later version. Currently user has option of printing 3-D time varying
pressure or saturation or both in a single NetCDF file containing
multiple time steps. User should configure ParFlow(pfsimulatior 
part) ``--with-netcdf`` option and link the appropriate NetCDF4 library. Naming
convention of output files is analogues to binary file names. Following
options are available for NetCDF4 output along with various performance
tuning options. User is advised to explore NetCDF4 chunking and ROMIO
hints option for better I/O performance.

**HDF5 Library version 1.8.16 or higher is required for NetCDF4 parallel
I/O**

*integer* **NetCDF.NumStepsPerFile** This key sets number of time steps
user wishes to output in a NetCDF4 file. Once the time step count
increases beyond this number, a new file is automatically created.

.. container:: list

   ::

      pfset NetCDF.NumStepsPerFile    5      ## TCL syntax
      <runname>.NetCDF.NumStepsPerFile = 5   ## Python syntax

*string* **NetCDF.WritePressure** False This key sets pressure variable
to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WritePressure    True     ## TCL syntax
      <runanme>.NetCDF.WritePressure = True  ## Python syntax

*string* **NetCDF.WriteSaturation** False This key sets saturation
variable to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteSaturation    True      ## TCL syntax
      <runname>.NetCDF.WriteSaturation = True   ## Python syntax

*string* **NetCDF.WriteMannings** False This key sets Mannings
coefficients to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteMannings	    True    ## TCL syntax
      <runname>.NetCDF.WriteMannings = True  ## Python syntax

*string* **NetCDF.WriteSubsurface** False This key sets subsurface
data (permeabilities, porosity, specific storage) to be written in
NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteSubsurface	    True    ## TCL syntax
      <runname>.NetCDF.WriteSubsurface	= True   ## Python syntax

*string* **NetCDF.WriteSlopes** False This key sets x and y slopes to be
written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteSlopes	    True    ## TCL syntax
      <runname>.NetCDF.WriteSlopes = True    ## Python syntax

*string* **NetCDF.WriteMask** False This key sets mask to be written in
NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteMask True         ## TCL syntax
      <runname>.NetCDF.WriteMask	= True   ## Python syntax 

*string* **NetCDF.WriteDZMultiplier** False This key sets DZ multipliers
to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteDZMultiplier True          ## TCL syntax
      <runname>.NetCDF.WriteDZMultiplier = True    ## Python syntax

*string* **NetCDF.WriteEvapTrans** False This key sets Evaptrans to be
written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteEvapTrans True          ## TCL syntax
      <runname>.NetCDF.WriteEvapTrans = True    ## Python syntax

*string* **NetCDF.WriteEvapTransSum** False This key sets Evaptrans sum
to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteEvapTransSum True          ## TCL syntax
      <runname>.NetCDF.WriteEvapTransSum = True    ## Python syntax

*string* **NetCDF.WriteOverlandSum** False This key sets overland sum to
be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteOverlandSum	True        ## TCL syntax
      <runname>.NetCDF.WriteOverlandSum = True  ## Python syntax

*string* **NetCDF.WriteOverlandBCFlux** False This key sets overland bc
flux to be written in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.WriteOverlandBCFlux	True        ## TCL syntax
      <runname>.NetCDF.WriteOverlandBCFlux = True  ## Python syntax

NetCDF4 Chunking
~~~~~~~~~~~~~~~~

Chunking may have significant impact on I/O. If this key is not set,
default chunking scheme will be used by NetCDF library. Chunks are
hypercube(hyperslab) of any dimension. When chunking is used, chunks are
written in single write operation which can reduce access times. For
more information on chunking, refer to NetCDF4 user guide.

*string* **NetCDF.Chunking** False This key sets chunking for each time
varying 3-D variable in NetCDF4 file.

.. container:: list

   ::

      pfset NetCDF.Chunking    True       ## TCL syntax
      <runname>.NetCDF.Chunking = True    ## Python syntax

Following keys are used only when **NetCDF.Chunking** is set to true.
These keys are used to set chunk sizes in x, y and z direction. A
typical size of chunk in each direction should be equal to number of
grid points in each direction for each processor. e.g. If we are using a
grid of 400(x)X400(y)X30(z) with 2-D domain decomposition of 8X8, then
each core has 50(x)X50(y)X30(z) grid points. These values can be used to
set chunk sizes each direction. For unequal distribution, chunk sizes
should as large as largest value of grid points on the processor. e.g.
If one processor has grid distribution of 40(x)X40(y)X30(z) and another
has 50(x)X50(y)X30(z), the later values should be used to set chunk
sizes in each direction.

*integer* **NetCDF.ChunkX** None This key sets chunking size in
x-direction.

.. container:: list

   ::

      pfset NetCDF.ChunkX    50     ## TCL syntax
      <runname>.NetCDF.ChunkX = 50  ## Python syntax

*integer* **NetCDF.ChunkY** None This key sets chunking size in
y-direction.

.. container:: list

   ::

      pfset NetCDF.ChunkY    50     ## TCL syntax
      <runname>.NetCDF.ChunkY = 50  ## Python syntax

*integer* **NetCDF.ChunkZ** None This key sets chunking size in
z-direction.

.. container:: list

   ::

      pfset NetCDF.ChunkZ    30        ## TCL syntax
      <runname>.NetCDF.ChunkZ = 30     ## Python syntax


NetCDF4 Compression
~~~~~~~~~~~~~~~~~~~

*integer* **NetCDF.Compression** False This key enables in-transit
deflate compression for all NetCDF variables using zlib. To use this
feature, NetCDF4 v4.7.4 must be available, which supports the necessary
parallel zlib compression. The compression quality can be influenced by
the chunk sizes and the overall data distribution. Compressed variables
in NetCDF files can be opened in serial mode also within older versions
of NetCDF4.

::

   pfset NetCDF.Compression True          ## TCL syntax
   <runname>.NetCDF.Compression = True    ## Python syntax

*integer* **NetCDF.CompressionLevel** 1 This key sets the deflate
compression level (if **NetCDF.Compression** is enabled), which influence
the overall compression quality. zlib supports values between 0 (no
compression), 1 (fastest compression) - 9 (slowest compression, smallest
files).

::

   pfset NetCDF.CompressionLevel 1           ## TCL syntax
   <runname>.NetCDF.CompressionLevel = 1     ## Python syntax


ROMIO Hints
~~~~~~~~~~~

ROMIO is a poratable MPI-IO implementation developed at Argonne National
Laboratory, USA. Currently it is released as a part of MPICH. ROMIO sets
hints to optimize I/O operations for MPI-IO layer through MPI_Info
object. This object is passed on to NetCDF4 while creating a file. ROMIO
hints are set in a text file in "key" and "value" pair. *For correct
settings contact your HPC site administrator*. As in chunking, ROMIO
hints can have significant performance impact on I/O.

*string* **NetCDF.ROMIOhints** None This key sets ROMIO hints file to be
passed on to NetCDF4 interface. If this key is set, the file must be
present and readable in experiment directory.

.. container:: list

   ::

      pfset NetCDF.ROMIOhints "romio.hints"         ## TCL syntax
      <runname>.NetCDF.ROMIOhints = "romio.hints"   ## Python syntax

An example ROMIO hints file looks as follows.

.. container:: list

   ::

      romio_ds_write disable
      romio_ds_read disable
      romio_cb_write enable
      romio_cb_read enable
      cb_buffer_size 33554432

Node Level Collective I/O
~~~~~~~~~~~~~~~~~~~~~~~~~

A node level collective strategy has been implemented for I/O. One
process on each compute node gathers the data, indices and counts from
the participating processes on same compute node. All the root processes
from each compute node open a parallel NetCDF4 file and write the data.
e.g. If ParFlow is running on 3 compute nodes where each node consists
of 24 processors(cores); only 3 I/O streams to filesystem would be
opened by each root processor each compute node. This strategy could be
particularly useful when ParFlow is running on large number of
processors and every processor participating in I/O may create a
bottleneck. **Node level collective I/O is currently implemented for 2-D
domain decomposition and variables Pressure and Saturation only. All the
other ParFlow NetCDF output Tcl flags should be set to false(default
value). CLM output is independently handled and not affected by this
key. Moreover on speciality architectures, this may not be a portable
feature. Users are advised to test this feature on their machine before
putting into production.**

*string* **NetCDF.NodeLevelIO** False This key sets flag for node level
collective I/O.

.. container:: list

   ::

      pfset NetCDF.NodeLevelIO   True        ## TCL syntax
      <runname>.NetCDF.NodeLevelIO = True    ## Python syntax

NetCDF4 Initial Conditions: Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analogues to ParFlow binary files, NetCDF4 based option can be used to
set the initial conditions for pressure to be read from an “nc" file
containing single time step of pressure. The name of the variable in
“nc" file should be “pressure". A sample NetCDF header of an initial
condition file looks as follows. The names of the dimensions are not
important. The order of dimensions is important e.g. *(time, lev, lat,
lon) or (time,z, y, x)*

.. container:: list

   ::

      netcdf initial_condition {
      dimensions:
      	x = 200 ;
      	y = 200 ;
      	z = 40 ;
      	time = UNLIMITED ; 
      variables:
      	double time(time) ;
      	double pressure(time, z, y, x) ;
      }

**Node level collective I/O is currently not implemented for setting
initial conditions.**

*string* **ICPressure.Type** no default This key sets flag for initial
conditions to be read from a NetCDF file.

NetCDF4 files may have more than one timestep in the file.   By default the first
timestep will be read.  The TimeStep attribute is used to specify the timestep
to be used for the initial pressure.   Negative values are allowed to index
from the end.   "-1" is often a useful index to read the last timestep
in the file.

.. container:: list

   ::

      pfset ICPressure.Type   "NCFile"        ## TCL syntax  
      pfset Geom.domain.ICPressure.FileName "initial_condition.nc" ## TCL syntax
      pfset Geom.domain.ICPressure.TimeStep -1 ## TCL syntax

      <runname>.ICPressure.Type = "NCFile"    ## Python syntax
      <runname>.Geom.domain.ICPressure.FileName = "initial_condition.nc" ## Python syntax
      <runname>.Geom.domain.ICPressure.TimeStep = -1 ## Python syntax

NetCDF4 Slopes
~~~~~~~~~~~~~~

NetCDF4 based option can be used slopes to be read from an “nc" file
containing single time step of slope values. The name of the variable in
“nc" file should be “slopex" and “slopey" A sample NetCDF header of
slope file looks as follows. The names of the dimensions are not
important. The order of dimensions is important e.g. *(time, lat, lon)
or (time, y, x)*

.. container:: list

   ::

      netcdf slopex {
      dimensions:
      	time = UNLIMITED ; // (1 currently)
      	lon = 41 ;
      	lat = 41 ;
      variables:
      	double time(time) ;
      	double slopex(time, lat, lon) ;
      }
      netcdf slopey {
      dimensions:
      	time = UNLIMITED ; // (1 currently)
      	lon = 41 ;
      	lat = 41 ;
      variables:
      	double time(time) ;
      	double slopey(time, lat, lon) ;
      }

The two NetCDF files can be merged into one single file and can be used
with tcl flags. The variable names should be exactly as mentioned above.
Please refer to “slopes.nc" under Little Washita test case. **Node level
collective I/O is currently not implemented for setting initial
conditions.**

*string* **TopoSlopesX.Type** no default This key sets flag for slopes
in x direction to be read from a NetCDF file.

.. container:: list

   ::

      pfset TopoSlopesX.Type "NCFile"              ## TCL syntax
      pfset TopoSlopesX.FileName "slopex.nc"       ## TCL syntax

      <runname>.TopoSlopesX.Type = "NCFile"        ## Python syntax
      <runname>.TopoSlopesX.FileName = "slopex.nc" ## Python syntax

*string* **TopoSlopesY.Type** no default This key sets flag for slopes
in y direction to be read from a NetCDF file.

.. container:: list

   ::

      pfset TopoSlopesY.Type "NCFile"              ## TCL syntax
      pfset TopoSlopesy.FileName "slopey.nc"       ## TCL syntax

      <runname>.TopoSlopesY.Type = "NCFile"        ## Python syntax
      <runname>.TopoSlopesy.FileName = "slopey.nc" ## Python syntax

NetCDF4 Transient EvapTrans Forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following keys can be used for NetCDF4 based transient evaptrans
forcing. The file should contain forcing for all time steps. For a given
time step, if the forcing is null, zero values could be filled for the
given time step in the “.nc" file. The format of the sample file looks
as follows. The names of the dimensions are not important. The order of
dimensions is important e.g. *(time, lev, lat, lon) or (time,z, y, x)*

.. container:: list

   ::

      netcdf evap_trans {
      dimensions:
      	time = UNLIMITED ; // (1000 currently)
      	x = 72 ;
      	y = 72 ;
      	z = 3 ;
      variables:
      	double evaptrans(time, z, y, x) ;
      }

**Node level collective I/O is currently not implemented for transient
evaptrans forcing.**

*string* **NetCDF.EvapTransFileTransient** False This key sets flag for
transient evaptrans forcing to be read from a NetCDF file.

.. container:: list

   ::

      pfset NetCDF.EvapTransFileTransient True           ## TCL syntax
      <runname>.NetCDF.EvapTransFileTransient = True     ## Python syntax

*string* **NetCDF.EvapTrans.FileName** no default This key sets the name
of the NetCDF transient evaptrans forcing file.

.. container:: list

   ::

      pfset NetCDF.EvapTrans.FileName "evap_trans.nc"          ## TCL syntax
      <runname>.NetCDF.EvapTrans.FileName = "evap_trans.nc"    ## Python syntax

NetCDF4 CLM Output
~~~~~~~~~~~~~~~~~~

Similar to ParFlow binary and silo, following keys can be used to write
output CLM variables in a single NetCDF file containing multiple time
steps.

*integer* **NetCDF.CLMNumStepsPerFile** None This key sets number of
time steps to be written to a single NetCDF file.

.. container:: list

   ::

      pfset NetCDF.CLMNumStepsPerFile 24           ## TCL syntax
      <runname>.NetCDF.CLMNumStepsPerFile = 24     ## Python syntax

*string* **NetCDF.WriteCLM** False This key sets CLM variables to be
written in a NetCDF file.

.. container:: list

   ::

      pfset NetCDF.WriteCLM True          ## TCL syntax
      <runname>.NetCDF.WriteCLM = True    ## Python syntax

The output variables are:

.. container:: description

   ``eflx_lh_tot`` for latent heat flux total :math:`[W/m^2]` using the silo variable *LatentHeat*;

   ``eflx_lwrad_out`` for outgoing long-wave radiation :math:`[W/m^2]` using the silo variable *LongWave*;

   ``eflx_sh_tot`` for sensible heat flux total :math:`[W/m^2]` using the silo variable *SensibleHeat*;

   ``eflx_soil_grnd`` for ground heat flux :math:`[W/m^2]` using the silo variable *GroundHeat*;

   ``qflx_evap_tot`` for total evaporation :math:`[mm/s]` using the silo variable *EvaporationTotal*;

   ``qflx_evap_grnd`` for ground evaporation without condensation :math:`[mm/s]` using the silo 
   variable *EvaporationGroundNoSublimation*;

   ``qflx_evap_soi`` for soil evaporation :math:`[mm/s]` using the silo variable *EvaporationGround*;

   ``qflx_evap_veg`` for vegetation evaporation :math:`[mm/s]` using the silo variable *EvaporationCanopy*;

   ``qflx_tran_veg`` for vegetation transpiration :math:`[mm/s]` using the silo variable *Transpiration*;

   ``qflx_infl`` for soil infiltration :math:`[mm/s]` using the silo variable *Infiltration*;

   ``swe_out`` for snow water equivalent :math:`[mm]` using the silo variable *SWE*;

   ``t_grnd`` for ground surface temperature :math:`[K]` using the silo variable *TemperatureGround*; and

   ``t_soil`` for soil temperature over all layers :math:`[K]` using the silo variable *TemperatureSoil*.

NetCDF4 CLM Input/Forcing
~~~~~~~~~~~~~~~~~~~~~~~~~

| NetCDF based meteorological forcing can be used with following TCL
  keys. It is built similar to 2D forcing case for CLM with parflow
  binary files. All the required forcing variables must be present in
  one single NetCDF file spanning entire length of simulation. If the
  simulation ends before number of time steps in NetCDF forcing file,
  next cycle of simulation can be restarted with same forcing file
  provided it covers the time span of this cycle.
| e.g. If the NetCDF forcing file contains 100 time steps and simulation
  CLM-ParFlow simulation runs for 10 cycles containing 10 time steps in
  each cycle, the same forcing file can be reused. The user has to set
  correct value for the key ``Solver.CLM.IstepStart``
| The format of input file looks as follows. The variable names should
  match exactly as follows. The names of the dimensions are not
  important. The order of dimensions is important e.g. *(time, lev, lat,
  lon) or (time,z, y, x)*

.. container:: list

   ::

      netcdf metForcing {
      dimensions:
      	lon = 41 ;
      	lat = 41 ;
      	time = UNLIMITED ; // (72 currently)
      variables:
      	double time(time) ;
      	double APCP(time, lat, lon) ;
      	double DLWR(time, lat, lon) ;
      	double DSWR(time, lat, lon) ;
      	double Press(time, lat, lon) ;
      	double SPFH(time, lat, lon) ;
      	double Temp(time, lat, lon) ;
      	double UGRD(time, lat, lon) ;
      	double VGRD(time, lat, lon) ;

**Note: While using NetCDF based CLM forcing,** ``Solver.CLM.MetFileNT``
**should be set to its default value of 1**

*string* **Solver.CLM.MetForcing** no default This key sets
meteorological forcing to be read from NetCDF file.

.. container:: list

   ::

      pfset Solver.CLM.MetForcing "NC"          ## TCL syntax
      <runname>.Solver.CLM.MetForcing = "NC"    ## Python syntax

Set the name of the input/forcing file as follows.

.. container:: list

   ::

      pfset Solver.CLM.MetFileName "metForcing.nc"          ## TCL syntax
      <runname>.Solver.CLM.MetFileName = "metForcing.nc"    ## Python syntax

This file should be present in experiment directory. User may create
soft links in experiment directory in case where data can not be moved.

NetCDF Testing Little Washita Test Case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basic NetCDF functionality of output (pressure and saturation) and
initial conditions (pressure) can be tested with following tcl script.
CLM input/output functionality can also be tested with this case.

.. container:: list

   ::

      parflow/test/washita/tcl_scripts/LW_NetCDF_Test.tcl

This test case will be initialized with following initial condition
file, slopes and meteorological forcing.

.. container:: list

   ::

      parflow/test/washita/parflow_input/press.init.nc
      parflow/test/washita/parflow_input/slopes.nc
      parflow/test/washita/clm_input/metForcing.nc
