.. _Getting Started:

Getting Started
===============

This chapter is an introduction to setting up and running ParFlow. In
:ref:`ParFlow Solvers` we describe the solver options available
for use with ParFlow applications.

The ParFlow source code can be found on the `ParFlow GitHub <https://github.com/parflow/parflow>`_. 
Instructions for building ParFlow and Frequently-Asked Questions can be found on the 
`ParFlow Wiki <https://github.com/parflow/parflow/wiki>`_. 

.. _ParFlow Solvers:

ParFlow Solvers
---------------

ParFlow can operate using a number of different solvers. Two of these
solvers, IMPES (running in single-phase, fully-saturated mode, not
multiphase) and RICHARDS (running in variably-saturated mode, not
multiphase, with the options of land surface processes and coupled
overland flow) are detailed below. This is a brief summary of solver
settings used to simulate under three sets of conditions,
fully-saturated, variably-saturated and variably-saturated with overland
flow. A complete, detailed explanation of the solver parameters for
ParFlow may be found later in this manual. To simulate fully saturated,
steady-state conditions set the solver to IMPES, an example is given
below. This is also the default solver in ParFlow, so if no solver is
specified the code solves using IMPES.

::

   pfset Solver               Impes

To simulate variably-saturated, transient conditions, using Richards’
equation, variably/fully saturated, transient with compressible storage
set the solver to RICHARDS. An example is below. This is also the solver
used to simulate surface flow or coupled surface-subsurface flow.

::

   pfset Solver             Richards

To simulate overland flow, using the kinematic wave approximation to the
shallow-wave equations, set the solver to RICHARDS and set the upper
patch boundary condition for the domain geometry to OverlandFlow, an
example is below. This simulates overland flow, independently or coupled
to Richards’ Equation as detailed in :cite:p:`KM06`. The
overland flow boundary condition can simulate both uniform and
spatially-distributed sources, reading a distribution of fluxes from a
binary file in the latter case.

::

   pfset Patch.z-upper.BCPressure.Type	OverlandFlow

For this case, the solver needs to be set to RICHARDS::

   pfset Solver		Richards

ParFlow may also be coupled with the land surface model ``CLM`` [:cite:p:`DZD+03`]. 
This version of ``CLM`` has been extensively modified to be called 
from within ParFlow as a subroutine, to support parallel infrastructure including I/O and most importantly with modified physics 
to support coupled operation to best utilize the integrated hydrology in ParFlow[:cite:p:`MM05`, :cite:p:`KM08a`]. 
To couple ``CLM`` into ParFlow first the ``–with-clm`` 
option is needed in the ``./configure`` command as indicated in Installing ParFlow. Second, the ``CLM`` module needs 
to be called from within ParFlow, this is done using the following solver key::
   
    pfset Solver.LSM CLM

Note that this key is used to call ``CLM`` from within the nonlinear solver time loop and requires that the solver bet set to RICHARDS to work. 
Note also that this key defaults to *not* call ``CLM`` so if this line is omitted ``CLM`` will not be called from within 
ParFlow even if compiled and linked. Currently, ``CLM`` gets some of it’s information from ParFlow such as grid, 
topology and discretization, but also has some of it’s own input files for land cover, land cover types and atmospheric forcing.