.. _Introduction:

Introduction
============

`ParFlow <https://parflow.org>`_ (*PARallel FLOW*) is an integrated hydrology model that
simulates surface and subsurface flow. ParFlow :cite:p:`Ashby-Falgout90,
Jones-Woodward01,KM06,M13` is a parallel simulation platform that operates 
in three modes:

#. steady-state saturated;

#. variably saturated;

#. and integrated-watershed flow.

ParFlow is especially suitable for large scale problems on a range of
single and multi-processor computing platforms. ParFlow simulates
saturated and variably saturated subsurface flow in heterogeneous
porous media in three spatial dimensions using a mulitgrid-preconditioned
conjugate gradient solver :cite:p:`Ashby-Falgout90` and a Newton-Krylov nonlinear 
solver :cite:`Jones-Woodward01`. ParFlow solves  
surface-subsurface flow to enable the simulation of hillslope runoff and channel 
routing in a truly integrated fashion :cite:p:`KM06`. ParFlow is also fully-coupled 
with the land surface model ``CLM`` :cite:p:`Dai03` as described in 
:cite:t:`MM05,KM08a`. The development and application of ParFlow has been 
on-going for more than 20 years :cite:p:`Meyerhoff14a,Meyerhoff14b,
Meyerhoff11,Mikkelson13,RMC10,Shrestha14, SNSMM10,Siirila12a,Siirila12b,
SMPMPK10,Williams11,Williams13,FM10,Keyes13,KRM10,Condon13a,Condon13b,
M13,KRM10,KRM10,SNSMM10,DMC10,AM10,MLMSWT10,M10,FM10,KMWSVVS10,SMPMPK10,
FFKM09,KCSMMB09,MTK09,dBRM08,MK08b,KM08b,KM08a,MK08a,MCT08,MCK07,
MWH07,KM06,MM05,TMCZPS05,MWT03,Teal02,WGM02,Jones-Woodward01,MCT00,
TCRM99,TBP99,TFSBA98,Ashby-Falgout90` and resulted in some of the most 
advanced numerical solvers and multigrid preconditioners for massively 
parallel computer environments that are available today. Many of the numerical 
tools developed within the ParFlow platform have been turned into or are 
from libraries that are now distributed and maintained at LLNL 
(*Hypre* and *SUNDIALS*, for example). An additional advantage of 
ParFlow is the use of a sophisticated octree-space partitioning 
algorithm to depict complex structures in three-space, such as 
topography, different hydrologic facies, and watershed boundaries. 
All these components implemented into ParFlow enable large scale, 
high resolution watershed simulations.

ParFlow is primarily written in *C*, uses a modular architecture 
and contains a flexible communications layer to encapsulate parallel 
process interaction on a range of platforms. ``CLM`` is fully-integrated 
into ParFlow as a module and has been parallelized (including I/O) 
and is written in *FORTRAN 90/95*. ParFlow is organized into a main 
executable ``pfdir/pfsimulator/parflow_exe`` and a 
library ``pfdir/pfsimulator/parflow_lib`` (where ``pfdir`` is 
the main directory location) and is comprised of more than 190 
separate source files. ParFlow is structured to allow it to be 
called from within another application (e.g. WRF, the Weather Research
and Forecasting atmospheric model) or as a stand-alone application. 
There is also a directory structure for the message-passing 
layer ``pfdir/pfsimulator/amps`` for the associated 
tools ``pfdir/pftools`` for ``CLM`` ``pfdir/pfsimulator/clm`` and a 
directory of test cases ``pfdir/test``.

.. _how to:

How to use this manual
----------------------

This manual describes how to use ParFlow, and is intended for
hydrologists, geoscientists, environmental scientists and engineers.
This manual is written assuming the reader has a basic understanding of
Linux / UNIX environments, how to compose and execute scripts in various
programming languages (e.g. TCL or Python), and is familiar with groundwater and
surface water hydrology, parallel computing, and numerical modeling in
general. In :ref:`Getting Started`, we describe how to install
ParFlow, including building the code and associated libraries. Then, we
lead the user through a simple ParFlow run and discuss the automated
test suite. In :ref:`The ParFlow System`, we describe the
ParFlow system in more detail. This chapter contains a lot of useful
information regarding how a run is constructed and most importantly
contains two detailed, annotated scripts that run two classical ParFlow
problems, a fully saturated, heterogeneous aquifer and a variably
saturated, transient, coupled watershed. Both test cases are published
in the literature and are a terrific initial starting point for a new
ParFlow user.

:ref:`Manipulating Data` describes data analysis and
processing. :ref:`Model_Equations` provides the basic
equations solved by ParFlow. :ref:`ParFlow Files` describes
the formats of the various files used by ParFlow. These chapters are
really intended to be used as reference material. This manual provides
some overview of ParFlow some information on building the code, examples
of scripts that solve certain classes of problems and a compendium of
keys that are set for code options.