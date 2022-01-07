.. _Introduction:

Introduction
============

ParFlow (*PARallel FLOW*) is an integrated hydrology model that
simulates surface and subsurface flow. ParFlow
:raw-latex:`\cite{Ashby-Falgout90, Jones-Woodward01, KM06, M13}` is a
parallel simulation platform that operates in three modes:

#. steady-state saturated;

#. variably saturated;

#. and integrated-watershed flow.

ParFlow is especially suitable for large scale problems on a range of
single and multi-processor computing platforms. ParFlow simulates
saturated and variably saturated subsurface flow in heterogeneous porous
media in three spatial dimensions using a mulitgrid-preconditioned
conjugate gradient solver :raw-latex:`\cite{Ashby-Falgout90}` and a
Newton-Krylov nonlinear solver :raw-latex:`\cite{Jones-Woodward01}`.
ParFlow has recently been extended to coupled surface-subsurface flow to
enable the simulation of hillslope runoff and channel routing in a truly
integrated fashion :raw-latex:`\cite{KM06}`. ParFlow is also
fully-coupled with the land surface model ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12
‘=̃12 ‘=̂12
``CLM  as described in . The development and application of ParFlow has been on-going for more than 20 years  and resulted in some of the most advanced numerical solvers and multigrid preconditioners for massively parallel computer environments that are available today. Many of the numerical tools developed within the ParFlow platform have been turned into or are from libraries that are now distributed and maintained at LLNL (Hypre and SUNDIALS, for example). An additional advantage of ParFlow is the use of a sophisticated octree-space partitioning algorithm to depict complex structures in three-space, such as topography, different hydrologic facies, and watershed boundaries. All these components implemented into ParFlow enable large scale, high resolution watershed simulations.``

ParFlow is primarily written in *C*, uses a modular architecture and
contains a flexible communications layer to encapsulate parallel process
interaction on a range of platforms. ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12
‘=̂12
``CLM is fully-integrated into ParFlow as a module and has been parallelized (including I/O) and is written in FORTRAN 90/95. ParFlow is organized into a main executable ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/pfsimulator/parflow_exe and a library ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/pfsimulator/parflow_lib (where ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir is the main directory location) and is comprised of more than 190 separate source files. ParFlow is structured to allow it to be called from within another application (e.g. WRF, the Weather Research and Forecasting atmospheric model) or as a stand-alone application. There is also a directory structure for the message-passing layer ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/pfsimulator/amps for the associated tools ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/pftools for ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 CLM ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/pfsimulator/clm and a directory of test cases ‘#=12 ‘$=12 ‘%=12 ‘&=12 ‘_=12 ‘=̃12 ‘=̂12 pfdir/test.``

.. _how to:

How to use this manual
----------------------

This manual describes how to use ParFlow, and is intended for
hydrologists, geoscientists, environmental scientists and engineers.
This manual is written assuming the reader has a basic understanding of
Linux / UNIX environments, how to compose and execute scripts in various
programming languages (e.g. TCL), and is familiar with groundwater and
surface water hydrology, parallel computing, and numerical modeling in
general. In Chapter `2 <#Getting Started>`__, we describe how to install
ParFlow, including building the code and associated libraries. Then, we
lead the user through a simple ParFlow run and discuss the automated
test suite. In Chapter `3 <#The ParFlow System>`__, we describe the
ParFlow system in more detail. This chapter contains a lot of useful
information regarding how a run is constructed and most importantly
contains two detailed, annotated scripts that run two classical ParFlow
problems, a fully saturated, heterogeneous aquifer and a variably
saturated, transient, coupled watershed. Both test cases are published
in the literature and are a terrific initial starting point for a new
ParFlow user.

Chapter `4 <#Manipulating Data>`__ describes data analysis and
processing. Chapter `5 <#Model_Equations>`__ provides the basic
equations solved by ParFlow. Chapter `6 <#ParFlow Files>`__ describes
the formats of the various files used by ParFlow. These chapters are
really intended to be used as reference material. This manual provides
some overview of ParFlow some information on building the code, examples
of scripts that solve certain classes of problems and a compendium of
keys that are set for code options.