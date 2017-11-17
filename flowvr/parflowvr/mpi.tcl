#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

#TODO: rename this file into something like parflow tcl header...., cleanup comments!
source ./common.tcl
pfset NetCDF.WritePressure			False
pfset NetCDF.WritePressure			False

pfset FlowVR                    True
pfset FlowVR.Event              True
pfset FlowVR.ServeFinalState    True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#done in do.sh
pfwritedb mpi
#pfrun mpi

# done in undist.tcl:
#pfundist mpi
