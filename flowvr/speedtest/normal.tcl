#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

file delete -force results_normal
file mkdir results_normal
cd results_normal

source ../common.tcl

pfset NetCDF.WritePressure			True
#
#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#pfwritedb mpi
pfrun normal

pfundist normal
