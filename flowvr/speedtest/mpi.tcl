#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

source ./common.tcl
source ./Ns.tcl
pfset NetCDF.WritePressure			False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#done in doChecks.sh
pfwritedb mpi
#pfrun mpi

# done in undist.tcl:
#pfundist mpi
