#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

source ./_common.tcl
source ./_Ns.tcl
pfset NetCDF.WritePressure			False
pfset FlowVR True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
#done in doChecks.sh
pfwritedb default_richards
