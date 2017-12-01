# This is the FlowVR starter to run problem.tcl within FlowVR
#
#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

source ./problem.tcl
pfset NetCDF.WritePressure			False

pfset FlowVR                    True
pfset FlowVR.ServeFinalState    True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfwritedb [lindex $argv 0]
# pfrun - done in do.sh by flowvr
# pfundist - done in scripts/undist.tcl:
