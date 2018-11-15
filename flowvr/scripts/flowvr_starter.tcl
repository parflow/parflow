#!/usr/bin/tclsh
# This is the FlowVR starter to run .tcl's within FlowVR
#
#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

source ./[lindex $argv 3].tcl

pfset FlowVR                    True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
pfwritedb [lindex $argv 3]
# pfrun - done in do.sh by flowvr
# pfundist - done in scripts/undist.tcl:
