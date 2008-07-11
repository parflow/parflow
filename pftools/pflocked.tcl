#!/bin/sh
# the next line restarts using tclsh \
exec tclsh "$0" "$@"              

#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.5 $
#EHEADER***********************************************************************

source $env(PARFLOW_DIR)/bin/evap.tcl
source $env(PARFLOW_DIR)/bin/dirt.tcl

#
# Set up some global command names
#
set RLOG "rlog"
set CI   "ci"
set CO   "co"

proc checklocked { username } {
    if { [file isdirectory "RCS"] } {
	global RLOG
	global opt_single_line
	if [string length "[glob -nocomplain RCS/*]"] {
	    set locked [eval exec $RLOG -L -R -l$username [glob RCS/*]]
	    if { [llength $locked] } {
		regsub -all RCS\/ $locked "" locked
		regsub -all ,v $locked "" locked
		if { [info exists opt_single_line] } {
		    foreach i $locked {
			puts [pwd]/$i
		    }
		} {
		    puts "# Directory [pwd]"
		    puts $locked
		}
	    }
	}
    }
}

set PDT \
{PDT pflocked
  recursive, R: switch
  single_line, s: switch
PDTEND optional_file_list}

set MM { pflocked

Determine which files are locked by specified users

Examples:
      pflocked 
           all files that are locked in the current directory
      pflocked -R
           all files locked in the current directory and subdirectories

      pflocked steve
           all files locked by user steve

.recursive
Recurses into all subdirectories

.single_line
Puts file names on same line as the directory
(useful for some scripts)}

set PDT [split $PDT "\n"]
set MM [split $MM "\n"]

evap $PDT $MM

if {$argc > 0} {

    foreach i $argv {
	puts "# Files locked by: $i"
	#
	# Check the current directory
	#
	checklocked $i

	if { [info exists opt_recursive] } { 
	    #
	    # recurse into subdirectories
	    # 
	    eval dirtraverse \{ checklocked $i \}
	}
    }
} else {
    puts "# Files locked by: any"

    checklocked ""
    
    if { [info exists opt_recursive] } { 
	#
	# recurse into subdirectories
	# 
	dirtraverse {checklocked  ""}
    }
}






