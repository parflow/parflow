#!/bin/sh
# the next line restarts using tclsh \
exec tclsh "$0" "$@"              

#BHEADER**********************************************************************
#
#  Copyright (c) 1995-2024, Lawrence Livermore National Security,
#  LLC. Produced at the Lawrence Livermore National Laboratory. Written
#  by the Parflow Team (see the CONTRIBUTORS file)
#  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
#
#  This file is part of Parflow. For details, see
#  http://www.llnl.gov/casc/parflow
#
#  Please read the COPYRIGHT file or Our Notice and the LICENSE file
#  for the GNU Lesser General Public License.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License (as published
#  by the Free Software Foundation) version 2.1 dated February 1999.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
#  and conditions of the GNU General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
#  USA
#**********************************************************************EHEADER

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






