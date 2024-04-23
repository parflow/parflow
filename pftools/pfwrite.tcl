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

proc checkwritable { } {
    global opt_single_line
    global opt_warning
    global env

    set first_hit 1

    #
    # If this is not an RCS directory, don't bother.
    # Don't do object files or libraries
    # Don't do emacs/vi tempfiles
    # Don't do directories
    # Don't do links
    set files [glob -nocomplain *]
    if { [file exists RCS] } {
	foreach i $files {
	    if { [ file writable $i] && \
		    ([ file extension $i] != ".o") && \
		    ([ file extension $i] != ".a") && \
		    ([string last "~" $i] != [expr [string length $i]-1]) && \
		    (![file isdirectory $i]) && \
		    [file type $i] != "link" } {

		set break_out 0
		if { [file exists RCS/$i,v] } {
		    if { ![string length [exec rlog -L -R -l$env(LOGNAME) $i]]} {
			if [info exists opt_warning] {
			    set warning "Warning file is writable but not checked-out "
			} {
			    set warning ""
			}
		    } {
			# file is writable RCS so don't bother
			set break_out 1
		    }
		} {
		    set warning ""
		}
		
		if { !$break_out } {
		    if { [info exists opt_single_line] } { 
			puts "$warning[pwd]/$i"
		    } {
			if { ![info exists opt_single_line] && $first_hit } { 
			    puts "# Directory [pwd]"
			    set first_hit 0
			}
			puts "$warning$i"
		    }
		}
	    }
	}
    }
}

set PDT \
{PDT pfwrite
  recursive, R: switch
  single_line, s: switch
  warning, w: switch
PDTEND optional_file_list}

set MM { pfwrite

Determine which files are writable in the current directory tree.


Excludes: directories not using RCS
          .o files 
          .a files 
          files ending in "~"
          files that are locked by the user

Examples:
      pfwrite
           files that are in the current directory
      pfwrite -R
           files locked in pthe current directory and subdirectories

.recursive
Recurses into all subdirectories

.single_line
Puts file names on same line as the directory
(useful for some scripts)

.warning
Warns about files that are under RCS control and writeable but
not checked out.
}

set PDT [split $PDT "\n"]
set MM [split $MM "\n"]

evap $PDT $MM

checkwritable
    
if { [info exists opt_recursive] } { 
    #
    # recurse into subdirectories
    # 
    dirtraverse {checkwritable}
}







