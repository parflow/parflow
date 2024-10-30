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

package provide parflow 1.0

namespace eval Parflow {
    variable PFDB
    array set PFDB {FileVersion -1}

    variable IsArchUnix

    # Determine which Arch we are running on
    if { [ string compare $tcl_platform(platform) "windows" ] } {
	set Parflow::IsArchUnix 1

    } {
	set Parflow::IsArchUnix 0
    }

    #
    # Fix up filenames for Win32
    #
    proc FixupFilename { filename } {

	if $Parflow::IsArchUnix {
	    set new_filename $filename
	} {
	    regsub -all \\\\ $filename "/" new_filename
	}

	return $new_filename
    }

    variable PARFLOW_DIR [Parflow::FixupFilename $::env(PARFLOW_DIR)]

    namespace export pfget pfset pfrun pfundist

    namespace export pfStructuredPoints

    #
    # Export names from the shared library
    #
    namespace export pfloadsds
    namespace export pfsavesds
    namespace export pfbfcvel
    namespace export pfgetsubbox
    namespace export pfenlargebox
    namespace export pfload
    namespace export pfreload
    namespace export pfreloadall
    namespace export pfdist
    namespace export pfsave
    namespace export pfvtksave
    namespace export pfpatchysolid
    namespace export pfsolidfmtconvert
    namespace export pfgetelt
    namespace export pfgridtype
    namespace export pfgetgrid
    namespace export pfsetgrid
    namespace export pfcvel
    namespace export pfvvel
    namespace export pfvmag
    namespace export pfhhead
    namespace export pfphead
    namespace export pfflux
    namespace export pfnewlabel
    namespace export pfaxpy
    namespace export pfsum
    namespace export pfcellsum
    namespace export pfcelldiff
    namespace export pfcellmult
    namespace export pfcelldiv
    namespace export pfcellsumconst
    namespace export pfcelldiffconst
    namespace export pfcellmultconst
    namespace export pfcelldivconst
    namespace export pfgetstats
    namespace export pfmdiff
    namespace export pfdiffelt
    namespace export pfsavediff
    namespace export pfgetlist
    namespace export pfnewgrid
    namespace export pfnewlabel
    namespace export pfdelete
    namespace export pfcomputetop
    namespace export pfcomputebottom
    namespace export pfextracttop
    namespace export pfcomputedomain
    namespace export pfprintdomain
    namespace export pfextract2Ddomain
    namespace export pfbuilddomain
    namespace export pfdistondomain
    namespace export pfsurfacestorage
    namespace export pfsubsurfacestorage
    namespace export pfgwstorage
    namespace export pfsurfacerunoff
    namespace export pfwatertabledepth
    namespace export pfwritedb

    namespace export pfslopex
    namespace export pfslopey
    namespace export pfupstreamarea
    namespace export pffillflats
    namespace export pfpitfilldem
    namespace export pfmovingavgdem
    namespace export pftopodeficit
    namespace export pfsattrans
    namespace export pfeffectiverecharge
    namespace export pftoporecharge
    namespace export pftopoindex
    namespace export pftopowt
    namespace export pfhydrostatic
    namespace export pfslopexD4
    namespace export pfslopeyD4
    namespace export pfslopeD8
    namespace export pfsegmentD8
    namespace export pfchildD8
    namespace export pfflintslaw
    namespace export pfflintslawfit
    namespace export pfflintslawbybasin

    namespace export pfprintdata
    namespace export pfprintdiff
    namespace export pfprintlist
    namespace export pfprintelt
    namespace export pfprintgrid
    namespace export pfprintstats
    namespace export pfprintmdiff
    namespace export pfhelp
}

#
# Output a string that can containg blanks etc to a file
#
proc Parflow::PFWriteComplexString {file string} {
    puts $file [string length $string]
    puts $file $string
}

#
# Write an array from a file
#
proc Parflow::PFWriteArray {file name} {
    upvar $name a

    puts $file [array size a]
    foreach el [array names a] {
	PFWriteComplexString $file $el
	PFWriteComplexString $file $a($el)
    }
}

#
# Save the current state to file
#
proc Parflow::pfwritedb {name} {

    #
    # SGS: if file exists we should probably prompt to overwrite
    #

    set file [open [FixupFilename $name.pfidb] "w"]

    foreach i "Parflow::PFDB" {
	PFWriteArray $file $i
    }

    close $file
}


#
# Sets a value in the database
#
proc Parflow::pfset { key value } {

    set Parflow::PFDB($key) "$value"
}

#
# Retreives a value from the DataBase
#
proc Parflow::pfget { key } {

    return $Parflow::PFDB($key)
}

#
# Tests to see if a value exists
#
proc Parflow::pfexists { key } {
    return [info exists Parflow::PFDB($key)]
}

#
# Run parflow
# Should also provide option for output name?
#
proc Parflow::pfrun { runname args } {

    #
    # Process the optional args
    #
    set run_args ""
    set debug 0
    set state flag
    foreach arg $args {
	switch -- $state {
	    flag {
		switch -glob -- $arg {
		    -g*   {
			set debug 1
			set state debug
		    }
		}
	    }
	    debug {
		set run_args "-g \"$arg\" "
		set state flag
	    }
 	}
    }

    #
    # Write out the current state of the database
    #

    pfwritedb $runname

    if [pfexists Process.Topology.P] {
	set P [pfget Process.Topology.P]
    } {
	set P 1
    }

    if [pfexists Process.Topology.Q] {
	set Q [pfget Process.Topology.Q]
    } {
	set Q 1
    }

    if [pfexists Process.Topology.R] {
	set R [pfget Process.Topology.R]
    } {
	set R 1
    }

    set NumProcs [expr $P * $Q * $R]

    # Run parflow
    if [pfexists Process.Command] {
	set command [pfget Process.Command]
	puts [format "Using command : %s" [format $command $NumProcs $runname]]
	puts [eval exec [format $command $NumProcs $runname]]
    } {
	puts [eval exec sh $Parflow::PARFLOW_DIR/bin/run  $runname $NumProcs]
    }

}


#
# undistribute files
#
proc Parflow::pfundist { runname } {

    global PARFLOW_DIR

    # first check if this is a single file if so just work on it
    if [file exists $runname.dist] {
	file delete $runname.dist

	return
    }

    if [file exists $runname.00000] {
	set files [lsort [glob -nocomplain $runname.\[0-9\]*]]

	file delete $runname
	eval exec /bin/cat $files > $runname
	eval file delete $files

	return
    }


    # if not a single file assume it is a runname and unload all
    set root "$runname.out"

    set filelist ""

    foreach  postfix ".00000 .dist" {
	append filelist [glob -nocomplain $root.perm_x.*$postfix] " "
	append filelist [glob -nocomplain $root.perm_y.*$postfix] " "
	append filelist [glob -nocomplain $root.perm_z.*$postfix] " "
	append filelist [glob -nocomplain $root.porosity.*$postfix] " "
        append filelist [glob -nocomplain $root.specific_storage.*$postfix] " "

	append filelist [glob -nocomplain $root.press.*$postfix] " "
	append filelist [glob -nocomplain $root.density.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.satur.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.satur.?.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.concen.??.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.concen.?.??.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.phasex.?.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.phasey.?.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.phasez.?.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.temp.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.et.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.obf.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.mask.?????.*$postfix] " "
	append filelist [glob -nocomplain $root.mask.*$postfix] " "

	append filelist [glob -nocomplain $root.velx.*$postfix] " "
	append filelist [glob -nocomplain $root.vely.*$postfix] " "
	append filelist [glob -nocomplain $root.velz.*$postfix] " "
    }

    foreach i $filelist {
	if [string compare [file extension $i] ".dist"] {
	    set files [lsort [glob -nocomplain [file rootname $i].\[0-9\]*]]

	    file delete [file rootname $i]
	    eval exec /bin/cat $files > [file rootname $i]
	    eval file delete $files
	} {
	    file delete $i
	}
    }
}

# Procedure pfreloadall - This procedure is used reload all of the
#           current datasets.
#           executes the pfload command to load it.
#
# Parameters - none
#
# Return value - NOne
proc Parflow::pfreloadall {} {

    foreach str [pfgetlist] {
	if {[catch "pfreload  [lindex $str 0]"]} {
	    ErrorDialog {There is not enough memory to store the data.  Try deleting a few data sets to free up space.}
	    return
	}
    }
}
