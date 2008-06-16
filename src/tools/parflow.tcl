#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

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
    namespace export pfgetelt
    namespace export pfgridtype
    namespace export pfgetgrid
    namespace export pfcvel
    namespace export pfvvel
    namespace export pfvmag
    namespace export pfhhead
    namespace export pfphead
    namespace export pfflux
    namespace export pfnewlabel
    namespace export pfaxpy
    namespace export pfgetstats
    namespace export pfmdiff
    namespace export pfdiffelt
    namespace export pfsavediff
    namespace export pfgetlist
    namespace export pfnewgrid
    namespace export pfnewlabel
    namespace export pfdelete

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
proc Parflow::PFWriteDB {name} {

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
    
    PFWriteDB $runname

    #
    # If user does not set the hostname then use this machine
    # as the default
    # 
    if [pfexists Process.Hostnames] {
	set machines [pfget Process.Hostnames]
    } {
	set machines [info hostname]
    }

    set file [open .hostfile "w" ]
    foreach name "$machines" {
	puts $file $name
    }
    close $file

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
   set NumNodes [expr round(($NumProcs+.01) / 2) ]


    puts [exec $Parflow::PARFLOW_DIR/bin/bootmc $NumProcs]
    puts [exec $Parflow::PARFLOW_DIR/bin/getmc $NumProcs]
    #
    # SGS this change done at some point breaks the pattern for how Parflow was setup to execute the "run" script. 
    # Not all of the run scripts currently understand the arg change and even the ones that do are broken.
    #
    ##puts [eval exec $Parflow::PARFLOW_DIR/bin/run $run_args $runname]
    puts [eval exec $Parflow::PARFLOW_DIR/bin/run  $runname $NumProcs $NumNodes  > pfout.txt]
    puts [exec $Parflow::PARFLOW_DIR/bin/freemc]
    puts [exec $Parflow::PARFLOW_DIR/bin/killmc]
    
    # Need to add stuff to run parflow here
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
