#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

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

set RemoteHostName gwaihir
set RemoteDir graphics
set RemoteFiles ""

proc GetRemoteFiles {} {
    global RemoteHostName
    global RemoteDir
    global RemoteFiles

    set remotefiles [exec rsh $RemoteHostName ls -leR $RemoteDir]
    set remotefiles [split $remotefiles "\n"]

    set CWD ""
    set RemoteFiles ""
    foreach i $remotefiles {
	puts $i

	# if regexp ends in : then this is a directory and change the current
	# directory
	if [expr [llength $i] == 1] {
	    set CWD [string trimright $i ":"]
	    puts "Changing cwd=$CWD"
	} elseif [expr [llength $i] == 2] { 
	} {
	    set month [lindex $i 6]
	    set day   [lindex $i 7]
	    set time  [lindex $i 8]
	    set year   [lindex $i 9]
	    set filename   [lindex $i 10]
	    append RemoteFiles "$CWD$filename $month $day $time $year"
	}
    }
}

set LocalFiles ""

proc ListFiles {} {
    global LocalFiles

    set CWD [pwd]
    
    foreach i [glob *] {
	append LocalFiles "$CWD/$i\n"
    }
}

proc GetLocalFiles {} {
    dirtraverse { ListFiles }
}


#
# Is there a better way to include things like this?
#
source $env(PARFLOW_DIR)/bin/fsbox.tcl
source $env(PARFLOW_DIR)/bin/dirt.tcl



proc ScanFiles {} {
    global env
    global ParFlowSrc
    global LocalFiles

    . configure -cursor watch
    update

    .rcslocked.list delete 0 end
    .writable.list delete 0 end
    .warning.list delete 0 end

    cd $ParFlowSrc
    set EffectiveParFlowDir [pwd]

    #
    # Strip off the PARFLOW_DIR so display is less cluttered
    #
    regsub -all $EffectiveParFlowDir $files "" $LocalFiles
    set files [split $files "\n"]

    #
    # Find list to put them in
    #
    foreach i $files {
	puts $i
	if ![string match {#*} $i ] {
	    .rcslocked.list insert end $i
	}
    }

    . configure -cursor xterm
    update
}


proc Quit {} {

    exit 0
}

if [ string length [array names env PARFLOW_SRC] ] {
    set ParFlowSrc $env(PARFLOW_SRC)
} {
    set ParFlowSrc $env(PARFLOW_DIR)
}

set button 0

#
# Button bar for commands
#
frame .buttonbar -relief groove 
button .scanfiles -text "Scan Files" -command ScanFiles
button .difffiles -text "Diff Files" -command DiffFiles
button .done -text "Done" -command Done
button .quit -text "Quit" -command Quit

pack .scanfiles .difffiles .done .quit -side left -expand 1 -fill both -in .buttonbar 

pack .buttonbar

#
# List of Files
#
frame .rcslocked
pack .rcslocked -side top -expand yes -fill y -fill x

label .rcslocked.title	-text "Locked Files"
pack .rcslocked.title -side top

scrollbar .rcslocked.yscroll -command ".rcslocked.list yview"
scrollbar .rcslocked.xscroll -orient horizontal -command ".rcslocked.list xview" 

listbox .rcslocked.list -height 5 -width 78 -relief raised -borderwidth 2 -yscrollcommand ".rcslocked.yscroll set" -xscrollcommand ".rcslocked.xscroll set" -selectmode multiple -exportselection no
    
pack .rcslocked.yscroll -side right -fill y
pack .rcslocked.xscroll -side bottom -fill x
pack .rcslocked.list -expand yes -fill y

#
# List of writable files
#

frame .writable
pack .writable -side top -expand yes -fill y -fill x

label .writable.title	-text "Files that are writable and not in repository"
pack .writable.title -side top

scrollbar .writable.yscroll -command ".writable.list yview"
scrollbar .writable.xscroll -orient horizontal -command ".writable.list xview" 

listbox .writable.list -height 5 -width 78 -relief raised -borderwidth 2 -yscrollcommand ".writable.yscroll set" -xscrollcommand ".writable.xscroll set" -exportselection no -selectmode multiple

    
pack .writable.yscroll -side right -fill y
pack .writable.xscroll -side bottom -fill x
pack .writable.list -expand yes -fill y

#
# List of writable files
#

frame .warning
pack .warning -side top -expand yes -fill y -fill x

label .warning.title	-text "Warning: RCS Files that are writable but not checked out"
pack .warning.title -side top

scrollbar .warning.yscroll -command ".warning.list yview"
scrollbar .warning.xscroll -orient horizontal -command ".warning.list xview" 

listbox .warning.list -height 3 -width 78 -relief raised -borderwidth 2 -yscrollcommand ".warning.yscroll set" -xscrollcommand ".warning.xscroll set" -exportselection no -selectmode multiple

    
pack .warning.yscroll -side right -fill y
pack .warning.xscroll -side bottom -fill x
pack .warning.list -expand yes -fill y

#
# Text for the checkin notes
#
frame .checkinnotes

label .checkinnotes.title -text "CheckIn Message"
text .checkinnotes.text -height 3 -width 80 
pack .checkinnotes.title
pack .checkinnotes.text
pack .checkinnotes

#
# Text for the checkin notes
#
frame .emailnotes

label .emailnotes.title -text "Email Message"
text .emailnotes.text -height 10 -width 80 -yscrollcommand ".emailnotes.yscroll set"

scrollbar .emailnotes.yscroll -command ".emailnotes.text yview"


pack .emailnotes.title
pack .emailnotes.yscroll -side right -fill y
pack .emailnotes.text

pack .emailnotes

puts "Exiting"






