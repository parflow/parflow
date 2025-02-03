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

#
# Is there a better way to include things like this?
#
source $env(PARFLOW_DIR)/bin/fsb.tcl

proc ReadEmailFile {} {

    set filters {{"All Files" *}}

    set filename [FSBox "Email Text File" $filters "loading"]

    set file [open $filename r]

    while { [gets $file line] >= 0 } {
	.emailnotes.text insert end $line 
	.emailnotes.text insert end "\n"
    }

    close $file
}

proc DiffFiles {} {
    global env
    global ParFlowSrc

    set fileindices [.rcslocked.list curselection]

    foreach i $fileindices {
	set filename [.rcslocked.list get $i]
	cd $ParFlowSrc[file dirname $filename]
	exec $env(PARFLOW_DIR)/bin/tkdiff [file tail $filename]
    }
}


proc ScanFiles {} {
    global env
    global ParFlowSrc

    . configure -cursor watch
    update

    .rcslocked.list delete 0 end
    .writable.list delete 0 end
    .warning.list delete 0 end

    cd $ParFlowSrc
    set EffectiveParFlowDir [pwd]

    set files [exec pflocked -s -R $env(LOGNAME)]
    
    #
    # Strip off the PARFLOW_DIR so display is less cluttered
    #
    regsub -all $EffectiveParFlowDir $files "" files

    #
    # Strip extra information from output
    #
    set files [split $files "\n"]

    
    foreach i $files {
	if ![string match {#*} $i ] {
	    .rcslocked.list insert end $i
	}
    }
    
    set files [exec pfwrite -s -w -R ]

    #
    # Strip off the PARFLOW_DIR so display is less cluttered
    #
    regsub -all $EffectiveParFlowDir $files "" files

    #
    # Strip extra information from output
    #
    set files [split $files "\n"]
    
    foreach i $files {
	if ![string match {#*} $i ] {
	    if  [string match {Warning*} $i] {
		.warning.list insert end [lindex $i [expr [llength $i]-1]]
	    } {
		.writable.list insert end $i
	    }
	}
    }

    . configure -cursor xterm
    update
}

proc CheckInFiles {} {
    global FilesCheckedIn
    global FileCheckInMessage
    global FileGroup
    

    incr FileGroup 1

    set FilesCheckedIn($FileGroup) ""

    set FileCheckInMessage($FileGroup) [.checkinnotes.text get 1.0 end]

    set fileindices [lsort -decreasing -integer [.rcslocked.list curselection]]

    foreach i $fileindices {
	set filename [.rcslocked.list get $i]
	set FilesCheckedIn($FileGroup) [lappend FilesCheckedIn($FileGroup) $filename]
	.rcslocked.list delete $i
    }

    .rcslocked.list selection clear 0 end

    set fileindices [lsort -decreasing -integer [.writable.list curselection]]

    foreach i $fileindices {
	set filename [.writable.list get $i]
	set FilesCheckedIn($FileGroup) [lappend FilesCheckedIn($FileGroup) $filename]
	.writable.list delete $i
    }
    
    .writable.list selection clear 0 end
}

proc Done {} {
    global FilesCheckedIn
    global FileCheckInMessage
    global FileGroup
    global env
    global UserToNotify
    global EmailAppend
    global CheckinErrors
    global ParFlowSrc

    if {$FileGroup == 0} {
	after idle {.dialog1.msg configure -wraplength 4i}
	set i [tk_dialog .dialog1 "No Checkins" {There are no checkins to do, Do you wish to Exit?} info 0 OK Cancel]
	
	switch $i {
	    0 {exit}
	    1 {return}
	}
    } 

    set VersionNotLocked 1

    while { $VersionNotLocked } {
	
	cd $ParFlowSrc/config
        if { [catch {exec pfco -q -l version.h} msg] } {
	    puts $msg
	    after idle {.dialog1.msg configure -wraplength 4i}
	    set i [tk_dialog .dialog1 "Repository Locked" {Someone else is updating the repository...wait and try again} info 0 OK ]
	} {
	    set VersionNotLocked 0
	}
    }

    for { set j 1} { $j <= $FileGroup } {incr j 1} {
	
	foreach i $FilesCheckedIn($j) {
	    
	    set directory $ParFlowSrc[file dirname $i]
	    
	    cd $directory
	    catch {exec ci -u [file tail $i] << "$FileCheckInMessage($j)"} msg

	    # 
	    # Do some error checking on the returned message
	    #
	    if {[string first "file is unchanged" $msg] != -1} {
		# No need to include in list no changes were made
		set email 0
	    } {
		if {[string first "new revision" $msg] != -1} {
		    # A good checkin of a new revision
		    set email 1
		} {
		    if {[string first "initial revision" $msg] != -1} {
			# A good checkin of an initial revision
			set email 1
		    } {
			append CheckinErrors "$msg"
			set email 1
		    }
		}
	    }

	    # change the group and perms on the checked in file

	    exec chmod ug+r RCS/[file tail $i],v

	    # if user can execute make make repository item executable
	    if { [file executable RCS/[file tail $i],v ] } {
		exec chmod ug+x RCS/[file tail $i],v
	    }
	    exec chmod o-rwx RCS/[file tail $i],v
	    exec chgrp parflow RCS/[file tail $i],v
	    
	    if $email {
		# Add commands to email notification
		append EmailAppend "cd \$PARFLOW_SRC[file dirname $i]\n"
		append EmailAppend "pfco [file tail $i]\n"
	    }
	}
    }

    cd $ParFlowSrc/config
    catch {exec ci -f -u version.h << "xpfci update"} msg

    catch {exec chgrp parflow RCS/version.h,v } msg

    global button
    set button 1

    if [string length $CheckinErrors] {


	toplevel .error



	label .error.errorlabel -text "The following error conditions were found"
	frame .error.tframe

	text .error.tframe.errortext -height 10 -width 80 -yscrollcommand ".error.tframe.yscroll set"

	scrollbar .error.tframe.yscroll -command ".error.tframe.errortext yview"


	pack .error.errorlabel
	pack .error.tframe
	pack .error.tframe.yscroll -side right -fill y
	pack .error.tframe.errortext

	.error.tframe.errortext insert end $CheckinErrors

	label .error.label -text "Do you want to send the email notification?"
	pack .error.label

	#
	# Button bar for commands
	#
	frame .error.buttonbar -relief groove 
	button .error.buttonbar.continue -text "Continue" -command "set button 1"
	button .error.buttonbar.cancel -text "Cancel" -command "set button 0"
	
	pack .error.buttonbar.continue -side left
	pack .error.buttonbar.cancel -side left
	pack .error.buttonbar

	tkwait variable button 

	destroy .error
    }

    if $button { 
	set emailnotes ""

	append emailnotes "[.emailnotes.text get 1.0 end]"

	# Append the version.h file so we get the checkin number and data
	append emailnotes "\n\nVersion Information\n"
	append emailnotes "=============================================================================\n"
	set file [open $ParFlowSrc/config/version.h r]

	while { [gets $file line] >= 0 } {
	    append emailnotes "$line"
	    append emailnotes "\n"
	}

	close $file
	
	append emailnotes "\n\n\nCommands to checkout changed files\n"
	append emailnotes "=============================================================================\n"
	
	append emailnotes $EmailAppend

	if [catch {exec Mail -s "ParFlow Update" $UserToNotify << "$emailnotes"} msg] {
	    # need a popup here
	    puts "Error when mailing"
	    puts $msg
	} 

	#
	# Reset the lists to null
	#
	
	unset FilesCheckedIn
	unset FileCheckInMessage

	set FileGroup 0
	set FilesCheckedIn(1) ""
	set FileCheckInMessage(1) ""
	set EmailAppend ""
	set CheckinErrors ""
    }
}

proc Quit {} {

    global FileGroup

    if {$FileGroup > 0} {
	after idle {.dialog1.msg configure -wraplength 4i}
	set i [tk_dialog .dialog1 "Checkins Not Completed" {There are checkins which have not been completed, Do you wish to Exit?} info 0 OK Cancel]
	
	switch $i {
	    0 {exit}
	    1 {}
	}
    } {exit}
}

set FileGroup 0
set FilesCheckedIn(1) ""
set FileCheckInMessage(1) ""

set UserToNotify "parflow@mercury.llnl.gov $env(LOGNAME)"
set EmailAppend ""
set CheckinErrors ""

if [ string length [array names env PARFLOW_SRC] ] {
    set ParFlowSrc $env(PARFLOW_SRC)
} {
    set ParFlowSrc $env(PARFLOW_DIR)
}


set button 0

frame .mbar -relief raised -bd 2
pack .mbar -side top -fill x
menubutton .mbar.file -text File -underline 0 \
		-menu .mbar.file.menu
menubutton .mbar.help -text Help -underline 0 \
		-menu .mbar.help.menu
pack .mbar.file -side left
pack .mbar.help -side right
tk_menuBar .mbar .mbar.file .mbar.help

set m .mbar.file.menu
menu $m
$m add command -label "Open file for mail..." -command "ReadEmailFile"
$m add separator
$m add command -label "Quit" -command "Quit"

set m .mbar.help.menu
menu $m
$m add command -label "Help not here yet.." -command {error "No help at this time....\" entry"}

#
# Button bar for commands
#
frame .buttonbar -relief groove 
button .scanfiles -text "Scan Files" -command ScanFiles
button .difffiles -text "Diff Files" -command DiffFiles
button .checkinfiles -text "CheckIn Files" -command CheckInFiles
button .done -text "Done" -command Done
button .quit -text "Quit" -command Quit

pack .scanfiles .difffiles .checkinfiles .done .quit -side left -expand 1 -fill both -in .buttonbar 

pack .buttonbar

#
# List of locked Files
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






