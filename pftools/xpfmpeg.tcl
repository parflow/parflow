#!/bin/sh
# the next line restarts using tclsh \
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


proc Quit {} {
    puts "Exiting...."
    exit
}

proc AddFile {} {
    set filters {{"All Files" *}}

    set filename [FSBox "Add File" $filters "loading"]

    .frames.list insert end [file tail $filename]
}

proc AddFiles {} {

    global button
    global FileSpec
    global StartInt
    global StopInt
    global IncInt

    set w .addpopup

    toplevel $w

    frame $w.file
    label $w.file.label -text "File specifier:" -width 15 -anchor w
    entry $w.file.entry -width 40 -textvariable FileSpec
    pack $w.file.label $w.file.entry -side left
    pack $w.file

    frame $w.start
    label $w.start.label -text "Starting integer:" -width 15 -anchor w
    entry $w.start.entry -width 40 -textvariable StartInt
    pack $w.start.label $w.start.entry -side left
    pack $w.start

    frame $w.stop
    label $w.stop.label -text "Stoping integer:" -width 15 -anchor w
    entry $w.stop.entry -width 40 -textvariable StopInt
    pack $w.stop.label $w.stop.entry -side left
    pack $w.stop

    frame $w.inc
    label $w.inc.label -text "Increment integer:" -width 15 -anchor w
    entry $w.inc.entry -width 40 -textvariable IncInt
    pack $w.inc.label $w.inc.entry -side left
    pack $w.inc

    focus $w.file.entry

    #
    # Button bar for commands
    #
    frame $w.buttonbar -relief groove 
    button $w.buttonbar.continue -text "Continue" -command "set button 1"
    button $w.buttonbar.cancel -text "Cancel" -command "set button 0"
    
    pack $w.buttonbar.continue -side left
    pack $w.buttonbar.cancel -side left
    pack $w.buttonbar
    
    tkwait variable button 
    
    destroy $w

    if $button { 
	.frames.list insert end "$FileSpec \[$StartInt-$StopInt+$IncInt\]"
    }
}

proc DelFiles {} {

    set fileindices [lsort -decreasing -integer [.frames.list curselection]]

    foreach i $fileindices {
	.frames.list delete $i
    }
}

proc DoConvert {file} {
    set ext [file extension $file]
    set root [file rootname $file]

    if [regexp {(\[[0-9]+)} $file start] {
	set start [string trimleft $start \[]
	set format_string "%.[string length $start]d"

	regexp {(-[0-9]+)} $file end
	set end [string trimleft $end -]

	regexp {(\+[0-9]+)} $file inc
	set inc [string trimleft $inc \+]

	set filesname [string range $file 0 [string first " " $file]]
	
	set suffix [file extension $filesname]

	set prefix [file rootname $filesname]

	set ret "$prefix.ppm \[$start-$end+$inc\]"

	#
	# Strip of lead 0 since this means octal to tcl
	#
	set start [string trimleft $start 0]
	if [expr [string length $start] == 0] {
	    set start 0
	}
	set end [string trimleft $end 0]
	set inc [string trimleft $inc 0]

	for { set i $start} { [expr $i <= $end] } {incr i $inc} {
	    regsub {\*}  $filesname [format $format_string $i] filename
	    DoConvert [string trim $filename " "]
	}

	return $ret
    }
	    
    if { "$ext" == ".ppm" } {
	#
	# No conversion needed
	#
	return $file
    } {

	set convert 1
	if { [file exists $root.ppm] } {
	    if { [file mtime $file] < [file mtime $root.ppm] } {
		set convert 0
	    }
	}
	

	if $convert {
	    #
	    # Need to do a conversion
	    # 
	    exec convert $file $root.ppm
	}

	return $root.ppm
    }
}

proc CreateMpeg {} {

    global env
    global MPEGFileName

    set filters {{"MPEG" {.mpg}}
                 {"All Files" *}}

    set filename [FSBox "MPEG output" $filters "saving"]

    if [string length filename] {
	set MPEGFileName [file tail $filename]

	. configure -cursor watch
	update

	set files ""
	set num [.frames.list size]
	for { set i 0} { $i < $num } {incr i 1} {
	    append files "[DoConvert [.frames.list get $i]]\n"
	}
	set files [string trim $files "\n"]

	. configure -cursor watch
	update

	#
	# Read in the parameter file
	#
	set params ""
	set file [open $env(PARFLOW_DIR)/bin/xpfmpeg.param r]
	while { [gets $file line] >= 0 } {
	    append params "$line\n"
	}
	close $file
	
	regsub -all "VALUE_OUTPUT_FILENAME" $params $MPEGFileName params
	regsub -all "VALUE_INPUT_FILENAMES" $params $files params

	set file [open xpfmpeg.temp.param w]
	puts $file "$params"
	close $file

	. configure -cursor watch
	update

	catch {exec mpeg_encode xpfmpeg.temp.param} msg
	puts "$msg"

	. configure -cursor xterm
	update


    } {
	puts "Cancel"
    }
}


set button 0

set MPEGFileName ""

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
$m add command -label "Quit" -command "Quit"

set m .mbar.help.menu
menu $m
$m add command -label "Help not here yet.." -command {error "No help at this time....\" entry"}

#
# Button bar for commands
#
frame .buttonbar -relief groove 
button .addfile -text "Add Single File" -command AddFile
button .addfiles -text "Add File Sequence" -command AddFiles
button .delfiles -text "Delete Files" -command DelFiles
button .creatempeg -text "Create Movie" -command CreateMpeg
button .quit -text "Quit" -command Quit

pack .addfile .addfiles .delfiles .creatempeg .quit -side left -expand 1 -fill both -in .buttonbar 

pack .buttonbar

#
# List of locked Files
#
frame .frames
pack .frames -side top -expand yes -fill y -fill x

label .frames.title	-text "Current Files"
pack .frames.title -side top

scrollbar .frames.yscroll -command ".frames.list yview"
scrollbar .frames.xscroll -orient horizontal -command ".frameslist xview" 

listbox .frames.list -height 10 -width 78 -relief raised -borderwidth 2 -yscrollcommand ".frames.yscroll set" -xscrollcommand ".frames.xscroll set" -selectmode multiple -exportselection no
    
pack .frames.yscroll -side right -fill y
pack .frames.xscroll -side bottom -fill x
pack .frames.list -expand yes -fill y







