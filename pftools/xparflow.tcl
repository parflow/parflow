#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

#
# Notes:
#
# have autoload/unload flag on run panel check if files 
#     have been loaded or not
#

#
# add deletion of unused variables so files don't get huge with unused
# items
#

#
# Fix up filenames for Win32
# 
proc FixupFilename { filename } {
    global IsArchUnix

    if $IsArchUnix {
	set new_filename $filename
    } {
	regsub -all \\\\ $filename "/" new_filename
    }
    
    return $new_filename
}


# Determine which Arch we are running on
if { [ string compare $tcl_platform(platform) "windows" ] } {
    set IsArchUnix 1

} {
    set IsArchUnix 0
}

set PARFLOW_DIR [FixupFilename $env(PARFLOW_DIR)]
#
# Is there a better way to include things like this?
# yes there is....
#
#
#SGS: change this to Jims FileBox so we reduce number of items used.
#
source $PARFLOW_DIR/bin/fsb.tcl

#
# Clears the current state and gets an new project name
#
proc New {} {
    global Vars

    set filters {
	{"ParFlow Project" {.xpf}}
	{"All files"	*}
    }

    #
    # Get name for the project
    #
    set filename [FSBox "New Project" $filters "saving" ]

    set CurrentProject [file root [file tail $filename]]

    #
    # SGS note:
    # really should check to see if file exists and prompt if they
    # want the same name?
    #

    #
    # Clean up the existing state
    #
    foreach i {NumProcsP NumProcsQ NumProcsR \
	    NumProcsVM NumProcs NumPhases} {
	set Vars($i) 1
    }
    
    foreach i { "Wells" } {
	set Vars($i) ""
    }

    foreach i {"NumPhaseSource" "NumWells" \
	    "xlower" "ylower" "zlower" \
	    "xupper" "yupper" "zupper" \
	    "spacingx" "spacingy" "spacingz" \
	    "grippointsx" "gridpointsy" "gridpointsz" } {
	set Vars($i) 0
    }

    set w .new
    
    #
    # Prompt the user to enter the number of phases for this project
    # This will determine the dims of several other things so needs to
    # be set for the entire project
    # 
    toplevel $w
    wm title $w "New Project"

    label $w.label -text "Number of Phases"

    entry $w.entry -relief sunken -width 20 -textvariable Vars(NumPhases)

    frame $w.buttons
    pack  $w.buttons -side bottom -expand y -fill x -pady 2m
    button $w.buttons.new -text "OK" -command \
	    "destroy $w; Save $CurrentProject"
    pack $w.buttons.new

    pack $w.label $w.entry
    
}

#
# 
#
proc ProcessPhaseSource {} {
    global Vars
    global Unknowns

    if [expr $Vars(NumPhaseSource) == 0] {
	set Vars(PhaseSourceData) "0.0\n"
    }

    set temp "$Vars(NumPhaseSource)\n$Vars(PhaseSourceData)"

    #
    # Replicate for Each phase
    # 
    set Vars(PhaseSourceData) ""
    for { set j 0} { $j < $Vars(NumPhases) } {incr j} {
	append Vars(PhaseSourceData) $temp
    }
}

#
# SGS Need to convert things to double !!!!
#

#
# 
#
proc PrettyList {format list} {

    return [join $list "     "]
}

#
# Snaps values to grid point
#
proc SnapTo { axis value } {
    global Vars

    return [expr (double(round( ($value - $Vars(${axis}lower)) \
	    / $Vars(spacing${axis}))) * $Vars(spacing${axis})) \
	    + $Vars(${axis}lower)]
}

#
# Prepare wells for output
#
proc ProcessWells {} {
    global Vars

    set Vars(WellData) ""
    set Vars(NumWellData) 0

    foreach i $Vars(Wells) {
	set name [lindex $i 0]

	set X [SnapTo "x" [lindex $i 1]]
	set Y [SnapTo "y" [lindex $i 2]]
	set ZLower [SnapTo "z" [lindex $i 3]]
	set ZUpper [SnapTo "z" [lindex $i 4]]

	set Value [lindex $i 6]


	if [expr [string compare [lindex $i 5] "Pumping Rate"] == 0]  {
	    append Vars(WellData) "# Well <$name>\n" 
	    set Z $ZLower
	    while {[expr $Z <= $ZUpper]} {
		append Vars(WellData) "$X $Y $Z $Value\n" 
		incr Vars(NumWellData)
		set Z [expr $Z + $Vars(spacingz)]
	    } 
	} {
	    append Vars(PhaseSourceData) "# Well <$name>\n" 
	    # lower corner
	    append Vars(PhaseSourceData) "1\n"
	    set CX [expr $X - ($Vars(spacingx)/2.0)]
	    set CY [expr $Y - ($Vars(spacingy)/2.0)]
	    set CZ [expr $ZLower - ($Vars(spacingz)/2.0)]
	    append Vars(PhaseSourceData) "$CX $CY $CZ\n"

	    # upper corner
	    set CX [expr $X + ($Vars(spacingx)/2.0)]
	    set CY [expr $Y + ($Vars(spacingy)/2.0)]
	    set CZ [expr $ZLower + ($Vars(spacingz)/2.0)]
	    append Vars(PhaseSourceData) "$CX $CY $CZ\n"
	    append Vars(PhaseSourceData) "$Value\n"

	    incr Vars(NumPhaseSource)
	}
    }

}

#
# 
#
proc PopulateWells {} {
    global Temp
    global Vars
    
    set w .wells
    
    $w.scrolllist.list delete 0 end

    foreach i $Vars(Wells) {
	$w.scrolllist.list insert end [PrettyList "String Real Real Real Real" $i]
    }    
}

#
#
#
proc DisplayWell { listy } {
    global Temp
    global Vars
    
    set w .wells
    
    set index [$w.scrolllist.list nearest $listy]

    if {$index >= 0} {
	set list [lindex $Vars(Wells) $index]

	set index 0
	foreach i {"Well Label" "X" "Y" "Z Lower" "Z Upper" "WellOption" "Value"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    set  Temp($widget) [lindex $list $index]
	    incr index
	}
    } {
	puts "something else clear?"
    }
}

#
# Adds a new well
#
proc AddWellToList {} {
    global Temp
    global Vars
    
    set w .wells

    foreach i {"Well Label" "X" "Y" "Z Lower" "Z Upper" "WellOption" "Value"} {
	set widget [string tolower $i]
	regsub -all " " $widget "" widget
	lappend newwell $Temp($widget)
    }

    set found 0
    set index 0
    foreach i $Vars(Wells) {
	set name [lindex $i 0]
	if [expr [string compare $name $Temp(welllabel)] == 0] {
	    set found 1
	    break
	}
	incr index
    }

    if $found { 
	$w.scrolllist.list delete $index
	$w.scrolllist.list insert $index  [PrettyList "String Real Real Real Real" $newwell]
	set Vars(Wells) [lreplace $Vars(Wells) $index $index $newwell]
    } {
	$w.scrolllist.list insert end [PrettyList "String Real Real Real Real" $newwell]
	lappend Vars(Wells) $newwell

	set Vars(NumWells) [expr $Vars(NumWells) + 1]
    }
}

#
# Delete a well
#
proc DelWell {} {
    global Vars

    set w .wells
    
    set delindices [lsort -integer [$w.scrolllist.list curselection]]

    foreach i $delindices {
	set Vars(Wells) [lreplace $Vars(Wells) $i $i]
	set Vars(NumWells) [expr $Vars(NumWells) - 1]
    }

    DeleteFromList $w.scrolllist.list    
}

#
# The wells subpanel
#
proc Wells {} {
    global Vars
    set w .wells

    if { [winfo exists $w] } {
	wm deiconify $w
	PopulateWells
    } {
	toplevel $w
	wm title $w "Wells Setup"

	label $w.label -text "Current Wells"
	pack $w.label
	frame $w.scrolllist
	pack $w.scrolllist 
	scrollbar $w.scrolllist.listscroll -command "$w.scrolllist.list yview"
	listbox $w.scrolllist.list -width 40 -yscrollcommand "$w.scrolllist.listscroll set"
	bind $w.scrolllist.list <ButtonPress-1> "DisplayWell %y"

	pack $w.scrolllist.listscroll -side right -fill y
	pack $w.scrolllist.list -expand yes -fill y -fill x

	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.new -text "Add/Update" -command " AddWellToList"
	button $w.buttons.del -text "Delete" -command "DelWell"
	button $w.buttons.ok -text OK -command "wm withdraw $w"
	pack  $w.buttons.new $w.buttons.del \
		$w.buttons.ok -side left -expand 1
	
	foreach i {"Well Label" "X" "Y" "Z Lower" "Z Upper"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $w.$widget -bd 2
	    entry $w.$widget.entry -relief sunken -width 20 -textvariable Temp($widget)
	    label $w.$widget.label -text $i
	    pack $w.$widget -fill x
	    pack $w.$widget.label -side left
	    pack $w.$widget.entry -side right
	}

	frame $w.wellopts
	pack $w.wellopts -side top -expand yes 

	foreach option {"Pumping Rate" "Hydraulic Head"} {
	    set lower [string tolower $option]
	    radiobutton $w.wellopts.$lower -text $option -variable Temp(welloption) \
		    -relief flat -value $option
	    pack $w.wellopts.$lower -side top -pady 2 -anchor w
	}
	
	foreach i {"Value"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $w.$widget -bd 2
	    entry $w.$widget.entry -relief sunken -width 20 -textvariable Temp($widget)
	    label $w.$widget.label -text $i
	    pack $w.$widget -fill x
	    pack $w.$widget.entry -side right
	    pack $w.$widget.label -side left
	}

	PopulateWells
    }
}

proc ProcessGridSpacing {axis} {
    global Vars

    set Vars(gridpoints${axis}) [expr round( ($Vars(${axis}upper) - \
	    $Vars(${axis}lower)) / $Vars(spacing${axis}))+1]

    ProcessComputationGrid
}

#
# Prepare computation grid variables for output
#
proc ProcessComputationGrid {} {
    global Vars
    foreach i {"x" "y" "z"} {
	set gridvar gridpoints${i}
	set spacingvar spacing${i}
	set upper ${i}upper
	set lower ${i}lower

	set Vars($spacingvar) [expr ($Vars($upper) - \
		$Vars($lower))/($Vars($gridvar))]
    }
}

#
# Computation grid subpanel
#
proc ComputationGrid {} {
    global Vars
    set w .computationgrid

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Computation Grid Setup"

	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.ok -text OK -command "wm withdraw $w; ProcessComputationGrid"
	pack $w.buttons.ok -side left -expand 1

	set l $w.left
	set r $w.right

	frame $l
	frame $r
	pack $l $r -side left -expand yes 

	foreach i {"Grid Points X"  "Grid Points Y" "Grid Points Z"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $l.$widget -bd 2
	    entry $l.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
	    bind $l.$widget.entry <FocusOut> +{ProcessComputationGrid}
	    bind $l.$widget.entry <KeyPress-Return> +{ProcessComputationGrid}
	    label $l.$widget.label -text $i
	    pack $l.$widget
	    pack $l.$widget.entry -side right
	    pack $l.$widget.label -side left
	}

	foreach i {"X" "Y" "Z"} {
	    set widget [string tolower spacing$i]
	    frame $r.$widget -bd 2
	    entry $r.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
	    bind $r.$widget.entry <FocusOut> "ProcessGridSpacing [string tolower $i]"
	    bind $r.$widget.entry <KeyPress-Return> "ProcessGridSpacing [string tolower $i]"
	    label $r.$widget.label -text "Spacing $i"
	    pack $r.$widget
	    pack $r.$widget.entry -side right
	    pack $r.$widget.label -side left
	}
    }
}

#
# Domain subpanel
#
proc Domain {} {
    global Vars
    set w .domain

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Domain Bounding Box"
	wm iconname $w "Domain Bounding Box"

	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.ok -text OK -command "wm withdraw $w"
	pack $w.buttons.ok -side left -expand 1

	set l $w.left
	set r $w.right

	frame $l
	frame $r
	pack $l $r -side left -expand yes 

	foreach i {"X lower" "Y lower" "Z lower"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $l.$widget -bd 2
	    entry $l.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
	    bind $l.$widget.entry <FocusOut> +{ProcessComputationGrid}
	    bind $l.$widget.entry <KeyPress-Return> +{ProcessComputationGrid}
	    label $l.$widget.label -text $i
	    pack $l.$widget
	    pack $l.$widget.entry -side right
	    pack $l.$widget.label -side left
	}

	foreach i {"X upper" "Y upper" "Z upper"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $r.$widget -bd 2
	    entry $r.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
	    bind $r.$widget.entry <FocusOut> +{ProcessComputationGrid}
	    bind $r.$widget.entry <KeyPress-Return> +{ProcessComputationGrid}
	    label $r.$widget.label -text $i
	    pack $r.$widget
	    pack $r.$widget.entry -side right
	    pack $r.$widget.label -side left
	}
    }
}

#
# 
#
proc ProcessMisc {} {
    global Vars
    
#    set Vars(Density) ""
#    set Vars(Viscosity) ""
#    for { set j 0} { $j < $Vars(NumPhases) } {incr j} {
#	set i "Phase Density ${j}"
#	set widget [string tolower $i]
#	regsub -all " " $widget "" widget
#	append Vars(Density) $Vars($widget)

#	set i "Phase Viscosity ${j}"
#	set widget [string tolower $i]
#	regsub -all " " $widget "" widget
#	append Vars(Viscosity) $Vars($widget)
#    }

}


#
# File popup to prompt for a new problem template
#
proc SetProblemTemplate {} {
    global Vars

    set filters {
	{"ParFlow Problem" {.pfin}}
	{"All files"	*}
    }

    set Vars(CurrentProblemTemplate) [FSBox "Problem Template" \
	    $filters "loading" ]
}

#
# Misc subpanel
#
proc ProblemTemplate {} {
    global Vars
    set w .misc

    toplevel $w
    wm title $w "Misc Setup"
    wm iconname $w "Misc Setup"
    
    frame $w.buttons
    pack  $w.buttons -side bottom -expand y -fill x -pady 2m
    button $w.buttons.ok -text OK -command "destroy $w"
    pack $w.buttons.ok -side left -expand 1

    set i "Problem Template"

    set widget [string tolower $i]
    regsub -all " " $widget "" widget
    frame $w.$widget -bd 2

    entry $w.$widget.entry -relief sunken -width 40 -textvariable Vars(CurrentProblemTemplate)
    button $w.$widget.label -text $i -command SetProblemTemplate
    pack $w.$widget
    pack $w.$widget.entry -side right
    pack $w.$widget.label -side left
    
#    set i "Gravity"
#    set widget [string tolower $i]
#    regsub -all " " $widget "" widget
#    frame $w.$widget -bd 2
#    entry $w.$widget.entry -relief sunken -width 20 -textvariable Vars($i)
#    label $w.$widget.label -text $i
#    pack $w.$widget
#    pack $w.$widget.entry -side right
#    pack $w.$widget.label -side left

#    set l $w.left
#    set r $w.right
    
#    frame $l
#    frame $r
#    pack $l $r -side left -expand yes 
    
#    for { set j 0} { $j < $Vars(NumPhases) } {incr j} {
#	set i "Phase Density ${j}"
#	set widget [string tolower $i]
#	regsub -all " " $widget "" widget
#	frame $l.$widget -bd 2
#	entry $l.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
#	label $l.$widget.label -text $i
#	pack $l.$widget
#  	pack $l.$widget.entry -side right
#  	pack $l.$widget.label -side left
#  	
#  	set i "Phase Viscosity ${j}"
#  	set widget [string tolower $i]
#  	regsub -all " " $widget "" widget
#  	frame $r.$widget -bd 2
#  	entry $r.$widget.entry -relief sunken -width 20 -textvariable Vars($widget)
#  	label $r.$widget.label -text $i
#  	pack $r.$widget
#  	pack $r.$widget.entry -side right
#  	pack $r.$widget.label -side left
#      }

}

#
# Problem setup subpanel.  Invokes different subpanels to setup
# values.
#
proc ProblemSetup {} {

    set w .problem

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Problem Setup"
	wm iconname $w "Problem Setup"

	frame $w.buttonbar -relief groove 
	button $w.problemtemplate -text "Problem Template" -command ProblemTemplate
	button $w.domainbutton -text "Domain Bounding Box" -command Domain
	button $w.compgrid -text "Computation Grid" -command ComputationGrid
#	button $w.wells -text "Wells" -command Wells
	button $w.unknown -text "Set Unknowns" -command Unknown
	button $w.ok -text "OK" -command "wm withdraw $w"
	pack $w.buttonbar

#	pack $w.domainbutton $w.compgrid $w.unknown $w.ok 
#	pack $w.domainbutton $w.compgrid $w.wells $w.problemtemplate $w.unknown 

	pack $w.domainbutton $w.compgrid $w.unknown $w.problemtemplate $w.ok \
		-fill x -expand y -side top -in $w.buttonbar
    }
}

proc ProcessTolerance {} {
}

proc Tolerance {} {
    global Vars

    set w .tolerance

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Solver Parameters"

	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.ok -text OK -command "wm withdraw $w; ProcessTolerance"
	pack $w.buttons.ok -side left -expand 1

	set l $w.left
	set r $w.right

	frame $l
	frame $r
	pack $l $r -side left -expand yes 

	set names(SAdvectionOrder) SolverImpessadvectionorder
	set names(AdvectionOrder) SolverImpesadvectionorder
	set names(MaxIterations) SolverImpesmaxiter
	set names(CFL) SolverImpesCFL
	set names(Relative) SolverImpesreltol
	set names(Absolute) SolverImpesabstol
	set names(Drop) SolverImpesdroptol

	foreach i {"SAdvectionOrder" "AdvectionOrder" "CFL" "MaxIterations" "Relative" "Absolute" "Drop"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $l.$widget -bd 2
	    entry $l.$widget.entry -relief sunken -width 20 -textvariable Vars($names($i))
	    label $l.$widget.label -text $i
	    pack $l.$widget.entry -side right
	    pack $l.$widget.label -side left
	    pack $l.$widget
	}
    }    
}

proc SolverSetup {} {

    set w .solver

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Solver Setup"
	wm iconname $w "Solver Setup"

	frame $w.buttonbar -relief groove 

	button $w.configure -text "Advanced Configuration" \
		-command {exec xpfconfig 1 $Vars(CurrentConfig)}
	button $w.tolerance -text "Solver Parameters" \
		-command Tolerance
	button $w.output -text "File Output Options" \
		-command OutputOptions

	button $w.ok -text "OK" -command "wm withdraw $w"

	pack $w.buttonbar

	pack $w.configure $w.tolerance $w.output $w.ok \
		-fill x -expand y -side top -in $w.buttonbar
    }
}

#
# Output a string that can containing blanks etc to a file
#
proc WriteComplexString {file string} {
    puts $file [string length $string]
    puts $file $string
}

#
# Read a string that can containing blanks etc to a file
#
proc ReadComplexString {file stringname} {
    upvar $stringname string

    gets $file len
    set string [read $file $len]
    # read the newline from the puts
    read $file 1
}

#
# Read an array from a file
#
proc ReadArray {file name} {
    upvar $name a

    foreach el [array names a] {
	unset a($el)
    }

    gets $file num
    for { set j 0} { $j < $num } {incr j} {
	gets $file el
	ReadComplexString $file a($el)
    }
}


#
# Write an array from a file
#
proc WriteArray {file name} { 
    upvar $name a

    puts $file [array size a]
    foreach el [array names a] {
	puts $file $el
	WriteComplexString $file $a($el)
    }
}


#
# Save the current state to file
#
proc Save {name} {
    global Vars
    global Unknowns

    #
    # SGS: if file exists we should probably prompt to overwrite
    #

    set file [open [FixupFilename $name.xpf] "w"]
    
    foreach i "Vars Unknowns" {
	WriteArray $file $i
    }

    close $file
}

#
# Prompts user to enter the name of an existing project
# Defaults to current if an error occurs.
#
proc ProjectPrompt { operation } {
    global CurrentProject

    set filters {
	{"ParFlow Project" {.xpf}}
	{"All files"	*}
    }

    set filename [FSBox "Select Project" $filters $operation]    

    if { $operation == "loading"} {
	if [file isfile $filename] {
	    set CurrentProject [file root [file tail $filename]]
	    
	    .projectlabel configure -text $CurrentProject
    
	    return $filename
	} {
	    # What should we return on failure
	    return $CurrentProject.xpf
	}
    } {
	if { $filename == "" } {
	    # What should we return on failure
	    return $CurrentProject.xpf
	} {
	    set CurrentProject [file root [file tail $filename]]
	    
	    .projectlabel configure -text $CurrentProject
    
	    return $filename
	}
    }
}

#
# Load a project from a file
#
proc LoadProject {filename} {
    global Vars
    global Unknowns

    set file [open [FixupFilename $filename] "r"]

    foreach i "Vars Unknowns" {
	ReadArray $file $i
    }
    close $file
}

#
# Prompt user for project and load it, overwriting current
# state.
#
proc Load {} {
    set filename [ProjectPrompt loading]

    #
    # SGS if the filename is the same then we should probably not
    # overwrite what is already here!
    # 
    
    LoadProject $filename
}

#
# Prompt user for a filename and save the current project to that 
# file.
#
proc SaveAs {} {
    global CurrentProject

    set filename [ProjectPrompt saving]

    Save $CurrentProject
}

#
# Run a command.
# Output is directed to the output window.
#
proc RunCmd {command} {
    global OutputText
    global Output
    global IsArchUnix

    . configure -cursor watch
    $Output configure -cursor watch
    $OutputText configure -cursor watch
    update

    set command [FixupFilename $command]

    # Put the command in the output window so we can see what was started
    $OutputText insert end $command\n 
    $OutputText see end



    if $IsArchUnix {
	set command "$command |& cat"
	set buildproc [open |$command {RDONLY NOCTTY} ]
    } {
	set command "$command"
	set buildproc [open |$command {RDONLY } ]
    }

    #
    # Put the output from the command to the screen
    #
    while { [gets $buildproc line] >= 0 } {
	$OutputText insert end $line\n
	$OutputText see end
	update
    }

    close $buildproc

    . configure -cursor xterm
    $Output configure -cursor xterm
    $OutputText configure -cursor xterm
    update
}

#
# Prompts the user to enter values for currently unknown variables
# in all input templates
#
proc Unknown { } {
    global CurrentProject
    global Vars

    ProcessUnknown solver.template $CurrentProject.in.solver
    ProcessUnknown $Vars(CurrentProblemTemplate) $CurrentProject.in.problem
}

#
# Creates an outputfile with the variables in the input file replaces
# with their current values.  If an unknown variable is encountered 
# the user will be prompted to enter a value for it.
#
proc ProcessUnknown {inputfilename outputfilename} {
    global Vars
    
    set CommentState 0


    set inputfile [open [FixupFilename $inputfilename] "r"]
    set outputfile [open [FixupFilename $outputfilename] "w"]
    # sgs error checking here

    while { [gets $inputfile line] >= 0 } {

	# Store comment lines for unknown variables
	if { [string first "#" $line] == 0 } {
	    if $CommentState { 
		append comments "$line\n"
	    } {
		set comments $line
		set CommentState 1
	    }
	} {
	    set CommentState 0
	}
	
	# Process the line for variables
	while { [regexp {\$[a-zA-Z]+} $line varname] } {
	    set newvar "\\$varname"
	    set rootvar [string range $varname 1 end]
	    if [string length [array names Vars $rootvar]] {
		regsub $newvar $line $Vars($rootvar) line
	    } {
		# Prompt user to input the value
		regsub $newvar $line [InputUnknown $rootvar $comments] line 
	    }
	}
	puts $outputfile $line
    }

    close $inputfile
    close $outputfile
}

#
# Input panel to prompt user to enter the a value for an unknown variable.
# Displays the comments leading up to it as help for the format and values
# which are expected.
#
proc InputUnknown {variable comments} {
    global Unknowns

    set w .unknown

    set button 0

    if { [winfo exists $w] } {
	wm deiconify $w
	$w.value.title configure -text "Enter value for $variable"
    } {
	toplevel $w
	wm title $w "Run ParFlow"
	wm iconname $w "Run ParFlow"

	frame $w.comments
	label $w.comments.title -text "Help Information"
	text $w.comments.text -height 10 -width 80 \
		-yscrollcommand "$w.comments.yscroll set"
	scrollbar $w.comments.yscroll -command "$w.comments.text yview"

	frame $w.value
	label $w.value.title -text "Enter value for $variable"
	text $w.value.text -height 10 -width 80 \
		-yscrollcommand "$w.value.yscroll set"

	scrollbar $w.value.yscroll -command "$w.value.text yview"

	button $w.continue -text "Continue" -command "set button 1"

	pack $w.comments
	pack $w.comments.title
	pack $w.comments.yscroll -side right -fill y
	pack $w.comments.text

	pack $w.value
	pack $w.value.title
	pack $w.value.yscroll -side right -fill y
	pack $w.value.text
	pack $w.continue
    }

    $w.comments.text delete 1.0 end
    $w.comments.text insert end $comments

    $w.value.text delete 1.0 end
    if [string length [array names Unknowns $variable]] {
	$w.value.text insert end $Unknowns($variable)
    }

    tkwait variable button 
    wm withdraw $w

    return [set Unknowns($variable) [string trimright [$w.value.text get 1.0 end] "\n"]]
}

# SGS need to catch the output somewhere ala xparflow proc
proc ReplaceVariables {inputfilename outputfilename} {
    global Vars
    global Unknowns
    
    set CommentState 0


    set inputfile [open [FixupFilename $inputfilename] "r"]
    set outputfile [open [FixupFilename $outputfilename] "w"]
    # sgs error checking here

    while { [gets $inputfile line] >= 0 } {

	# Store comment lines for unknown variables
	if { [string first "#" $line] == 0 } {
	    if $CommentState { 
		append comments "$line\n"
	    } {
		set comments $line
		set CommentState 1
	    }
	} {
	    set CommentState 0
	}
	
	# Process the line for variables
	while { [regexp {\$[a-zA-Z]+} $line varname] } {
	    set newvar "\\$varname"
	    set rootvar [string range $varname 1 end]
	    if [string length [array names Vars $rootvar]] {
		# Need some checking if a math function?
		#regsub $newvar $line [expr $Vars($rootvar)] line
		regsub $newvar $line $Vars($rootvar) line
	    } {
		if [string length [array names Unknowns $rootvar]] {
		    regsub $newvar $line $Unknowns($rootvar) line 
		} {
		    regsub $newvar $line [InputUnknown $rootvar $comments] line
		}
	    }

	}
	puts $outputfile $line
    }

    close $inputfile
    close $outputfile
}

#
#
#
proc ProcessRelativePermeability {} {
    global Vars

    if $Vars(NumPhaseSource) {
	set Vars(RelativePermeability) ""
	for { set j 0} { $j < $Vars(NumPhases) } {incr j} {
	    append Vars(RelativePermeability) "# Phase $j\n1\n0.0\n2.0\n"
	}
    }
	set Vars(RelativePermeability) "0\n1.0\n"
}

proc ProcessPorosity {} {
    global Vars

    set Vars(Porosity) "0\n1.0\n"
}

proc ProcessCapillaryPressure {} {
    global Vars

    set Vars(CapillaryPressure) ""
    if $Vars(NumPhaseSource) {
	for { set j 0} { $j < [expr $Vars(NumPhases) - 1] } {incr j} {
	    append Vars(CapillaryPressure) "# Phase $j\n0\n1.0\n"
	}
    } 
}

proc CreateExe {} {
    global CurrentProject
        global Vars

    #
    # Process Remaining items
    #
    # Initialize phase source data to null since it is set up in several
    # places
    set Vars(PhaseSourceData) ""
    set Vars(NumPhaseSource) 0

    #
    # Compute the well extents based on user input
    # 

    # ProcessWells

    # ProcessRelativePermeability

    # ProcessPorosity

    # ProcessCapillaryPressure

    # ProcessMisc

    #
    # Final processing for the Phase sources
    #
    # ProcessPhaseSource
    
    ReplaceVariables solver.template $CurrentProject.in.solver
    ReplaceVariables $Vars(CurrentProblemTemplate) $CurrentProject.in.problem
}
proc CreateInputParFlow {} {
    global Vars
    global CurrentProject
    
    CreateExe
}

#
# Prompts user for a file using a selection box and loads that file
# onto the current machine configuration.
#
proc PfLoad { } {
    global CurrentProject


   set filters {
	{"ParFlow Binary" {.pfb}}
	{"ParFlow Scattered Binary" {.pfsb}}
	{"All files"	*}
    }

    #
    # Get file from user
    #
    set filename [FSBox "Load File" $filters "loading" ]    

    #
    # Need to make sure that the solver file is current
    #
    # SGS note:
    # Perhaps we should pass the args we need into pfload directly, 
    # rather than pulling them out of solver?
    #
    ReplaceVariables solver.template $CurrentProject.in.solver

    #
    # Perform the loading onto the current machine configuration
    #
    exec pfload $CurrentProject $filename $filename
}

proc PfUnload { } {

   set filters {
	{"ParFlow Binary" {.pfb}}
	{"ParFlow Scattered Binary" {.pfsb}}
	{"All files"	*}
    }

    set filename [FSBox "Unload File" $filters "saving" ]

    exec pfunload_file $filename 
}

proc PfUnloadAll { } {
    global CurrentProject

    exec pfunload $CurrentProject.out
}


proc RunParFlow {hostlist} {
    global BootMCFlag
    global GetMCFlag
    global Vars
    global CurrentProject
    global env
    global IsArchUnix
    global PARFLOW_DIR
    
    CreateExe

    set hostfile [open [FixupFilename .hostfile] "w"]
    # sgs add error here

    set num [$hostlist size]
    
    for { set j 0} { $j < $num } {incr j} {
	puts $hostfile [$hostlist get $j]
    }
    close $hostfile

    if { $BootMCFlag } { 
	if $IsArchUnix {
	    RunCmd "$PARFLOW_DIR/bin/killmc"
	    RunCmd "$PARFLOW_DIR/bin/bootmc $Vars(NumProcsVM)"
	} {
	    RunCmd "killmc.bat"
	    RunCmd "bootmc.bat $Vars(NumProcsVM)"
	}
	set BootMCFlag 0 
    }

    if { $GetMCFlag } {
	if $IsArchUnix {
	    RunCmd "$PARFLOW_DIR/bin/freemc"
	    RunCmd "$PARFLOW_DIR/bin/getmc $Vars(NumProcs)"
	} {
	    RunCmd "freemc.bat"
	    RunCmd "getmc.bat $Vars(NumProcs)"
	}
	set GetMCFlag 0
    }

    if $IsArchUnix {
	RunCmd "$PARFLOW_DIR/bin/run $CurrentProject"
    } {
	RunCmd "run.bat $CurrentProject"
    }
}

proc BatchParFlow {hostlist} {
    global BootMCFlag
    global GetMCFlag
    global Vars
    global CurrentProject
    global env
    global IsArchUnix
    global PARFLOW_DIR
    
    CreateExe

    set hostfile [open [FixupFilename .hostfile] "w"]
    # sgs add error here

    set num [$hostlist size]
    
    for { set j 0} { $j < $num } {incr j} {
	puts $hostfile [$hostlist get $j]
    }
    close $hostfile

    if $IsArchUnix {
	RunCmd "$PARFLOW_DIR/bin/batchmc $Vars(NumProcs) $CurrentProject"
    } {
	puts "Not working"
    }
}


proc ReadVMData {hostlistVM hostlist} {
    global env
    global Vars
    global PARFLOW_DIR

    set HostName [info hostname]

    set vmfile [open [FixupFilename $PARFLOW_DIR/bin/vmdata]]
    # sgs add error

    gets $vmfile num

    for { set j 0} { $j < $num } {incr j} {
	gets $vmfile HostNameVM
	gets $vmfile MaxNumProcsVM
	
	gets $vmfile NumSystemsVM
	if { $NumSystemsVM } {
	    for { set i 0} { $i < $NumSystemsVM } {incr i} {
		gets $vmfile SystemsVM($i)
	    }
	}
	
	gets $vmfile DefNumProcsVM

	if { $NumSystemsVM } {
	    gets $vmfile NumDefSystemsVM
	    for { set i 0} { $i < $NumDefSystemsVM } {incr i} {
		gets $vmfile DefSystemsVM($i)
	    }
	}
	
	if { ![string compare $HostNameVM $HostName] } {
	    break
	}
    }

    if { $j >= $num } {
	#
	# If we did not find this machine in the VMdatabase file then
	# use some defaults
	#
	set HostNameVM $HostName

	set MaxNumProcsVM 8
	set NumSystemsVM 1
	set SystemsVM(0) $HostName 

	set DefNumProcsVM 1
	set NumDefSystemsVM 1
	set DefSystemsVM(0) $HostName 


    }

    set Vars(NumProcsVM) $DefNumProcsVM

    for { set i 0} { $i < $NumSystemsVM } {incr i} {
	$hostlistVM insert end $SystemsVM($i)
    }

    set w .run
    $w.nodes.cvmnodes configure -to $MaxNumProcsVM

    for { set i 0} { $i < $NumDefSystemsVM } {incr i} {
	$hostlist insert end $DefSystemsVM($i)
    }
    
    close $vmfile
}

proc SetScaleMinMax { } {
    global Vars

    set w .run
    set nodes $w.nodes 
    
    $w.nodes.cmnodes configure -to $Vars(NumProcsVM) 

    set remainder [expr $Vars(NumProcsP) + ($Vars(NumProcsVM) - $Vars(NumProcs))/($Vars(NumProcsQ)*$Vars(NumProcsR))]
    $nodes.cvmnodesp configure -to $remainder 
    $nodes.cvmnodesp set $Vars(NumProcsP)

    set remainder [expr $Vars(NumProcsQ) + ($Vars(NumProcsVM) - $Vars(NumProcs))/($Vars(NumProcsP)*$Vars(NumProcsR))]
    $nodes.cvmnodesq configure -to $remainder 
    $nodes.cvmnodesq set $Vars(NumProcsQ)

    set remainder [expr $Vars(NumProcsR) + ($Vars(NumProcsVM) - $Vars(NumProcs))/($Vars(NumProcsP)*$Vars(NumProcsQ))]
    $nodes.cvmnodesr configure -to $remainder 
    $nodes.cvmnodesr set $Vars(NumProcsR)

}

proc ComputeDims {N} {
    set dim 0
    set num(0) 1
    set num(1) 1
    set num(2) 1
    set factor 2
    set temp 0

    global Vars

    while { $factor <= $N } {
	if { $N % $factor } {
	    incr factor
	} {
	    set num($dim) [expr $num($dim) * $factor]
	    set dim [expr ($dim +1) % 3]
	    set N [expr $N/$factor]
	}
    }
    
    if { $num(0) > $num(2) } {
	set temp $num(0)
	set num(0) $num(2)
	set num(2) $temp
    }

    if { $num(0) > $num(1)} {
	set temp $num(0)
	set num(0) $num(1)
	set num(1) $temp
    }

    if { $num(1) > $num(2)} {
	set temp $num(1)
	set num(1) $num(2)
	set num(2) $temp
    }

    set Vars(NumProcsP) $num(0)
    set Vars(NumProcsQ) $num(1)
    set Vars(NumProcsR) $num(2)

}

proc NodesChanged {value} {
    global Vars
    global GetMCFlag

    set Vars(NumProcs) $value

    ComputeDims $Vars(NumProcs)
    
    SetScaleMinMax
    
    set GetMCFlag 1
}

proc CPUDimChanged {varname value} {
    global GetMCFlag 
    global Vars
    global Vars

    set $varname $value

    set Vars(NumProcs) [expr $Vars(NumProcsP) * $Vars(NumProcsQ) * $Vars(NumProcsR) ]
    SetScaleMinMax
  
    set GetMCFlag 1
}

proc CPUVMChanged {value} {
    global Vars
    global Vars
    global GetMCFlag
    global BootMCFlag

    set Vars(NumProcsVM) $value

    if { $Vars(NumProcs) > $Vars(NumProcsVM) } {
	set $Vars(NumProcs) $Vars(NumProcsVM)
	NodesChanged $Vars($NumProcs)
    }

    SetScaleMinMax

    set BootMCFlag 1
    set GetMCFlag 1
}

proc DeleteFromList {list} {
    $list delete [lsort -decreasing -integer [$list curselection]]
}

proc AddFromList {fromlist tolist} {
    set fileindices [$fromlist curselection]
    foreach i $fileindices {
	$tolist insert end [$fromlist get $i]
    }
}

proc OutputOptions {} {
    global Vars

    set w .outputoptions

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "File Output Options"

	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.ok -text OK -command "wm withdraw $w; ProcessComputationGrid"
	pack $w.buttons.ok -side left -expand 1

	set l $w.left
	set r $w.right

	frame $l
	frame $r
	pack $l $r -side left -expand yes 

	set names(Permeability) SolverImpesprintsubsurfdata
	set names(Pressure) SolverImpesprintpress
	set names(Saturation) SolverImpesprintsatur
	set names(Concentration) SolverImpesprintconcen
	set names(Well) SolverImpesprintwells

	foreach i {"Permeability" "Pressure" "Saturation" "Concentration" "Well"} {
	    set widget [string tolower $i]
	    regsub -all " " $widget "" widget
	    frame $l.$widget -bd 2
	    entry $l.$widget.entry -relief sunken -width 20 -textvariable Vars($names($i))
	    label $l.$widget.label -text $i
	    pack $l.$widget.entry -side right
	    pack $l.$widget.label -side left
	    pack $l.$widget
	}
    }    
}

proc Run {} {
    global CurrentProject
    global NumProcs 
    global Vars

    set w .run

    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Run ParFlow"
	wm iconname $w "Run ParFlow"
	
	label $w.machineslabel -text "Virtual Machine Setup:" -anchor w
	pack $w.machineslabel -side top -fill x -expand yes
	
	set l $w.cm
	set r $w.am
	set nodes $w.nodes 
	frame $l
	frame $r
	frame $nodes
	frame $w.buttonbar -relief groove 
	pack $w.buttonbar -side bottom
	pack $nodes -side bottom
	pack $l $r -side left -expand yes 
	
	
	#
	# Button bar for commands
	#
	button $w.buttonbar.run -text "Run" -command \
		"wm withdraw $w; RunParFlow $l.scrolllist.mlist"
	button $w.buttonbar.batch -text "Batch" -command "wm withdraw $w; BatchParFlow $l.scrolllist.mlist"
	button $w.buttonbar.input -text "Create Input" -command "wm withdraw $w; CreateInputParFlow"
	button $w.buttonbar.cancel -text "Cancel" -command "wm withdraw $w"
	pack $w.buttonbar.run $w.buttonbar.batch $w.buttonbar.input \
		$w.buttonbar.cancel \
		-side left
	
	label $l.label -text "Current Machines"
	pack $l.label
	frame $l.scrolllist
	pack $l.scrolllist 
	button $l.button -text "Delete from Current" -command \
		"DeleteFromList $l.scrolllist.mlist"
	pack $l.button -side bottom -fill y
	scrollbar $l.scrolllist.mlistscroll -command "$l.scrolllist.mlist yview"
	listbox $l.scrolllist.mlist -yscrollcommand "$l.scrolllist.mlistscroll set"
	pack $l.scrolllist.mlistscroll -side right -fill y
	pack $l.scrolllist.mlist -expand yes -fill y
	
	label $r.label -text "Available Machines"
	pack $r.label
	frame $r.scrolllist
	pack $r.scrolllist 
	button $r.button -text "Add to Current" -command \
		"AddFromList $r.scrolllist.mlist $l.scrolllist.mlist"
	pack $r.button -side bottom -fill y
	scrollbar $r.scrolllist.mlistscroll -command "$r.scrolllist.mlist yview"
	listbox $r.scrolllist.mlist -yscrollcommand "$r.scrolllist.mlistscroll set"
	pack $r.scrolllist.mlistscroll -side right -fill y
	pack $r.scrolllist.mlist -expand yes -fill y
	
	scale $nodes.cvmnodes -orient horizontal -from 1 -to 64 \
		-command "CPUVMChanged" -length 284 \
		-label  "Nodes in VM"
	pack $nodes.cvmnodes
	
	scale $nodes.cmnodes -orient horizontal -from 1 -to $Vars(NumProcsVM) \
		-command "NodesChanged" -length 284 \
		-label  "Total Nodes"
	pack $nodes.cmnodes
	
	scale $nodes.cvmnodesp -orient horizontal -from 1 -to 64 \
	    -command "CPUDimChanged Vars(NumProcsP)" -length 284 \
	    -label  "P"
	pack $nodes.cvmnodesp 
	
	scale $nodes.cvmnodesq -orient horizontal -from 1 -to 64 \
		-command "CPUDimChanged Vars(NumProcsQ)" -length 284 \
		-label "Q"
	pack $nodes.cvmnodesq 
	
	scale $nodes.cvmnodesr -orient horizontal -from 1 -to 64 \
		-command "CPUDimChanged Vars(NumProcsR)" -length 284 \
		-label  "R"
	pack $nodes.cvmnodesr 
	
	ReadVMData $r.scrolllist.mlist $l.scrolllist.mlist
    }
}

#
# Execute the xpftools app
#
proc PfTools { } {
    global IsArchUnix

    if $IsArchUnix {
	exec xpftools &
    } {
	exec xpftools.bat &
    }



}

#
# Execute the GMS app
#
proc GMS {} {
    exec gms &
}

#
# Execute the AVS app
#
proc AVS {} {
    exec pfavs &
}

#
# Run the command to build ParFlow 
#
proc BuildParFlow {} {
    global Vars
    global env
    global PARFLOW_DIR

    #
    # Arrays with build options
    #
    set CompileOptions(Optimize) -O
    set CompileOptions(Debug) -g
    set CompileOptions(Profile) -profile

    set TimingOptions(1) -time
    set TimingOptions(0) ""

    if { [string compare "default" $Vars(CompileConfigFile)] } {
	RunCmd "$PARFLOW_DIR/bin/build -config $Vars(CompileConfigFile) $CompileOptions($Vars(CompileOption)) $TimingOptions($Vars(TimingOption)) install"
    } {
	RunCmd "$PARFLOW_DIR/bin/build $CompileOptions($Vars(CompileOption)) $TimingOptions($Vars(TimingOption)) install"

    }
}

#
# The build subpanel.  Get build options from user and then
# build.
#
proc Build {} {
    global Vars
    global CurrentProject

    set w .build
    
    if { [winfo exists $w] } {
	wm deiconify $w
    } {
	toplevel $w
	wm title $w "Build ParFlow"
	wm iconname $w "Build ParFlow"

	#
	# Invoke xpfconfig if user wants to edit callgraph
	#
	button $w.configure -text "Advanced Configuration" \
		-command {exec xpfconfig 0 $Vars(CurrentConfig)}
	pack $w.configure -side top -expand y -fill x

	#
	# Code generation options
	#
	label $w.compileoptslabel -text "Compile Options:" -anchor w
	pack $w.compileoptslabel -side top -fill x -expand yes
	
	frame $w.compileopts
	pack $w.compileopts -side top -expand yes 
	
	foreach option {Optimize Debug Profile} {
	    set lower [string tolower $option]
	    radiobutton $w.compileopts.$lower -text $option -variable Vars(CompileOption) \
		    -relief flat -value $option
	    pack $w.compileopts.$lower -side top -pady 2 -anchor w
	}
	

	#
	# Does user want to do timing
	#
	label $w.timingoptslabel -justify left -text "Timing Options:" -anchor w
	pack $w.timingoptslabel -side top -fill x
	
	frame $w.timingopts
	pack $w.timingopts -side top -expand yes

	foreach option {Timing} {
	    set lower [string tolower $option]
	    checkbutton $w.timingopts.$lower -text $option -variable Vars(TimingOption) \
		    -relief flat 
	    pack $w.timingopts.$lower -side top -pady 2 -anchor w
	}

	#
	# Config file
	#
	# Allow user to specify a different configuration file if they
	# want to override the defaults
	frame $w.configframe
	
	button $w.configframe.label -text "Compile Config File" -command { set Vars(CompileConfigFile) [FSBox "Compiler Config File" {{"Config Files" {.config}} {"All files" *}} "loading" ] }

	entry $w.configframe.entry -relief sunken -width 20 -textvariable Vars(CompileConfigFile)

	pack $w.configframe
	pack $w.configframe.entry -side right
	pack $w.configframe.label -side left

	#
	# Command buttons
	#
	frame $w.buttons
	pack  $w.buttons -side bottom -expand y -fill x -pady 2m
	button $w.buttons.ok -text OK -command "eval { wm withdraw $w; BuildParFlow}"
	button $w.buttons.cancel -text Cancel -command " wm withdraw $w"
	pack $w.buttons.ok $w.buttons.cancel -side left -expand 1
    }
}

proc Exit {} {
    exit
}

#
# Need to get better precision than the default 6 
#
set tcl_precision 17

set Vars(CompileOption) "Debug"
set Vars(TimingOption) 1

set Vars(NumProcsVM) 1
set Vars(NumProcs) 1

set Unknowns(dummy) ""

set Vars(NumProcsP) 1
set Vars(NumProcsQ) 1
set Vars(NumProcsR) 1

set GlobalMCFlag 1

set Vars(CurrentConfig) "./default_single.xpfc"
set Vars(CurrentProblemTemplate) "default_single.pfin"

set CurrentProject "default_single"
if [file isfile $CurrentProject.xpf] {
    # Load the default project
    LoadProject ./$CurrentProject.xpf
} {
    set CurrentProject ""
}

#
# Global Data Setup Not initialized by Project
#

foreach i { projecttitle filebuttonbar setupbuttonbar parflowbuttonbar pffopsbuttonbar appbuttonbar } {
    frame .$i -relief groove -borderwidth 4 -relief ridge
}

label .projectheader -text "Current Project:"
label .projectlabel -text "$CurrentProject"

pack .projectheader .projectlabel -side left -in .projecttitle

button .pfloadbutton -text "Load" -command PfLoad
button .pfunloadbutton -text "Unload" -command PfUnload
button .pfunloadallbutton -text "UnloadAll" -command PfUnloadAll
button .pftoolsbutton -text "PFTools" -command PfTools
button .buildbutton -text "Build" -command Build
button .gmsbutton -text "GMS" -command GMS
button .setupbutton -text "Problem Setup" -command ProblemSetup
button .solversetupbutton -text "Solver Setup" -command SolverSetup
button .runbutton -text "Run" -command Run
button .avsbutton -text "AVS" -command AVS
button .outputbutton -text "View Output" -command {
    #
    # lower then raise to get on top....is there a command that will raise?
    #
    wm iconify .buildmsg
    wm deiconify .buildmsg
}

button .newbutton -text "New" -command "New"
button .loadbutton -text "Open" -command "Load"
button .savebutton -text "Save" -command {Save $CurrentProject}
button .saveasbutton -text "SaveAs" -command "SaveAs"
button .exitbutton -text "Exit" -command Exit

pack .newbutton .loadbutton .savebutton .saveasbutton \
	.exitbutton \
	-fill x -expand y -side left -in .filebuttonbar

pack .gmsbutton .setupbutton .solversetupbutton \
	-fill x -expand y -side left -in .setupbuttonbar

pack .pfloadbutton .pfunloadbutton .pfunloadallbutton \
	-fill x -expand y -side left -in .pffopsbuttonbar

pack .buildbutton .runbutton .outputbutton \
	-fill x -expand y -side left -in .parflowbuttonbar

pack  .pftoolsbutton .avsbutton \
	-fill x -expand y -side left -in .appbuttonbar

foreach i { projecttitle filebuttonbar setupbuttonbar parflowbuttonbar pffopsbuttonbar appbuttonbar } {
    pack .$i -fill x -expand y -side top -padx 2m -pady 2m
}

wm title . "ParFlow Control"
wm iconname . "ParFlow Control"

set Output .buildmsg
toplevel $Output
wm title $Output "ParFlow Output"
wm iconname $Output "ParFlow Output"

label $Output.label -text "Output from processes run by XParFlow"
pack $Output.label

frame $Output.tframe
pack $Output.tframe

set OutputText $Output.tframe.text
text $OutputText -height 20 -width 80 -yscrollcommand "$Output.tframe.yscroll set"

scrollbar $Output.tframe.yscroll -command "$Output.tframe.text yview"
pack $Output.tframe.yscroll -side right -fill y
pack $Output.tframe.text

wm withdraw $Output

#
# Button bar for commands
#
frame $Output.buttonbar -relief groove 
pack $Output.buttonbar
button $Output.buttonbar.clear -text "Clear" -command "$OutputText delete 1.0 end"
pack $Output.buttonbar.clear -side left
button $Output.buttonbar.continue -text "Hide" -command "wm withdraw $Output"
pack $Output.buttonbar.continue -side left
