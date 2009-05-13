proc getCoord_ui {root args} {
    global tcl_dir
        source chunk_global.tcl
        global base

	# this treats "." as a special case

	if {$root == "."} {
	    set base ""
	} else {
	    set base $root
	}
#
#  Major Frames
#
        frame $base.frameGrid -relief groove -borderwidth 3   
        frame $base.frameAxes -relief groove -borderwidth 3   
        frame $base.frameOKPS
	grid $base.frameGrid -row 0 -column 0 \
                   -padx 5 -pady 5 -ipadx 5 -ipady 5 
	grid $base.frameAxes -row 1 -column 0 \
                   -padx 5 -pady 5 -ipadx 5 -ipady 5 
	grid $base.frameOKPS -row 2 -column 0 -sticky ew  
#
# Grid Frame
#
	label $base.gridLabel -text "  GRID  " -font {times 12 bold} 
        label $base.gridOrigin -text Origin -font {times 12 bold} 
        label $base.gridCellSize -text "Cell Size" -font {times 12 bold}
        label $base.gridScale -text "Scale (Units/Inch)" -font {times 12 bold}
        label $base.gridX -text X -font {times 12 bold}
        label $base.gridY -text Y -font {times 12 bold}
        label $base.gridZ -text Z -font {times 12 bold}


	entry $base.gridMinX -textvariable block_min_x -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridMinY -textvariable block_min_y -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridMinZ -textvariable block_min_z -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridDX -textvariable block_d_x -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridDY -textvariable block_d_y -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridDZ -textvariable block_d_z -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridScaleX -textvariable block_scale_x -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridScaleY -textvariable block_scale_y -width 20 \
           -background #bdd -highlightcolor #f00
	entry $base.gridScaleZ -textvariable block_scale_z -width 20 \
           -background #bdd -highlightcolor #f00

	grid $base.gridLabel -in $base.frameGrid -row 1 -column 0 -sticky nw   
	grid $base.gridOrigin -in $base.frameGrid -row 1 -column 1 -sticky w  
	grid $base.gridCellSize -in $base.frameGrid -row 1 -column 2 -sticky w 
	grid $base.gridScale -in $base.frameGrid -row 1 -column 3 -sticky w 
	grid $base.gridX -in $base.frameGrid -row 2 -column 0  
	grid $base.gridY -in $base.frameGrid -row 3 -column 0  
	grid $base.gridZ -in $base.frameGrid -row 4 -column 0  
        grid $base.gridMinX -in $base.frameGrid -row 2 -column 1 
        grid $base.gridMinY -in $base.frameGrid -row 3 -column 1 
        grid $base.gridMinZ -in $base.frameGrid -row 4 -column 1 
        grid $base.gridDX -in $base.frameGrid -row 2 -column 2 
        grid $base.gridDY -in $base.frameGrid -row 3 -column 2 
        grid $base.gridDZ -in $base.frameGrid -row 4 -column 2 
        grid $base.gridScaleX -in $base.frameGrid -row 2 -column 3 
        grid $base.gridScaleY -in $base.frameGrid -row 3 -column 3 
        grid $base.gridScaleZ -in $base.frameGrid -row 4 -column 3 
#
#   Axes Frame
#
	label $base.axesName -text " AXES " -font {times 12 bold} 
        label $base.axesX -text X -font {times 12 bold}
        label $base.axesY -text Y -font {times 12 bold}
        label $base.axesZ -text Z -font {times 12 bold}
	label $base.axesLabel -text "Coordinate Name" -font {times 12 bold} 
        label $base.axesSpacing -text "Label Spacing" -font {times 12 bold} 
        label $base.axesTics -text "Tics Per Label" -font {times 12 bold}
        label $base.axesDec -text ".Decimal Places" -font {times 12 bold}

        entry $base.axesLabelX -textvariable label_x -width 16 \
           -background #bdd -highlightcolor #f00
        entry $base.axesLabelY -textvariable label_y -width 16 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesLabelZ -textvariable label_z -width 16 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesSpacingX -textvariable label_space_x -width 15 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesSpacingY -textvariable label_space_y -width 15 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesSpacingZ -textvariable label_space_z -width 15 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesTicsX -textvariable label_tic_per_x -width 2 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesTicsY -textvariable label_tic_per_y -width 2 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesTicsZ -textvariable label_tic_per_z -width 2 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesDecX -textvariable label_decimals_x -width 1 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesDecY -textvariable label_decimals_y -width 1 \
           -background #bdd -highlightcolor #f00 
        entry $base.axesDecZ -textvariable label_decimals_z -width 1 \
           -background #bdd -highlightcolor #f00 
        
        radiobutton $base.axesDecX0 -text 0 -value 0 -variable label_decimals_x
        radiobutton $base.axesDecX1 -text 1 -value 1 -variable label_decimals_x
        radiobutton $base.axesDecX2 -text 2 -value 2 -variable label_decimals_x
        radiobutton $base.axesDecX3 -text 3 -value 3 -variable label_decimals_x
        radiobutton $base.axesDecY0 -text 0 -value 0 -variable label_decimals_y
        radiobutton $base.axesDecY1 -text 1 -value 1 -variable label_decimals_y
        radiobutton $base.axesDecY2 -text 2 -value 2 -variable label_decimals_y
        radiobutton $base.axesDecY3 -text 3 -value 3 -variable label_decimals_y
        radiobutton $base.axesDecZ0 -text 0 -value 0 -variable label_decimals_z
        radiobutton $base.axesDecZ1 -text 1 -value 1 -variable label_decimals_z
        radiobutton $base.axesDecZ2 -text 2 -value 2 -variable label_decimals_z
        radiobutton $base.axesDecZ3 -text 3 -value 3 -variable label_decimals_z

     radiobutton $base.axesTicsX1 -text 1 -value 1 -variable label_tic_per_x
     radiobutton $base.axesTicsX2 -text 2 -value 2 -variable label_tic_per_x
     radiobutton $base.axesTicsX5 -text 5 -value 5 -variable label_tic_per_x
     radiobutton $base.axesTicsX10 -text 10 -value 10 -variable label_tic_per_x
     radiobutton $base.axesTicsY1 -text 1 -value 1 -variable label_tic_per_y
     radiobutton $base.axesTicsY2 -text 2 -value 2 -variable label_tic_per_y
     radiobutton $base.axesTicsY5 -text 5 -value 5 -variable label_tic_per_y
     radiobutton $base.axesTicsY10 -text 10 -value 10 -variable label_tic_per_y
     radiobutton $base.axesTicsZ1 -text 1 -value 1 -variable label_tic_per_z
     radiobutton $base.axesTicsZ2 -text 2 -value 2 -variable label_tic_per_z
     radiobutton $base.axesTicsZ5 -text 5 -value 5 -variable label_tic_per_z
     radiobutton $base.axesTicsZ10 -text 10 -value 10 -variable label_tic_per_z

	grid $base.axesName -in $base.frameAxes -row 1 -column 0 -sticky nw  
	grid $base.axesLabel -in $base.frameAxes -row 1 -column 1 -sticky w  
	grid $base.axesSpacing -in $base.frameAxes -row 1 -column 2 -sticky w 
	grid $base.axesTics -in $base.frameAxes -row 1 -column 3 \
             -columnspan 5 -sticky w 
	grid $base.axesDec -in $base.frameAxes -row 1 -column 8 \
             -columnspan 4 -sticky w  
        grid $base.axesX -in $base.frameAxes -row 2 -column 0
        grid $base.axesY -in $base.frameAxes -row 3 -column 0
        grid $base.axesZ -in $base.frameAxes -row 4 -column 0
        grid $base.axesLabelX -in $base.frameAxes -row 2 -column 1
        grid $base.axesLabelY -in $base.frameAxes -row 3 -column 1
        grid $base.axesLabelZ -in $base.frameAxes -row 4 -column 1
        grid $base.axesSpacingX -in $base.frameAxes -row 2 -column 2
        grid $base.axesSpacingY -in $base.frameAxes -row 3 -column 2
        grid $base.axesSpacingZ -in $base.frameAxes -row 4 -column 2
        grid $base.axesTicsX -in $base.frameAxes -row 2 -column 3 
        grid $base.axesTicsY -in $base.frameAxes -row 3 -column 3 
        grid $base.axesTicsZ -in $base.frameAxes -row 4 -column 3 

        grid $base.axesTicsX1 -in $base.frameAxes -row 2 -column 4 
        grid $base.axesTicsX2 -in $base.frameAxes -row 2 -column 5 
        grid $base.axesTicsX5 -in $base.frameAxes -row 2 -column 6 
        grid $base.axesTicsX10 -in $base.frameAxes -row 2 -column 7 
        grid $base.axesTicsY1 -in $base.frameAxes -row 3 -column 4 
        grid $base.axesTicsY2 -in $base.frameAxes -row 3 -column 5 
        grid $base.axesTicsY5 -in $base.frameAxes -row 3 -column 6 
        grid $base.axesTicsY10 -in $base.frameAxes -row 3 -column 7 
        grid $base.axesTicsZ1 -in $base.frameAxes -row 4 -column 4 
        grid $base.axesTicsZ2 -in $base.frameAxes -row 4 -column 5 
        grid $base.axesTicsZ5 -in $base.frameAxes -row 4 -column 6 
        grid $base.axesTicsZ10 -in $base.frameAxes -row 4 -column 7 

        grid $base.axesDecX -in $base.frameAxes -row 2 -column 8 
        grid $base.axesDecY -in $base.frameAxes -row 3 -column 8 
        grid $base.axesDecZ -in $base.frameAxes -row 4 -column 8 
        grid $base.axesDecX0 -in $base.frameAxes -row 2 -column 9 
        grid $base.axesDecX1 -in $base.frameAxes -row 2 -column 10 
        grid $base.axesDecX2 -in $base.frameAxes -row 2 -column 11 
        grid $base.axesDecY0 -in $base.frameAxes -row 3 -column 9
        grid $base.axesDecY1 -in $base.frameAxes -row 3 -column 10 
        grid $base.axesDecY2 -in $base.frameAxes -row 3 -column 11 
        grid $base.axesDecZ0 -in $base.frameAxes -row 4 -column 9 
        grid $base.axesDecZ1 -in $base.frameAxes -row 4 -column 10 
        grid $base.axesDecZ2 -in $base.frameAxes -row 4 -column 11 
#
#   OKPS buttons
#
        button $base.buttonOK \
                -text "OK" \
                -font {times 12 bold} \
                -background "#9f9" \
                -borderwidth 4 \
                -command "destroy $base"
        button $base.buttonPS \
                -background #ff8040 \
                -borderwidth 4 \
                -command "source $chunk_run_file" \
                -text {Generate PS/EPS} \
                -font {times 12 bold} 
        pack $base.buttonOK -in $base.frameOKPS \
           -side left -expand true -fill x
        pack $base.buttonPS  -in $base.frameOKPS \
           -side right



# additional interface code
# end additional interface code

}

# Allow interface to be run "stand-alone" for testing
catch {
    if [info exists embed_args] {
        # we are running in the plugin
        getCoord_ui .
    } else {
        # we are running in stand-alone mode
        if {$argv0 == [info script]} {
            wm title . "Coord"
            getCoord_ui .
        }
    }
}
