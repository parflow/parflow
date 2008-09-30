proc getEPS_ui {root args} {
        global global_file
        source $global_file

	# this treats "." as a special case

	if {$root == "."} {
	    set base ""
	} else {
	    set base $root
	}

        label $base.labelEPS -text "Create EPS?" -font {times 12 bold} 
	radiobutton $base.radioEPSyes \
		-text yes \
		-value eps \
		-variable ps_file_format \
                -command "set ps_file [file rootname $ps_file]; \
                          append ps_file . eps"

	radiobutton $base.radioEPSno \
		-text no \
		-value ps \
		-variable ps_file_format \
                -command "set ps_file [file rootname $ps_file]; \
                          append ps_file . ps"
        grid $base.labelEPS -row 0 -column 0 
        grid $base.radioEPSyes -row 0 -column 1 -sticky w
        grid $base.radioEPSno -row 0 -column 2 -sticky w

        label $base.labelMinY -text "min Y" -font {times 12 bold}
        label $base.labelMaxY -text "max Y" -font {times 12 bold}
        scale $base.minY -from 0 \
            -to $eps_BoundingBox_max_y -length 396 \
            -variable eps_BoundingBox_min_y -orient horizontal \
            -tickinterval 100 -showvalue true -resolution 1 -digits 1 
        scale $base.maxY -from $eps_BoundingBox_min_y -to 792 -length 396 \
            -variable eps_BoundingBox_max_y -orient horizontal \
            -tickinterval 100 -showvalue true -resolution 1 -digits 1 
        grid $base.labelMinY -row 6 -column 0 -sticky w
        grid $base.minY -row 6 -column 1 -sticky w -columnspan 2
        grid $base.labelMaxY -row 7 -column 0 -sticky w
        grid $base.maxY -row 7 -column 1 -sticky w -columnspan 2
        label $base.labelMinX -text "min X" -font {times 12 bold}
        label $base.labelMaxX -text "max X" -font {times 12 bold}
        scale $base.minX -from 0 -to $eps_BoundingBox_max_x -length 306 \
            -variable eps_BoundingBox_min_x -orient horizontal \
            -tickinterval 100 -showvalue true -resolution 1 -digits 1 
        scale $base.maxX -from $eps_BoundingBox_min_x -to 612 -length 306 \
            -variable eps_BoundingBox_max_x -orient horizontal \
            -tickinterval 100 -showvalue true -resolution 1 -digits 1 
        grid $base.labelMinX -row 4 -column 0 -sticky w
        grid $base.minX -row 4 -column 1 -sticky w -columnspan 2
        grid $base.labelMaxX -row 5 -column 0 -sticky w
        grid $base.maxX -row 5 -column 1 -sticky w -columnspan 2
    
        label $base.creator -text "Creator: " -font {times 12 bold}
        label $base.date -text "Date: " -font {times 12 bold}
        label $base.time -text "Time: " -font {times 12 bold}
        entry $base.entryCreator -textvariable eps_creator -width 40 \
           -background #bdd -highlightcolor #f00
        entry $base.entryDate -textvariable eps_date -width 40 \
           -background #bdd -highlightcolor #f00
        entry $base.entryTime -textvariable eps_time -width 40 \
           -background #bdd -highlightcolor #f00
        grid $base.creator  -row 1 -column 0 -sticky w -columnspan 2
        grid $base.entryCreator -row 1 -column 1 -sticky w -columnspan 2
        grid $base.date -row 2 -column 0 -sticky w
        grid $base.entryDate -row 2 -column 1 -sticky w -columnspan 2
        grid $base.time -row 3 -column 0 -sticky w 
        grid $base.entryTime -row 3 -column 1 -sticky w -columnspan 2

        frame $base.frameOKPS
        grid $base.frameOKPS -row 8 -column 0 -columnspan 5 -sticky ew
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

}
