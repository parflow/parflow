
proc getFieldLegendCategorical_ui {root args} {
    global tcl_dir
    source chunk_global.tcl


    # this treats "." as a special case

    if {$root == "."} {
	set base ""
    } else {
	set base $root
    }
    #
    #  Big Frames
    #
    frame $base.frameTop -relief groove -borderwidth 3
    frame $base.frameMiddle -relief groove -borderwidth 3
    frame $base.frameOKPS
    grid $base.frameTop -row 0 -column 0 -padx 5 -pady 5 -ipadx 5 -ipady 5
    grid $base.frameMiddle -row 1 -column 0 -padx 5 -pady 5 -ipadx 5
    grid $base.frameOKPS -row 2 -column 0 -sticky ew
    #
    #  Labels
    #
    label $base.categoryN -text "Number of Categories to Display:" \
	-font {times 12 bold}
    label $base.category -text "Category #" -font {times 12 bold}
    label $base.label -text "Label" -font {times 12 bold}
    set j 0
    foreach x {#Rows: #Columns: "Height (in):" "Width (in):" \
		   "Translate (pts):" } {
	label $base.label$x -text $x -font {times 12 bold} 
	grid $base.label$x -in $base.frameMiddle -row $j -column 0 -sticky w
	set j [expr $j + 1]
    } 
    label $base.labelY -text Y -font {times 12 bold} 
    grid $base.labelY -in $base.frameMiddle -row $j -column 1
    set j [expr $j - 1]
    label $base.labelX -text X -font {times 12 bold} 
    grid $base.labelX -in $base.frameMiddle -row $j -column 1 
    grid $base.categoryN -in $base.frameTop -row 0 -column 0 -columnspan 2
    grid $base.category -in $base.frameTop -row 1 -column 0
    grid $base.label -in $base.frameTop -row 1 -column 1
    #
    #  Entries
    # 
    entry $base.entryNCategories -textvariable field_legend_n_categories \
	-width 2 -background #bdd -highlightcolor #f00
    grid $base.entryNCategories -in $base.frameTop -row 0 -column 2
    for {set i 1} {$i<=$field_n_cutoffs} {incr i 1} {
	entry $base.entryCategory$i -textvariable field_legend_category($i) \
	    -width 2 -background #bdd -highlightcolor #f00
	entry $base.entryFieldLegendCategoryName($i) \
	    -textvariable field_legend_category_name($i) \
	    -background #bdd -highlightcolor #f00
	set j [expr $i + 1]
	grid $base.entryCategory$i -in $base.frameTop -row $j -column 0
	grid $base.entryFieldLegendCategoryName($i) \
	    -in $base.frameTop -row $j -column 1
    }
    #
    #       frames
    #
    frame $base.frameRows
    frame $base.frameColumns
    #
    # Scales
    #     
    scale $base.scaleHeight -from 0 -to 5.0 -length 200 \
	-variable field_legend_height -orient horizontal \
	-tickinterval 1.0 -showvalue true -resolution 0.1 -digits 2 
    scale $base.scaleWidth -from 0 -to 8.0 -length 320 \
	-variable field_legend_width -orient horizontal \
	-tickinterval 1.0 -showvalue true -resolution 0.1 -digits 2 
    scale $base.scaleTransX -from 0 -to 600 -length 300 \
	-variable field_legend_translation_x -orient horizontal \
	-tickinterval 100 -showvalue true -resolution 1  
    scale $base.scaleTransY -from 0 -to 600 -length 300 \
	-variable field_legend_translation_y -orient horizontal \
	-tickinterval 100 -showvalue true -resolution 1  
    grid $base.frameRows -in $base.frameMiddle -row 0 -column 2 -sticky w
    grid $base.frameColumns -in $base.frameMiddle -row 1 -column 2 -sticky w
    grid $base.scaleHeight -in $base.frameMiddle -row 2 -column 2 -sticky w
    grid $base.scaleWidth -in $base.frameMiddle -row 3 -column 2 -sticky w
    grid $base.scaleTransX -in $base.frameMiddle -row 4 -column 2 -sticky w
    grid $base.scaleTransY -in $base.frameMiddle -row 5 -column 2 -sticky w
    #
    # Radiobuttons
    #
    foreach j {1 2 3 4 5 6 7 8 9 10}  {
        radiobutton $base.radioRow$j -text $j -value $j \
            -variable field_legend_n_rows 
        radiobutton $base.radioColumn$j -text $j -value $j \
            -variable field_legend_n_columns 
        grid $base.radioRow$j -in $base.frameRows -row 0 -column $j
        grid $base.radioColumn$j -in $base.frameColumns -row 0 -column $j
    }
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


# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getFieldLegendCategorical_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getFieldLegendCategorical_ui"
	    getFieldLegendCategorical_ui .
	}
    }
}
