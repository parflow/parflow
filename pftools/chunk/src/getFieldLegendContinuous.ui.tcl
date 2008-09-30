
proc getFieldLegendContinuous_ui {root args} {
        global chunk_run_file

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
        frame $base.frameOKPS
        grid $base.frameTop -row 0 -column 0 -padx 5 -pady 5 -ipadx 5 -ipady 5
        grid $base.frameOKPS -row 1 -column 0 -sticky ew
#
#  Labels
#       
        label $base.range -text "Range:" -font {times 12 bold}      
        label $base.size -text "Size (inches):" -font {times 12 bold}      
        label $base.labelSp -text "Label Spacing:" -font {times 12 bold}      
        label $base.dec -text ".Decimal Places:" -font {times 12 bold}      
        label $base.ticsPer -text "Tics Per Label:" -font {times 12 bold}      
        label $base.log10 -text "Log10 the scale?" -font {times 12 bold}      
        label $base.title -text "Title:" -font {times 12 bold}      
        label $base.translate -text "Translate (pts):" -font {times 12 bold}   
        label $base.transX -text "X" -font {times 12 bold}      
        label $base.transY -text "Y" -font {times 12 bold}      
#
#  Frames
#
        frame $base.ranges
        frame $base.sizes
        frame $base.labelSps
        frame $base.decs
        frame $base.ticsPers
        frame $base.log10s
#
#  Entries
#
     entry $base.entryMin -textvariable field_legend_min -width 20 \
           -background #bdd -highlightcolor #f00
     entry $base.entryMax -textvariable field_legend_max -width 20 \
           -background #bdd -highlightcolor #f00
     entry $base.entrySp -textvariable field_legend_label_spacing -width 20 \
           -background #bdd -highlightcolor #f00
     entry $base.entryDec -textvariable field_legend_label_decimals -width 1 \
           -background #bdd -highlightcolor #f00
     entry $base.entryTic -textvariable field_legend_label_tics_per -width 2 \
           -background #bdd -highlightcolor #f00
     entry $base.entryTitle -textvariable field_legend_title -width 60 \
           -background #bdd -highlightcolor #f00
#
# Radiobuttons
#
        radiobutton $base.dec0 -text 0 -value 0 -variable \
                field_legend_label_decimals
        radiobutton $base.dec1 -text 1 -value 1 -variable \
                field_legend_label_decimals
        radiobutton $base.dec2 -text 2 -value 2 -variable \
                field_legend_label_decimals
        radiobutton $base.tic1 -text 1 -value 1 -variable \
                field_legend_label_tics_per
        radiobutton $base.tic2 -text 2 -value 2 -variable \
                field_legend_label_tics_per
        radiobutton $base.tic5 -text 5 -value 5 -variable \
                field_legend_label_tics_per
        radiobutton $base.tic10 -text 10 -value 10 -variable \
                field_legend_label_tics_per
        radiobutton $base.logyes -text yes -value yes -variable field_legend_log 
        radiobutton $base.logno -text no -value no -variable field_legend_log 
#
# Scales
# 
        scale $base.scaleTransX -from 0 -to 600 -length 300 \
            -variable field_legend_translation_x -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scaleTransY -from 0 -to 800 -length 400 \
            -variable field_legend_translation_y -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scaleWidth -from -0 -to 8.0 -length 200 \
            -variable field_legend_width -orient horizontal \
            -tickinterval 2.0 -showvalue true -resolution 0.1 -digit 2
        scale $base.scaleHeight -from -0 -to 1.0 -length 125 \
            -variable field_legend_height -orient horizontal \
            -tickinterval 0.50 -showvalue true -resolution 0.05 -digit 3
#
#  Grid main
#
        grid $base.range -in $base.frameTop -row 0 -column 0 \
               -sticky w -ipadx 10
        grid $base.ranges -in $base.frameTop -row 0 -column 2 -sticky w 
        grid $base.size -in $base.frameTop -row 1 -column 0 \
               -sticky w -ipadx 10
        grid $base.sizes -in $base.frameTop -row 1 -column 2 -sticky w
        grid $base.labelSp -in $base.frameTop -row 2 -column 0 \
               -sticky w -ipadx 10
        grid $base.entrySp -in $base.frameTop -row 2 -column 2 -sticky w 
        grid $base.dec -in $base.frameTop -row 3 -column 0  \
               -sticky w -ipadx 10
        grid $base.decs -in $base.frameTop -row 3 -column 2 -sticky w 
        grid $base.ticsPer -in $base.frameTop -row 4 -column 0 \
               -sticky w -ipadx 10
        grid $base.ticsPers -in $base.frameTop -row 4 -column 2 -sticky w 
        grid $base.log10 -in $base.frameTop -row 5 -column 0 \
               -sticky w -ipadx 10
        grid $base.log10s -in $base.frameTop -row 5 -column 2 -sticky w 
        grid $base.title -in $base.frameTop -row 6 -column 0 \
               -sticky w -ipadx 10
        grid $base.entryTitle -in $base.frameTop -row 6 -column 2 -sticky w 
        grid $base.translate -in $base.frameTop -row 7 -column 0 \
               -sticky w -ipadx 10

        label $base.min -text "min" -font {times 12 bold}        
        label $base.max -text " max" -font {times 12 bold}       
        grid $base.entryMin -in $base.ranges -row 0 -column 1
        grid $base.entryMax -in $base.ranges -row 0 -column 3
        grid $base.min -in $base.ranges -row 0 -column 0 
        grid $base.max -in $base.ranges -row 0 -column 2

        label $base.width -text "width" -font {times 12 bold}      
        label $base.height -text " height" -font {times 12 bold}      
        grid $base.width -in $base.sizes -row 0 -column 0
        grid $base.height -in $base.sizes -row 0 -column 2
        grid $base.scaleWidth -in $base.sizes -row 0 -column 1
        grid $base.scaleHeight -in $base.sizes -row 0 -column 3

        grid $base.entryDec -in $base.decs -row 0 -column 0 -sticky w
        grid $base.dec0 -in $base.decs -row 0 -column 1
        grid $base.dec1 -in $base.decs -row 0 -column 2
        grid $base.dec2 -in $base.decs -row 0 -column 3

        grid $base.entryTic -in $base.ticsPers -row 0 -column 0
        grid $base.tic1 -in $base.ticsPers -row 0 -column 1
        grid $base.tic2 -in $base.ticsPers -row 0 -column 2
        grid $base.tic5 -in $base.ticsPers -row 0 -column 3
        grid $base.tic10 -in $base.ticsPers -row 0 -column 4

        grid $base.logyes -in $base.log10s -row 0 -column 0
        grid $base.logno -in $base.log10s -row 0 -column 1

        grid $base.scaleTransX -in $base.frameTop -row 7 -column 2 -sticky w
        grid $base.transX -in $base.frameTop -row 7 -column 1
        grid $base.scaleTransY -in $base.frameTop -row 8 -column 2 -sticky w
        grid $base.transY -in $base.frameTop -row 8 -column 1 
#
# OKPS buttons
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
	getFieldLegendContinuous_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Legend for Continuous Variable"
	    getFieldLegendContinuous_ui .
	}
    }
}
