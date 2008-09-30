proc getFieldCutoffs_ui {root args} {

        global field_n_cutoffs
        global field_cutoff
        global field_cutoff_color
        global chunk_run_file

	# this treats "." as a special case

	if {$root == "."} {
	    set base ""
	} else {
	    set base $root
	}

        frame $base.frameTop
        frame $base.frameOKPS
        grid $base.frameTop -row 0 -column 0 
        grid $base.frameOKPS -row 1 -column 0  -sticky ew
 
        label $base.labelCutoff -text "#" -font {times 12 bold} 
        label $base.labelValue -text "Cutoff" -font {times 12 bold}
        label $base.labelColor -text "Color" -font {times 12 bold}
        label $base.labelCode -text "Code" -font {times 12 bold}
        grid $base.labelCutoff -in $base.frameTop -row 0 -column 0
        grid $base.labelValue -in $base.frameTop -row 0 -column 1
        grid $base.labelColor -in $base.frameTop -row 0 -column 2
        grid $base.labelCode -in $base.frameTop -row 0 -column 3
        for {set i 1 } {$i <= $field_n_cutoffs} { incr i 1} {
          label $base.labelCutoff$i -text $i
          entry $base.entryValue$i -width 12 -textvariable field_cutoff($i) \
             -background #bdd -highlightcolor #f00
          entry $base.entryColorCode$i -width 7 \
             -textvariable field_cutoff_color($i)
          set color "#ffffff"
          catch { set color $field_cutoff_color($i) } 
          button $base.buttonColor($i) -background $color -padx 10 \
               -command "buttoncolor $base.buttonColor($i) $i $base"
          grid $base.labelCutoff$i -in $base.frameTop  -row $i -column 0
          grid $base.entryValue$i -in $base.frameTop -row $i -column 1
          grid $base.buttonColor($i) -in $base.frameTop -row $i -column 2
          grid $base.entryColorCode$i -in $base.frameTop -row $i -column 3

          set rgb [winfo rgb $root $field_cutoff_color($i) ]
          set denom_r [lindex [winfo rgb $root red] 0]
          set denom_g [lindex [winfo rgb $root green] 1]
          set denom_b [lindex [winfo rgb $root blue] 2]
          set r_color [expr 100 * [lindex $rgb 0] / $denom_r]
          set g_color [expr 100 * [lindex $rgb 1] / $denom_g]
          set b_color [expr 100 * [lindex $rgb 2] / $denom_b]
          set field_cutoff_r($i) [expr $r_color / 100.0 ]
          set field_cutoff_g($i) [expr $g_color / 100.0 ]
          set field_cutoff_b($i) [expr $b_color / 100.0 ]

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

          

proc buttoncolor {w k base} {
    global field_cutoff_color 
    set old_color [$base.buttonColor($k) cget -background]
    set color [tk_chooseColor -parent $w -initialcolor $old_color \
                        -title "Line $k Color"]
    if {$color != ""} {
                set field_cutoff_color($k) $color
                $w configure -background $color
    }
    return
}




# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getFieldCutoffs_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getFieldCutoffs_ui"
	    getFieldCutoffs_ui .
	}
    }
}
