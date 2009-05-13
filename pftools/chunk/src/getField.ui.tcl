
proc getField_ui {root args} {
    global global_file tcl_dir
    source $global_file 
    # this treats "." as a special case

    if {$root == "."} {
	set base ""
    } else {
	set base $root
    }

    #
    #   Big Frames
    #
    frame $base.top
    frame $base.specs -relief groove -borderwidth 3
    frame $base.display -relief groove -borderwidth 3
    frame $base.frameOKPS

    grid $base.top -row 0 -column 0 -sticky w -padx 5 -pady 5
    grid $base.specs -row 1 -column 0 \
	-ipadx 5 -ipady 5 -padx 5 -pady 5 -sticky w 
    grid $base.display -row 2 -column 0 \
	-ipadx 5 -ipady 5 -padx 5 -pady 5 -sticky w 
    grid $base.frameOKPS -row 3 -column 0 -sticky ew
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

    #
    #  Top
    #
    label $base.plot -text "Plot a Field ?" -font {times 12 bold}
    radiobutton $base.fieldYes -text yes -value yes -variable field_plot
    radiobutton $base.fieldNo -text no -value no -variable field_plot
    
    grid $base.plot -in $base.top -row 0 -column 0
    grid $base.fieldYes -in $base.top -row 0 -column 1
    grid $base.fieldNo -in $base.top -row 0 -column 2
    #
    #  Specifications
    #
    frame $base.formats
    frame $base.dataTypes
    frame $base.frameCategorical -relief raised -borderwidth 1 \
	-background "beige"
    frame $base.frameContinuous -relief raised -borderwidth 1 \
	-background "beige"
    frame $base.log10s
    frame $base.legends

    label $base.labelSpecs -text "FIELD SPECIFICATIONS" \
	-font {times 12 bold}
    label $base.format -text "Format:" -font {times 12 bold}
    label $base.data -text "Data:" -font {times 12 bold} 
    label $base.labelMin -text "min" -background "beige" \
	-font {times 12 bold} 
    label $base.labelMax -text "max" -background "beige" \
	-font {times 12 bold} 
    label $base.log10 -text "Log10?" -font {times 12 bold}
    label $base.labelTimestep -text timestep 
    label $base.nCutoffs -text "Number of Cutoffs:" -background "beige" \
	-font {times 12 bold} 

    button $base.buttonFieldFile \
	-background "#e8e" -borderwidth 4 -text "File:" \
	-command "OpenFieldFile .get_Field"  -font {times 12 bold}
    source getFieldCutoffs.ui.tcl
    button $base.buttonFieldCutoffs \
	-background "#e8e" -borderwidth 4 -text "Cutoffs" \
	-font {times 12 bold} \
	-command "toplevel .getField_Cutoffs; \
                      corner $base .getField_Cutoffs; \
                      focus .getField_Cutoffs; \
                      getFieldCutoffs_ui .getField_Cutoffs; \
                      grab .getField_Cutoffs; \
                      wm title .getField_Cutoffs \
                      {Field Cutoffs} ; \
                      tkwait window .getField_Cutoffs" 


    proc OpenFieldFile {root args } {
	global field_file 
	set field_dir [file dirname $field_file]
	set field_file_temp [tk_getOpenFile \
				 -initialdir $field_dir \
				 -initialfile [file tail $field_file] \
				 -parent $root \
				 -title {Open Field File}]
	if {$field_file_temp != ""} {
	    set field_file $field_file_temp
	} 
    }     

    label $base.labelColorTable -textvariable field_color_table  \
	-background "beige"
    button $base.fieldColor \
	-background #e8e -borderwidth 4 \
	-command "OpenColorTableFile .get_Field " \
	-text "Color Table:" -font {times 12 bold}
    proc OpenColorTableFile {root args} {
	global field_color_table
	set typelist {
	    {"CTB Files" {".ctb"} {"CTB "}}
	}
	set field_color_table_temp \
	    [tk_getOpenFile -defaultextension .ctb \
		 -filetypes $typelist \
		 -initialdir [file dirname $field_color_table] \
		 -initialfile [file tail $field_color_table] \
		 -parent $root \
		 -title "Open Color Table File for Field"]
	if {$field_color_table_temp != ""} {
	    set field_color_table $field_color_table_temp
	}
    }


    entry $base.fieldFile -textvariable field_file -width 80 \
	-background #bdd -highlightcolor #f00
    entry $base.timestep -textvariable field_nuft_timestep -width 3 \
	-background #bdd -highlightcolor #f00
    entry $base.entryMin -textvariable field_value_min -width 10 \
	-background #bdd -highlightcolor #f00
    entry $base.entryMax -textvariable field_value_max -width 10 \
	-background #bdd -highlightcolor #f00

    scale $base.fieldNCut -from 0 -to 20 -length 150 \
	-variable field_n_cutoffs -orient horizontal \
	-tickinterval 5 -showvalue true -resolution 1 -digits 1 \
	-command "set field_legend_n_categories [lindex $field_n_cutoffs 2]" \
	-background "beige"

    foreach x {ascii hdf hdf_int1 bgr_int1 pfb cnb nuft} {
	radiobutton $base.$x -text $x -value $x \
	    -variable field_format 
    }

    radiobutton $base.logYes -text yes -value yes \
	-variable field_log10 
    radiobutton $base.logNo -text no -value no \
	-variable field_log10
    radiobutton $base.legendYes -text yes -value yes \
	-variable field_legend
    radiobutton $base.legendNo -text no -value no \
	-variable field_legend
    radiobutton $base.continuous -text continuous -font {times 12 bold} \
	-value continuous -variable field_data_type -background "beige" 
    radiobutton $base.categorical -text categorical -font {times 12 bold} \
	-value categorical -variable field_data_type -background "beige" 
    #
    #  Legend Button
    #
    source getFieldLegendContinuous.ui.tcl
    source getFieldLegendCategorical.ui.tcl
    button $base.fieldLegend \
	-background #e8e -borderwidth 4 \
	-text Legend?  -font {times 12 bold} \
	-command "get_legend" 

    proc get_legend {args} {
	global field_data_type
	switch -exact -- $field_data_type {\
					       categorical { \
								 toplevel .get_FieldLegendCategorical;\
								 corner .get_Field .get_FieldLegendCategorical ; \
								 focus .get_FieldLegendCategorical;\
								 wm title .get_FieldLegendCategorical \
								 {Field Legend for Categorical Variable} ; \
								 getFieldLegendCategorical_ui \
								 .get_FieldLegendCategorical; \
								 grab .get_FieldLegendCategorical; \
								 tkwait window .get_FieldLegendCategorical \
							     } \
					       continuous { \
								toplevel .get_FieldLegendContinuous;\
								corner .get_Field .get_FieldLegendContinuous ; \
								focus .get_FieldLegendContinuous;\
								wm title .get_FieldLegendContinuous \
								{Field Legend for Continuous Variable} ; \
								getFieldLegendContinuous_ui \
								.get_FieldLegendContinuous; \
								grab .get_FieldLegendContinuous; \
								tkwait window .get_FieldLegendContinuous \
							    }\
					   }
	return
    }
    #
    #  Grid
    #
    grid $base.formats -in $base.specs -row 2 -column 1 -sticky w
    grid $base.dataTypes -in $base.specs -row 4 -column 0 \
	-columnspan 2 -sticky ew
    grid $base.frameContinuous -in $base.dataTypes \
	-row 0 -column 0 -padx 5 -pady 5 -ipadx 5 -ipady 5
    grid $base.frameCategorical -in $base.dataTypes \
	-row 0 -column 1 -padx 5 -pady 5 -ipadx 5 -ipady 5
    grid $base.log10s -in $base.specs -row 5 -column 1 -sticky w
    grid columnconfigure $base.log10s 2 -minsize 30
    grid $base.legends -in $base.log10s -row 0 -column 4 -sticky w

    grid $base.labelSpecs -in $base.specs -row 0 -column 0 \
	-sticky nw -columnspan 2
    grid $base.buttonFieldFile  -in $base.specs -row 1 -column 0 \
	-sticky w -padx 10
    grid $base.fieldFile  -in $base.specs -row 1 -column 1
    grid $base.format -in $base.specs -row 2 -column 0 -sticky w -padx 10
    set i 1
    foreach x {ascii hdf hdf_int1 bgr_int1 pfb cnb nuft} {
	grid $base.$x -in $base.formats -row 0 -column $i 
	set i [expr $i + 1] }
    grid $base.labelTimestep -in $base.formats -row 0 -column $i  
    set i [expr $i + 1] 
    grid $base.timestep -in $base.formats -row 0 -column $i  
    grid $base.data -in $base.specs -row 3 -column 0 -sticky nw -padx 10

    grid $base.continuous -in $base.frameContinuous \
	-row 0 -column 0 -columnspan 2 -sticky w 
    grid $base.fieldColor -in $base.frameContinuous \
	-row 0 -column 2 -columnspan 2 -sticky w -pady 5 
    grid $base.labelColorTable -in $base.frameContinuous \
	-row 1 -column 0  -columnspan 4 -sticky nw
    grid $base.labelMin -in $base.frameContinuous \
	-row 2 -column 0 -sticky e -pady 10
    grid $base.entryMin -in $base.frameContinuous \
	-row 2 -column 1 -sticky w
    grid $base.labelMax -in $base.frameContinuous \
	-row 2 -column 2 -sticky e
    grid $base.entryMax -in $base.frameContinuous \
	-row 2 -column 3 -sticky w

    grid $base.categorical -in $base.frameCategorical \
	-row 0 -column 0 -sticky w 
    grid $base.buttonFieldCutoffs -in $base.frameCategorical \
	-row 0 -column 1 -sticky ew -pady 5
    grid $base.nCutoffs -in $base.frameCategorical \
	-row 1 -column 0 -sticky e
    grid $base.fieldNCut -in $base.frameCategorical \
	-row 1 -column 1 -sticky w

    grid $base.log10 -in $base.specs -row 5 -column 0 -sticky w -padx 10 
    grid $base.logYes -in $base.log10s -row 0 -column 0
    grid $base.logNo -in $base.log10s -row 0 -column 1
    grid $base.fieldLegend -in $base.log10s -row 0 -column 3 \
	-sticky e -padx 10
    grid $base.legendYes -in $base.legends -row 0 -column 0
    grid $base.legendNo -in $base.legends -row 0 -column 1
    #
    #   Display
    #
    frame $base.colors
    frame $base.shadings
    frame $base.nBits
    label $base.labelDisplay -text "POSTSCRIPT DISPLAY" \
	-font {times 12 bold}
    label $base.colorgray -text "Color or Grayscale?" -font {times 12 bold}
    label $base.shading -text Shading: -font {times 12 bold}
    label $base.corner -text "Corner Brightness:" -font {times 12 bold} 
    label $base.labelShadingXY -text XY -font {times 12 bold} 
    label $base.labelShadingXZ -text XZ -font {times 12 bold}
    label $base.labelShadingYZ -text YZ -font {times 12 bold}
    label $base.nBit -text "Color Resolution:" -font {times 12 bold}
    #       source getFieldColors.ui.tcl
    radiobutton $base.color -text color -value 1 \
	-variable field_color
    radiobutton $base.grayscale -text grayscale -value 0 \
	-variable field_color
    radiobutton $base.nBit2 -text 2-bit \
	-value 2 -variable field_n_bit
    radiobutton $base.nBit4 -text 4-bit \
	-value 4 -variable field_n_bit
    radiobutton $base.nBit8 -text 8-bit \
	-value 8 -variable field_n_bit
    scale $base.shadingXY -from -8 -to 8 -length 100 \
	-variable field_shade_xy -orient horizontal \
	-tickinterval 8 -showvalue true -resolution 1 -digits 1 
    scale $base.shadingXZ -from -8 -to 8 -length 100 \
	-variable field_shade_xz -orient horizontal \
	-tickinterval 8 -showvalue true -resolution 1 -digits 1 
    scale $base.shadingYZ -from -8 -to 8 -length 100 \
	-variable field_shade_yz -orient horizontal \
	-tickinterval 8 -showvalue true -resolution 1 -digits 1 
    scale $base.cornerBrightness -from 0.0 -to 1.0 -length 320 \
	-variable field_corner_bright -orient horizontal \
	-tickinterval 0.25 -showvalue true -resolution 0.01 -digits 3 

    grid $base.labelDisplay -in $base.display -row 0 -column 0 \
	-sticky nw -columnspan 2
    grid $base.colorgray -in $base.display -row 1 -column 0 \
	-sticky w -padx 10
    grid $base.colors -in $base.display -row 1 -column 1 \
	-sticky w  
    grid columnconfigure $base.colors 2 -minsize 30
    grid $base.color -in $base.colors -row 0 -column 0
    grid $base.grayscale -in $base.colors -row 0 -column 1
    grid $base.nBits -in $base.colors -row 0 -column 4 -sticky w 
    grid $base.nBit -in $base.colors -row 0 -column 3 -sticky e -padx 10  
    grid $base.nBit2 -in $base.nBits -row 0 -column 0  
    grid $base.nBit4 -in $base.nBits -row 0 -column 1  
    grid $base.nBit8 -in $base.nBits -row 0 -column 2  
    grid $base.shading -in $base.display -row 3 -column 0 \
	-sticky w -padx 10 
    grid $base.shadings -in $base.display -row 3 -column 1 -sticky w 
    grid $base.shadingXY -in $base.shadings -row 0 -column 1
    grid $base.labelShadingXY -in $base.shadings -row 0 -column 0
    grid $base.shadingXZ -in $base.shadings -row 0 -column 3
    grid $base.labelShadingXZ -in $base.shadings -row 0 -column 2
    grid $base.shadingYZ -in $base.shadings -row 0 -column 5
    grid $base.labelShadingYZ -in $base.shadings -row 0 -column 4
    grid $base.corner -in $base.display -row 4 -column 0 \
	-sticky w -padx 10
    grid $base.cornerBrightness -in $base.display -row 4 -column 1 -sticky w 

}


# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getField_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getField_ui"
	    getField_ui .
	}
    }
}
