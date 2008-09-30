proc getPlot_ui {root args} {
        source chunk_global.tcl
        global base
	# this treats "." as a special case

	if {$root == "."} {
	    set base ""
	} else {
	    set base $root
	}
#
#   Frames
#
        frame $base.framePlot -relief groove -borderwidth 2 
        frame $base.frameTitle -relief groove -borderwidth 2 
        frame $base.frameOKPS 
    
#
#   Labels
#
        label $base.labelPlot -text "PICTURE" -font {times 12 bold}
        label $base.labelOrientation -text "Orientation:" -font {times 12 bold}
        label $base.labelBaseFontSize -text "Base Font Size:" \
             -font {times 12 bold}
       label $base.labelPlotTransX -text "Translate:   X" -font {times 12 bold} 
        label $base.labelPlotTransY -text "Y" -font {times 12 bold}
       label $base.labelTitleTransX -text "Translate:   X" -font {times 12 bold}
        label $base.labelTitleTransY -text "Y" -font {times 12 bold}
        label $base.labelColors -text "Colors:" -font {times 12 bold}
        label $base.labelTitle -text "TITLE" -font {times 12 bold}
        label $base.labelTitle1 -text "Line 1:" -font {times 12 bold}
        label $base.labelTitle2 -text "Line 2:" -font {times 12 bold}
        label $base.labelJustification -text "Justification:" -font {times 12 bold}
#
#   Entries
#
	entry $base.entryBaseFontSize -textvariable plot_textsize -width 2 \
               -background #bdd -highlightcolor #f00
	entry $base.entryTitle1 -textvariable plot_title_1 -width 45 \
               -background #bdd -highlightcolor #f00 
	entry $base.entryTitle2 -textvariable plot_title_2 -width 45 \
               -background #bdd -highlightcolor #f00
	entry $base.entryPlotTransX -textvariable plot_translation_x -width 5 \
               -background #bdd -highlightcolor #f00
	entry $base.entryPlotTransY -textvariable plot_translation_y -width 5 \
               -background #bdd -highlightcolor #f00
	entry $base.entryTitleTransX -textvariable plot_title_translation_x \
               -width 5 -background #bdd -highlightcolor #f00
	entry $base.entryTitleTransY -textvariable plot_title_translation_y \
               -width 5 -background #bdd -highlightcolor #f00
	entry $base.entryFontSize -textvariable plot_textsize -width 40 \
               -background #bdd -highlightcolor #f00
#
#  Buttons
#
        button $base.buttonBackgroundColor \
              -background $plot_background_color \
              -foreground "#000000" \
              -borderwidth 4 \
              -command {
               set color [tk_chooseColor -initialcolor $plot_background_color \
                            -title "Background Color"]
                 if {$color != ""} {
                    $base.buttonBackgroundColor configure -background $color
                    set plot_background_color $color
                 }
               } \
              -text "Background" -font {times 12 bold}
        button $base.buttonTextColor \
              -background $plot_text_color \
              -foreground "#ffffff" \
              -borderwidth 4 \
              -command {
                 set color [tk_chooseColor -initialcolor $plot_text_color \
                            -title "Text Color"]
                 if {$color != ""} {
                    $base.buttonTextColor configure -background $color
                    set plot_text_color $color
                 }
               } \
              -text "Text" -font {times 12 bold}
        button $base.buttonAxesColor \
              -background $plot_axes_color \
              -foreground "#ffffff" \
              -borderwidth 4 \
              -command {
                 set color [tk_chooseColor -initialcolor $plot_axes_color \
                            -title "Axes Color"]
                 if {$color != ""} {
                    $base.buttonAxesColor configure -background $color
                    set plot_axes_color $color
                 }
               } \
              -text "Axes" -font {times 12 bold}
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

#
#  Radiobuttons
#
        radiobutton $base.radiobuttonPortrait \
                -text Portrait \
                -value no \
                -variable plot_landscape
        radiobutton $base.radiobuttonLandscape \
                -text Landscape \
                -value yes \
                -variable plot_landscape
        radiobutton $base.radiobuttonTitleLeft \
                -text left \
                -value left \
                -variable plot_title_justification
        radiobutton $base.radiobuttonTitleCenter \
                -text center \
                -value center \
                -variable plot_title_justification
        radiobutton $base.radiobuttonTitleRight \
                -text right \
                -value right \
                -variable plot_title_justification
#
#  ListBox for fontsize
#
        frame $base.frameFontSize -borderwidth .1c
        scrollbar $base.frameFontSize.scroll \
                  -command "$base.frameFontSize.list yview"
        listbox $base.frameFontSize.list \
                  -selectmode single \
                  -selectbackground "#00f" \
                  -selectforeground "#fff" \
                  -yscroll "$base.frameFontSize.scroll set" \
                  -setgrid 1 -height 3 
        pack $base.frameFontSize.scroll -side right -fill y 
        pack $base.frameFontSize.list -side left -expand 1 -fill both
        $base.frameFontSize.list insert 0 4 5 6 7 8 9 10 12 14 16 18 20 24 30
        bind $base.frameFontSize.list <ButtonPress-1> \
             {set plot_textsize [$base.frameFontSize.list get anchor]}
#
#  Translate Scale
# 
        scale $base.scalePlotTransX -from -200 -to 200 -length 200 \
            -variable plot_translation_x -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scalePlotTransY -from -200 -to 200 -length 200 \
            -variable plot_translation_y -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scaleTitleTransX -from -200 -to 200 -length 200 \
            -variable plot_title_translation_x -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scaleTitleTransY -from -200 -to 200 -length 200 \
            -variable plot_title_translation_y -orient horizontal \
            -tickinterval 100 -showvalue true
#
#  Pack Grid
#
        pack $base.buttonOK -in $base.frameOKPS \
           -side left -expand true -fill x
        pack $base.buttonPS  -in $base.frameOKPS \
           -side right

        grid $base.framePlot -row 0 -column 0 -sticky ew \
               -padx 5 -pady 5 -ipadx 2 -ipady 2
        grid $base.frameTitle -row 1 -column 0 -sticky ew \
               -padx 5 -pady 5 -ipadx 2 -ipady 2
        grid $base.frameOKPS -row 2 -column 0 -sticky ew

        grid $base.labelPlot -in $base.framePlot \
            -row 0 -column 0 -sticky w 
        grid $base.labelOrientation -in $base.framePlot \
            -row 1 -column 0 -sticky w -padx 10 
        grid $base.radiobuttonPortrait -in $base.framePlot \
            -row 1 -column 1 -sticky ew 
        grid $base.radiobuttonLandscape -in $base.framePlot \
            -row 1 -column 2 -sticky ew 
        grid $base.labelColors -in $base.framePlot \
            -row 2 -column 0 -sticky w -padx 10 
        grid $base.buttonBackgroundColor -in $base.framePlot \
            -row 2 -column 1 -sticky ew -pady 4 
        grid $base.buttonTextColor -in $base.framePlot \
            -row 2 -column 2 -sticky ew -pady 4
        grid $base.buttonAxesColor -in $base.framePlot \
            -row 2 -column 3 -sticky ew -pady 4 
        grid $base.labelBaseFontSize -in $base.framePlot \
            -row 3 -column 0 -sticky w -padx 10 
        grid $base.entryBaseFontSize -in $base.framePlot \
            -row 3 -column 1 -sticky w  
        grid $base.frameFontSize -in $base.framePlot \
            -row 3 -column 2 -sticky nsew -columnspan 2
        grid $base.labelPlotTransX -in $base.framePlot \
            -row 4 -column 0 -sticky e 
        grid $base.entryPlotTransX -in $base.framePlot \
            -row 4 -column 1 -sticky w 
        grid $base.scalePlotTransX -in $base.framePlot \
            -row 4 -column 2 -sticky ew -columnspan 3 
        grid $base.labelPlotTransY -in $base.framePlot \
            -row 5 -column 0 -sticky e 
        grid $base.entryPlotTransY -in $base.framePlot \
            -row 5 -column 1 -sticky w 
        grid $base.scalePlotTransY -in $base.framePlot \
            -row 5 -column 2 -sticky ew -columnspan 3 

        grid $base.labelTitle -in $base.frameTitle \
            -row 0 -column 0 -sticky w 
        grid $base.labelTitle1 -in $base.frameTitle \
            -row 1 -column 0 -sticky w -padx 10 
	grid $base.entryTitle1 -in $base.frameTitle \
            -row 1 -column 1 -columnspan 4 -sticky w 
        grid $base.labelTitle2 -in $base.frameTitle \
            -row 2 -column 0 -sticky w -padx 10 
	grid $base.entryTitle2 -in $base.frameTitle \
            -row 2 -column 1 -columnspan 4 -sticky w 
        grid $base.labelJustification -in $base.frameTitle \
            -row 3 -column 0 -sticky w -padx 10 
        grid $base.radiobuttonTitleLeft -in $base.frameTitle \
            -row 3 -column 1 -sticky ew 
        grid $base.radiobuttonTitleCenter -in $base.frameTitle \
            -row 3 -column 2 -sticky ew 
        grid $base.radiobuttonTitleRight -in $base.frameTitle \
            -row 3 -column 3 -sticky ew 
        grid $base.labelTitleTransX -in $base.frameTitle \
            -row 4 -column 0 -sticky e 
        grid $base.entryTitleTransX -in $base.frameTitle \
            -row 4 -column 1 -sticky w 
        grid $base.scaleTitleTransX -in $base.frameTitle \
            -row 4 -column 2 -sticky ew -columnspan 3 
        grid $base.labelTitleTransY -in $base.frameTitle \
            -row 5 -column 0 -sticky e 
        grid $base.entryTitleTransY -in $base.frameTitle \
            -row 5 -column 1 -sticky w 
        grid $base.scaleTitleTransY -in $base.frameTitle \
            -row 5 -column 2 -sticky ew -columnspan 3 
        
  


}


# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getPlot_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getPlot_ui"
	    getPlot_ui .
	}
    }
}
