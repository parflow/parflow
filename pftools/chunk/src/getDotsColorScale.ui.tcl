
proc getDotsColorScale_ui {root i args} {
        global chunk_run_file
        global global_file
        source $global_file  

        # this treats "." as a special case

        if {$root == "."} {
            set base ""
        } else {
            set base $root
        }
#
#  Big Frames
#
        frame $base.top 
        frame $base.frameLegend -relief groove -borderwidth 3
        frame $base.frameOKPS
        grid $base.top -row 0 -column 0 -padx 5 -pady 5 -ipadx 5 -ipady 5
        grid $base.frameLegend -row 1 -column 0 \
             -padx 5 -pady 5 -ipadx 5 -ipady 5
        grid $base.frameOKPS -row 2 -column 0 -sticky ew
#
# Top Frame
#        
        label $base.range -text "Range:" -font {times 12 bold}
        label $base.rangeMin -text "min" -font {times 12 bold}
        label $base.rangeMax -text "max" -font {times 12 bold}
        label $base.scaleFactor -text "Scale Factor" -font {times 12 bold}
        entry $base.entryRangeMin -textvariable dots_value_min($i) -width 10 \
           -background #bdd -highlightcolor #f00
        entry $base.entryRangeMax -textvariable dots_value_max($i) -width 10 \
           -background #bdd -highlightcolor #f00
        entry $base.entryScaleFactor -textvariable dots_time_scale($i) \
           -width 10 -background #bdd -highlightcolor #f00
        grid $base.range -in $base.top -row 0 -column 0
        grid $base.rangeMin -in $base.top -row 0 -column 1 -sticky e
        grid $base.entryRangeMin -in $base.top -row 0 -column 2 -sticky w
        grid $base.rangeMax -in $base.top -row 0 -column 3 -sticky e
        grid $base.entryRangeMax -in $base.top -row 0 -column 4 -sticky w
        grid $base.scaleFactor -in $base.top -row 0 -column 5 -sticky e
        grid $base.entryScaleFactor -in $base.top -row 0 -column 6 -sticky w
        button $base.colorTable -text "Color Table:" -background #e8e \
             -borderwidth 4 \
             -command "OpenDotsColorTableFile $base $i" -font {times 12 bold}  
        label $base.labelColorTable -textvariable dots_color_table($i) 
        grid $base.colorTable -in $base.top -row 1 -column 0
        grid $base.labelColorTable -in $base.top -row 1 -column 1 \
             -columnspan 6 -sticky w 

        proc OpenDotsColorTableFile {root i args} {
                      global dots_color_table
                      set typelist {
                          {"CTB Files" {".ctb"} {"CTB "}}
                      }
                      set color_table_dir [file dirname $dots_color_table($i)]
                      set dots_color_table_temp \
                      [tk_getOpenFile -defaultextension .ctb \
                      -filetypes $typelist \
                      -initialdir $color_table_dir \
                      -initialfile [file tail $dots_color_table($i)] \
                      -parent $root \
                      -title {"Open color table file for dots $i"}]
                      if {$dots_color_table_temp != ""} {
                        set dots_color_table($i) $dots_color_table_temp
                      }
                    }

#
#  Legend Frame
# 
        frame $base.legend1
        grid $base.legend1 -in $base.frameLegend -row 0 -column 0 -sticky w
        label $base.labelLegend -text "Legend?" -font {times 12 bold}
        radiobutton $base.radioLegendYes -text yes -value yes \
            -variable dots_legend($i)
        radiobutton $base.radioLegendNo -text no -value no \
            -variable dots_legend($i)
        label $base.legendTitle -text "    Title" -font {times 12 bold}
        entry $base.entryLegendTitle -textvariable dots_legend_title($i) \
           -width 60 -background #bdd -highlightcolor #f00
        grid $base.labelLegend -in $base.legend1 -row 0 -column 0
        grid $base.radioLegendYes -in $base.legend1 -row 0 -column 1
        grid $base.radioLegendNo -in $base.legend1 -row 0 -column 2
        grid $base.legendTitle -in $base.legend1 -row 0 -column 3 
        grid $base.entryLegendTitle -in $base.legend1 -row 0 -column 4

        frame $base.legend2
        grid $base.legend2 -in $base.frameLegend -row 1 -column 0 -sticky w
        label $base.labelSpacing -text "Label Spacing" -font {times 12 bold}
        entry $base.entrySpacing -textvariable dots_legend_label_spacing($i) \
           -width 10 -background #bdd -highlightcolor #f00
       label $base.decimalPlaces -text "  .Decimal Places" -font {times 12 bold}
        entry $base.entryDecimal -textvariable dots_legend_label_decimals($i) \
           -width 1 -background #bdd -highlightcolor #f00
        radiobutton $base.radioDec0 -text 0 -value 0 \
            -variable dots_legend_label_decimals($i)
        radiobutton $base.radioDec1 -text 1 -value 1 \
            -variable dots_legend_label_decimals($i)
        radiobutton $base.radioDec2 -text 2 -value 2 \
            -variable dots_legend_label_decimals($i)
        label $base.ticsPerLabel -text "  Tics Per Label" -font {times 12 bold}
        entry $base.entryTicsPerLabel \
            -textvariable dots_legend_label_tics_per($i) \
            -width 1 -background #bdd -highlightcolor #f00
        radiobutton $base.ticsPer1 -text 1 -value 1 \
            -variable dots_legend_label_tics_per($i) 
        radiobutton $base.ticsPer2 -text 2 -value 2 \
            -variable dots_legend_label_tics_per($i) 
        radiobutton $base.ticsPer5 -text 5 -value 5 \
            -variable dots_legend_label_tics_per($i) 
        radiobutton $base.ticsPer10 -text 10 -value 10 \
            -variable dots_legend_label_tics_per($i) 
        grid $base.labelSpacing -in $base.legend2 -row 0 -column 0
        grid $base.entrySpacing -in $base.legend2 -row 0 -column 1
        grid $base.decimalPlaces -in $base.legend2 -row 0 -column 2
        grid $base.entryDecimal -in $base.legend2 -row 0 -column 3
        grid $base.radioDec0 -in $base.legend2 -row 0 -column 4 
        grid $base.radioDec1 -in $base.legend2 -row 0 -column 5 
        grid $base.radioDec2 -in $base.legend2 -row 0 -column 6 
        grid $base.ticsPerLabel -in $base.legend2 -row 0 -column 7
        grid $base.entryTicsPerLabel -in $base.legend2 -row 0 -column 8
        grid $base.ticsPer1 -in $base.legend2 -row 0 -column 9 
        grid $base.ticsPer2 -in $base.legend2 -row 0 -column 10 
        grid $base.ticsPer5 -in $base.legend2 -row 0 -column 11 
        grid $base.ticsPer10 -in $base.legend2 -row 0 -column 12 

        frame $base.legend3
        grid $base.legend3 -in $base.frameLegend -row 2 -column 0 -sticky w
        label $base.translate -text "Translate:" -font {times 12 bold}
        label $base.translateX -text "  X" -font {times 12 bold}
        label $base.translateY -text "  Y" -font {times 12 bold}
        scale $base.scaleTransX -from 0 -to 500 -length 250 \
            -variable dots_legend_translation_x($i) -orient horizontal \
            -tickinterval 100 -showvalue true
        scale $base.scaleTransY -from 0 -to 500 -length 250 \
            -variable dots_legend_translation_y($i) -orient horizontal \
            -tickinterval 100 -showvalue true
        grid $base.translate -in $base.legend3 -row 0 -column 0
        grid $base.translateX -in $base.legend3 -row 0 -column 1
        grid $base.scaleTransX -in $base.legend3 -row 0 -column 2
        grid $base.translateY -in $base.legend3 -row 0 -column 3
        grid $base.scaleTransY -in $base.legend3 -row 0 -column 4

 
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
     
 
