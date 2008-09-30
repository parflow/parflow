     proc getSurfaces_ui {root args }  {
        global global_file
        source $global_file
        global color
        # this treats "." as a special case

        if {$root == "."} {
            set base ""
        } else {
            set base $root
        }
#
#  Main frames
#
        frame $base.top
        frame $base.topVShade -relief groove -borderwidth 4 
        frame $base.frameOKPS
        grid $base.top -row 0 -column 0
        grid $base.topVShade -row 1 -column 0 -padx 5 -pady 5 
        grid $base.frameOKPS -row 2 -column 0 -sticky ew
#
#  Surfaces
# 
        for {set i 1} {$i <= $surface_n_files} {incr i 1} {
#
#   File Boxes
#
           frame $base.file$i -relief groove -borderwidth 4 
           grid $base.file$i -in $base.top -column 0 -row $i \
               -padx 5 -pady 5 -ipadx 3 -ipady 3
           label $base.labelFile$i -text "FILE $i" -font {times 12 bold}
           grid $base.labelFile$i -in $base.file$i -row 0 -column 0
#
#  Surface Boxes
#
           frame $base.surfaces$i
           grid $base.surfaces$i -in $base.file$i -column 1 -row 0 
           for {set j 1} {$j <= $surface_n_cutoffs($i)} {incr j 1} {
              frame $base.surface$i$j -relief raised -borderwidth 2
              grid $base.surface$i$j -in $base.surfaces$i -row $j -column 0 \
                -pady 2
              frame $base.left$i$j
              frame $base.right$i$j
              frame $base.firstRow$i$j
              frame $base.secondRow$i$j
              frame $base.thirdRow$i$j
              grid $base.left$i$j -in $base.surface$i$j -column 1 -row 0 \
                   -padx 2 
              grid $base.right$i$j -in $base.surface$i$j -column 2 -row 0 \
                   -padx 2 
              grid $base.firstRow$i$j -in $base.left$i$j -column 0 -row 0 \
                     -sticky w  -padx 2 -pady 2
              grid $base.secondRow$i$j -in $base.left$i$j -column 0 -row 1 \
                     -sticky w -padx 2 -pady 2
              grid $base.thirdRow$i$j -in $base.left$i$j -column 0 -row 2 \
                     -sticky w -padx 2 -pady 2

              label $base.labelSurface$i$j -text $j -font {times 16 bold}
              label $base.plot$i$j -text "Plot?" -font {times 12 bold} 
              checkbutton $base.cplot$i$j -variable surface_plot($i,$j) \
                          -onvalue yes -offvalue no
              label $base.label$i$j -text "     Label" -font {times 12 bold} 
              entry $base.elabel$i$j -textvariable surface_legend_label($i,$j) \
                 -width 18  -background #bdd -highlightcolor #f00
           grid $base.labelSurface$i$j -in $base.surface$i$j -row 0 -column 0 \
                 -rowspan "$surface_n_cutoffs($i)" -sticky n
           grid $base.plot$i$j -in $base.firstRow$i$j -row 0 -column 1 -sticky w
           grid $base.cplot$i$j -in $base.firstRow$i$j -row 0 -column 2 \
               -sticky w
           grid $base.label$i$j -in $base.firstRow$i$j -row 0 -column 3 \
               -sticky e
           grid $base.elabel$i$j -in $base.firstRow$i$j -row 0 -column 4 \
               -sticky e

              label $base.range$i$j -text "Range:" -font {times 12 bold}
              label $base.min$i$j -text "min" -font {times 12 bold} 
              entry $base.emin$i$j -textvariable surface_value_min($i,$j) \
                 -width 10  -background #bdd -highlightcolor #f00
              label $base.max$i$j -text "max" -font {times 12 bold} 
              entry $base.emax$i$j -textvariable surface_value_max($i,$j) \
                 -width 10  -background #bdd -highlightcolor #f00
              grid $base.range$i$j -in $base.secondRow$i$j -row 0 -column 0 \
                -sticky w
              grid $base.min$i$j -in $base.secondRow$i$j -row 0 -column 1
              grid $base.emin$i$j -in $base.secondRow$i$j -row 0 -column 2
              grid $base.max$i$j -in $base.secondRow$i$j -row 0 -column 3
              grid $base.emax$i$j -in $base.secondRow$i$j -row 0 -column 4

          label $base.labelColors$i$j -text "Colors:" -font {times 12 bold} 
          grid $base.labelColors$i$j -in $base.thirdRow$i$j -row 0 -column 0 \
             -sticky w
          set m 0
          foreach k {out in edge} {
             set m [expr $m + 1 ]
             set color "#ffffff"
             catch {set color $surface_color($i,$j,$k)}
             button $base.buttonColor($i,$j,$k) \
              -background $color -padx 10 \
              -text "$k" -font {times 12 bold} \
              -command "buttoncolor $base.buttonColor($i,$j,$k) $i $j $k"
             checkbutton $base.check$i$j$k -variable surface_plot($i,$j,$k) \
                   -onvalue -1 -offvalue 0 
             grid $base.buttonColor($i,$j,$k) -in $base.thirdRow$i$j \
                   -row 0 -column [expr 2 * $m + 1] 
             grid $base.check$i$j$k -in $base.thirdRow$i$j \
                   -row 0 -column [expr 2 * $m + 2]
          } 

proc buttoncolor {w i j k} {
    global surface_color
    set old_color [$w cget -background]
    set color [tk_chooseColor -parent $w -initialcolor $old_color \
                        -title "$k color"]
    if {$color != ""} { 
        $w configure -background $color 
        set surface_color($i,$j,$k) $color
    }
    return
}


        label $base.shading$i$j -text "Shading:" -font {times 12 bold} 
        label $base.xy$i$j -text "XY" -font {times 12 bold} 
        label $base.xz$i$j -text "XZ" -font {times 12 bold} 
        label $base.yz$i$j -text "YZ" -font {times 12 bold} 
        scale $base.shadingXY$i$j -from -8 -to 8 -length 60 \
            -variable surface_shade_xy($i,$j) -orient horizontal \
            -tickinterval 8 -showvalue true -resolution 1 -digits 1 \
            -sliderlength 16
        scale $base.shadingXZ$i$j -from -8 -to 8 -length 60 \
            -variable surface_shade_xz($i,$j) -orient horizontal \
            -tickinterval 8 -showvalue true -resolution 1 -digits 1 \
            -sliderlength 16
        scale $base.shadingYZ$i$j -from -8 -to 8 -length 60 \
            -variable surface_shade_yz($i,$j) -orient horizontal \
            -tickinterval 8 -showvalue true -resolution 1 -digits 1 \
            -sliderlength 16
        grid $base.shading$i$j -in $base.right$i$j -row 0 -column 0 \
            -columnspan 6 -sticky ew
        grid $base.xy$i$j -in $base.right$i$j -row 1 -column 0           
        grid $base.shadingXY$i$j -in $base.right$i$j -row 1 -column 1
        grid $base.xz$i$j -in $base.right$i$j -row 1 -column 2           
        grid $base.shadingXZ$i$j -in $base.right$i$j -row 1 -column 3
        grid $base.yz$i$j -in $base.right$i$j -row 1 -column 4           
        grid $base.shadingYZ$i$j -in $base.right$i$j -row 1 -column 5
        label $base.transparency$i$j -text "Transparency:" -font {times 12 bold}
        scale $base.scaleTransparency$i$j -from 0.00 -to 1.00 -length 100 \
            -variable transparency($i,$j) -orient horizontal \
            -tickinterval 1.0 -showvalue true -resolution 0.01 -digits 3 \
            -sliderlength 16
        grid $base.transparency$i$j -in $base.right$i$j -row 0 -column 6 \
            -padx 5 -sticky ew
        grid $base.scaleTransparency$i$j -in $base.right$i$j -row 1 -column 6

        frame $base.trans_dia$i$j
        grid $base.trans_dia$i$j -in $base.right$i$j -row 0 -column 7 \
            -rowspan 2
        foreach k {2 3 5} {
          radiobutton $base.trans_dia$i$j$k -text $k -value $k \
            -variable surface_transparent_dia($i,$j)
          grid $base.trans_dia$i$j$k -in $base.trans_dia$i$j -row $k -column 0
        } 
    }
}

#
#  Shading
#
        label $base.labelVShade -text "VERTICAL SHADING" -font {times 12 bold}
        label $base.labelBottom -text "    Bottom Cell:" -font {times 12 bold}
        label $base.labelTop -text "Top Cell:" -font {times 12 bold}
        entry $base.entryTopCell -textvariable surface_z_top -width 4 \
              -background #bdd -highlightcolor #f00
        entry $base.entryBottomCell -textvariable surface_z_bottom -width 4 \
              -background #bdd -highlightcolor #f00
        scale $base.shadingBottom -from -15 -to 15 -length 140 \
            -variable surface_shade_bottom -orient horizontal \
            -tickinterval 15 -showvalue true
        scale $base.shadingTop -from -15 -to 15 -length 140 \
            -variable surface_shade_top -orient horizontal \
            -tickinterval 15 -showvalue true
        grid $base.labelVShade -in $base.topVShade -column 0 -row 0 \
              -sticky nw
        grid $base.labelTop -in $base.topVShade -column 1 -row 0
        grid $base.entryTopCell -in $base.topVShade -column 2 -row 0
        grid $base.shadingTop -in $base.topVShade -column 3 -row 0
        grid $base.labelBottom -in $base.topVShade -column 4 -row 0
        grid $base.entryBottomCell -in $base.topVShade -column 5 -row 0
        grid $base.shadingBottom -in $base.topVShade -column 6 -row 0
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

 



