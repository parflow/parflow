        proc getLines_ui {root args }  {
            global color
            global i
            global base
            global global_file
            source $global_file
        # this treats "." as a special case
        if {$root == "."} {
            set base ""
        } else {
            set base $root
        }
#
#  Main frames
#
        frame $base.lines
        grid $base.lines -row 0 -column 0
        frame $base.frameOKPS
        grid $base.frameOKPS -row 1 -column 0 -sticky ew
#
#  top frame
#
        for {set i 1} {$i<=$line_n_files} {incr i 1} {
              frame $base.top$i 
              grid $base.top$i -in $base.lines -row $i -column 0
              button $base.getLineFile$i -background #e8e -borderwidth 4 \
                   -text "File $i" -font {times 12 bold} \
                   -command "OpenLineFile $base $i"
              grid $base.getLineFile$i -in $base.top$i -row 0 -column 0 \
                   -sticky e
              entry $base.lineFileName$i -textvariable line_file($i) \
                  -width 80 -background #bdd -highlightcolor #f00
              grid $base.lineFileName$i -in $base.top$i -row 0 -column 1
              frame $base.lineFormats$i
              grid $base.lineFormats$i -in $base.top$i -row 1 -column 0 \
                   -sticky nw
              label $base.labelFormat$i -text Format: -font {times 12 bold}
              grid $base.labelFormat$i -in $base.lineFormats$i \
                   -row 0 -column 0 -sticky w
              set j 1
              foreach x {slim nts nuft ind arrow} {
                 if {$j==1} {set lab "zero val."}
                 if {$j==2} {set lab "# points"}
                 if {$j==3} {set lab "weight ch."}
                 if {$j==4} {set lab "geoeas"}
                 if {$j==5} {set lab "arrow"}
                 radiobutton $base.$i$x -text $lab -value $x \
                   -variable line_format($i)
                 grid $base.$i$x -in $base.lineFormats$i -row $j -column 0 \
                   -sticky nw 
                 set j [expr $j + 1]
              }
          
#
#  Frame Appearance
#
          frame $base.appearance$i -relief groove -borderwidth 3
          grid $base.appearance$i -in $base.top$i -row 1 -column 1 \
            -padx 1 -pady 1 -ipadx 1 -ipady 1  
          frame $base.lowerleft$i 
          grid $base.lowerleft$i -in $base.appearance$i -row 1 -column 0 \
             -rowspan 2 -columnspan 2
          label $base.labelAppearance$i -text "LINE $i APPEARANCE" \
               -font {times 12 bold}
          grid $base.labelAppearance$i -in $base.appearance$i -row 0 -column 0 \
               -sticky nw -columnspan 3
          label $base.outside$i -text "Outside:" -font {times 12 bold}
          label $base.inside$i -text "Inside:" -font {times 12 bold}
          grid $base.outside$i -in $base.appearance$i -row 1 -column 2 \
            -ipadx 5 -sticky e
          grid $base.inside$i -in $base.appearance$i -row 2 -column 2 \
            -ipadx 5 -sticky e
          label $base.width$i -text "Width:" -font {times 12 bold}
          label $base.minwidth$i -text "Min Width:" -font {times 12 bold}
          label $base.dash$i -text "Dash:" -font {times 12 bold}
          label $base.marker$i -text "Marker:" -font {times 12 bold}
          grid $base.width$i -in $base.appearance$i -row 0 -column 3 
          grid $base.minwidth$i -in $base.appearance$i -row 0 -column 4 
          grid $base.dash$i -in $base.appearance$i -row 0 -column 5 
          grid $base.marker$i -in $base.appearance$i -row 0 -column 6 
          scale $base.outsideWidth$i -from 0.0 -to 10.0 -length 72 \
              -variable line_outside_linewidth($i)  -orient horizontal \
              -tickinterval 10.0 -showvalue true -resolution 0.5 -digits 3 \
              -sliderlength 18 
          scale $base.insideWidth$i -from 0.0 -to 10.0 -length 72 \
              -variable line_inside_linewidth($i)  -orient horizontal \
              -tickinterval 10.0 -showvalue true -resolution 0.5 -digits 3 \
              -sliderlength 18 
          scale $base.outsideMinWidth$i -from 0.0 -to 10.0 -length 72 \
              -variable line_outside_linewidth_min($i)  -orient horizontal \
              -tickinterval 10.0 -showvalue true -resolution 0.5 -digits 3 \
              -sliderlength 18 
          scale $base.insideMinWidth$i -from 0.0 -to 10.0 -length 72 \
              -variable line_inside_linewidth_min($i)  -orient horizontal \
              -tickinterval 10.0 -showvalue true -resolution 0.5 -digits 3 \
              -sliderlength 18
          scale $base.outsideDash$i -from 0 -to 20 -length 72 \
              -variable line_outside_dash($i)  -orient horizontal \
              -tickinterval 10 -showvalue true -resolution 1 \
              -sliderlength 18 
          scale $base.insideDash$i -from 0 -to 20 -length 72 \
              -variable line_inside_dash($i)  -orient horizontal \
              -tickinterval 10 -showvalue true -resolution 1 \
              -sliderlength 18 
          grid $base.outsideWidth$i -in $base.appearance$i -row 1 -column 3 \
             -padx 3 -ipadx 5
          grid $base.insideWidth$i -in $base.appearance$i -row 2 -column 3 \
             -padx 3 -ipadx 5
          grid $base.outsideMinWidth$i -in $base.appearance$i -row 1 -column 4 \
             -padx 3 -ipadx 5
          grid $base.insideMinWidth$i -in $base.appearance$i -row 2 -column 4 \
             -padx 3 -ipadx 5
          grid $base.outsideDash$i -in $base.appearance$i -row 1 -column 5 \
             -padx 3 -ipadx 5
          grid $base.insideDash$i -in $base.appearance$i -row 2 -column 5 \
             -padx 3 -ipadx 5
#
#  Markers 
#
        frame $base.frameInsideMarker$i  
        grid $base.frameInsideMarker$i -in $base.appearance$i -row 2 -column 6 

        set j 0
        foreach marker {o + x } {
          set j [expr $j + 1]
          radiobutton $base.pickMarkerIn1$i$j -text $marker \
                  -variable line_inside_marker($i) \
                  -value $j
          grid $base.pickMarkerIn1$i$j -in $base.frameInsideMarker$i \
                  -row 1 -column $j -sticky ew \
        }
        set j 3
        foreach marker {bx di tr } {
          set j [expr $j + 1]
          set j2 [expr $j - 3]
          radiobutton $base.pickMarkerIn2$i$j -text $marker \
                  -variable line_inside_marker($i) \
                  -value $j
          grid $base.pickMarkerIn2$i$j -in $base.frameInsideMarker$i \
                  -row 2 -column $j2 -sticky ew \
        }


        checkbutton $base.checkInsideMarker$i \
           -variable line_inside_sign($i) \
           -text only -onvalue -1 -offvalue 1
        radiobutton $base.radioInsideMarker$i -text "none" \
           -variable line_inside_marker($i) -value 0   

        grid $base.radioInsideMarker$i -in $base.frameInsideMarker$i \
             -row 0 -column 1 -sticky w -columnspan 2
        grid $base.checkInsideMarker$i -in $base.frameInsideMarker$i \
             -row 0 -column 2 -sticky e -columnspan 2

        frame $base.frameOutsideMarker$i  
        grid $base.frameOutsideMarker$i -in $base.appearance$i -row 1 -column 6 

        set j 0
        foreach marker {o + x } {
          set j [expr $j + 1]
          radiobutton $base.pickMarkerOut1$i$j -text $marker \
                  -variable line_outside_marker($i) \
                  -value $j
          grid $base.pickMarkerOut1$i$j -in $base.frameOutsideMarker$i \
                  -row 1 -column $j -sticky ew \
        }
        set j 3
        foreach marker {bx di tr } {
          set j [expr $j + 1]
          set j2 [expr $j - 3]
          radiobutton $base.pickMarkerOut2$i$j -text $marker \
                  -variable line_outside_marker($i) \
                  -value $j
          grid $base.pickMarkerOut2$i$j -in $base.frameOutsideMarker$i \
                  -row 2 -column $j2 -sticky ew \
        }
        checkbutton $base.checkOutsideMarker$i \
           -variable line_outside_sign($i) \
           -text only -onvalue -1 -offvalue 1
        radiobutton $base.radioOutsideMarker$i -text "none" \
           -variable line_outside_marker($i) -value 0   

        grid $base.radioOutsideMarker$i -in $base.frameOutsideMarker$i \
             -row 0 -column 1 -sticky w -columnspan 2
        grid $base.checkOutsideMarker$i -in $base.frameOutsideMarker$i \
             -row 0 -column 2 -sticky e -columnspan 2

        label $base.origin$i -text Origin: -font {times 12 bold}
        label $base.x$i -text X -font {times 12 bold}
        label $base.y$i -text Y -font {times 12 bold}
        label $base.z$i -text Z -font {times 12 bold}
        grid $base.origin$i -in $base.lowerleft$i -row 1 -column 0
        grid $base.x$i -in $base.lowerleft$i -row 0 -column 1 -sticky e
        grid $base.y$i -in $base.lowerleft$i -row 1 -column 1 -sticky e
        grid $base.z$i -in $base.lowerleft$i -row 2 -column 1 -sticky e
        entry $base.entryX$i -textvariable line_origin_x($i) \
                  -width 12 -background #bdd -highlightcolor #f00
        entry $base.entryY$i -textvariable line_origin_y($i) \
                  -width 12 -background #bdd -highlightcolor #f00
        entry $base.entryZ$i -textvariable line_origin_z($i) \
                  -width 12 -background #bdd -highlightcolor #f00
        grid $base.entryX$i -in $base.lowerleft$i -row 0 -column 2
        grid $base.entryY$i -in $base.lowerleft$i -row 1 -column 2
        grid $base.entryZ$i -in $base.lowerleft$i -row 2 -column 2
         

          set color "#fff"
          catch {set color $line_color($i)}
          button $base.color$i -text Color -background $color -borderwidth 4 \
             -font {times 12 bold} \
             -command "buttoncolor $base.color$i $i" 

proc buttoncolor {w k} {
    global line_color
    set old_color [$w cget -background]
    set color [tk_chooseColor -parent $w -initialcolor $old_color \
                        -title "Line $k Color"]
    if {$color != ""} {
      $w configure -background $color 
      set line_color($k) $color
    }
    return
}

          source getLineColorScale.ui.tcl
          button $base.scalecolor$i -text "Color Scale" -background "#e8e" \
               -borderwidth 4 -font {times 12 bold} \
               -command "toplevel .getLineColorScale; \
                      focus .getLineColorScale; \
                      corner .get_Lines .getLineColorScale; \
                      wm title .getLineColorScale {Color Scale for Line $i}; \
                      getLineColorScale_ui .getLineColorScale $i; \
                      grab .getLineColorScale; \
                      tkwait window .getLineColorScale"

          label $base.winnow$i -text "Winnow" -font {times 12 bold}
          entry $base.entryWinnow$i -textvariable line_winnow($i) \
                  -width 4 -background #bdd -highlightcolor #f00
          frame $base.radioWinnow$i
          grid $base.radioWinnow$i -in $base.lowerleft$i -row 4 -column 2
          radiobutton $base.winnow10$i -text 10 -value 10 \
                   -variable line_winnow($i)
          radiobutton $base.winnow100$i -text 100 -value 100 \
                   -variable line_winnow($i)
          grid $base.winnow10$i -in $base.radioWinnow$i -row 0 -column 0
          grid $base.winnow100$i -in $base.radioWinnow$i -row 0 -column 1
          grid $base.color$i -in $base.lowerleft$i -row 3 -column 0 
          grid $base.scalecolor$i -in $base.lowerleft$i -row 3 -column 1 \
                  -columnspan 2 -sticky w 
          grid $base.winnow$i -in $base.lowerleft$i -row 4 -column 0 -sticky e 
          grid $base.entryWinnow$i -in $base.lowerleft$i -row 4 -column 1 \
               -sticky w
          
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

     proc getMarker_ui {root m args} {
        global line_outside_marker 
        # this treats "." as a special case
        if {$root == "."} {
            set base ""
        } else {
            set base $root
        }
         set j 1
         foreach marker {3p tr 5p pe 6p ci } {
         set line_outside_marker($m) $marker 
         button $base.buttonOK \
                -text "OK" \
                -font {times 12 bold} \
                -background "#9f9" \
                -borderwidth 4 \
                -command "destroy $base"
         grid $base.buttonOK -column 0 -row [expr $j + 1]
         }
     }  

        proc OpenLineFile {w k } {
                          global line_file
                          global home_dir
                          if {[file exists $line_file($k)] == 1} {
                           set line_dir [file dirname $line_file($k)]
                          } else {
                           set line_dir $home_dir
                          }
                          set line_file_temp [tk_getOpenFile \
                          -initialdir $line_dir \
                          -initialfile [file tail $line_file($k)] \
                          -parent $w \
                          -title "Open Line File $k"]
                          if {$line_file_temp != ""} {
                            set line_file($k) $line_file_temp
                          }
        }   
