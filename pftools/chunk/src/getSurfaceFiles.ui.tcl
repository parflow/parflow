        proc get_Surface_Files_ui {root args }  {
            global i
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
        frame $base.top
        frame $base.frameOKPS
        grid $base.top -row 0 -column 0
        grid $base.frameOKPS -row 1 -column 0 -sticky ew
            for {set i 1} {$i<=$surface_n_files} {incr i 1} {
              frame $base.surface$i -relief groove -borderwidth 3
              grid $base.surface$i -in $base.top -row $i -column 0 \
                 -padx 5 -pady 5 -ipadx 5 -ipady 5
              button $base.getSurfaceFile$i -background #e8e -borderwidth 4 \
                   -text "File $i" -font {times 12 bold} \
                   -command "OpenSurfaceFile $base $i" 
              grid $base.getSurfaceFile$i -in $base.surface$i -row 0 -column 0 \
                   -sticky e
              entry $base.surfaceFileName$i -textvariable surface_file($i) \
                  -width 80 -background #bdd -highlightcolor #f00
              grid $base.surfaceFileName$i -in $base.surface$i -row 0 -column 1
              label $base.labelFormat$i -text Format: -font {times 12 bold}
              frame $base.surfaceFormats$i
              grid $base.labelFormat$i -in $base.surface$i -row 1 -column 0 \
                   -sticky w
              grid $base.surfaceFormats$i -in $base.surface$i -row 1 -column 1 \
                   -sticky w
              set j 0
              foreach x {ascii hdf hdf_int1 bgr_int1 pfb cnb nuft} {
                 radiobutton $base.$i$x -text $x -value $x \
                   -variable surface_format($i)
                 grid $base.$i$x -in $base.surfaceFormats$i -row 0 -column $j
                 set j [expr $j + 1]
              }
              label $base.labelNuftTimestep$i -text "NUFT Timestep:" \
                  -font {times 12 bold}
              grid $base.labelNuftTimestep$i -in $base.surfaceFormats$i \
                    -row 0 -column $j 
              entry $base.entryNuftTimestep$i \
                    -textvariable surface_nuft_timestep($i)  -width 4\
                    -background #bdd -highlightcolor #f00
              set j [expr $j + 1]
              grid $base.entryNuftTimestep$i -in $base.surfaceFormats$i \
                    -row 0 -column $j 
              frame $base.surfaceNumberOf$i
              grid $base.surfaceNumberOf$i -in $base.surface$i \
                -row 2 -column 1 -sticky w 
              label $base.labelSurfaceNumberOf$i -text "# Surfaces:" \
                -font {times 12 bold}
              grid $base.labelSurfaceNumberOf$i -in $base.surface$i \
                -row 2 -column 0 -sticky w 
              entry $base.entrySurfaceNumberOf$i -width 1 \
                  -textvariable surface_n_cutoffs($i) \
                  -background #bdd -highlightcolor #f00
              grid $base.entrySurfaceNumberOf$i -in $base.surfaceNumberOf$i \
                -row 0 -column 0 -sticky w 
              foreach m {1 2 3 4 5} {\
                  radiobutton $base.surfaceCutoff$i$m -text $m -value $m \
                   -variable surface_n_cutoffs($i) 
                  set m2 [expr $m + 1] 
                  grid $base.surfaceCutoff$i$m -in $base.surfaceNumberOf$i \
                  -row 0 -column $m  
              }
              
              frame $base.surfaceLegend$i -relief raised -borderwidth 2
              grid $base.surfaceLegend$i -in $base.surface$i -row 3 -column 0 \
                -columnspan 2 -pady 5
              label $base.legend$i -text "Legend?" -font {times 12 bold}
              radiobutton $base.legendYes$i -text yes -value yes \
                 -variable surface_legend($i)
              radiobutton $base.legendNo$i -text no -value no \
                 -variable surface_legend($i)
              grid $base.legend$i -in $base.surfaceLegend$i -row 0 -column 0
              grid $base.legendYes$i -in $base.surfaceLegend$i -row 0 -column 1
              grid $base.legendNo$i -in $base.surfaceLegend$i -row 0 -column 2 \
                  -sticky w
              label $base.legendTranslate$i -text "  Translate:" \
                 -font {times 12 bold}
              grid $base.legendTranslate$i -in $base.surfaceLegend$i \
                  -row 0 -column 4
              label $base.legendX$i -text " X" -font {times 12 bold}
              label $base.legendY$i -text " Y" -font {times 12 bold}
              grid $base.legendX$i -in $base.surfaceLegend$i -row 0 -column 5
              grid $base.legendY$i -in $base.surfaceLegend$i -row 0 -column 7
              scale $base.translateX$i -from 0 -to 600 -length 150 \
                -variable surface_legend_translation_x($i) -orient horizontal \
                -tickinterval 300 -showvalue true -resolution 1
              scale $base.translateY$i -from 0 -to 800 -length 200 \
                -variable surface_legend_translation_y($i) -orient horizontal \
                -tickinterval 400 -showvalue true -resolution 1
             grid $base.translateX$i -in $base.surfaceLegend$i -row 0 -column 6\
                  -sticky w
             grid $base.translateY$i -in $base.surfaceLegend$i -row 0 -column 8\
                  -sticky w
             button $base.surfaces$i -text Surfaces -borderwidth 4 \
                 -background #e8e
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

        proc OpenSurfaceFile {w k } {
                          global surface_file
                          global home_dir
                          if {[file exists $surface_file($k)] == 1} {
                           set surface_dir [file dirname $surface_file($k)]
                          } else {
                           set surface_dir $home_dir
                          }
                          set surface_file_temp [tk_getOpenFile \
                          -initialdir $surface_dir \
                          -initialfile [file tail $surface_file($k)] \
                          -parent $w \
                          -title "Open Surface File $k"]
                          if {$surface_file_temp != ""} {
                            set surface_file($k) $surface_file_temp
                          }
        }    


