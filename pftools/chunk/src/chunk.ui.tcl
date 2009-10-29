#!/usr/local/bin/wish
#   root     is the parent window for this user interface

#
#  Initialize home directory and defaults
#
source $env(PARFLOW_DIR)/pftools/chunk/src/chunk_global.tcl
set home_dir $env(PARFLOW_DIR)/pftools/chunk
set src_dir $home_dir/src
set tcl_dir $home_dir/tcl
set code_dir $home_dir/code
cd $src_dir

source "chunk_defaults.tcl"

proc chunk_ui {root args } {
    global tcl_dir
        # this treats "." as a special case

	if {$root == "."} { set base ""} else { set base $root }
#
#  Source sub-window processes
#
        source chunk_global.tcl
        source getFiles.ui.tcl
        source getPlot.ui.tcl
        source getCoord.ui.tcl
        source getField.ui.tcl
        source getChunk.ui.tcl
        source getSurfaces.ui.tcl
        source getLines.ui.tcl
        source getDots.ui.tcl
        source getEPS.ui.tcl
#
#  Define Labels
#
        label $base.view -text "Configure:" -font {times 12 bold}
        label $base.data -text "Data:" -font {times 12 bold}
        label $base.postscript -text "PostScript:" -font {times 12 bold}
    
#
#  Define buttons
#    
	button $base.fileButton \
                -command "toplevel .get_Files; \ 
                         wm title .get_Files {File Names}; \
                         corner . .get_Files ; \
                         focus .get_Files; getFiles_ui \
                        .get_Files; grab .get_Files; tkwait window .get_Files" \
		-background #e8e -borderwidth 4 \
                -text "I/O" -font {times 12 bold}

        proc corner { root base } {
            set xw [winfo rootx $root]
            set yw [winfo rooty $root] 
            wm geometry $base "+$xw+$yw" 
        }
         

	button $base.plotButton \
                -command "toplevel .get_Plot; \ 
                         wm title .get_Plot {Plot Configuration}; \
                         corner . .get_Plot ; \
                         focus .get_Plot; getPlot_ui \
                         .get_Plot; grab .get_Plot; tkwait window .get_Plot" \
		-background #e8e -borderwidth 4 \
		-text Plot -font {times 12 bold}

	button $base.blockButton \
                -command "toplevel .get_Coord; \
                         wm title .get_Coord {Coordinate System}; \
                         corner . .get_Coord ; \
                         focus .get_Coord; getCoord_ui \
                        .get_Coord; grab .get_Coord; tkwait window .get_Coord" \
		-background #e8e -borderwidth 4 \
		-text Coord -font {times 12 bold} 

	button $base.chunkButton \
                -command "toplevel .get_Chunk; \
                         wm title .get_Chunk {Chunk Configuration}; \
                         corner . .get_Chunk ; \
                         focus .get_Chunk; getChunk_ui \
                        .get_Chunk; grab .get_Chunk; tkwait window .get_Chunk" \
		-background #e8e -borderwidth 4 \
		-text Chunk -font {times 12 bold}

        button .epsButton \
                -background #e8e \
                -borderwidth 4 \
                -command "toplevel .get_EPS; \
                       wm title .get_EPS {Make EPS PostScript File}; \
                         corner . .get_EPS ; \
                         focus .get_EPS; getEPS_ui \
                        .get_EPS; grab .get_EPS; tkwait window .get_EPS" \
                -text {EPS} -font {times 12 bold}
        button .psButton \
                -background #ff8040 \
                -borderwidth 4 \
                -command "source $chunk_run_file" \
                -text {Generate PS/EPS} -font {times 12 bold}
        button .gsButton \
                -background #f01010 \
                -borderwidth 4 \
                -command {GSview_run "$ps_file"} \
                -text {GSview} -font {times 12 bold}
  
        proc GSview_run {ps_file} {
            set fileId [open GSview_exe.txt r]
            gets $fileId GSview_exe
            exec $GSview_exe $ps_file & 
            close $fileId
        }

        proc GeneratePS {args} {
                          global global_file
                          source "$global_file"
                          source "$chunk_run_file"
                          }
#
#  Frames
#
        frame .field -relief groove -borderwidth 4
        frame .surfaces -relief groove -borderwidth 4
        frame .lines -relief groove -borderwidth 4
        frame .dots -relief groove -borderwidth 4
#
#  Geometry Management
#
        grid .view -row 0 -column 0 -sticky w
        grid .data -row 1 -column 0 -sticky nw -ipady 10
        grid .postscript -row 2 -column 0 -sticky w 
	grid .fileButton -row 0 -column 1 -sticky nesw 
	grid .plotButton -row 0 -column 2 -sticky nesw 
	grid .blockButton -row 0 -column 3 -sticky nesw 
	grid .chunkButton -row 0 -column 4 -sticky nesw
	grid .field -row 1 -column 1 -sticky nesw -padx 2 -pady 2
	grid .surfaces -row 1 -column 2 -sticky nesw -padx 2 -pady 2
	grid .lines  -row 1 -column 3 -sticky nesw -padx 2 -pady 2
	grid .dots -row 1 -column 4 -sticky nesw -padx 2 -pady 2
	grid .epsButton -row 2 -column 1 -sticky nesw
	grid .psButton -row 2 -column 2 -columnspan 2 -sticky nesw
	grid .gsButton -row 2 -column 4 -sticky nesw
#
# Field Frame 
#
	button .fieldButton \
                -command "toplevel .get_Field; wm title .get_Field {Field}; \
                         corner . .get_Field ; \
                         focus .get_Field; getField_ui \
                        .get_Field; grab .get_Field; tkwait window .get_Field" \
		-background #8ee -borderwidth 4 \
		-text Field -font {times 12 bold}
      label .plot -text "Plot?" -font {times 12 bold}
      radiobutton .fieldYes -text yes -value yes -variable field_plot 
      radiobutton .fieldNo -text no -value no -variable field_plot 

	pack .fieldButton -in .field -side top  
      pack .plot -in .field -side top 
      pack .fieldYes -in .field -side top -anchor w
      pack .fieldNo -in .field -side top -anchor w
#
# Surfaces Frame
#
	button .surfacesButton \
                -command "toplevel .get_Surfaces; \
                         wm title .get_Surfaces {Surfaces}; \
                         corner . .get_Surfaces ; \
                         focus .get_Surfaces; \
                         getSurfaces_ui .get_Surfaces; \
                         grab .get_Surfaces; tkwait window .get_Surfaces" \
		-background #8ee -borderwidth 4 \
		-text Surfaces -font {times 12 bold}
	grid .surfacesButton -in .surfaces -row 0 -column 0 
        button .plotNSurfaceFiles -text "Files" \
            -font {times 12 bold} \
            -command "toplevel .get_Surface_Files; \
                      wm title .get_Surface_Files {Surface Files}; \
                      corner . .get_Surface_Files ; \
                      focus .get_Surface_Files; get_Surface_Files_ui \
                     .get_Surface_Files; grab .get_Surface_Files; \
                     tkwait window .get_Surface_Files" \
            -borderwidth 4 \
            -background "#e8e"
        grid .plotNSurfaceFiles -in .surfaces -row 2 -column 0
        source getSurfaceFiles.ui.tcl

        scale $base.scaleNSurfaceFiles -from 0 -to 4 -length 60 \
            -variable surface_n_files -orient horizontal \
            -tickinterval 4 -showvalue true -sliderlength 16
        grid .scaleNSurfaceFiles -in .surfaces \
                   -row 1 -column 0 
#
# Lines Frame
#
	button .linesButton \
                -command "toplevel .get_Lines; wm title .get_Lines {Lines}; \
                        corner . .get_Lines ; \
                        focus .get_Lines; getLines_ui \
                        .get_Lines; grab .get_Lines; tkwait window .get_Lines" \
		-background #8ee -borderwidth 4 \
		-text Lines -font {times 12 bold}

	pack .linesButton  -in .lines -side top 

      scale $base.scaleNLineFiles -from 0 -to 4 -length 60 \
            -variable line_n_files -orient horizontal \
            -tickinterval 4 -showvalue true -sliderlength 16
      pack .scaleNLineFiles -in .lines -side top 
      source getLines.ui.tcl
#
# Dots Frame
#
	button .dotsButton \
                -command "toplevel .get_Dots; wm title .get_Dots {Dots}; \
                          corner . .get_Dots ; \
                          focus .get_Dots; getDots_ui \
                         .get_Dots; grab .get_Dots; tkwait window .get_Dots" \
		-background #8ee -borderwidth 4 \
		-text Dots -font {times 12 bold}
	pack .dotsButton -in .dots -side top
      scale $base.scaleNDotsFiles -from 0 -to 4 -length 60 \
            -variable dots_n_files -orient horizontal \
            -tickinterval 4 -showvalue true -sliderlength 16
      pack .scaleNDotsFiles -in .dots -side top
      source getDots.ui.tcl 
}

wm title . "Chunk - Version 4.0 - All Rights Reserved"
wm geometry . "+0+0"
chunk_ui . 
