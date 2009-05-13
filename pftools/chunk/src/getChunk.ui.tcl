proc getChunk_ui {root args} {
    global tcl_dir
    source chunk_global.tcl
    global base
    # this treats "." as a special case
        
    if {$root == "."} {
	set base ""
    } else {
	set base $root
    }

#
#  Major Frames
#

    frame $base.frameChunk -relief groove -borderwidth 3
    frame $base.frameConcat -relief groove -borderwidth 3
    frame $base.frameOKPS 
    grid $base.frameChunk -row 0 -column 0 \
	-padx 5 -pady 5 -ipadx 5 -ipady 5
    grid $base.frameConcat -row 1 -column 0 \
	-padx 5 -pady 5 -ipadx 5 -ipady 5
    grid $base.frameOKPS -row 2 -column 0 -sticky ew 
    #
    #  Chunk
    #
    frame $base.chunkSpecs
    frame $base.chunkErase
    grid $base.chunkSpecs -in $base.frameChunk -row 0 -column 0
    grid $base.chunkErase -in $base.frameChunk -row 1 -column 0
    
    label $base.chunk -text "CHUNKS" -font {times 12 bold}
    label $base.chunkX -text X -font {times 12 bold}
    label $base.chunkY -text Y -font {times 12 bold}
    label $base.chunkZ -text Z -font {times 12 bold}
    label $base.chunkNumber -text Number -font {times 12 bold}
    label $base.chunkSpace -text "Space (units)" -font {times 12 bold}
    label $base.chunkCropMin -text "Crop Min" -font {times 12 bold}
    label $base.chunkCropMax -text "Crop Max" -font {times 12 bold}
    #        label $base.chunkNumberErase -text "Number of chunks to erase:" -font {times 12 bold}
    label $base.chunkListErase -text "List of chunks to erase:" -font {times 12 bold}
    source getChunkErase.ui.tcl
    button $base.chunkNumberErase -text "Chunks to Erase" \
	-font {times 12 bold} -borderwidth 4 -background "#e8e" \
	-command "toplevel .getChunkErase; focus .getChunkErase; \
                      corner .get_Chunk .getChunkErase; \
                      getChunkErase_ui .getChunkErase; \
                      grab .getChunkErase; \
                      wm title .getChunkErase {Chunks to Erase} ; \
                      tkwait window .getChunkErase" 
    
    grid $base.chunk -in $base.frameChunk -row 0 -column 0
    grid $base.chunkX -in $base.frameChunk -row 2 -column 0 
    grid $base.chunkY -in $base.frameChunk -row 3 -column 0 
    grid $base.chunkZ -in $base.frameChunk -row 4 -column 0 
    grid $base.chunkNumber -in $base.frameChunk -row 1 -column 1 \
	-columnspan 4 -sticky w
    grid $base.chunkSpace -in $base.frameChunk -row 1 -column 5 \
	-padx 10 -sticky w 
        grid $base.chunkCropMin -in $base.frameChunk -row 1 -column 6 \
	-columnspan 4 -sticky w
    grid $base.chunkCropMax -in $base.frameChunk -row 1 -column 10 \
	-columnspan 4 -sticky w
    
    entry $base.chunkNX -textvariable chunk_n_x -width 2 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkNY -textvariable chunk_n_y -width 2 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkNZ -textvariable chunk_n_z -width 2 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkSpaceX -textvariable chunk_space_x -width 12 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkSpaceY -textvariable chunk_space_y -width 12 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkSpaceZ -textvariable chunk_space_z -width 12 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMinX -textvariable chunk_crop_x_min -width 4 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMinY -textvariable chunk_crop_y_min -width 4 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMinZ -textvariable chunk_crop_z_min -width 4 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMaxX -textvariable chunk_crop_x_max -width 4 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMaxY -textvariable chunk_crop_y_max -width 4 \
	-background #bdd -highlightcolor #f00
    entry $base.chunkCropMaxZ -textvariable chunk_crop_z_max -width 4 \
	-background #bdd -highlightcolor #f00
    
    grid $base.chunkNX -in $base.frameChunk -row 2 -column 1
    grid $base.chunkNY -in $base.frameChunk -row 3 -column 1
    grid $base.chunkNZ -in $base.frameChunk -row 4 -column 1
    grid $base.chunkSpaceX -in $base.frameChunk -row 2 -column 5 -padx 10
    grid $base.chunkSpaceY -in $base.frameChunk -row 3 -column 5 -padx 10
    grid $base.chunkSpaceZ -in $base.frameChunk -row 4 -column 5 -padx 10
    grid $base.chunkCropMinX -in $base.frameChunk -row 2 -column 6
    grid $base.chunkCropMinY -in $base.frameChunk -row 3 -column 6
        grid $base.chunkCropMinZ -in $base.frameChunk -row 4 -column 6
    grid $base.chunkCropMaxX -in $base.frameChunk -row 2 -column 10 
    grid $base.chunkCropMaxY -in $base.frameChunk -row 3 -column 10 
    grid $base.chunkCropMaxZ -in $base.frameChunk -row 4 -column 10 
    
    grid $base.chunkNumberErase -in $base.frameChunk -row 5 -column 0 \
	-columnspan 13 -sticky sw -ipady 5
    
    radiobutton $base.chunkNX1 -text 1 -value 1 -variable chunk_n_x
    radiobutton $base.chunkNX2 -text 2 -value 2 -variable chunk_n_x
    radiobutton $base.chunkNX3 -text 3 -value 3 -variable chunk_n_x
    radiobutton $base.chunkNY1 -text 1 -value 1 -variable chunk_n_y
    radiobutton $base.chunkNY2 -text 2 -value 2 -variable chunk_n_y
    radiobutton $base.chunkNY3 -text 3 -value 3 -variable chunk_n_y
    radiobutton $base.chunkNZ1 -text 1 -value 1 -variable chunk_n_z
    radiobutton $base.chunkNZ2 -text 2 -value 2 -variable chunk_n_z
    radiobutton $base.chunkNZ3 -text 3 -value 3 -variable chunk_n_z
    radiobutton $base.chunkCropMinX0 -text 0 -value 0 \
	-variable chunk_crop_x_min
    radiobutton $base.chunkCropMinX1 -text 1 -value 1 \
	-variable chunk_crop_x_min
    radiobutton $base.chunkCropMinX10 -text 10 -value 10 \
	-variable chunk_crop_x_min
    radiobutton $base.chunkCropMinY0 -text 0 -value 0 \
	-variable chunk_crop_y_min
    radiobutton $base.chunkCropMinY1 -text 1 -value 1 \
	-variable chunk_crop_y_min
    radiobutton $base.chunkCropMinY10 -text 10 -value 10 \
                      -variable chunk_crop_y_min
    radiobutton $base.chunkCropMinZ0 -text 0 -value 0 \
                      -variable chunk_crop_z_min
    radiobutton $base.chunkCropMinZ1 -text 1 -value 1 \
	-variable chunk_crop_z_min
    radiobutton $base.chunkCropMinZ10 -text 10 -value 10 \
	-variable chunk_crop_z_min
    radiobutton $base.chunkCropMaxX0 -text 0 \
	-value 0 -variable chunk_crop_x_max
    radiobutton $base.chunkCropMaxX1 -text 1 \
	-value 1 -variable chunk_crop_x_max
    radiobutton $base.chunkCropMaxX10 -text 10 \
	-value 10 -variable chunk_crop_x_max
    radiobutton $base.chunkCropMaxY0 -text 0 \
	-value 0 -variable chunk_crop_y_max
        radiobutton $base.chunkCropMaxY1 -text 1 \
	-value 1 -variable chunk_crop_y_max
    radiobutton $base.chunkCropMaxY10 -text 10 \
	-value 10 -variable chunk_crop_y_max
    radiobutton $base.chunkCropMaxZ0 -text 0 \
                      -value 0 -variable chunk_crop_z_max
    radiobutton $base.chunkCropMaxZ1 -text 1 \
	-value 1 -variable chunk_crop_z_max
    radiobutton $base.chunkCropMaxZ10 -text 10 \
	-value 10 -variable chunk_crop_z_max
    
    #
    #  Grid the Chunk parameters
    #
    grid $base.chunkNX1 -in $base.frameChunk -row 2 -column 2
    grid $base.chunkNX2 -in $base.frameChunk -row 2 -column 3
    grid $base.chunkNX3 -in $base.frameChunk -row 2 -column 4
    grid $base.chunkNY1 -in $base.frameChunk -row 3 -column 2
    grid $base.chunkNY2 -in $base.frameChunk -row 3 -column 3
    grid $base.chunkNY3 -in $base.frameChunk -row 3 -column 4
    grid $base.chunkNZ1 -in $base.frameChunk -row 4 -column 2
    grid $base.chunkNZ2 -in $base.frameChunk -row 4 -column 3
    grid $base.chunkNZ3 -in $base.frameChunk -row 4 -column 4
    grid $base.chunkCropMinX0 -in $base.frameChunk -row 2 -column 7
    grid $base.chunkCropMinX1 -in $base.frameChunk -row 2 -column 8
    grid $base.chunkCropMinX10 -in $base.frameChunk -row 2 -column 9
    grid $base.chunkCropMinY0 -in $base.frameChunk -row 3 -column 7
    grid $base.chunkCropMinY1 -in $base.frameChunk -row 3 -column 8
    grid $base.chunkCropMinY10 -in $base.frameChunk -row 3 -column 9
    grid $base.chunkCropMinZ0 -in $base.frameChunk -row 4 -column 7
    grid $base.chunkCropMinZ1 -in $base.frameChunk -row 4 -column 8
    grid $base.chunkCropMinZ10 -in $base.frameChunk -row 4 -column 9
    grid $base.chunkCropMaxX0 -in $base.frameChunk -row 2 -column 11 
    grid $base.chunkCropMaxX1 -in $base.frameChunk -row 2 -column 12 
    grid $base.chunkCropMaxX10 -in $base.frameChunk -row 2 -column 13 
    grid $base.chunkCropMaxY0 -in $base.frameChunk -row 3 -column 11 
    grid $base.chunkCropMaxY1 -in $base.frameChunk -row 3 -column 12 
    grid $base.chunkCropMaxY10 -in $base.frameChunk -row 3 -column 13 
    grid $base.chunkCropMaxZ0 -in $base.frameChunk -row 4 -column 11 
    grid $base.chunkCropMaxZ1 -in $base.frameChunk -row 4 -column 12 
    grid $base.chunkCropMaxZ10 -in $base.frameChunk -row 4 -column 13 
    #
    #   Concat Parameters
    #
    label $base.concat -text "PERSPECTIVE" -font {times 12 bold}
    label $base.concatXX -text "X to X':" -font {times 12 bold}
    label $base.concatXY -text "X to Y':" -font {times 12 bold}
    label $base.concatYY -text "Y to Y':" -font {times 12 bold}
    label $base.concatYX -text "Y to X':" -font {times 12 bold}
    
    frame $base.radio
    radiobutton $base.radio3D -text "3-D" -font {times 12 bold}\
	-variable chunk_view -value 3D \
	-command "set block_transform_xx 0.93; \
                         set block_transform_xy -0.40; \
                         set block_transform_yy  0.40; \
                         set block_transform_yx  0.93 "  
    radiobutton $base.radioXY -text "XY plane" -font {times 12 bold} \
	-variable chunk_view -value XY \
	-command "set block_transform_xx  1.00;\
                         set block_transform_xy  0.00; \
                         set block_transform_yy  1.00; \
                         set block_transform_yx  0.00; \
                         set chunk_erase_n 0 "  
    radiobutton $base.radioXZ -text "XZ plane" -font {times 12 bold} \
	-variable chunk_view -value XZ \
	-command "set block_transform_xx 1.00; \
                         set block_transform_xy  0.00; \
                         set block_transform_yy  0.00; \
                         set block_transform_yx  0.00; \
                         set chunk_erase_n 0 "  
    radiobutton $base.radioYZ -text "YZ plane" -font {times 12 bold} \
	-variable chunk_view -value YZ \
	-command "set block_transform_xx 0.00; \
                         set block_transform_xy  0.00; \
                         set block_transform_yy  0.00; \
                         set block_transform_yx  1.00; \  
                         set chunk_erase_n 0 "  
    entry $base.entryConcatXX -textvariable block_transform_xx -width 6 \
	-background #bdd -highlightcolor #f00
    entry $base.entryConcatXY -textvariable block_transform_xy -width 6 \
	-background #bdd -highlightcolor #f00
    entry $base.entryConcatYY -textvariable block_transform_yy -width 6 \
	-background #bdd -highlightcolor #f00
    entry $base.entryConcatYX -textvariable block_transform_yx -width 6 \
	-background #bdd -highlightcolor #f00
    
    scale $base.scaleConcatXX -from 0.00 -to 1.00 -length 400 \
	-variable block_transform_xx -orient horizontal \
	-tickinterval 0.2 -showvalue true -digits 3 -resolution 0.01
    scale $base.scaleConcatXY -from -1.00 -to 0.00 -length 400 \
	-variable block_transform_xy -orient horizontal \
	-tickinterval 0.2 -showvalue true -digits 3 -resolution 0.01
    scale $base.scaleConcatYY -from 0.00 -to 1.00 -length 400 \
	-variable block_transform_yy -orient horizontal \
	-tickinterval 0.2 -showvalue true -digits 3 -resolution 0.01
    scale $base.scaleConcatYX -from 0.00 -to 1.00 -length 400 \
	-variable block_transform_yx -orient horizontal \
            -tickinterval 0.2 -showvalue true -digits 3 -resolution 0.01
    grid $base.concat -in $base.frameConcat -row 0 -column 0
    grid $base.radio -in $base.frameConcat -row 0 -column 2
    grid $base.radio3D -in $base.radio -row 0 -column 0
    grid $base.radioXY -in $base.radio -row 0 -column 1
    grid $base.radioXZ -in $base.radio -row 0 -column 2
    grid $base.radioYZ -in $base.radio -row 0 -column 3
    
    grid $base.concatXX -in $base.frameConcat -row 1 -column 0
    grid $base.concatXY -in $base.frameConcat -row 2 -column 0
    grid $base.concatYY -in $base.frameConcat -row 3 -column 0
    grid $base.concatYX -in $base.frameConcat -row 4 -column 0
    grid $base.entryConcatXX -in $base.frameConcat -row 1 -column 1
    grid $base.entryConcatXY -in $base.frameConcat -row 2 -column 1
    grid $base.entryConcatYY -in $base.frameConcat -row 3 -column 1
    grid $base.entryConcatYX -in $base.frameConcat -row 4 -column 1
    grid $base.scaleConcatXX -in $base.frameConcat -row 1 -column 2
    grid $base.scaleConcatXY -in $base.frameConcat -row 2 -column 2
    grid $base.scaleConcatYY -in $base.frameConcat -row 3 -column 2
    grid $base.scaleConcatYX -in $base.frameConcat -row 4 -column 2
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
    
    
    # additional interface code
    # end additional interface code
    
}


# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getChunk_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getChunk_ui"
	    getChunk_ui .
	}
    }
}
