proc getChunkErase_ui {root args} {
    global tcl_dir
     source chunk_global.tcl
        # this treats "." as a special case

        if {$root == "."} {
            set base ""
        } else {
            set base $root
        }  
     set nx $chunk_n_x
     set ny $chunk_n_y
     set nz $chunk_n_z
     set nxy [expr $chunk_n_x * $chunk_n_y]
     for {set k 1 } {$k <= $nz} {incr k 1} { 
       frame $base.z$k -relief groove -borderwidth 2
       grid $base.z$k -row 0 -column $k -padx 5 -pady 5
       label $base.zLevel$k -text "Z = $k" -font {times 12 bold}
       grid $base.zLevel$k -in $base.z$k -row 0 -column 1 -columnspan $chunk_n_x
       for {set j 1 } {$j  <= $ny} {incr j 1} { 
         for {set i 1 } {$i <= $nx} {incr i 1} { 
           checkbutton $base.xyz($i,$j,$k) \
              -onvalue 1 -offvalue 0 -variable chunk_num($i,$j,$k) 
           grid $base.xyz($i,$j,$k) -in $base.z$k -row $j -column $i
         }
       }
     }
     
#
#   OKPS buttons
#
     frame $base.frameOKPS
     grid $base.frameOKPS -row 1 -column 1 -columnspan $nz -sticky ew
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
