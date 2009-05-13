proc getFiles_ui {root args} {
#    
#       set global variables    
#
        global global_file
        source $global_file
# this treats "." as a special case
        if {$root == "."} {set base ""} else { set base $root}
#
#  Initialize Parameters
#
        set tcl_typelist "{{{TCL Files} {.tcl} {}}}"
        set ps_typelist "{\
                        {{PS Files} {.ps} {}} \
                        {{EPS Files} {.eps} {}}\
                        }" 
        set par_typelist "{{{Parameter Files} {.par} {}}}"
        set exe_typelist "{{{Executable Files} {.exe} {}}}"
#
#  define buttons
#    
        button $base.buttonOpenTcl \
                -background #e9e \
                -command "OpenTcl $base $tcl_typelist" \
                -borderwidth 4 \
                -text "tcl:" -font {times 12 bold}

        button $base.buttonOpenPS \
                -background #e9e \
                -command "OpenPS $base $ps_typelist" \
                -borderwidth 4 \
                -text "PS/EPS:" -font {times 12 bold} 

        button $base.buttonOpenExe \
                -background #e9e \
                -command "OpenExe $base $exe_typelist" \
                -borderwidth 4 \
                -text "exe:" -font {times 12 bold}

        button $base.buttonOpenPar \
                -background #e9e \
                -command "OpenPar $base $par_typelist" \
                -borderwidth 4 \
                -text "par:" -font {times 12 bold} 
     
        button $base.buttonSaveTcl \
                -background "#ff8" \
                -command "source save_tcl_file.tcl" \
                -borderwidth 4 \
                -text "Save" -font {times 12 bold} 

        button $base.buttonSaveAsTcl \
                -background "#e9e" \
                -borderwidth 4 \
                -command "\
                      SaveAsTclFile $base $tcl_typelist" \
                -text "Save As" -font {times 12 bold} 

        button $base.buttonFilesEnter \
                -background "#9f9" \
                -text OK -font {times 12 bold} \
                -borderwidth 4 \
                -command "destroy $root"

        button $base.psButton \
                -background #ff8040 \
                -borderwidth 4 \
                -command "source $chunk_run_file" \
                -text {Generate PS/EPS} -font {times 12 bold}

#
#  Entries
#
        entry $base.entryTcl -textvariable tcl_file -width 80 \
           -background #bdd -highlightcolor #f00  
        entry $base.entryPS  -textvariable ps_file -width 80 \
           -background #bdd -highlightcolor #f00  
        entry $base.entryExe -textvariable exe_file -width 80 \
           -background #bdd -highlightcolor #f00  
        entry $base.entryPar -textvariable par_file -width 80 \
           -background #bdd -highlightcolor #f00  

#
#  Grid
#
     grid $base.buttonOpenTcl $base.entryTcl \
          $base.buttonSaveTcl $base.buttonSaveAsTcl -sticky ew
     grid $base.buttonOpenPS $base.entryPS -sticky ew 
     grid $base.buttonOpenPar $base.entryPar -sticky ew 
     grid $base.buttonOpenExe -row 3 -column 0 -sticky ew
     grid $base.entryExe -row 3 -column 1 -sticky ew
     grid $base.psButton -row 1 -column 2 -columnspan 2 -sticky ew
     grid $base.buttonFilesEnter -row 2 -column 2 -columnspan 2 -rowspan 2 \
         -sticky nsew
}

#
# Processes
#
        proc OpenTcl {base typelist} {
            global global_file
            source $global_file
            set tcl_file_temp [tk_getOpenFile -defaultextension .tcl \
                    -filetypes $typelist \
                     -initialdir [file dirname $tcl_file] \
                     -initialfile [file tail $tcl_file] \
                     -parent $base \
                     -title {"TCL Files:"}]
            if {$tcl_file_temp != ""} {set tcl_file $tcl_file_temp}
            source $tcl_file
        }
        proc OpenPS {base typelist} {
            global ps_file
            set ps_file_temp [tk_getSaveFile -defaultextension .ps \
                    -filetypes $typelist \
                    -initialdir [file dirname $ps_file] \
                    -initialfile [file tail $ps_file] \
                    -parent $base \
                    -title {"TCL Files:"}]
            if {$ps_file_temp != ""} {set ps_file $ps_file_temp}
        }
        proc OpenPar {base typelist} {
            global par_file
            set par_file_temp [tk_getOpenFile -defaultextension .par \
                    -filetypes $typelist \
                    -initialdir [file dirname $par_file] \
                    -initialfile [file tail $par_file] \
                    -parent $base \
                    -title {"PAR Files:"}]
            if {$par_file_temp != ""} {set par_file $par_file_temp}
        }
        proc OpenExe {base typelist} {
            global exe_file
            set exe_file_temp [tk_getOpenFile -defaultextension .par \
                    -filetypes $typelist \
                    -initialdir [file dirname $exe_file] \
                    -initialfile [file tail $exe_file] \
                    -parent $base \
                    -title {"EXE Files:"}]
            if {$exe_file_temp != ""} {set exe_file $exe_file_temp}
        }

        proc SaveAsTclFile {base typelist} {
               global global_file
               source $global_file
               set file_temp [tk_getSaveFile -defaultextension ".tcl" \
                          -filetypes $typelist \
                          -initialdir [file dirname $tcl_file] \
                          -initialfile [file tail $tcl_file] \
                          -parent $base \
                          -title "Save As TCL" ] 
               if {$file_temp != ""} {
                 set tcl_file $file_temp
                 source save_tcl_file.tcl
               }
               return $tcl_file
        }

catch {
    if [info exists embed_args] {
        # we are running in the plugin
        getFiles_ui .
    } else {
        # we are running in stand-alone mode
        if {$argv0 == [info script]} {
            wm title . "Files"
            getFiles_ui .
        }
    }
}

