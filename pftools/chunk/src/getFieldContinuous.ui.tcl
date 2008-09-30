proc getFieldContinuous_ui {root args} {

      global field_value_min
      global field_value_max
      global field_color_table
      global field_n_cutoffs

      set field_n_cutoffs 0

	# this treats "." as a special case

	if {$root == "."} {
	    set base ""
	} else {
	    set base $root
	}
    
	frame $base.frame#1

	label $base.label#5 \
		-text {Field Data Range:}

	label $base.label#3 \
		-text min

	entry $base.entry#2 \
		-textvariable field_value_min \
		-width 8

	label $base.label#4 \
		-text max

	entry $base.entry#3 \
		-textvariable field_value_max \
		-width 8

	button $base.button#2 \
		-background green \
            -command "destroy $base" \
		-text enter

        button $base.buttonFieldColorTable \
                -background magenta \
                -command "OpenColorTableFile .get_Field" \
                -text "color table:" 
        proc OpenColorTableFile {root args} {
                          global global_file
                          source $global_file
                          set typelist {
                              {"CTB Files" {".ctb"} {"CTB "}}
                          }
                          set field_color_table [tk_getOpenFile -defaultextension .ctb \
                          -filetypes $typelist \
                          -initialdir $color_table_dir \
                          -initialfile $field_color_table \
                          -parent $root \
                          -title {"open color table file for field"}]
                          }

	entry $base.entryFieldColorTable \
		-textvariable field_color_table \
		-width 60


	# Geometry management

	grid $base.frame#1 -in $root	-row 1 -column 1 
	grid $base.label#5 -in $base.frame#1	-row 1 -column 1 
	grid $base.label#3 -in $base.frame#1	-row 1 -column 2  \
		-sticky e
	grid $base.entry#2 -in $base.frame#1	-row 1 -column 3 
	grid $base.label#4 -in $base.frame#1	-row 1 -column 4  \
		-sticky e
	grid $base.entry#3 -in $base.frame#1	-row 1 -column 5 
	grid $base.button#2 -in $base.frame#1	-row 1 -column 6  \
		-padx 30 \
		-sticky s
	grid $base.buttonFieldColorTable -in $root	-row 2 -column 1  \
		-sticky sw
	grid $base.entryFieldColorTable -in $root	-row 3 -column 1 

	# Resize behavior management

	grid rowconfigure $root 1 -weight 0 -minsize 30
	grid rowconfigure $root 2 -weight 0 -minsize 30
	grid rowconfigure $root 3 -weight 0 -minsize 30
	grid columnconfigure $root 1 -weight 0 -minsize 30

	grid rowconfigure $base.frame#1 1 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 1 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 2 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 3 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 4 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 5 -weight 0 -minsize 30
	grid columnconfigure $base.frame#1 6 -weight 0 -minsize 30
# additional interface code
# end additional interface code

}


# Allow interface to be run "stand-alone" for testing

catch {
    if [info exists embed_args] {
	# we are running in the plugin
	getFieldContinuous_ui .
    } else {
	# we are running in stand-alone mode
	if {$argv0 == [info script]} {
	    wm title . "Testing getFieldContinuous_ui"
	    getFieldContinuous_ui .
	}
    }
}
