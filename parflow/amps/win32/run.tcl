#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

proc pfstrip {input_filename output_filename} {

    set input_file [open $input_filename r]
    set output_file [open $output_filename w]

    while { [gets $input_file line] >= 0} {
	if {![regexp "(^#.*$)|(^\ *$)" $line]} {
	    puts $output_file $line
	}
    }

    close $input_file
    close $output_file
}

set program "parflow"

set runname [lindex $argv 0]

# If user did not specify process group assume none
if {$argc > 1} {
    set processgroup [lindex $argv 1]
} {
    set processgroup ""
}


# Get the number of processor to run on
set file [open .amps.info.$processgroup]
gets $file num_procs
close $file

#pfstrip $runname.in.solver $runname.in.solver.strp
#pfstrip $runname.in.problem $runname.in.problem.strp

exec $env(PARFLOW_DIR)/bin/$program $num_procs $runname

#file delete $runname.in.solver.strp $runname.in.problem.strp
