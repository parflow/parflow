#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

proc pfstrip {input_filename output_filename} {

    set input_file [open [lindex $argv 0] r]
    set output_file [open [lindex $argv 1] w]

    puts [lindex $argv 0]
    puts [lindex $argv 1]
    
    while { [gets $input_file line] >= 0} {
	if {![regexp "(^#.*$)|(^\ *$)" $line]} {
	    puts $output_file $line
	}
    }

    close $input_file
    close $output_file
}

set runname [lindex $argv 0]

pfstrip $runname.in.solver $runname.in.solver.strp
pfstrip $runname.in.problem $runname.in.problem.strp
