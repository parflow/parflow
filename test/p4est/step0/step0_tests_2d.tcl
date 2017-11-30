lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

# Test cases for step 0 in parflow-p4est integration.
# The goal of this case is enable to enable parflow to
# run in serial with multiple subgrids. To run this test
# suite just execute tclsh step0_tests.tcl in the current
# directory.

# The "test_brick_2d" script sets up the database for
# to execute the default version of Parflow and take
# the number of grid points (Nx,Ny) and the Processor
# grid (P,Q) as argument.
#
# The "test_brick_2d_with_p4est"  sets up the database to execute
# the Parflow version with p4est integrated and
# take as arguments the number of grid points (Nx,Ny) and
# subgrid points in each coordinate direction (Mx, My).

#	TEST 1
#
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=1     	Mx=6 My=12,    --> 2 Trees and initial level 0

#	TEST 2
#
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=2     	Mx=6 My=6     --> 1 Trees and initial level 1

#	TEST 3
#  +-----+-----+-----+
#  |     |     |     |
#  |     |     |     |
#  +-----+-----+-----+
#  |     |     |     |
#  |     |     |     |
#  +-----+-----+-----+
#  |     |     |     |
#  |     |     |     |
#  +-----+-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=3, Q=3     	Mx=4 My=4    --> 9 Trees and initial level 0

# each quoted text is a combination of Nx,Ny as described above
array set pf_param_nxy {
        1 "12 12"
        2 "12 12"
	3 "12 12"
}

# each quoted text is a combination of P,Q as described above
array set pf_param_pq {
        1 "2 1"
	2 "2 2"
	3 "3 3"
}

# each quoted text is a combination of Mx,My as described above
array set pf_param_mxy {
        1 "6 12"
        2 "6 6"
	3 "4 4"
}

# Required number of processors for stantard ParFLow
array set np_arr {
	1 "2"
        2 "4"
	3 "9"
}

for {set i 1} {$i < 4} {incr i} {

	#Purge output from previous runs
	exec rm -rf "output_2d_test${i}"
	exec mkdir  output_2d_test${i}

	#Update parflow data base for each case
        set  Nxy [split $pf_param_nxy($i) ]
        set  PQ [split $pf_param_pq($i) ]
	exec tclsh test_brick_2d.tcl\
                [lindex $Nxy 0] [lindex $Nxy 1]\
                [lindex $PQ 0] [lindex $PQ 1]
        set  Mxy [split $pf_param_mxy($i)]
	exec tclsh test_brick_2d_with_p4est.tcl\
                [lindex $Nxy 0] [lindex $Nxy 1]\
                [lindex $Mxy 0] [lindex $Mxy 1]
	
        #Move the created data base to output directory
	exec mv {*}[glob *.pfidb] output_2d_test${i}/

	#run each example
	puts "Running step0_2d_test $i"
	cd  output_2d_test${i}/
        exec mpirun -np $np_arr($i) $Parflow::PARFLOW_DIR/bin/parflow test_brick_2d
	#We will run them in serial with p4est
        exec mpirun -np 1 $Parflow::PARFLOW_DIR/bin/parflow test_brick_2d_with_p4est

	#Colapse paralell output in single files
	pfundist test_brick_2d
	pfundist test_brick_2d_with_p4est
	
	#Compare pressure output file for both test cases	
        source ../compare_files.tcl
	set passed 1
	
	foreach t "00000 00001" {
                if ![pftestFile test_brick_2d.out.press.$t.pfb \
                test_brick_2d_with_p4est.out.press.$t.pfb \
                "Max difference in Pressure for timestep $t" $sig_digits] {
	    		set passed 0
		}
	}

        if $passed {
                puts "\n\n\n\nstep0_2d_test $i : PASSED"
        } else {
                puts "\n\n\n\nstep0_2d_test $i : FAILED"
	}
	puts "*****************************************************************************"

        cd ..
}
