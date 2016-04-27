lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

# Test cases for step 1 in parflow-p4est integration.
#
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
#  +-----+-----+-----+-----+
#  |     |     |     |     |
#  |     |     |     |     |
#  +-----+-----+-----+-----+
#
#  Nx = Ny = 14
#  PARLOW		PARFLOW + P4EST
#  P=4, Q=1     	Mx=3 My=14,    --> 4 Trees and initial level 0
#
#  This case is important to test that the lower left corner of each
#  Subgrid has been computed correctly.

#	TEST 3
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

#	TEST 4
#
#  +-----+-----+-----+-----+
#  |     |     |     |	   |
#  |     |     |     |	   |
#  +-----+-----+-----+-----+
#  |     |     |     |	   |
#  |     |     |     |	   |
#  +-----+-----+-----+-----+
#  |     |     |     |     |
#  |     |     |     |     |
#  +-----+-----+-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=4, Q=3     	Mx=3 My=4    --> 12 Trees and initial level 0


#	TEST 5
#  +-----+-----+-----+-----+
#  |     |     |     |	   |
#  |     |     |     |	   |
#  +-----+-----+-----+-----+
#  |     |     |     |	   |
#  |     |     |     |	   |
#  +-----+-----+-----+-----+
#  |     |     |     |	   |
#  |     |     |     |	   |
#  +-----+-----+-----+-----+
#  |     |     |     |     |
#  |     |     |     |     |
#  +-----+-----+-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=4, Q=4     	Mx=3 My=3    --> 1 Trees and initial level 2


# each quoted text is a combination of Nx,Ny as described above
array set pf_param_nxy {
        1 "12 12"
        2 "14 14"
        3 "12 12"
        4 "12 12"
        5 "12 12"
}

# each quoted text is a combination of P,Q as described above
array set pf_param_pq {
        1 "2 1"
        2 "4 1"
        3 "2 2"
        4 "4 3"
        5 "4 4"
}

# each quoted text is a combination of Mx,My as described above
array set pf_param_mxy {
        1 "6 12"
        2 "3 14"
        3 "6 6"
        4 "3 4"
        5 "3 3"
}

# Required number of processors for each case
array set np_arr {
	1 "2"
        2 "4"
        3 "4"
        4 "12"
        5 "16"
}

for {set i 1} {$i < 6} {incr i} {

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
	puts "Running TEST $i"
	cd  output_2d_test${i}/
        exec mpirun -np $np_arr($i) ../../test_grid test_brick_2d
        exec mpirun -np $np_arr($i) ../../test_grid test_brick_2d_with_p4est

	#Colapse paralell output in single files
	pfundist test_brick_2d
	pfundist test_brick_2d_with_p4est
	
	#Compare pressure output file for both test cases	
        source ../compare_files.tcl
	set passed 1
	
	foreach t "00000 00001" {
		if ![pftestFile test_brick_2d.out.press.$t.pfb \
		test_brick_2d.out.press.$t.pfb \
		"Max difference in Pressure for timestep $t" $sig_digits] {
	    		set passed 0
		}
	}

	if $passed {
                puts "PASSED\n"
	} {
                puts "FAILED\n"
	}


        cd ..
}
