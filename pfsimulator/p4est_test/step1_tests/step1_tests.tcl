lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

# Test cases for step 1 in parflow-p4est integration.
#
# The "test_brick_2d" script sets up the database for
# to execute the default version of Parflow and take the Processor
# grid (P,Q,R) as argument.
#
# The "test_brick_2d_with_p4est"  sets up the database to execute
# the Parflow version with p4est integrated and
# take as arguments the number of subgrid points in each
# coordinate direction (Mx, My ,Mz)

#	TEST 1
#
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=1, R=1	Mx=6 My=12, Mz=1  --> 2 Trees and initial level 0

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
#  P=2, Q=2, R=1	Mx=6 My=6, Mz=1  --> 1 Trees and initial level 1

#	TEST 3
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
#  P=4, Q=3, R=1	Mx=3 My=4, Mz=1 --> 12 Trees and initial level 0


#	TEST 4
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
#  P=4, Q=4, R=1	Mx=3 My=3, Mz=1 --> 1 Trees and initial level 2


# each quoted text is a combination of P,Q,R as described above
array set pf_param { 
	1 "2 1 1" 
	2 "2 2 1" 
	3 "4 3 1"
	4 "4 4 1"
}

# each quoted text is a combination of Mx,My,Mz as described above
array set pf_param_with_p4est { 
	1 "6 12 1" 
	2 "6 6 1" 
	3 "3 4 1" 
	4 "3 3 1"
}

# Required number of processors for each case
array set np_arr {
	1 "2"
	2 "4"
	3 "12"
	4 "16" 
}

for {set i 1} {$i < 5} {incr i} {

	#Purge output from previous runs
        exec rm -rf "output_test${i}"
        exec mkdir  output_test${i}

	#Update parflow data base for each case
	set  PQR [split $pf_param($i) ]
	exec tclsh test_brick_2d.tcl\
		[lindex $PQR 0] [lindex $PQR 1] [lindex $PQR 2]
	set  Mxyz [split $pf_param_with_p4est($i)]
	exec tclsh test_brick_2d_with_p4est.tcl\
		[lindex $Mxyz 0] [lindex $Mxyz 1] [lindex $Mxyz 2]
	
        #Move the created data base to output directory
        exec mv {*}[glob *.pfidb] output_test${i}/

	#run each example
	puts "Running TEST $i"
        cd  output_test${i}/
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
		puts "PASSED"
	} {
		puts "FAILED"
	}


        cd ..
}
