lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

# Test cases for step 1 in parflow-p4est integration.
#
# The "test_brick_3d" script sets up the database for
# to execute the default version of Parflow and take
# the number of grid points (Nx,Ny,Nz) and the Processor
# grid (P,Q,R) as argument.
#
# The "test_brick_3d_with_p4est"  sets up the database to execute
# the Parflow version with p4est integrated and
# take as arguments the number of grid points (Nx,Ny,Nz) and
# subgrid points in each coordinate direction (Mx, My, Mz).

#	TEST 1
#
#  Nx = Ny = NZ = 12
#  PARLOW		PARFLOW + P4EST
#  P=1, Q=2 R=2    	Mx=12 My=6,, Mz=6  --> 4 Trees and initial level 0

#	TEST 2
#
#
#  Nx = Ny = NZ = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=2, R=2     	Mx=6 My=6 Mz=6    --> 1 Trees and initial level 1

#	TEST 3
#
#  Nx = Ny = Nz = 14
#  PARLOW		PARFLOW + P4EST
#  P=4, Q=4, R=1     	Mx=3 My=3 Mz=14    --> 16 Trees and initial level 2
#
#  This case is important to test that the lower left corner of each
#  Subgrid has been computed correctly. Additionally, test the case with
#  multiple trees and initial level > 0

# each quoted text is a combination of Nx,Ny as described above
array set pf_param_nxyz {
	1 "12 12 12"
	2 "12 12 12"
	3 "14 14 14"
}

# each quoted text is a combination of P,Q as described above
array set pf_param_pqr {
	1 "1 2 2"
	2 "2 2 2"
	3 "4 4 1"
}

# each quoted text is a combination of Mx,My as described above
array set pf_param_mxyz {
	1 "12 6 6"
	2 "6 6 6"
	3 "3 3 14"
}

# Required number of processors for each case
array set np_arr {
	1 "4"
	2 "8"
	3 "16"
}

for {set i 1} {$i < 4} {incr i} {

	#Purge output from previous runs
	exec rm -rf "output_3d_test${i}"
	exec mkdir  output_3d_test${i}

	#Update parflow data base for each case
	set  Nxyz [split $pf_param_nxyz($i) ]
	set  PQR [split $pf_param_pqr($i) ]
	exec tclsh test_brick_3d.tcl\
		[lindex $Nxyz 0] [lindex $Nxyz 1] [lindex $Nxyz 2]\
		[lindex $PQR 0] [lindex $PQR 1] [lindex $PQR 2]
	set  Mxyz [split $pf_param_mxyz($i)]
	exec tclsh test_brick_3d_with_p4est.tcl\
		[lindex $Nxyz 0] [lindex $Nxyz 1] [lindex $Nxyz 2]\
		[lindex $Mxyz 0] [lindex $Mxyz 1] [lindex $Mxyz 2]

	#Move the created data base to output directory
	exec mv {*}[glob *.pfidb] output_3d_test${i}/

	#run each example
	puts "Running TEST $i"
	cd  output_3d_test${i}/
	exec mpirun -np $np_arr($i) parflow test_brick_3d
	exec mpirun -np $np_arr($i) parflow test_brick_3d_with_p4est

	#Colapse paralell output in single files
	pfundist test_brick_3d
	pfundist test_brick_3d_with_p4est

	#Compare pressure output file for both test cases
	source ../compare_files.tcl
	set passed 1
        set sig_digits 4

	foreach t "00000 00001" {
                if  ![pftestFile test_brick_3d.out.press.$t.pfb \
                test_brick_3d_with_p4est.out.press.$t.pfb \
                "Max difference in Pressure for timestep $t" $sig_digits] {
			set passed 0
		}
	}

        if $passed {
		puts "PASSED\n"
        } else {
		puts "FAILED\n"
	}


	cd ..
}
