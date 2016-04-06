lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

# each quoted text is a combination of P,Q,R
array set pf_param { 
	1 "2 1 1" 
	2 "2 2 1" 
	3 "4 3 1"
	4 "4 4 1"
}

# each quoted text is a combination of Mx,My,Mz
array set pf_param_with_p4est { 
	1 "6 12 1" 
	2 "6 6 1" 
	3 "3 4 1" 
	4 "3 3 1"
}

# Required number of processors for the mpirun
array set np_arr {
	1 "2"
	2 "4"
	3 "12"
	4 "16" 
}

for {set i 3} {$i < 5} {incr i} {

	#Purge output from previous runs
	exec rm -rf "output_test${i}"
	exec mkdir -p output_test${i}/test_brick_2d.out/
	exec mkdir -p output_test${i}/test_brick_2d_with_p4est.out/

	#Update parflow data base for each case
	set  PQR [split $pf_param($i) ]
	exec tclsh test_brick_2d.tcl\
		[lindex $PQR 0] [lindex $PQR 1] [lindex $PQR 2]
	set  Mxyz [split $pf_param_with_p4est($i)]
	exec tclsh test_brick_2d_with_p4est.tcl\
		[lindex $Mxyz 0] [lindex $Mxyz 1] [lindex $Mxyz 2]
	
	#run each example
	puts "Running TEST $i"
	exec mpirun -np $np_arr($i) ../test_grid test_brick_2d
	exec mpirun -np $np_arr($i) ../test_grid test_brick_2d_with_p4est

	#Colapse paralell output in single files
	pfundist test_brick_2d
	pfundist test_brick_2d_with_p4est
	
	#Compare pressure output file for both test cases	
	source compare_files.tcl
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

	# move output to suitable subdirectories for further use
	exec mv -f test_brick_2d.out/\
		output_test${i}/test_brick_2d.out/
	exec mv -f test_brick_2d_with_p4est.out/\
		output_test${i}/test_brick_2d_with_p4est.out
	exec mv {*}[glob *.out*] output_test${i}
}
