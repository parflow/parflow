#! /bin/sh

#	TEST 1 
#
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#
#  Nx = Ny = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=1, R=1	Mx=6 My=12, Mz=1

#Update data parflow data base for each case
tclsh test_brick.tcl 2 1 1 
tclsh test_brick_with_p4est.tcl 6 12 1

#run example and retrieve pressure l2 error
mpirun -np 2 ../test_grid test_brick | grep pressure_l2_err > l2_err.out
mpirun -np 2 ../test_grid test_brick_with_p4est | grep pressure_l2_err > l2_err_p4.out

#Compare l2 error from both simulations
L1=$(cat l2_err.out | awk 'NR==1 {print $6}') 
L2=$(cat l2_err_p4.out | awk 'NR==1 {print $6}') 

echo | awk -v n1=$L1-$L2 '{if (n1>0) printf ( "TEST 1 FAILED \n");\
		else printf ("TEST 1 PASSED \n");}'

#	TEST 2 
#
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
#  |     |     |
#  |     |     |
#  +-----+-----+
