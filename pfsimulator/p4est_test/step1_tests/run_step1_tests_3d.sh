#! /bin/bash

# Test cases for step 1 in parflow-p4est integration.
#
# The "test_brick_3d" script sets up the database for
# to execute the default version of Parflow and take the Processor
# grid (P,Q,R) as argument.
#
# The "test_brick_3d_with_p4est"  sets up the database to execute
# the Parflow version with p4est integrated and
# take as arguments the number of subgrid points in each
# coordinate direction (Mx, My ,Mz)

#	TEST 1 
#
#
#  Nx = Ny = Nz = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=2, R=2	Mx=6 My=6, Mz=6  --> 1 Trees and initial level 1
#	TEST 2
#
#  Nx = Ny = Nz = 12
#  PARLOW		PARFLOW + P4EST
#  P=2, Q=4, R=2	Mx=6 My=3, Mz=6  --> 2 Trees and initial level 1

#	TEST 3
#
#
#  Nx = Ny = Nz = 12
#  PARLOW		PARFLOW + P4EST
#  P=3, Q=3, R=3	Mx=4 My=4, Mz=4 --> 27 Trees and initial level 0

# This set the tolerance for the squared difference of the pressure L2 error
# computed by parflow with and without using p4est.
EPS=1e-6

# each quoted text is a combination of P,Q,R
pf_param=( "2 2 2" "2 4 2" "4 3 1")

# each quoted text is a combination of Mx,My,Mz
pf_param_with_p4est=( "6 6 6" "6 3 6" "4 4 4")

# Required number of processors for the mpirun
np_arr=( 8 16 27 )

for (( i=0;i<3;i++ )); do
	#Update parflow data base for each case
	tclsh test_brick_3d.tcl ${pf_param[i]}
	tclsh test_brick_3d_with_p4est.tcl ${pf_param_with_p4est[i]}

	T=$(echo "${i}+1" | bc -l)
	#run example and retrieve pressure l2 error
	echo  "Running TEST ${T}"
	mpirun -np ${np_arr[i]} ../test_grid test_brick_3d\
		| grep pressure_l2_err > l2_err.out
	mpirun -np ${np_arr[i]} ../test_grid test_brick_3d_with_p4est\
		| grep pressure_l2_err > l2_err_p4.out

	#Compare l2 error from both simulations
	E1=$(cat l2_err.out	| awk 'NR==1 {print $6}')
	E2=$(cat l2_err_p4.out	| awk 'NR==1 {print $6}')
	echo | awk -v E=$E1-$E2, -v TOL=$EPS '{if (E*E>TOL)\
		printf ( "FAILED \n\n"); else printf ("PASSED \n\n");}'
done
