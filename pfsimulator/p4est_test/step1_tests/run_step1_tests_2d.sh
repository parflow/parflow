#! /bin/bash

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
#  P=2, Q=1     	Mx=6 My=12    --> 2 Trees and initial level 0

#	TEST 2
#
#  +-----+-----+-----+-----+
#  |     |     |     |     |
#  |     |     |     |     |
#  +-----+-----+-----+-----+
#
#  Nx = Ny = 14
#  PARLOW		PARFLOW + P4EST
#  P=4, Q=1     	Mx=3 My=14    --> 4 Trees and initial level 0
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
#  P=2, Q=2     	Mx=6 My=6   --> 1 Trees and initial level 1

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
#  P=4, Q=3     	Mx=3 My=4   --> 12 Trees and initial level 0


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
#  P=4, Q=4     	Mx=3 My=3   --> 1 Trees and initial level 2

# This set the tolerance for the squared difference of the pressure L2 error
# computed by parflow with and without using p4est.
EPS=1e-6

# each quoted text is a combination of Nx,Ny
pf_param_nxy=("12 12" "14 14" "12 12" "12 12" "12 12" "12 12")

# each quoted text is a combination of P,Q
pf_param_pq=("2 1" "4 1" "2 2" "4 3" "4 4")

# each quoted text is a combination of Mx,My
pf_param_mxy=("6 12" "3 14" "6 6" "3 4" "3 3")

# Required number of processors for the mpirun
np_arr=( 2 4 4 12 16 )

for (( i=0;i<5;i++ )); do

	T=$(echo "${i}+1" | bc -l)

	#Delete output from previous runs
	rm -rf output_test${T}
	mkdir  output_test${T}

	#Update parflow data base for each case
        tclsh test_brick_2d.tcl ${pf_param_nxy[i]} ${pf_param_pq[i]}
        tclsh test_brick_2d_with_p4est.tcl ${pf_param_nxy[i]} ${pf_param_mxy[i]}

	#Move the created data base to output directory
	mv *.pfidb output_test${T}/

	#run example and retrieve pressure l2 error
	echo  "Running TEST ${T}"
	cd  output_test${T}/
	mpirun -np ${np_arr[i]} ../../test_grid test_brick_2d\
		| grep pressure_l2_err > l2_err.out
	mpirun -np ${np_arr[i]} ../../test_grid test_brick_2d_with_p4est\
		| grep pressure_l2_err > l2_err_p4.out

	#Compare l2 error from both simulations
	E1=$(cat l2_err.out	| awk 'NR==1 {print $6}')
	E2=$(cat l2_err_p4.out	| awk 'NR==1 {print $6}')
	echo | awk -v E=$E1-$E2, -v TOL=$EPS '{if (E*E>TOL)\
		printf ( "FAILED \n\n"); else printf ("PASSED \n\n");}'
	cd ..
done
