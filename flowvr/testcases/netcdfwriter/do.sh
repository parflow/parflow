#!/bin/bash -x



$PARFLOW_DIR/bin/parflowvr/cleanup-flowvr-xml.sh

killall flowvrd
flowvr-kill

rm *.nc

# put good results into results folder (run tclsh default_richards_witsh_netcdf.tcl 1 1 1 without flowvr once ;) )
tclsh ./default_richards_with_netcdf.tcl 1 1 1

errors=0

# _tests_noflowvr.tcl needs the correct_output folder:
ln -s ../../../../test/correct_output results/
tclsh ./scripts/_tests_noflowvr.tcl
errors=$errors+$?

rm -rf results_noFlowVR
mv results results_noFlowVR


# now lets run it with parflow

python ./parflowvr.py
flowvrd -s 4M &

# wait for flowvrd to startup
sleep 1

flowvr --batch-mode parflowvr


killall flowvrd

# _tests_noflowvr.tcl needs the correct_output folder:
ln -s ../../../../test/correct_output results/
tclsh ./scripts/_tests_noflowvr.tcl
errors=$errors+$?

# and compare the results!
echo Compare results. Diffs in Time are ok up to now as we transmit timestamps as floats atm. So we loose some prec. Diffs in the other variables are not ok..

$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./pressure.default_richards.out.00000.nc results/default_richards.out.00000.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./pressure.default_richards.out.00001.nc results/default_richards.out.00001.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./pressure.default_richards.out.00000.nc results_noFlowVR/default_richards.out.00000.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./pressure.default_richards.out.00001.nc results_noFlowVR/default_richards.out.00001.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./saturation.default_richards.out.00000.nc results/default_richards.out.00000.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./saturation.default_richards.out.00001.nc results/default_richards.out.00001.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./saturation.default_richards.out.00000.nc results_noFlowVR/default_richards.out.00000.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./saturation.default_richards.out.00001.nc results_noFlowVR/default_richards.out.00001.nc -vague
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./multi.default_richards.out.00000.nc results_noFlowVR/default_richards.out.00000.nc
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py ./multi.default_richards.out.00001.nc results_noFlowVR/default_richards.out.00001.nc
errors=$errors+$?


errorcount=`echo $errors | bc`
echo -------- $errorcount errors! -------
echo ------------ END! --------------

exit $errorcount
