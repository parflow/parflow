#!/bin/bash -x


../scripts/cleanup.sh

killall flowvrd
flowvr-kill

rm *.nc

rm -rf results
mkdir results

export START_TIME=0.0
export STOP_TIME=0.009

# put good results into results folder (run tclsh default_richards_witsh_netcdf.tcl 1 1 1 without flowvr once ;) )
tclsh ./default_richards_with_netcdf.tcl 1 1 1

tclsh ./scripts/_tests_noflowvr.tcl

rm -rf results_noFlowVR
cp -r results results_noFlowVR


# now lets run it with parflow
# wait, type stop manually /// shoot it
echo "Now starting flowVR . this will take some time. When the output does not change further, wait a few seconds and type \"stop\" or \"s\" and hit [enter]"
read -n1 -r -p "Press any key to continue..." key

python ./parFlowVR.py $START_TIME $STOP_TIME
flowvrd -s 4M &

# wait for flowvrd to startup
sleep 1

flowvr --batch-mode parflowvr


killall flowvrd

tclsh ./scripts/_tests_noflowvr.tcl
# and compare the results!
echo Compare results. Diffs in Time are ok up to now as we transmit timestamps as floats atm. So we loose some prec. Diffs in the other variables are not ok..

../scripts/compare_nc.py ./default_richards.out.00000.nc results/default_richards.out.00000.nc
../scripts/compare_nc.py ./default_richards.out.00001.nc results/default_richards.out.00001.nc
../scripts/compare_nc.py ./default_richards.out.00000.nc results_noFlowVR/default_richards.out.00000.nc
../scripts/compare_nc.py ./default_richards.out.00001.nc results_noFlowVR/default_richards.out.00001.nc

echo ------------ END! --------------
