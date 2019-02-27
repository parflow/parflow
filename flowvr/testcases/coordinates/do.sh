#!/bin/bash

here=$PWD

# create reference:
mkdir -p reference
cd reference
rm *.nc -f
tclsh ../hillslope_sens.tcl && $PARFLOW_DIR/bin/parflow hillslope_sens

cd $here

killall flowvrd
flowvr-kill
python hillslope_sens.py
tclsh $PARFLOW_DIR/bin/parflowvr/flowvr_starter.tcl 1 1 1 hillslope_sens  # does all the preparation...

flowvrd &


flowvr hillslope_sens --batch

killall flowvrd
flowvr-kill

echo .
echo .
err=`cat logger.log`
if [ "$err" != "error: 0.00000000E+00" ];
then
  echo ERROR!
  echo $err
  exit 1
fi;
echo PASSED!
exit 0
