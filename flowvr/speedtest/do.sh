#!/bin/bash

# for fancy logs: ./doChecks.sh >log 2>&1

../test/scripts/cleanup.sh

killall flowvrd
flowvr-kill

rm *.nc

P=3
Q=2
R=1


if [ -z "$START_TIME" ]; then
  export START_TIME=0.0
fi
if [ -z "$STOP_TIME" ]; then
  export STOP_TIME=0.01
fi

python ./mpi.py $P $Q $R $START_TIME $STOP_TIME
#flowvrd -s 3G & # do not need this line!
flowvrd &

# wait for flowvrd to startup
sleep 1
NumProcs=`echo $P*$Q*$R | bc`

tclsh ./mpi.tcl $P $Q $R --FlowVR # does all the preparation...


# replaces the call to pfrun:
sh $PARFLOW_DIR/bin/bootmc $NumProcs
sh $PARFLOW_DIR/bin/getmc $NumProcs

flowvr --batch-mode mpi

sh $PARFLOW_DIR/bin/freemc
sh $PARFLOW_DIR/bin/killmc


# replaces the call to pfundist
tclsh ./undist.tcl



killall flowvrd


echo ------------ END! --------------
