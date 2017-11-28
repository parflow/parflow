#!/bin/bash

# for fancy logs: ./doChecks.sh >log 2>&1

killall flowvrd
flowvr-kill

rm *.nc
rm parflowvr.net.xml
rm parflowvr.run.xml
rm parflowvr.cmd.xml

# TODO: if other simulation data provider for visit are running this will probably cause problems!?
rm ~/.visit/simulations/*.sim2

P=2
Q=1
R=1

#needed by common.tcl :
export START_TIME=0.0
export STOP_TIME=0.001

DEBUGFLAGS="-v -v -v"
DEBUGFLAGS=""

python ./parflowvr.py $P $Q $R $START_TIME $STOP_TIME
#flowvrd -s 3G & # do not need this line!
flowvrd $DEBUGFLAGS &

# wait for flowvrd to startup
sleep 1
NumProcs=`echo $P*$Q*$R | bc`

tclsh ./mpi.tcl $P $Q $R --FlowVR # does all the preparation...


# replaces the call to pfrun:
sh $PARFLOW_DIR/bin/bootmc $NumProcs
sh $PARFLOW_DIR/bin/getmc $NumProcs

flowvr parflowvr --batch-mode $DEBUGFLAGS

sh $PARFLOW_DIR/bin/freemc
sh $PARFLOW_DIR/bin/killmc


# replaces the call to pfundist
tclsh ./undist.tcl



killall flowvrd


echo ------------ END! --------------
