#!/bin/bash

# for fancy logs: ./do.sh >log 2>&1

killall flowvrd
flowvr-kill

rm *.out.*
rm parflowvr.net.xml
rm parflowvr.run.xml
rm parflowvr.cmd.xml

# TODO: if other simulation data provider for visit are running this will cause problems!
rm ~/.visit/simulations/*.sim2

P=2
Q=1
R=1
PROBLEMNAME=fvr

DEBUGFLAGS="-v -v -v"
DEBUGFLAGS=""

python ./parflowvr.py $PROBLEMNAME $P $Q $R
#flowvrd -s 3G & # do not need this line!
flowvrd $DEBUGFLAGS &

# wait for flowvrd to startup
sleep 1
NumProcs=`echo $P*$Q*$R | bc`

tclsh ./flowvr_starter.tcl $PROBLEMNAME $P $Q $R  # does all the preparation...


# replaces the call to pfrun:
sh $PARFLOW_DIR/bin/bootmc $NumProcs
sh $PARFLOW_DIR/bin/getmc $NumProcs

flowvr parflowvr --batch-mode $DEBUGFLAGS

sh $PARFLOW_DIR/bin/freemc
sh $PARFLOW_DIR/bin/killmc


# replaces the call to pfundist
tclsh ../scripts/undist.tcl $PROBLEMNAME



killall flowvrd


echo ------------ END! --------------
