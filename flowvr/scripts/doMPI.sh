#!/bin/bash
#
# run parflow in MPI with flowvr.
#
#
# need to get PROBLEMNAME, P, Q and R from parameters!
#
# for fancy logs: ./doMPI.sh >log 2>&1
#
#

if [ -z "$4" ];
then
  echo Error! could not find the needed amount of parameters!
  exit 1
fi

PROBLEMNAME=$1
P=$2
Q=$3
R=$4

killall flowvrd
flowvr-kill

rm $PROBLEMNAME.out.*
rm $PROBLEMNAME.pfidb
rm parflowvr.net.xml
rm parflowvr.run.xml
rm parflowvr.cmd.xml

# TODO: if other simulation data provider for visit are running this will cause problems!
rm ~/.visit/simulations/*.sim2


python ./parflowvr.py $PROBLEMNAME $P $Q $R
#flowvrd -s 3G & # do not need this line!
flowvrd $DAEMONFLAGS $DEBUGFLAGS &

# wait for flowvrd to startup
sleep 1
NumProcs=`echo $P*$Q*$R | bc`

tclsh $PARFLOW_DIR/bin/parflowvr/flowvr_starter.tcl $P $Q $R $PROBLEMNAME # does all the preparation...


# replaces the call to pfrun:
sh $PARFLOW_DIR/bin/bootmc $NumProcs
sh $PARFLOW_DIR/bin/getmc $NumProcs

flowvr parflowvr --batch $DEBUGFLAGS

sh $PARFLOW_DIR/bin/freemc
sh $PARFLOW_DIR/bin/killmc


# replaces the call to pfundist
tclsh $PARFLOW_DIR/bin/parflowvr/undist.tcl $PROBLEMNAME



killall flowvrd


echo ------------ END! --------------
