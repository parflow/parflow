#!/bin/bash
# generates a FlowVR dataflow graph and runs it
# first parameter: the  problemm name.
# all further parameters are forwarded to the execution of flowvr.
# so e.g. --batch-mode is possible

killall flowvrd
flowvr-kill
rm $1.net.xml
rm $1.run.xml
rm $1.cmd.xml
python $1.py
flowvrd&
flowvr $@
