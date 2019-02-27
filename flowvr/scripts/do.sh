#!/bin/bash
# generates a FlowVR dataflow graph and runs it
# first parameter: the  problemm name.
# all further parameters are forwarded to the execution of flowvr.
# so e.g. --batch is possible

killall flowvrd
flowvr-kill
rm parflowvr.net.xml
rm parflowvr.run.xml
rm parflowvr.cmd.xml
tclsh $1.tcl
python parflowvr.py
flowvrd&
flowvr parflowvr "${@:2}"
