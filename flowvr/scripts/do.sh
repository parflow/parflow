#!/bin/bash
# generates a FlowVR dataflow graph and runs it

killall flowvrd
flowvr-kill
rm $1.net.xml
rm $1.run.xml
rm $1.cmd.xml
python $1.py
flowvrd&
flowvr $1
