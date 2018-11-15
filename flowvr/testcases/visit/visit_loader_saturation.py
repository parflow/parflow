#!/usr/local/bin/ipython2 -i

# will open visit with the first simulation that is found in ~/.visit/simulations

from visit import *
import os
import time

Launch()

prepath = os.environ.get('HOME')+'/.visit/simulations'

simname = ""

v = GetView3D()
v.RotateAxis(0,80.)
v.RotateAxis(1,30.)
v.RotateAxis(2,10.)
SetView3D(v)



def load():
    global simname
    simname = prepath + '/' + os.listdir(prepath)[-1]
    OpenDatabase(simname)
    AddPlot("Pseudocolor", "saturation")
    DrawPlots()
    PseudocolorAtts = PseudocolorAttributes()
    PseudocolorAtts.scaling = PseudocolorAtts.Linear  # Linear, Log, Skew
    PseudocolorAtts.minFlag = 1
    PseudocolorAtts.min = 0.01
    PseudocolorAtts.opacityType=PseudocolorAtts.Constant
    PseudocolorAtts.opacity=1
    PseudocolorAtts.invertColorTable=1
    SetPlotOptions(PseudocolorAtts)


def unload():
    DeleteAllPlots()
    CloseDatabase(simname)


def reload():
    unload()
    load()

def triggerSnap():
    SendSimulationCommand("localhost", simname, "trigger snap")

def h():
    print("""
------------------------------------------------------------------------------
    h()           - print this help text
    triggerSnap() - trigger snap
    load()        - open latest sim file and draw pressure in Pseudocolor
    unload()      - delete plots and close sim file

    reload()      - use this to refresh view. does unload() and load()
------------------------------------------------------------------------------
    """)



load()
h()

while True:
    triggerSnap()
    time.sleep(0.5)



