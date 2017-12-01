#!/usr/local/bin/ipython2 -i
from visit import *
import os

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
    AddPlot("Pseudocolor", "pressure")
    DrawPlots()

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



