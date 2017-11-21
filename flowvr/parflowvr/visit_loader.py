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


load()



