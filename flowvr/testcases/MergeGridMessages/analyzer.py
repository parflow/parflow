#!/usr/bin/python2 -u
import pypfAnalyzer as pfa
import numpy as np

def onGridMessage(arr, m):
    # echo the received.
    print("gridmessage: n=(%d %d %d) N=(%d %d %d) i=(%d %d %d)" %
            (m.nx, m.ny, m.nz, m.grid.nX, m.grid.nY, m.grid.nZ, m.ix, m.iy, m.iz))
    print(arr.shape)
    print(m.variable)
    print(arr)
    pfa.SendSteer(pfa.ACTION_SET, m.variable, m.ix, m.iy, m.iz, arr)

pfa.SetGridMessageParser(onGridMessage)


# Run as a FlowVR module
pfa.run([])
