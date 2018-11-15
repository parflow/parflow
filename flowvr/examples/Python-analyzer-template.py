#!/usr/bin/python2 -u
import pypfAnalyzer as pfa
import numpy as np

def onGridMessage(arr, m):
    operand = np.ones(shape = arr.shape) * 42.
    for k in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for i in range(arr.shape[2]):
                # The coordinates x, y and z are grid coordinates in the problem domain
                x = i + m.ix;
                y = j + m.iy;
                z = k + m.iz;

                # Doing sample analysis and calculate the operand
                # Attention: the coordinates are not in the obvious order!
                operand[k,j,i] += x + y + z
    # Steer the simulation with the operand.
    # Depending on the FlowVR graph it is sometimes not guaranteed that the Steer
    # is performed for the next Simulation step already.
    pfa.SendSteer(pfa.ACTION_SET, pfa.VARIABLE_PRESSURE, m.ix, m.iy, m.iz, operand)

pfa.SetGridMessageParser(onGridMessage)


# Run as a FlowVR module
pfa.run([])

