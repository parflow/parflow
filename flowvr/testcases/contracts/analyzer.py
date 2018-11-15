#!/usr/local/bin/ipython2 -u -i
import pypfAnalyzer as pfa
import numpy as np
import random

import sys


def onGridMessage(arr, gm):
    # here we test if the contract timing is correct.
    print(gm.time)
    timestep = round(gm.time / 0.001)
    #print(timestep)
    #print(sys.argv)
    if ((timestep - float(sys.argv[1])) % float(sys.argv[2]) != 0):
        print("Error!")
        #raise
    print("YES")



print("setting it up!")
pfa.SetGridMessageParser(onGridMessage)

print("running!!")
pfa.run([])


