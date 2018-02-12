#!/usr/local/bin/ipython2 -u -i
import pypfAnalyzer as pfa
import numpy as np
import random

import netCDF4 as nc


A = nc.Dataset('reference/hillslope_sens.out.00001.nc')

steadystate = np.array(A.variables["pressure"])[-1]

def getError(arr):
    # if we would take from sklearn.metrics import mean_squared_error it would be probably
    # faster. but we do not want to depend from this package.
    global steadystate
    return np.sum((arr-steadystate)**2)

def SendLogs(logs):
    ll = []
    for log in logs:
        l = pfa.StampLog()
        l.stamp_name = log[0]
        l.value = log[1]
        ll.append(l)

    pfa.SendLog(ll)

count = 0
def onGridMessage(arr, gm):
    global count
    count += 1
    if count == 3:
        SendLogs([['error', getError(arr)]])
    print("check")




print("setting it up!")
pfa.SetGridMessageParser(onGridMessage)

print("running!!")
pfa.run(['error'])  # create an analyzer with a log port that can send pressure_probe values.


