#!/usr/local/bin/ipython2 -u -i
import pypfAnalyzer as pfa
import numpy as np
import random


def onGridMessage(arr, gm):
    print("Epic cool grid message! Let's do some stuff now!")
    t = pfa.StampLog()
    t.stampName = 't'
    t.value = gm.time
    print(gm.variable)
    b = pfa.StampLog()
    b.stampName = 'zweiundvierzig'
    b.value = 42.0
    pfa.SendLog([t,b])

    #r = 1 + random.random()* 0.2 - 0.4
    #arr *= r

    pfa.SendSteerMessage(pfa.ACTION_SET, gm.variable,
            gm.ix, gm.iy, gm.iz, arr)
    print("check")






print("setting it up!")
pfa.SetGridMessageParser(onGridMessage)

print("running!!")
pfa.run(['t', 'zweiundvierzig'])


