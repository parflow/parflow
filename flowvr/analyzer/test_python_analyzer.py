#!/usr/local/bin/ipython2 -u -i
import pypfcommon as pf
import numpy as np
import random

#pf.itest(m)

def onGridMessage(arr, gm):
    print("Epic cool grid message! Let's do some stuff now!")
    t = pf.StampLog()
    t.stampName = 't'
    t.value = gm.time
    print(gm.variable)
    b = pf.StampLog()
    b.stampName = 'zweiundvierzig'
    b.value = 42.0
    pf.SendLog([t,b])

    #r = 1 + random.random()* 0.2 - 0.4
    #arr *= r
    print(k)

    pf.SendSteerMessage(pf.ACTION_SET, gm.variable,
            gm.ix, gm.iy, gm.iz, arr)






print("setting it up!")
pf.SetGridMessageParser(onGridMessage)

print("running!!")
pf.run(['t', 'zweiundvierzig'])


