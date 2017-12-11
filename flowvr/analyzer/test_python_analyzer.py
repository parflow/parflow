#!/usr/local/bin/ipython2 -i
import pypfcommon as pf
import numpy as np

a = np.array([[[1,2,3],[4,5,6]],[[2,2,2],[3,3,3]]])

#pf.itest(m)

def onGridMessage(arr, gm):
    print("epic cool grid message! Let's do some stuff now!")




print("setting it up!")
pf.SetGridMessageParser(onGridMessage)

print("running!!")
pf.run()


