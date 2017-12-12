#!/usr/local/bin/ipython2 -i
import pypfcommon as pf
import numpy as np

a = np.array([[[1,2,3],[4,5,6]],[[2,2,2],[3,3,3]]])

#pf.itest(m)

def onGridMessage(arr, gm):
    print("epic cool grid message! Let's do some stuff now!")
    print(arr)
    a = pf.StampLog()
    a.stampName = 'a'
    a.value = 42.0
    b = pf.StampLog()
    b.stampName = 'b'
    b.value = 43.0
    cc = pf.StampLog()
    cc.stampName = 'cc'
    cc.value = 44.0
    print("sending log now")
    pf.SendLog([a,b,cc])

    #SendSteerMessage(pf.ACTION_SET, pf.VARIABLE_PRESSURE,
            #0,0,0, np.zeros(arr.shape))






print("setting it up!")
pf.SetGridMessageParser(onGridMessage)

print("running!!")
pf.run(['a','b','cc'])


