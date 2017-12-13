#!/usr/local/bin/ipython2 -u -i
import pypfcommon as pf
import numpy as np

a = np.array([[[1,2,3],[4,5,6]],[[2,2,2],[3,3,3]]])

#pf.itest(m)

def onGridMessage(arr, gm):
    print("epic cool grid message! Let's do some stuff now!")
    #print(arr)
    #a = pf.StampLog()
    #a.stampName = 'a'
    #a.value = 42.0
    #b = pf.StampLog()
    #b.stampName = 'b'
    #b.value = 43.0
    #cc = pf.StampLog()
    #cc.stampName = 'cc'
    #cc.value = 44.0
    print("sending log now")
    #pf.SendLog([a,b,cc])

    print("send steer message now")
    print(arr.shape)
    print(gm)
    shape = (gm.grid.nX, gm.grid.nY, gm.grid.nZ)
    print(shape)
    operator = np.ones(arr.shape)
    print(operator.sum())
    pf.SendSteerMessage(pf.ACTION_MULTIPLY, pf.VARIABLE_PRESSURE,
            gm.ix, gm.iy, gm.iz, operator)
    print("nice")






print("setting it up!")
pf.SetGridMessageParser(onGridMessage)

print("running!!")
#pf.run(['a','b','cc'])
pf.run([])


