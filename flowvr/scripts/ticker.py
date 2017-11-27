import flowvr
import time
import sys

# sends a message every second.

outport = flowvr.OutputPort("out")

inport = flowvr.InputPort("beginIt")  # really strange that we have to init it here...

v = flowvr.vectorPort()
v.append(inport)
v.append(outport)


tickerModule = flowvr.initModule(v)


print ("tttttt ticker now waiting!")
while tickerModule.wait() :
    print ("tttttt got beginIt")
    #rm = inport.get()
    #rm.clear()


    # I do not know how to make an own stamplist atm...
    m = flowvr.MessageWrite(outport.stamps)


    print("tttttttt setting stamp %f - %f" % (startTime, stopTime))

    m.data = tickerModule.alloc(0)
    outport.put(m)

    m.clear()

    print ("tttttt putting message")
    time.sleep(0.5)



print ("tttttt quit ticker")

tickerModule.close()
#tickerModule.abort()



