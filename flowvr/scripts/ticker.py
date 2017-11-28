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

size = 1
T = 0.5

if len(sys.argv) > 1:
    size = int(sys.argv[1])

if len(sys.argv) > 2:
    T = float(sys.argv[2])

print ("tttttt ticker now waiting!")
while tickerModule.wait() :
    print ("tttttt got beginIt")
    #rm = inport.get()
    #rm.clear()


    # I do not know how to make an own stamplist atm...
    m = flowvr.MessageWrite(outport.stamps)



    print(size)
    m.data = tickerModule.alloc(size)
    outport.put(m)

    m.clear()

    print ("tttttt putting message")
    time.sleep(T)



print ("tttttt quit ticker")

tickerModule.close()
#tickerModule.abort()



