import flowvr
import time

outport = flowvr.OutputPort("out")

stampStartTime = outport.addStamp("stampStartTime", float)
stampStopTime = outport.addStamp("stampStopTime", float)
inport = flowvr.InputPort("beginIt")  # really strange that we have to init it here...

v = flowvr.vectorPort()
v.append(inport)
v.append(outport)


simplestarterModule = flowvr.initModule(v)


print ("-------simple starter now waiting!")
while simplestarterModule.wait() :
    print ("-------got beginIt")
    #rm = inport.get()
    #rm.clear()


    # I do not know how to make an own stamplist atm...
    m = flowvr.MessageWrite(outport.stamps)


    print("----setting stamp")
    m.setStamp("stampStartTime", 0.0  )
    m.setStamp("stampStopTime",  0.010)

    # we need to initizlize data also for stamps messages ;)
    m.data = simplestarterModule.alloc(0)
    outport.put(m)

    m.clear()

    print ("-------putting message")
    time.sleep(1)
    break



print ("-----quit simple starter")

simplestarterModule.close()



