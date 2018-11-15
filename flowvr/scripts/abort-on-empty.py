#!/usr/bin/python

import flowvr


ports = flowvr.vectorPort()
port = flowvr.InputPort('in')
ports.push_back(port)

module = flowvr.initModule(ports)
while module.wait():
    message = port.get()
    if message.data.getSize() == 0:
        module.abort()


module.close()
