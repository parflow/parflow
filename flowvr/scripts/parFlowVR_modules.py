from flowvrapp import *

class Parflow(Component):
    def __init__(self, prefix, index, run, host):
        Component.__init__(self)
        pf = Module(prefix + "/" + str(index), run = run, host = host)


        pfEvent = Module(prefix + "/Event/" + str(index), run = run, host = host)
        p = pfEvent.addPort("triggerSnap", direction = 'in')
        #p = pf.addPort("triggerSnap", direction = 'in')
        p.blockstate='non blocking'
        self.ports["triggerSnap"] = p

        inportnames = ["in", "steerIn"]
        outportnames = ["pressure", "porosity", "saturation", "pressureSnap"]

        for inportname in inportnames:
            p = pf.addPort(inportname, direction = 'in') # TODO: unimplemented?, blockstate = 'nonblocking' if inportname=="triggerSnap" else 'blocking')
            self.ports[inportname] = p

        for outportname in outportnames:
            p = pf.addPort(outportname, direction = 'out')
            self.ports[outportname] = p


class ParflowMPI(Composite):
    """several instances of parflow module that generate pressure, porosity... for the
    next timeframe"""

    def __init__(self, hosts):
        Composite.__init__(self)

        prefix = "parflow"
        # hosts: string with host names, separated by spaces
        parflowrun = FlowvrRunOpenMPI("$PARFLOW_DIR/bin/parflow %s" % "mpi", hosts = hosts, prefix = prefix)
        #parflowrun = FlowvrRunOpenMPI("xterm -e gdb $PARFLOW_DIR/bin/parflow", hosts = hosts, prefix = prefix)

        # hosts_list: convert hosts to a list
        hosts_list = hosts.split(",")

        # nb of instances
        ninstance = len(hosts_list)

        for i in range(ninstance):
            parflow = Parflow(prefix, i, parflowrun, hosts_list[i])

            # collect ports
            for pname in parflow.ports:
                p = parflow.ports[pname]
                if not pname in self.ports:
                    self.ports[pname] = list()
                self.ports[pname].append(p)

class VisIt(Module):
    """Module that will abstract VisIt later"""
    def __init__(self, name):
        Module.__init__(self, name, cmdline = "$PARFLOW_DIR/bin/visit-connector")
        #Module.__init__(self, name, cmdline = "xterm -e gdb $PARFLOW_DIR/bin/visit-connector")
        self.addPort("triggerSnap", direction = 'out')
        self.addPort("pressureIn", direction = 'in')

class Controller(Module):
    """Module that will replace the simple starter"""
    def __init__(self, name):
        Module.__init__(self, name, cmdline = "echo TODO")
        self.addPort("out", direction = 'out')
        self.addPort("pressureIn", direction = 'in')

class Analyzer(Module):
    """Module that will analyze and give a Steer Proposition"""
    def __init__(self, name):
        Module.__init__(self, name, cmdline = "echo TODO")
        self.addPort("pressureIn", direction = 'in')
        p = self.addPort("steerOut", direction = 'out')#, blockstate='nonblocking') # send out thaat I need a trigger ;)
        p.blockstate='non blocking'


class Simplestarter(Module):
    """Module Simplestarter kicks of a nonsteered simple parflow simulation"""
    def __init__(self, name, starttime, stoptime):
        Module.__init__(self, name, cmdline = "python ../simplestarter/simplestarter.py %s %s" % (starttime, stoptime))
        #self.addPort("beginIt", direction = 'in')
        self.addPort("out", direction = 'out')

class NetCDFWriter(Module):
    """Module NetCDFWriter writes parflow output into netCDF files."""
    def __init__(self, name):
        # TODO: works with mpi too?!
        Module.__init__(self, name, cmdline = "$PARFLOW_DIR/bin/netcdf-writer")
        self.addPort("pressureIn", direction = 'in');
        self.addPort("out", direction = 'out');
        #self.run.options += '-x DISPLAY'
