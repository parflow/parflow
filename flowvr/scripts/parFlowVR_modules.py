from flowvrapp import *
from filters import *
import socket


class FilterMergeItExt(FilterWithManyInputs):
    """Merges messages received on input port into one message sent on output port.

    More precisely, when it receives a message on the 'order' port, it
    inspects the incoming message queue ('in' port), discard all
    messages with a non null 'scratch' stamp value, concatenate all
    other messages in one message sent on 'out' port. This message has
    its 'scratch' stamp set to 0 and its 'stamp' stamp set to the sum of
    all 'stamp' stamps of the concatenated messages. The name of the
    'scratch' and 'stamp' stamps is set from the component parameter.
    If forwardEmpty is set to "True" even empty mesages will be forwarded or if no message
    is available on the 'in' port an empty message will be generated.
    If forwardPresignal is set to "True" even Presignal messages (and messages with an it
    stamp < 0) will be forwarded

    Use newInputPort() to get a new input port (instead of getPort()).
    """

    def __init__(self, name, host = ''):
        self.messagetype = 'full'
        self.plugin_name = 'flowvr.plugins.MergeItExt'
        FilterWithManyInputs.__init__(self, name, host = host)
        self.addPort("order", direction = 'in', messagetype = 'stamps')


class Parflow(Module):
    def __init__(self, prefix, index=None, run=None, host=None, cmdline=None, outports=[], problemName=None):
        name = prefix + "/" + str(index) if index is not None else prefix

        if cmdline is None:
            cmdline = "$PARFLOW_DIR/bin/parflow %s" % problemName

        Module.__init__(self, name, run = run, host = host, cmdline = cmdline)

        inportnames = ["in"]
        outportnames = outports + ["snap"]


        for inportname in inportnames:
            p = self.addPort(inportname, direction = 'in')

        for outportname in outportnames:
            p = self.addPort(outportname, direction = 'out')

class ParflowMPI(Composite):
    """several instances of parflow module that generate pressure, porosity... for the
    next timeframe"""

    def __init__(self, hosts, problemName, outports=[], debugprefix=None):
        Composite.__init__(self)

        if debugprefix is None:
            debugprefix=''
        else:
            debugprefix+=' '

        prefix = "parflow"
        parflowrun=None
        cmdline=None

        # hosts: string with host names, separated by spaces
        if hosts.find(',') == -1:
            cmdline = "%s$PARFLOW_DIR/bin/parflow %s" % (debugprefix, problemName)
            # only one host. start locally!
        else:
            if socket.gethostname().find('frog') == 0:
                # I bet I'm on froggy ;)
                parflowrun = FlowvrRunOpenMPI("%s$HOME/bin/froggy_parflow %s" % (debugprefix, problemName), hosts = hosts, prefix = prefix, mpirunargs="--mca btl_sm_use_knem 0 --mca btl_vader_single_copy_mechanism none")
            else:
                parflowrun = FlowvrRunMPI("%s$PARFLOW_DIR/bin/parflow %s" % (debugprefix, problemName), hosts = hosts, prefix = prefix, mpistack="openmpi")


        # hosts_list: convert hosts to a list
        hosts_list = hosts.split(",")

        # nb of instances
        ninstance = len(hosts_list)

        for i in range(ninstance):
            parflow = Parflow(prefix, index=i, run=parflowrun, host=hosts_list[i],
                    cmdline=cmdline,
                    outports=outports)

            # collect ports
            for pname in parflow.ports:
                p = parflow.ports[pname]
                if not pname in self.ports:
                    self.ports[pname] = list()
                self.ports[pname].append(p)


class VisIt(Module):
    """Module that will abstract VisIt later"""
    def __init__(self, name, host=""):
        Module.__init__(self, name, cmdline = "$PARFLOW_DIR/bin/visit-connector", host=host)
        #Module.__init__(self, name, cmdline = "xterm -e gdb $PARFLOW_DIR/bin/visit-connector")
        self.addPort("triggerSnap", direction = 'out')
        self.addPort("in", direction = 'in')

class Analyzer(Module):
    """Module that will analyze GridMessages on the in Port and give a Steer Proposition
    on the out port. The log port can be used with the logger module to log and plot
    floating point outputs in realtime."""
    def __init__(self, name, cmdline, host=""):
        Module.__init__(self, name, cmdline = cmdline, host=host)
        self.addPort("in", direction = 'in')
        p = self.addPort("out", direction = 'out')#, blockstate='nonblocking') # send out thaat I need a trigger ;)
        #p.blockstate='non blocking'
        self.addPort("log", direction = 'out')

class Logger(Module):
    """Module that will log the given stamp value in a graph.
    stampNames: space seperated line of stamp names. Must be float stamps.
    The float content of those stamps will be logged and also plotted if showWindows is
    true.
    """
    def __init__(self, name, stampNames, showWindows=True, host=""):
        Module.__init__(self, name, cmdline = "python $PARFLOW_DIR/bin/parflowvr/logger.py %s %s" % (stampNames, "--show-windows" if showWindows else ""), host=host)
        self.addPort("in", direction = 'in')  #, messagetype='stamps')

class Ticker(Module):
    """Module sends a message every second"""
    def __init__(self, name, size=0, T=0.5, host=""):
        Module.__init__(self, name, cmdline = "python $PARFLOW_DIR/bin/parflowvr/ticker.py %d %f" % (size, T), host=host)
        #self.addPort("beginIt", direction = 'in')
        self.addPort("out", direction = 'out')

class NetCDFWriter(Module):
    """Module NetCDFWriter writes parflow output into netCDF files."""
    def __init__(self, name, fileprefix="", host=""):
        # TODO: works with mpi too?!
        Module.__init__(self, name, cmdline = "$PARFLOW_DIR/bin/netcdf-writer %s" % fileprefix, host=host)
        self.addPort("in", direction = 'in');
