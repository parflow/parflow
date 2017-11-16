import sys

from flowvrapp import *
from filters import *


class ParflowMPI(Composite):
  """several instances of parflow module that generate pressure, porosity... for the
    next timeframe"""

  def __init__(self, hosts):
    Composite.__init__(self)

    prefix = "parflow"
    # hosts: string with host names, separated by spaces
    parflowrun = FlowvrRunOpenMPI("$PARFLOW_DIR/bin/parflow %s" % "mpi", hosts = hosts, prefix = prefix)
# for debug:
    #parflowrun = FlowvrRunOpenMPI("xterm -e gdb $PARFLOW_DIR/bin/parflow", hosts = hosts, prefix = prefix)

    # hosts_list: convert hosts to a list
    hosts_list = hosts.split(",")

    # nb of instances
    ninstance = len(hosts_list)

    # collect ports
    all_in = []
    all_pressureout = []
    all_porosityout = []
    all_saturationout = []

# TODO: put in forloop!
    for i in range(ninstance):
      fluid = Module(prefix + "/" + str(i), run = parflowrun, host = hosts_list[i])
      fluid.addPort("in", direction = 'in');
      fluid.addPort("pressure", direction = 'out');
      fluid.addPort("porosity", direction = 'out');
      fluid.addPort("saturation", direction = 'out');

      all_in.append(fluid.getPort("in"))
      all_pressureout.append(fluid.getPort("pressure"))
      all_porosityout.append(fluid.getPort("porosity"))
      all_saturationout.append(fluid.getPort("saturation"))


    self.ports["in"] =  list(all_in)
    self.ports["pressure"] =  list(all_pressureout)
    self.ports["porosity"] =  list(all_porosityout)
    self.ports["saturation"] =  list(all_saturationout)


class Simplestarter(Module):
  """Module Simplestarter kicks of a nonsteered simple parflow simulation"""

  def __init__(self, name):
    Module.__init__(self, name, cmdline = "python ../simplestarter/simplestarter.py %s %s" % (sys.argv[4], sys.argv[5]))
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


# Main starts here ###########

P, Q, R = sys.argv[1:4]

simplestarter = Simplestarter("simplestarter")
# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
hosts = ("localhost,"*int(P)*int(Q)*int(R))[:-1]  # cut last ,
parflowmpi = ParflowMPI(hosts)
netcdfwriter = NetCDFWriter("netcdfwriter")

#spymodule = SpyModule("spy_module")
#parflowmpi.getPort("pressure")[0].link(spymodule.getPort("in"))

# SIMULATION TIME FRAME DATA TRANSFER

simplestarter.getPort("out").link(parflowmpi.getPort("in"))

# PRESSURE DATA TRANSFER
# link parflow (port pressure)  to netcdfwriter.   Comunication N to 1. Using MergeFilter to merge results

#TODO: shouldn't it work without this tree too?
# REM: if we would work on one core the tree will not work!
treeOut = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("pressure"), arity = 2)
treeOut.link(netcdfwriter.getPort("pressureIn"))

#spymodule = SpyModule("spy_module")
#treeOut.link(spymodule.getPort("in"))

#for p in parflowmpi.getPort("pressure"):
        #p.link(netcdfwriter.getPort("pressureIn"))

#pres = FilterPreSignal("Time_PreSignal", nb=1)  # will be inited with one token for the beginning. TODO set nb to 2 later!
#pres.getPort("out").link(simplestarter.getPort("beginIt"))
#netcdfwriter.getPort("out").link(pres.getPort("in"))

#import traces
#traces.add_traces(traces.select_primitives(["fluid", "gldens", "comNto1DensityMerge/node2", "comNto1DensityMerge/node1",
    #"comNto1DensityMerge/node0","comNto1DensityMerge/node3", "comNto1VelocityMerge/node0", "comNto1VelocityMerge/node1",
    #"comNto1VelocityMerge/node2", "comNto1VelocityMerge/node3"]), "fluidmpi")

app.generate_xml("mpi")
