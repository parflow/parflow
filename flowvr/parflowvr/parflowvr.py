import sys

from filters import *

sys.path.append('../scripts')
from parFlowVR_modules import *




# Main starts here ###########

P, Q, R = sys.argv[1:4]

pres = FilterPreSignal("PreSignal", nb=1)  # will be inited with one token for the beginning. TODO set nb to 2 later!
mergeIt = FilterMergeIt("parflow-controller")

# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1])  # cut last ,

visit = VisIt("visit")

#netcdfwriter = NetCDFWriter("netcdfwriter")
#analyzer = Analyzer("analyzer")

#spymodule = SpyModule("spy_module")

# SIMULATION TIME FRAME DATA TRANSFER

# PRESSURE DATA TRANSFER
# link parflow (port pressure)  to netcdfwriter.   Comunication N to 1. Using MergeFilter to merge results

#treePressure = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("pressure"), arity = 2)
#treePressure.link(netcdfwriter.getPort("pressureIn"))
#treePressure.link(analyzer.getPort("pressureIn"))

#analyzer.getPort("steerOut").link(parflowmpi.getPort("steerIn"))

treePressureSnap = generateNto1(prefix="comNto1PressureSnapMerge", in_ports = parflowmpi.getPort("pressureSnap"), arity = 2)
treePressureSnap.link(visit.getPort("pressureIn"))
visit.getPort("triggerSnap").link(mergeIt.getPort("in"))

parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))

pres.getPort("out").link(mergeIt.getPort("order"))
mergeIt.getPort("out").link(parflowmpi.getPort("in"))


#for p in parflowmpi.getPort("pressure"):
        #p.link(netcdfwriter.getPort("pressureIn"))

#pres.getPort("out").link(simplestarter.getPort("beginIt"))
#netcdfwriter.getPort("out").link(pres.getPort("in"))

#import traces
#traces.add_traces(traces.select_primitives(["fluid", "gldens", "comNto1DensityMerge/node2", "comNto1DensityMerge/node1",
    #"comNto1DensityMerge/node0","comNto1DensityMerge/node3", "comNto1VelocityMerge/node0", "comNto1VelocityMerge/node1",
    #"comNto1VelocityMerge/node2", "comNto1VelocityMerge/node3"]), "fluidmpi")

app.generate_xml("parflowvr")
