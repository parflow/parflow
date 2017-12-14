import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *


problemName, P, Q, R = sys.argv[1:5]

pres = FilterPreSignal("PreSignal", nb=1)  # will be inited with one token for the beginning. TODO set nb to 2 later!

mergeIt = FilterMergeItExt("parflow-controller", 2)

# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1], problemName)  # cut last ,

visit = VisIt("visit")

netcdfwriter = NetCDFWriter("netcdfwriter")
analyzer = Analyzer("analyzer")

logger = Logger("logger", "E M K")

treePressure = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("pressure"), arity = 2)
treePressure.link(netcdfwriter.getPort("pressureIn"))
treePressure.link(analyzer.getPort("pressureIn"))

analyzer.getPort("steerOut").link(mergeIt.getPort("in1"))
analyzer.getPort("log").link(logger.getPort("in"))

treePressureSnap = generateNto1(prefix="comNto1PressureSnapMerge", in_ports = parflowmpi.getPort("pressureSnap"), arity = 2)
treePressureSnap.link(visit.getPort("pressureIn"))

parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))

pres.getPort("out").link(mergeIt.getPort("order"))
visit.getPort("triggerSnap").link(mergeIt.getPort("in0"))
mergeIt.getPort("out").link(parflowmpi.getPort("in"))

app.generate_xml("parflowvr")
