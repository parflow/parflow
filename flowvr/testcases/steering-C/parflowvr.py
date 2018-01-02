import sys, os, socket

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *

problemName, P, Q, R = sys.argv[1:5]

pres = FilterPreSignal("PreSignal", nb=1)  # will be inited with one token for the beginning. TODO set nb to 2 later!

mergeIt = FilterMergeItExt("parflow-controller")

# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1],  # cut last ,
        problemName,
        ["out0"])  # ports as specified in tcl file


netcdfwriter = NetCDFWriter("netcdfwriter", fileprefix="_")
analyzer = Analyzer("C-Analyzer", "$PARFLOW_DIR/bin/parflowvr/analyzer_find_hillslope_K")


treePressure = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("out0"), arity = 2)
treePressure.link(netcdfwriter.getPort("in"))
treePressure.link(analyzer.getPort("in"))

analyzer.getPort("out").link(mergeIt.newInputPort())

logger = Logger("logger", "E M K")
analyzer.getPort("log").link(logger.getPort("in"))

parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))

pres.getPort("out").link(mergeIt.getPort("order"))
mergeIt.getPort("out").link(parflowmpi.getPort("in"))

spy = SpyModule("spy")
mergeIt.getPort("out").link(spy.getPort("in"))

app.generate_xml("parflowvr")
