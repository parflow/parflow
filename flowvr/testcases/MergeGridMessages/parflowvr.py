import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *



problemName, P, Q, R = sys.argv[1:5]


# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1],  # cut last ,
        problemName=problemName,
        outports=["pressure"])  # ports as specified in tcl file

analyzer = Analyzer("Python-Analyzer", "python -u ./analyzer.py")

treePressure = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("pressure"), arity = 2)
gridmessagemerger = FilterMergeGridMessages("gridMessageMerger")
treePressure.link(gridmessagemerger.getPort("in"))

gridmessagemerger.getPort("out").link(analyzer.getPort("in"))

pres = RoutingNode("pres")
analyzer.getPort("out").link(pres.getPort("in"))
pres.getPort("out").link(parflowmpi.getPort("in"))


nw1 = NetCDFWriter("normal", fileprefix="normal.")
nw2 = NetCDFWriter("merged", fileprefix="merged.")

gridmessagemerger.getPort("out").link(nw2.getPort("in"))
treePressure.link(nw1.getPort("in"))

app.generate_xml("parflowvr")
