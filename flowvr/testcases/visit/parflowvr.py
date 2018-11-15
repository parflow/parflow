import sys, os, socket

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *

problemName, P, Q, R = sys.argv[1:5]

pres = FilterPreSignal("PreSignal", nb=1)  # will be inited with one token for the beginning. TODO set nb to 2 later!

mergeIt = FilterMergeItExt("MergeItExt")

# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1],  # cut last ,
        problemName)


treeSnap = generateNto1(prefix="comNto1SnapMerge", in_ports = parflowmpi.getPort("snap"), arity = 2)

visit = VisIt("visit-connector")
treeSnap.link(visit.getPort("in"))
spy = SpyModule("spy")
treeSnap.link(spy.getPort("in"))

visit.getPort("triggerSnap").link(mergeIt.newInputPort())

parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))

pres.getPort("out").link(mergeIt.getPort("order"))
mergeIt.getPort("out").link(parflowmpi.getPort("in"))

app.generate_xml("parflowvr")
