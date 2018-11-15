import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *




# Main starts here ###########

problemName, P, Q, R = sys.argv[1:5]

pres = FilterPreSignal("PreSignal", nb=1)

mergeTasks = FilterMergeItExt("parflow-controller")

# Hostlist: comma separated for openmpi.  Add more hosts for more parallelism
# run all on localhost for the moment:
parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1], problemName,
        ["out0", "out1", "out2"])  # cut last ,


logger = Logger("logger", "t zweiundvierzig", False)

parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))

pres.getPort("out").link(mergeTasks.getPort("order"))
mergeTasks.getPort("out").link(parflowmpi.getPort("in"))

analyzer = Analyzer("Python-Analyzer", "python ./analyzer.py")
analyzer.getPort("out").link(mergeTasks.newInputPort())
analyzer.getPort("log").link(logger.getPort("in"))

mergeOuts = FilterMergeItExt("merge-outs")
parflowmpi.getPort("endIt")[0].link(mergeOuts.getPort("order"))

for i in range(3):
    tree = generateNto1(prefix="comNto1Out%d"%i, in_ports = parflowmpi.getPort("out%d"%i), arity = 2)
    tree.link(mergeOuts.newInputPort())

mergeOuts.getPort("out").link(analyzer.getPort("in"))


app.generate_xml("parflowvr")
