import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *




# Main starts here ###########

pres = FilterPreSignal("PreSignal", nb=1)

mergeTasks = FilterMergeItExt("parflow-controller")

# run all on localhost for the moment:
parflow = Parflow("parflow", problemName="hillslope_sens",
        outports=["pressure"])


logger = Logger("logger", "error", False)

parflow.getPort("endIt").link(pres.getPort("in"))

pres.getPort("out").link(mergeTasks.getPort("order"))
mergeTasks.getPort("out").link(parflow.getPort("in"))

analyzer = Analyzer("Python-Analyzer", "python -u ./analyzer.py")
analyzer.getPort("out").link(mergeTasks.newInputPort())
analyzer.getPort("log").link(logger.getPort("in"))

parflow.getPort("pressure").link(analyzer.getPort("in"))

app.generate_xml("hillslope_sens")
