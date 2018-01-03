import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *

parflow = Parflow("parflow",
        cmdline = "tclsh ./default_richards.tcl",
        outports = ["out0", "out1", "out2"])

analyzer0 = Analyzer("analyzer0", "python -u ./analyzer.py 0 3")
parflow.getPort("out0").link(analyzer0.getPort("in"))

analyzer1 = Analyzer("analyzer1", "python -u ./analyzer.py 5 10")
parflow.getPort("out1").link(analyzer1.getPort("in"))

analyzer2 = Analyzer("analyzer2", "python -u ./analyzer.py 1 2")
parflow.getPort("out2").link(analyzer2.getPort("in"))

app.generate_xml("default_richards")
