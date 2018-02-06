import sys
import os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *


# outports must be the same as specified in the tcl!
parflow = Parflow("parflow",
        cmdline = "tclsh ./default_richards_with_netcdf.tcl 1 1 1 --FlowVR",
        outports=["pressure", "saturation"])
# As nothing is connected to the parflow input ports, it will just run the simulation
# cycles as specified within the tcl file and parflow is not waiting for interaction

saturationwriter = NetCDFWriter("saturationwriter", "saturation.", abortOnEnd=False);
parflow.getPort("saturation").link(saturationwriter.getPort("in"))

pressurewriter = NetCDFWriter("pressurewriter", "pressure.", abortOnEnd=False)
parflow.getPort("pressure").link(pressurewriter.getPort("in"))

merge = FilterMerge("merge")
parflow.getPort("pressure").link(merge.newInputPort())
parflow.getPort("saturation").link(merge.newInputPort())

multiwriter = NetCDFWriter("multiwriter", "multi.", abortOnEnd=True)
merge.getPort("out").link(multiwriter.getPort("in"))

app.generate_xml("parflowvr")
