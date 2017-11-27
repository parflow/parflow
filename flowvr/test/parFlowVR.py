import sys

from filters import *

sys.path.append('../scripts')
from parFlowVR_modules import *


pres = FilterPreSignal("PreSignal", nb=1)
parflow = Parflow("parflow", cmdline = "tclsh ./default_richards_with_netcdf.tcl 1 1 1 --FlowVR")
parflow.getPort("endIt").link(pres.getPort("in"))

netcdfwriter = NetCDFWriter("netcdfwriter")
parflow.getPort("pressure").link(netcdfwriter.getPort("pressureIn"))

app.generate_xml("parflowvr")
