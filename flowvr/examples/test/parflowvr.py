from filters import *
import os, sys

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *

problemName, P, Q, R = sys.argv[1:5]

# Components:
pres = FilterPreSignal("PreSignal", nb=1)

parflowmpi = ParflowMPI(("localhost,"*int(P)*int(Q)*int(R))[:-1],  # cut last ,
        problemName,
        ["out0"])  # ports as specified in tcl file

analyzercmd = os.getenv('ANALYZER')

if analyzercmd:
    analyzer = Analyzer("Analyzer", analyzercmd)
else:
    #analyzer = Analyzer("Analyzer", "../a.out")
    analyzer = Analyzer("Analyzer", "../Python-analyzer-template.py")

# Connections:
controller = FilterMergeItExt("Controller")
analyzer.getPort("out").link(controller.newInputPort())

# This works as all the parflow instances are synchron
parflowmpi.getPort("endIt")[0].link(pres.getPort("in"))
pres.getPort("out").link(controller.getPort("order"))
controller.getPort("out").link(parflowmpi.getPort("in"))

treePressure = generateNto1(prefix="comNto1PressureMerge", in_ports = parflowmpi.getPort("out0"), arity = 2)
treePressure.link(analyzer.getPort("in"))


app.generate_xml("parflowvr")
