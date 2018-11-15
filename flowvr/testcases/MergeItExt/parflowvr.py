import sys, os

from filters import *

parflow_dir = os.getenv('PARFLOW_DIR')
sys.path.append(parflow_dir + '/bin/parflowvr')
from parFlowVR_modules import *

maxfreq = Ticker("ticker", 0)
t = Ticker("ticker2", 4, 1.0)
f = FilterMergeItExt("Mi")
spy = SpyModule("filter out")
spy2 = SpyModule("ticker out")

t.getPort("out").link(f.newInputPort())
maxfreq.getPort("out").link(f.getPort("order"))

f.getPort("out").link(spy.getPort("in"))
t.getPort("out").link(spy2.getPort("in"))


app.generate_xml("parflowvr")
