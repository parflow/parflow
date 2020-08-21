#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

from parflow import Run
from parflow.tools.utils import load_pfidb

pfidb_run = Run('pfidb_run', __file__)

input_pfidb = load_pfidb('pfidb_run.pfidb')

value = 1e-20
print(type(value))

for key, value in input_pfidb.items():
  print(int(value))
  pfidb_run.pfset(key, value)


pfidb_run.validate()