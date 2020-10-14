#-----------------------------------------------------------------------------
# Testing pfset with a .pfidb file
#-----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import cp

cp('$PF_SRC/test/python/new_features/write_check/dsingle.pfidb.ref')

dsingle = Run("dsingle", __file__)

dsingle.pfset(pfidbFile='dsingle.pfidb.ref')


#-----------------------------------------------------------------------------
# Write and compare the ParFlow database files
#-----------------------------------------------------------------------------

generatedFile, runArg = dsingle.write()

# Prevent regression
with open(generatedFile) as new, open(f'{generatedFile}.ref') as ref:
  if new.read() == ref.read():
    print('Success we have the same file')
  else:
    print('Files are different')
    sys.exit(1)
