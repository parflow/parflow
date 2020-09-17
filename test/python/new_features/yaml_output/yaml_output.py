#-----------------------------------------------------------------------------
# Testing hierarchical yaml output
#-----------------------------------------------------------------------------

import sys
from parflow import Run

yamlHierarchy = Run('hierarchical', __file__)
generatedFile, runFile = yamlHierarchy.write(file_format='yaml')

with open(generatedFile) as new, open(f'{generatedFile}.ref') as ref:
  if new.read() == ref.read():
    print('Success we have the same file')
  else:
    print('Files are different')
    sys.exit(1)
