#---------------------------------------------------------
# Testing Python clone function
#---------------------------------------------------------

import sys
import os
from parflow import Run

test = Run('full_clone', __file__)

test.pfset(yamlFile='./full_clone.yaml.ref', exit_if_undefined=True)

#-----------------------------------------------------------------------------

test.validate()
generatedFile, runFile = test.write(file_format='yaml')

# Prevent regression
with open(generatedFile) as new, open(f'{generatedFile}.ref') as ref:
  if new.read() == ref.read():
    print('Success we have the same file')
  else:
    print('Files are different')
    sys.exit(1)
