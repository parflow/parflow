import os
import sys
import numpy as np
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb, ParflowBinaryReader
from parflow.tools.fs import mkdir, get_absolute_path

# Check that the ParflowBinaryReader reads the correct values of p, q, r even
# when they don't exactly divide nx, ny, nz.

data = np.ones((89, 91))
run = Run("test_dist", __file__)
working_dir = get_absolute_path(os.path.join("test_output", "dist"))
mkdir(working_dir)

path = os.path.join(working_dir, 'data.pfb')
write_pfb(path, data)

run.dist(path, P=15, Q=15)
rd = ParflowBinaryReader(path, read_sg_info=True)
if rd.header['p'] != 15 or rd.header['q'] != 15:
    sys.exit(1)
    
run.dist(path, P=14, Q=15)
rd = ParflowBinaryReader(path, read_sg_info=True)
if rd.header['p'] != 14 or rd.header['q'] != 15:
    sys.exit(1)
