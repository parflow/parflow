import os
import numpy as np
from parflow import Run
from parflow.tools.io import read_pfb, write_pfb, ParflowBinaryReader
from parflow.tools.fs import mkdir, get_absolute_path

# Check that Run.dist() works even when grid P, Q, R don't exactly divide
# nx, ny, nz.

data = np.ones((89, 91))
run = Run("test_dist", __file__)
working_dir = get_absolute_path(os.path.join("test_output", "dist"))
mkdir(working_dir)

path = os.path.join(working_dir, 'data.pfb')
write_pfb(path, data)

run.dist(path, P=15, Q=15)

run.dist(path, P=14, Q=15)
