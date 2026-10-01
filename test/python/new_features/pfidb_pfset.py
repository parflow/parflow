# -----------------------------------------------------------------------------
# Testing pfset with a .pfidb file
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import get_absolute_path

run_name = "dsingle"
dsingle = Run(run_name, __file__)

dsingle.pfset(
    pfidb_file=get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
)

# -----------------------------------------------------------------------------
# Write and compare the ParFlow database files
# -----------------------------------------------------------------------------

generatedFile, runArg = dsingle.write()

passed = True

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
) as ref:
    if new.read() == ref.read():
        print("Files are the same")
    else:
        print("Files are different")
        passed = False

if passed:
    print(f"{run_name} : PASSED")
else:
    print(f"{run_name} : FAILED")
    sys.exit(1)
