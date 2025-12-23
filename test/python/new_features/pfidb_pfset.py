# -----------------------------------------------------------------------------
# Testing pfset with a .pfidb file
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import get_absolute_path

dsingle = Run("dsingle", __file__)

dsingle.pfset(
    pfidb_file=get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
)


# -----------------------------------------------------------------------------
# Write and compare the ParFlow database files
# -----------------------------------------------------------------------------

generatedFile, runArg = dsingle.write()

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
) as ref:
    if new.read() == ref.read():
        print("Success we have the same file")
    else:
        print("Files are different")
        sys.exit(1)
