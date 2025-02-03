# ---------------------------------------------------------
# Testing Python clone function
# ---------------------------------------------------------

import sys
import os
from parflow import Run
from parflow.tools.fs import get_absolute_path

test = Run("full_clone", __file__)

test.pfset(
    yaml_file="$PF_SRC/test/correct_output/full_clone.yaml.ref", exit_if_undefined=True
)

# -----------------------------------------------------------------------------

test.validate()
generatedFile, runFile = test.write(file_format="yaml")

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/full_clone.yaml.ref")
) as ref:
    if new.read() == ref.read():
        print("Success we have the same file")
    else:
        print("Files are different")
        sys.exit(1)
