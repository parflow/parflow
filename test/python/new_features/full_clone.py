# ---------------------------------------------------------
# Testing Python clone function
# ---------------------------------------------------------

import sys
import os
from parflow import Run
from parflow.tools.fs import get_absolute_path

run_name = "full_clone"
test = Run(run_name, __file__)

test.pfset(
    yaml_file="$PF_SRC/test/correct_output/full_clone.yaml.ref", exit_if_undefined=True
)

# -----------------------------------------------------------------------------

test.validate()
generatedFile, runFile = test.write(file_format="yaml")

passed = True

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/full_clone.yaml.ref")
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
