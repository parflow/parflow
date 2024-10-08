# -----------------------------------------------------------------------------
# example for pfsol generation
# checking for correct .asc file writing
# -----------------------------------------------------------------------------

import sys

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import (
    load_patch_matrix_from_sa_file,
    load_patch_matrix_from_asc_file,
    write_patch_matrix_as_sa,
)

asc_write = Run("asc_write", __file__)


# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

asc_write.GeomInput.Names = "domaininput"

asc_write.GeomInput.domaininput.GeomName = "domain"
asc_write.GeomInput.domaininput.GeomNames = "domain"

# -----------------------------------------------------------------------------
# PFSOL generation
# -----------------------------------------------------------------------------

sabino_mask = load_patch_matrix_from_sa_file(
    get_absolute_path("$PF_SRC/test/input/Sabino_Mask.sa")
)
# sabino_mask = load_patch_matrix_from_asc_file(get_absolute_path('Sabino_Mask.asc'))

sabino_mask_written = write_patch_matrix_as_sa(
    sabino_mask,
    get_absolute_path("$PF_SRC/test/python/new_features/sabino_mask_written.sa"),
)

with open(get_absolute_path("$PF_SRC/test/input/Sabino_Mask.sa"), "rt") as ref:
    with open(
        get_absolute_path("$PF_SRC/test/python/new_features/sabino_mask_written.sa"),
        "rt",
    ) as new:
        if new.read() == ref.read():
            print("Success we have the same file")
        else:
            print("Files are different")
            sys.exit(1)
