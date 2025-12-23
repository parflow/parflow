# -----------------------------------------------------------------------------
# example for pfsol generation from images
# -----------------------------------------------------------------------------

from parflow import Run
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import load_patch_matrix_from_image_file
from parflow.tools.builders import SolidFileBuilder

sabino = Run("sabino", __file__)

# -----------------------------------------------------------------------------
# PFSOL generation
# -----------------------------------------------------------------------------

color_to_patch = {
    "#FFFFFF": 0,  # Discard
    "#466CFF": 2,  # River
    "#C56C00": 3,  # Land
    "#1900FF": 4,  # Sea
    "#3DFFFF": 5,  # Lake
}

mask = load_patch_matrix_from_image_file(
    get_absolute_path("$PF_SRC/test/input/mask.png")
)

top_patches = load_patch_matrix_from_image_file(
    get_absolute_path("$PF_SRC/test/input/mask_top.png"), color_to_patch, fall_back_id=6
)

side_patches = load_patch_matrix_from_image_file(
    get_absolute_path("$PF_SRC/test/input/mask_side.png"),
    color_to_patch,
    fall_back_id=3,
)

# -----------------------------------------------------------------------------

SolidFileBuilder(bottom=1).mask(mask).top_ids(top_patches).side_ids(side_patches).write(
    "sabino_domain.pfsol", xllcorner=0, yllcorner=0, cellsize=90, vtk=True
)
