# builders.py
# functions for helping build ParFlow scripts
import os
import numpy as np
from parflow.tools.io import write_patch_matrix_as_asc, write_patch_matrix_as_sa
from parflow.tools.fs import get_absolute_path

class EcoSlimBuilder:

  def __init__(self, run_name):
      self.run_name = run_name

  def key_add(self, keys=['PrintVelocities']):
      self.run_name.Solver.PrintVelocities = True
      return self

  def write_slimin(self):
      return


class SolidFileBuilder:

    def __init__(self, top=1, bottom=2, side=3):
        self.name = None
        self.mask_array = None
        self.patch_ids_top = None
        self.patch_ids_bottom = None
        self.patch_ids_side = None
        self.top_id = top
        self.bottom_id = bottom
        self.side_id = side

    def mask(self, mask_array):
        self.mask_array = mask_array
        return self

    def top(self, patch_id):
        self.top_id = patch_id
        self.patch_ids_top = None
        return self

    def bottom(self, patch_id):
        self.bottom_id = patch_id
        self.patch_ids_top = None
        return self

    def side(self, patch_id):
        self.side_id = patch_id
        self.patch_ids_side = None
        return self

    def top_ids(self, top_patch_ids):
        self.patch_ids_top = top_patch_ids
        return self

    def bottom_ids(self, bottom_patch_ids):
        self.patch_ids_bottom = bottom_patch_ids
        return self

    def side_ids(self, side_patch_ids):
        self.patch_ids_side = side_patch_ids
        return self

    def write(self, name, xllcorner=0, yllcorner=0, cellsize=0, vtk=False):
        self.name = name
        output_file_path = get_absolute_path(name)
        if self.mask_array is None:
            raise Exception('No mask were define')

        jSize, iSize = self.mask_array.shape
        leftMask = np.zeros((jSize, iSize), dtype=np.int16)
        rightMask = np.zeros((jSize, iSize), dtype=np.int16)
        backMask = np.zeros((jSize, iSize), dtype=np.int16)
        frontMask = np.zeros((jSize, iSize), dtype=np.int16)
        bottomMask = np.zeros((jSize, iSize), dtype=np.int16)
        topMask = np.zeros((jSize, iSize), dtype=np.int16)

        for j in range(jSize):
            for i in range(iSize):
                if self.mask_array[j, i] != 0:
                    patch_value = 0 if self.patch_ids_side is None else self.patch_ids_side[j, i]
                    # Left (-x)
                    if i == 0 or self.mask_array[j, i-1] == 0:
                        leftMask[j, i] = patch_value if patch_value else self.side_id

                    # Right (+x)
                    if i + 1 == iSize or self.mask_array[j, i+1] == 0:
                        rightMask[j, i] = patch_value if patch_value else self.side_id

                    # Back (-y) (y flipped)
                    if j + 1 == jSize or self.mask_array[j+1, i] == 0:
                        backMask[j, i] = patch_value if patch_value else self.side_id

                    # Front (+y) (y flipped)
                    if j == 0 or self.mask_array[j-1, i] == 0:
                        frontMask[j, i] = patch_value if patch_value else self.side_id

                    # Bottom (-z)
                    patch_value = 0 if self.patch_ids_bottom is None else self.patch_ids_bottom[j, i]
                    bottomMask[j, i] = patch_value if patch_value else self.bottom_id

                    # Top (+z)
                    patch_value = 0 if self.patch_ids_top is None else self.patch_ids_top[j, i]
                    topMask[j, i] = patch_value if patch_value else self.top_id

        # Generate asc / sa files
        writeFn = write_patch_matrix_as_asc
        settings = {
            'xllcorner': xllcorner,
            'yllcorner': yllcorner,
            'cellsize': cellsize,
            'NODATA_value': 0,
        }
        short_name = name[:-6]

        left_file_path = get_absolute_path(f'{short_name}_left.asc')
        writeFn(leftMask, left_file_path, **settings)

        right_file_path = get_absolute_path(f'{short_name}_right.asc')
        writeFn(rightMask, right_file_path, **settings)

        front_file_path = get_absolute_path(f'{short_name}_front.asc')
        writeFn(frontMask, front_file_path, **settings)

        back_file_path = get_absolute_path(f'{short_name}_back.asc')
        writeFn(backMask, back_file_path, **settings)

        top_file_path = get_absolute_path(f'{short_name}_top.asc')
        writeFn(topMask, top_file_path, **settings)

        bottom_file_path = get_absolute_path(f'{short_name}_bottom.asc')
        writeFn(bottomMask, bottom_file_path, **settings)

        # Trigger conversion
        print('=== pfmask-to-pfsol ===: BEGIN')
        extra = []
        if vtk:
            extra.append('--vtk')
            extra.append(f'{output_file_path[:-6]}.vtk')
        os.system(f'$PARFLOW_DIR/bin/pfmask-to-pfsol --mask-top {top_file_path} --mask-bottom {bottom_file_path} --mask-left {left_file_path} --mask-right {right_file_path} --mask-front {front_file_path} --mask-back {back_file_path} --pfsol {output_file_path} {" ".join(extra)}')
        print('=== pfmask-to-pfsol ===: END')
        return self

    def for_key(self, geomItem):
        geomItem.InputType = 'SolidFile'
        geomItem.FileName = self.name
        return self
