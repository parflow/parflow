import os
import numpy as np


def make_solid_file(
    nx,
    ny,
    bottom_val,
    side_val,
    top_val1,
    top_val2,
    latsize,
    zdepth,
    river_dir,
    root_name,
    out_dir,
    pftools_path="./",
    top_val3=None,
    top_val4=None,
):
    """Create solid files for Tilted-V domain with optional seepage patches.

    Args:
        nx (int): number of grid cells in x
        ny (int): number of grid cells in y
        bottom_val (int): value for bottom patch
        side_val (int): value for side patch
        top_val1 (int): value for first top patch (the slopes)
        top_val2 (int): value for second top patch (the channel)
        top_val3 (int): value for third top patch (seepage one, optional.
            If its the only one, it will be in the middle of the channel.)
        top_val4 (int): value for fourth top patch (seepage two, optional)
        latsize (float): size of grid cell in lateral dimension
        zdepth (float): vertical thickness of grid cells
        river_dir (int): 1 for river in x direction and 2 for y
        root_name (str): root name for file outputs
        out_dir (str): directory to write files to
        pftools_path (str): path to the pfmask-to-pfsol utility

    Returns:
        None
    """

    # setup the asc file header
    header1 = "ncols          " + str(nx) + "\n"
    header2 = "nrows          " + str(ny) + "\n"
    header3 = "xllcorner          0" + "\n"
    header4 = "yllcorner          0" + "\n"
    header5 = "cellsize          " + str(latsize) + "\n"
    header6 = "NODATA_value          0"
    header = header1 + header2 + header3 + header4 + header5 + header6

    # Make top mask in for river
    mask = np.ones((ny, nx))
    patch = np.zeros((ny, nx))

    if river_dir == 1:
        patch[0 : int(np.floor(ny / 2)),] = top_val1
        patch[int(np.floor(ny / 2)),] = top_val2
        patch[(int(np.floor(ny / 2)) + 1) :,] = top_val1
        if top_val3 is not None and top_val4 is None:
            # Place one seepage patch in the middle of the river patch
            patch[
                int(np.floor(ny / 2)),
                int(np.floor(nx / 2)) : int(np.floor(nx / 2)) + 1,
            ] = top_val3
        elif top_val3 is not None and top_val4 is not None:
            # Place two seepage patches at 1/3 and 2/3 along the river patch
            patch[
                int(np.floor(ny / 2)),
                int(np.floor(nx / 3)) : int(np.floor(nx / 3)) + 1,
            ] = top_val3
            patch[
                int(np.floor(ny / 2)),
                int(np.floor(2 * nx / 3)) : int(np.floor(2 * nx / 3)) + 1,
            ] = top_val4
    else:
        patch[:, 0 : int(np.floor(nx / 2))] = top_val1
        patch[:, int(np.floor(nx / 2))] = top_val2
        patch[:, (int(np.floor(nx / 2)) + 1) :] = top_val1
        if top_val3 is not None and top_val4 is None:
            # Place one seepage patch in the middle of the river patch
            patch[
                int(np.floor(ny / 2)) : int(np.floor(ny / 2)) + 1,
                int(np.floor(nx / 2)),
            ] = top_val3
        elif top_val3 is not None and top_val4 is not None:
            # Place two seepage patches at 1/3 and 2/3 along the river patch
            patch[
                int(np.floor(ny / 3)) : int(np.floor(ny / 3)) + 1,
                int(np.floor(nx / 2)),
            ] = top_val3
            patch[
                int(np.floor(2 * ny / 3)) : int(np.floor(2 * ny / 3)) + 1,
                int(np.floor(nx / 2)),
            ] = top_val4

    #  Make arrays for asc files
    # front
    front = np.zeros(nx * ny)
    front[(nx * ny - nx) :] = side_val
    front_file = os.path.join(out_dir, (root_name + "_front.asc"))
    np.savetxt(front_file, front, fmt="%i", header=header, comments="")

    # back
    back = np.zeros(nx * ny)
    back[0:nx] = side_val
    back_file = os.path.join(out_dir, (root_name + "_back.asc"))
    np.savetxt(back_file, back, fmt="%i", header=header, comments="")

    # left
    left = np.zeros(nx * ny)
    left[::nx] = side_val
    left_file = os.path.join(out_dir, (root_name + "_left.asc"))
    np.savetxt(left_file, left, fmt="%i", header=header, comments="")

    # right
    right = np.zeros(nx * ny)
    right[(nx - 1) :: nx] = side_val
    right_file = os.path.join(out_dir, (root_name + "_right.asc"))
    np.savetxt(right_file, right, fmt="%i", header=header, comments="")

    # bottom
    bottom = np.ones(nx * ny) * bottom_val
    bottom_file = os.path.join(out_dir, (root_name + "_bottom.asc"))
    np.savetxt(bottom_file, bottom, fmt="%i", header=header, comments="")

    # top
    top = patch.reshape(nx * ny)
    top_file = os.path.join(out_dir, (root_name + "_top.asc"))
    np.savetxt(top_file, top, fmt="%i", header=header, comments="")

    # Run the pfmask-to-pfsol tool
    vtk_name = root_name + ".vtk"
    vtk_file = os.path.join(out_dir, vtk_name)

    pfsol_name = root_name + ".pfsol"
    pfsol_file = os.path.join(out_dir, pfsol_name)

    pftool_path = os.path.join(pftools_path, "pfmask-to-pfsol")

    pftools_command = (
        pftool_path
        + "  --mask-top "
        + top_file
        + " --mask-bottom "
        + bottom_file
        + " --mask-left "
        + left_file
        + " --mask-right "
        + right_file
        + "  --mask-front "
        + front_file
        + "  --mask-back "
        + back_file
        + " --z-top "
        + str(zdepth)
        + " --z-bottom 0.0"
        + " --vtk "
        + vtk_file
        + " --pfsol "
        + pfsol_file
    )
    os.system(pftools_command)
    print(pftools_command)

    # clean up the asc files
    os.remove(back_file)
    os.remove(front_file)
    os.remove(left_file)
    os.remove(right_file)
    os.remove(top_file)
    os.remove(bottom_file)
