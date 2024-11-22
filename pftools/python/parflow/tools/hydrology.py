"""hydrology module

Helper functions to calculate standard hydrology measures
"""

import numpy as np
import parflow.tools.io

# -----------------------------------------------------------------------------


def compute_hydraulic_head(pressure, z_0, dz):
    """
    Compute hydraulic head from a 3D pressure ndarray considering elevation.

    This function uses the formula hh = hp + hz, where hh is the hydraulic head,
    hp the pressure head, and hz the elevation head.

    Args:
        pressure (numpy.ndarray): 3D array of pressure values.
        z_0 (float): Elevation of the top layer.
        dz (float): Distance between z-layers.

    Returns:
        numpy.ndarray: 3D array of hydraulic head values.
    """
    num_layers = pressure.shape[0]
    elevation = z_0 + np.arange(num_layers) * dz + dz / 2
    hydraulic_head = pressure + elevation[:, np.newaxis, np.newaxis]
    return hydraulic_head


def compute_water_table_depth(saturation, top, dz):
    """Computes the water table depth as the first cell with a saturation=1 starting from top.

    Depth is depth below the top surface. Negative values indicate the water table was not found,
    either below domain or the column at (i,j) is outside of domain.

    Args:
        saturation: ndarray of shape (nz, nx, ny)
        top: ndarray of shape (1, nx, ny)
        dz: distance between grid points in the z direction
    Returns:
        A new ndarray water_table_depth of shape (1, nx, ny) with depth values at each (i,j) location.
    """
    nz, nx, ny = saturation.shape
    water_table_depth = np.ndarray((1, nx, ny))
    for j in range(ny):
        for i in range(nx):
            top_k = top[0, i, j]
            if top_k < 0:
                # Inactive column so set to bogus value
                water_table_depth[0, i, j] = -9999999.0
            elif top_k < nz:
                # Loop down until we find saturation >= 1
                k = top_k
                while k >= 0 and saturation[k, i, j] < 1:
                    k -= 1

                # Make sure water table was found in the column, set to bogus value if not
                if k >= 0:
                    water_table_depth[0, i, j] = (top_k - k) * dz
                else:
                    water_table_depth[0, i, j] = -9999999.0
            else:
                print(f"Error: Index in top (k={top_k}) is outside of domain (nz={nz})")


def calculate_water_table_depth(pressure, saturation, dz):
    """
    Calculate water table depth from the land surface
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-ny-by-nx ndarray of saturation values (bottom layer to top layer)
    :param dz: An ndarray of shape (nz,) of thickness values (bottom layer to top layer)
    :return: A ny-by-nx ndarray of water table depth values (measured from the top)
    """
    # Handle single-column pressure/saturation values
    if pressure.ndim == 1:
        pressure = pressure[:, np.newaxis, np.newaxis]
    if saturation.ndim == 1:
        saturation = saturation[:, np.newaxis, np.newaxis]

    domain_thickness = np.sum(dz)

    # Sentinel values padded to aid in subsequent calculations
    # A layer of thickness 0 at the top
    dz = np.hstack([dz, 0])
    # An unsaturated layer at the top
    # pad_width is a tuple of (n_before, n_after) for each dimension
    saturation = np.pad(
        saturation,
        pad_width=((0, 1), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Elevation of the center of each layer from the bottom (bottom layer to top layer)
    _elevation = np.cumsum(dz) - (dz / 2)
    # Make 3D with shape (nz, 1, 1) to allow subsequent operations
    _elevation = _elevation[:, np.newaxis, np.newaxis]

    """
    Indices of first unsaturated layer across the grid, going from bottom to top
    with 0 indicating the bottom layer.

    NOTE: np.argmax on a boolean array returns the index of the first True value it encounters.
    It returns a 0 if it doesn't find *any* True values.
    This would normally be a problem - however, since we added a sentinel 0 value to the sat array,
    we will not encounter this case.
    """
    z_indices = np.maximum(
        np.argmax(saturation < 1, axis=0)
        - 1,  # find first unsaturated layer; back up one cell
        0,  # clamp min. index value to 0
    )
    # Make 3D with shape (1, ny, nx) to allow subsequent operations
    z_indices = z_indices[np.newaxis, ...]

    saturation_elevation = np.take_along_axis(
        _elevation, z_indices, axis=0
    )  # shape (1, ny, nx)
    ponding_depth = np.take_along_axis(pressure, z_indices, axis=0)  # shape (1, ny, nx)

    wt_height = saturation_elevation + ponding_depth  # shape (1, ny, nx)
    wt_height = np.clip(
        wt_height, 0, domain_thickness
    )  # water table height clamped between 0<->domain thickness
    wtd = domain_thickness - wt_height  # shape (1, ny, nx)

    return wtd.squeeze(axis=0)  # shape (ny, nx)


# -----------------------------------------------------------------------------


def calculate_subsurface_storage(
    porosity, pressure, saturation, specific_storage, dx, dy, dz, mask=None
):
    """
    Calculate gridded subsurface storage across several layers.

    For each layer in the subsurface, storage consists of two parts
      - incompressible subsurface storage
        (porosity * saturation * depth of this layer) * dx * dy
      - compressible subsurface storage
        (pressure * saturation * specific storage * depth of this layer) * dx * dy

    :param porosity: A nz-by-ny-by-nx ndarray of porosity values (bottom layer to top layer)
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-ny-by-nx ndarray of saturation values (bottom layer to top layer)
    :param specific_storage: A nz-by-ny-by-nx ndarray of specific storage values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: A nz-by-ny-by-nx ndarray of subsurface storage values, spanning all layers (bottom to top)
    """
    if mask is None:
        mask = np.ones_like(porosity)

    mask = np.where(mask > 0, 1, 0)
    dz = dz[
        :, np.newaxis, np.newaxis
    ]  # make 3d so we can broadcast the multiplication below
    incompressible = porosity * saturation * dz * dx * dy
    compressible = pressure * saturation * specific_storage * dz * dx * dy
    compressible = np.where(pressure < 0, 0, compressible)
    total = incompressible + compressible
    total[mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


# -----------------------------------------------------------------------------


def calculate_surface_storage(pressure, dx, dy, mask=None):
    """
    Calculate gridded surface storage on the top layer.

    Surface storage is given by:
      Pressure at the top layer * dx * dy (for pressure values > 0)

    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: An ny-by-nx ndarray of surface storage values
    """
    if mask is None:
        mask = np.ones_like(pressure)

    mask = np.where(mask > 0, 1, 0)
    surface_mask = mask[-1, ...]
    total = pressure[-1, ...] * dx * dy
    total[total < 0] = 0  # surface storage is 0 when pressure < 0
    total[surface_mask == 0] = (
        0  # output values for points outside the mask are clamped to 0
    )
    return total


# -----------------------------------------------------------------------------


def calculate_evapotranspiration(et, dx, dy, dz, mask=None):
    """
    Calculate gridded evapotranspiration across several layers.

    :param et: A nz-by-ny-by-nx ndarray of evapotranspiration flux values with units 1/T (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: A nz-by-ny-by-nx ndarray of evapotranspiration values (units L^3/T), spanning all layers (bottom to top)
    """
    if mask is None:
        mask = np.ones_like(et)

    mask = np.where(mask > 0, 1, 0)
    dz = dz[
        :, np.newaxis, np.newaxis
    ]  # make 3d so we can broadcast the multiplication below
    total = et * dz * dx * dy
    total[mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


# -----------------------------------------------------------------------------


def _overland_flow(pressure_top, slopex, slopey, mannings, dx, dy):
    # Calculate fluxes across east and north faces

    # ---------------
    # The x direction
    # ---------------
    qx = (
        -(np.sign(slopex) * (np.abs(slopex) ** 0.5) / mannings)
        * (pressure_top ** (5 / 3))
        * dy
    )

    # Upwinding to get flux across the east face of cells - based on qx[i] if it is positive and qx[i+1] if negative
    qeast = np.maximum(0, qx[:, :-1]) - np.maximum(0, -qx[:, 1:])

    # Add the left boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[0] is negative
    qeast = np.hstack([-np.maximum(0, -qx[:, 0])[:, np.newaxis], qeast])

    # Add the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[-1] is positive
    qeast = np.hstack([qeast, np.maximum(0, qx[:, -1])[:, np.newaxis]])

    # ---------------
    # The y direction
    # ---------------
    qy = (
        -(np.sign(slopey) * (np.abs(slopey) ** 0.5) / mannings)
        * (pressure_top ** (5 / 3))
        * dx
    )

    # Upwinding to get flux across the north face of cells - based in qy[j] if it is positive and qy[j+1] if negative
    qnorth = np.maximum(0, qy[:-1, :]) - np.maximum(0, -qy[1:, :])

    # Add the top boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[0] is negative
    qnorth = np.vstack([-np.maximum(0, -qy[0, :]), qnorth])

    # Add the bottom boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[-1] is positive
    qnorth = np.vstack([qnorth, np.maximum(0, qy[-1, :])])

    return qeast, qnorth


# -----------------------------------------------------------------------------


def _overland_flow_kinematic(
    mask, pressure_top, slopex, slopey, mannings, dx, dy, epsilon
):
    ##### --- ######
    #     qx       #
    ##### --- ######

    # We will be tweaking the slope values for this algorithm, so we make a copy
    slopex = np.copy(slopex)
    slopey = np.copy(slopey)
    mannings = np.copy(mannings)

    # We're only interested in the surface mask, as an ny-by-nx array
    mask = mask[-1, ...]

    # Find all patterns of the form
    #  -------
    # | 0 | 1 |
    #  -------
    # and copy the slopex, slopey and mannings values from the '1' cells to the corresponding '0' cells
    _x, _y = np.where(np.diff(mask, axis=1, append=0) == 1)
    slopex[(_x, _y)] = slopex[(_x, _y + 1)]
    slopey[(_x, _y)] = slopey[(_x, _y + 1)]
    mannings[(_x, _y)] = mannings[(_x, _y + 1)]

    slope = np.maximum(epsilon, np.hypot(slopex, slopey))

    # Upwind pressure - this is for the north and east face of all cells
    # The slopes are calculated across these boundaries so the upper x boundaries are included in these
    # calculations. The lower x boundaries are added further down as q_x0
    pressure_top_padded = np.pad(
        pressure_top[:, 1:],
        (
            (
                0,
                0,
            ),
            (0, 1),
        ),
    )  # pad right
    pupwindx = np.maximum(0, np.sign(slopex) * pressure_top_padded) + np.maximum(
        0, -np.sign(slopex) * pressure_top
    )

    flux_factor = np.sqrt(slope) * mannings

    # Flux across the x directions
    q_x = -slopex / flux_factor * pupwindx ** (5 / 3) * dy

    # Fix the lower x boundary
    # Use the slopes of the first column
    q_x0 = (
        -slopex[:, 0]
        / flux_factor[:, 0]
        * np.maximum(0, np.sign(slopex[:, 0]) * pressure_top[:, 0]) ** (5 / 3)
        * dy
    )
    qeast = np.hstack([q_x0[:, np.newaxis], q_x])

    ##### --- ######
    #   qy   #
    ##### --- ######

    # Find all patterns of the form
    #  ---
    # | 0 |
    # | 1 |
    #  ---
    _x, _y = np.where(np.diff(mask, axis=0, append=0) == 1)
    slopey[(_x, _y)] = slopey[(_x + 1, _y)]
    slopex[(_x, _y)] = slopex[(_x + 1, _y)]
    mannings[(_x, _y)] = mannings[(_x + 1, _y)]

    slope = np.maximum(epsilon, np.hypot(slopex, slopey))

    # Upwind pressure - this is for the north and east face of all cells
    # The slopes are calculated across these boundaries so the upper y boundaries are included in these
    # calculations. The lower y boundaries are added further down as q_y0
    pressure_top_padded = np.pad(
        pressure_top[1:, :],
        (
            (
                0,
                1,
            ),
            (0, 0),
        ),
    )  # pad bottom
    pupwindy = np.maximum(0, np.sign(slopey) * pressure_top_padded) + np.maximum(
        0, -np.sign(slopey) * pressure_top
    )

    flux_factor = np.sqrt(slope) * mannings

    # Flux across the y direction
    q_y = -slopey / flux_factor * pupwindy ** (5 / 3) * dx

    # Fix the lower y boundary
    # Use the slopes of the first row
    q_y0 = (
        -slopey[0, :]
        / flux_factor[0, :]
        * np.maximum(0, np.sign(slopey[0, :]) * pressure_top[0, :]) ** (5 / 3)
        * dx
    )
    qnorth = np.vstack([q_y0, q_y])

    return qeast, qnorth


# -----------------------------------------------------------------------------


def calculate_overland_fluxes(
    pressure,
    slopex,
    slopey,
    mannings,
    dx,
    dy,
    flow_method="OverlandKinematic",
    epsilon=1e-5,
    mask=None,
):
    """
    Calculate overland fluxes across grid faces

    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param slopex: ny-by-nx
    :param slopey: ny-by-nx
    :param mannings: a scalar value, or a ny-by-nx ndarray
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
        'OverlandKinematic' by default.
    :param epsilon: Minimum slope magnitude for solver. Only applicable if flow_method='OverlandKinematic'.
        This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: A 2-tuple:
        qeast - A ny-by-(nx+1) ndarray of overland flux values
        qnorth - A (ny+1)-by-nx ndarray of overland flux values

    """

    """
    Numpy array origin is at the top left.
    The cardinal direction along axis 0 (rows) is North (going down!!).
    The cardinal direction along axis 1 (columns) is East (going right).
    qnorth (ny+1,nx) and qeast (ny,nx+1) values are to be interpreted as follows.

    +-------------------------------------> (East)
    |
    |                           qnorth_i,j (outflow if negative)
    |                                  +-----+------+
    |                                  |     |      |
    |                                  |     |      |
    |  qeast_i,j (outflow if negative) |-->  v      |---> qeast_i,j+1 (outflow if positive)
    |                                  |            |
    |                                  | Cell  i,j  |
    |                                  +-----+------+
    |                                        |
    |                                        |
    |                                        v
    |                           qnorth_i+1,j (outflow if positive)
    v
    (North)
    """

    # Handle cases when expected 2D arrays come in as 3D
    if slopex.ndim == 3:
        assert slopex.shape[0] == 1, f"Expected shape[0] of 3D ndarray {slopex} to be 1"
        slopex = slopex.squeeze(axis=0)
    if slopey.ndim == 3:
        assert slopey.shape[0] == 1, f"Expected shape[0] of 3D ndarray {slopey} to be 1"
        slopey = slopey.squeeze(axis=0)
    mannings = np.array(mannings)
    if mannings.ndim == 3:
        assert (
            mannings.shape[0] == 1
        ), f"Expected shape[0] of 3D ndarray {mannings} to be 1"
        mannings = mannings.squeeze(axis=0)

    pressure_top = pressure[-1, ...].copy()
    pressure_top = np.nan_to_num(pressure_top)
    pressure_top[pressure_top < 0] = 0

    assert flow_method in ("OverlandFlow", "OverlandKinematic"), "Unknown flow method"
    if flow_method == "OverlandKinematic":
        if mask is None:
            mask = np.ones_like(pressure)
        mask = np.where(mask > 0, 1, 0)
        qeast, qnorth = _overland_flow_kinematic(
            mask, pressure_top, slopex, slopey, mannings, dx, dy, epsilon
        )
    else:
        qeast, qnorth = _overland_flow(pressure_top, slopex, slopey, mannings, dx, dy)

    return qeast, qnorth


# -----------------------------------------------------------------------------


def calculate_overland_flow_grid(
    pressure,
    slopex,
    slopey,
    mannings,
    dx,
    dy,
    flow_method="OverlandKinematic",
    epsilon=1e-5,
    mask=None,
):
    """
    Calculate overland outflow per grid cell of a domain

    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param slopex: ny-by-nx
    :param slopey: ny-by-nx
    :param mannings: a scalar value, or a ny-by-nx ndarray
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
        'OverlandKinematic' by default.
    :param epsilon: Minimum slope magnitude for solver. Only applicable if kinematic=True.
        This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: A ny-by-nx ndarray of overland flow values
    """
    mask = np.where(mask > 0, 1, 0)
    qeast, qnorth = calculate_overland_fluxes(
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        flow_method=flow_method,
        epsilon=epsilon,
        mask=mask,
    )

    # Outflow is a positive qeast[i,j+1] or qnorth[i+1,j] or a negative qeast[i,j], qnorth[i,j]
    outflow = (
        np.maximum(0, qeast[:, 1:])
        + np.maximum(0, -qeast[:, :-1])
        + np.maximum(0, qnorth[1:, :])
        + np.maximum(0, -qnorth[:-1, :])
    )

    # Set the outflow values outside the mask to 0
    outflow[mask[-1, ...] == 0] = 0

    return outflow


# -----------------------------------------------------------------------------


def calculate_overland_flow(
    pressure,
    slopex,
    slopey,
    mannings,
    dx,
    dy,
    flow_method="OverlandKinematic",
    epsilon=1e-5,
    mask=None,
):
    """
    Calculate overland outflow out of a domain

    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param slopex: ny-by-nx
    :param slopey: ny-by-nx
    :param mannings: a scalar value, or a ny-by-nx ndarray
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
        'OverlandKinematic' by default.
    :param epsilon: Minimum slope magnitude for solver. Only applicable if flow_method='OverlandKinematic'.
        This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
        If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
    :return: A float value representing the total overland flow over the domain.
    """
    qeast, qnorth = calculate_overland_fluxes(
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        flow_method=flow_method,
        epsilon=epsilon,
        mask=mask,
    )

    if mask is not None:
        mask = np.where(mask > 0, 1, 0)
        surface_mask = mask[-1, ...]  # shape ny, nx
    else:
        surface_mask = np.ones_like(slopex)  # shape ny, nx

    # Important to typecast mask to float to avoid values wrapping around when performing a np.diff
    surface_mask = surface_mask.astype("float")
    # Find edge pixels for our surface mask along each face - N/S/W/E
    # All of these have shape (ny, nx) and values as 0/1

    # find forward difference of +1 on axis 0
    edge_south = np.maximum(0, np.diff(surface_mask, axis=0, prepend=0))
    # find forward difference of -1 on axis 0
    edge_north = np.maximum(0, -np.diff(surface_mask, axis=0, append=0))
    # find forward difference of +1 on axis 1
    edge_west = np.maximum(0, np.diff(surface_mask, axis=1, prepend=0))
    # find forward difference of -1 on axis 1
    edge_east = np.maximum(0, -np.diff(surface_mask, axis=1, append=0))

    # North flux is the sum of +ve qnorth values (shifted up by one) on north edges
    flux_north = np.sum(
        np.maximum(0, np.roll(qnorth, -1, axis=0)[np.where(edge_north == 1)])
    )
    # South flux is the negated sum of -ve qnorth values for south edges
    flux_south = np.sum(np.maximum(0, -qnorth[np.where(edge_south == 1)]))
    # West flux is the negated sum of -ve qeast values of west edges
    flux_west = np.sum(np.maximum(0, -qeast[np.where(edge_west == 1)]))
    # East flux is the sum of +ve qeast values (shifted left by one) for east edges
    flux_east = np.sum(
        np.maximum(0, np.roll(qeast, -1, axis=1)[np.where(edge_east == 1)])
    )

    flux = flux_north + flux_south + flux_west + flux_east
    return flux
