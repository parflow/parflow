import numpy as np


def compute_top(mask):
    """Python version of the C ComputeTop function.

    Computes the top indices of the computation domain as defined by
    the mask values. Mask has values 0 outside of domain so first
    non-zero entry is the top.

    Args:
        mask (numpy.ndarray): mask for the computation domain with shape (nz, nx, ny)

    Returns:
        A new numpy ndarray with dimensions (1, mask.shape[0], mask.shape[1])
        with (z) indices of the top surface for each i, j location.
    """
    nz, nx, ny = mask.shape
    top = np.ndarray((1, nx, ny))

    for j in range(ny):
        for i in range(nx):
            k = nz - 1
            while k >= 0:
                if mask[k, i, j] > 0.0:
                    break
                k -= 1
            top[0, i, j] = k

    return top


def extract_top(data, top):
    """Python version of the C ExtractTop function.

    Extracts the top values of a dataset based on a top dataset
    (which contains the z indices that define the top of the domain).

    Args:
        data (numpy.ndarray): array of data values with shape (nz, nx, ny)
        top (numpy.ndarray): array of z indices for each i, j location with shape (1, nx, ny)

    Returns:
        Returns a ndarray with top values extracted for each i,j location.

    Raises:
        ValueError: If a z index in top is outside the range of nz.
    """
    nz, nx, ny = data.shape
    top_values_of_data = np.ndarray((1, nx, ny))

    for j in range(ny):
        for i in range(nx):
            k = int(top[0, i, j])
            if k < 0:
                top_values_of_data[0, i, j] = 0.0
            elif k < nz:
                top_values_of_data[0, i, j] = data[k, i, j]
            else:
                raise ValueError(f"Index in top (k={k}) is outside of data (nz={nz})")

    return top_values_of_data
