import os
from parflow.tools.io import read_pfb, read_pfsb


def pf_test_equal(a, b, message):
    pf_eps = 1e-5
    if abs(a - b) > pf_eps:
        print(f"FAILED : {message} {a} is not equal to {b}")
        return False

    return True


def msig_diff(data1, data2, m, abs_zero=0.0):
    """Python version of the C MSigDiff function.

    Two nd arrays are given as the first two arguments.
    The grid point at which the number of digits in agreement
    (significant digits) is fewest is determined.  If m >= 0
    then the coordinate whose two values differ in more than
    m significant digits will be computed.  If m < 0 then the
    coordinate whose values have a minimum number of significant
    digits will be computed. The number of the fewest significant
    digits is determined, and the maximum absolute difference is
    computed. The only coordinates that will be considered will be
    those whose differences are greater than absolute zero.

    Args:
        data1 (numpy.ndarray): first ndarray
        data2 (numpy.ndarray): second ndarray
        m (int): number of significant digits
        abs_zero (float): threshold below which all numbers are considered to be 0

    Returns:
        A list of the following form is returned upon success:

               [[i j k s] max_adiff]

        where i, j, and k are the coordinates computed, sd is the
        minimum number of significant digits computed, and max_adiff
        is the maximum absolute difference computed.

    Raises:
        ValueError: If data1 and data2 do not have the same shape.
    """
    assert isinstance(abs_zero, float)
    assert abs_zero >= 0

    if not data1.shape == data2.shape:
        raise ValueError("Error: Data arrays must have the same dimensions.")

    nx, ny, nz = data1.shape

    if m >= 0:
        sig_dig_rhs = 0.5 / 10**m
    else:
        sig_dig_rhs = 0.0

    m_sig_digs_everywhere = True
    max_sdiff = 0.0
    max_adiff = 0.0

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                adiff = abs(data1[i, j, k] - data2[i, j, k])
                amax = max(abs(data1[i, j, k]), abs(data2[i, j, k]))

                if max_adiff < adiff:
                    max_adiff = adiff

                m_sig_digs = True
                if amax > abs_zero:
                    sdiff = adiff / amax
                    if sdiff > sig_dig_rhs:
                        m_sig_digs = False

                if not m_sig_digs:
                    if sdiff > max_sdiff:
                        max_sdiff = sdiff
                        mi, mj, mk = i, j, k

                    m_sig_digs_everywhere = False

    result = []
    if not m_sig_digs_everywhere:
        sig_digs = 0
        sdiff = max_sdiff
        while sdiff <= 0.5e-01:
            sdiff *= 10.0
            sig_digs += 1

        result.append([mi, mj, mk, sig_digs])
        result.append(max_adiff)

    return result


def pf_test_file(file, correct_file, message, sig_digits=6):
    """Python version of the tcl pftestFile procedure.

    Two file paths are given as the first two arguments.
    The function reads them into ndarrays and calls a comparison
    function (msig_diff) to check if the files differ by
    more than sig_digits significant digits in any coordinate.
    If they do, the function prints an error message and the
    coordinate in which the files have the greatest difference.

    Args:
        file (string): path to test output file
        correct_file (string): path to reference output file
        message (string): message to print in case of test failure
        sig_digits (int): number of significant digits

    Returns:
        bool: True if files differ in no more than sig_digits
              significant digits in all coordinates. False otherwise.

    Raises:
        FileNotFoundError: If file or correct_file do not exist.
        Exception: If the function fails to read either file into an ndarray.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"FAILED : output file <{file}> not created")

    if not os.path.exists(correct_file):
        raise FileNotFoundError(
            f"FAILED : regression check output file <{correct_file}> does not exist"
        )

    try:
        if file.endswith(".pfb"):
            data = read_pfb(file)
            correct_data = read_pfb(correct_file)
        elif file.endswith(".pfsb"):
            data = read_pfsb(file)
            correct_data = read_pfsb(correct_file)
        else:
            raise ValueError("Unknown parflow file type.")
    except Exception as e:
        print("Error: Failed to load data from files...", e)
        return False

    result = msig_diff(data, correct_data, sig_digits)
    if (len(result)) == 0:
        return True

    m_sig_digs, max_abs_diff = result

    i, j, k, sig_digs = m_sig_digs

    print(f"FAILED : {message}")
    print(f"\tMinimum significant digits at ({i:3d}, {j:3d}, {k:3d}) = {sig_digs:2d}")
    print(f"\tCorrect value {correct_data[i, j, k]:e}")
    print(f"\tComputed value {data[i, j, k]:e}")

    elt_diff = abs(data[i, j, k] - correct_data[i, j, k])
    print(f"\tDifference {elt_diff:e}")

    print(f"\tMaximum absolute difference = {max_abs_diff:e}")

    return False


def pf_test_file_with_abs(file, correct_file, message, abs_value, sig_digits=6):
    """Python version of the tcl pftestFileWithAbs procedure.

    Two file paths are given as the first two arguments.
    The function reads them into ndarrays and calls a comparison
    function (msig_diff) to check if the files differ by
    more than sig_digits significant digits in any coordinate.
    If they do, the function checks if the difference in that
    coordinate is greater than abs_value. If it is, it prints
    an error message and the coordinate in which the files have
    the greatest difference.

    Args:
        file (string): path to test output file
        correct_file (string): path to reference output file
        message (string): message to print in case of test failure
        abs_value (float): threshold to determine if two files differ
                           enough to fail the test
        sig_digits (int): number of significant digits

    Returns:
        bool: True if files differ in no more than sig_digits
              significant digits in all coordinates. False otherwise.

    Raises:
        FileNotFoundError: If file or correct_file do not exist.
        Exception: If the function fails to read either file into an ndarray.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"FAILED : output file <{file}> not created")

    if not os.path.exists(correct_file):
        raise FileNotFoundError(
            f"FAILED : regression check output file <{correct_file}> does not exist"
        )

    try:
        data = read_pfb(file)
        correct_data = read_pfb(correct_file)
    except Exception as e:
        print("Error: Failed to load data from files.", e)
        return False

    result = msig_diff(data, correct_data, sig_digits)
    if len(result) != 0:
        m_sig_digs, max_abs_diff = result

        i, j, k, sig_digs = m_sig_digs

        elt_diff = abs(data[i, j, k] - correct_data[i, j, k])

        if elt_diff > abs_value:
            print(f"FAILED : {message}")
            print(
                f"\tMinimum significant digits at ({i:3d}, {j:3d}, {k:3d}) = {sig_digs:2d}"
            )
            print(f"\tCorrect value {correct_data[i, j, k]:e}")
            print(f"\tComputed value {data[i, j, k]:e}")
            print(f"\tDifference {elt_diff:e}")
            print(f"\tMaximum absolute difference = {max_abs_diff:e}")
            return False

    return True
