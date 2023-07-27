import os
from parflow.tools.io import read_pfb

def py_MSigDiff(file1, file2, m, abs_zero=0.0):
    assert isinstance(abs_zero, float)
    assert abs_zero >= 0

    # Load data from the files and create xarray DataArrays
    try:
        data1 = read_pfb(file1)
        data2 = read_pfb(file2)
    except Exception as e:
        print("Error: Failed to load data from files.", e)
        return

    if not data1.shape == data2.shape:
        print("Error: Data arrays must have the same dimensions.")
        return
        
    nx, ny, nz = data1.shape

    if m >= 0:
        sig_dig_rhs = 0.5 / 10 ** m
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

    result_list = []
    if not m_sig_digs_everywhere:
        sig_digs = 0
        sdiff = max_sdiff
        while sdiff <= 0.5e-01:
            sdiff *= 10.0
            sig_digs += 1

        result_list.append([mi, mj, mk, sig_digs])
        result_list.append(max_adiff)

    return result_list

def py_pftestFile(file, correct_file, message, sig_digits=6):
    if not os.path.exists(file):
        print(f"FAILED : output file <{file}> not created")
        return False

    if not os.path.exists(correct_file):
        print(f"FAILED : regression check output file <{correct_file}> does not exist")
        return False

    result = py_MSigDiff(correct_file, file, sig_digits)
    if len(result) != 0:
        mSigDigs, maxAbsDiff = result

        i, j, k, sig_digs = mSigDigs

        print(f"FAILED : {message}")
        print(f"\tMinimum significant digits at ({i:3d}, {j:3d}, {k:3d}) = {sig_digs:2d}")
        print(f"\tCorrect value {xr.open_dataarray(correct_file)[i, j, k]:e}")
        print(f"\tComputed value {xr.open_dataarray(file)[i, j, k]:e}")

        elt_diff = abs(xr.open_dataarray(correct_file)[i, j, k] - xr.open_dataarray(file)[i, j, k])
        print(f"\tDifference {elt_diff:e}")

        print(f"\tMaximum absolute difference = {maxAbsDiff:e}")

        return False
    else:
        return True

def py_pftestFileWithAbs(file, correct_file, message, abs_value, sig_digits=6):
    if not os.path.exists(file):
        print(f"FAILED : output file <{file}> not created")
        return False

    if not os.path.exists(correct_file):
        print(f"FAILED : regression check output file <{correct_file}> does not exist")
        return False

    result = py_MSigDiff(correct_file, file, sig_digits)
    if len(result) != 0:
        mSigDigs, maxAbsDiff = result

        i, j, k, sig_digs = mSigDigs

        elt_diff = abs(xr.open_dataarray(correct_file)[i, j, k] - xr.open_dataarray(file)[i, j, k])

        if elt_diff > abs_value:
            print(f"FAILED : {message}")
            print(f"\tMinimum significant digits at ({i:3d}, {j:3d}, {k:3d}) = {sig_digs:2d}")
            print(f"\tCorrect value {xr.open_dataarray(correct_file)[i, j, k]:e}")
            print(f"\tComputed value {xr.open_dataarray(file)[i, j, k]:e}")
            print(f"\tDifference {elt_diff:e}")
            print(f"\tMaximum absolute difference = {maxAbsDiff:e}")
            return False

    return True
