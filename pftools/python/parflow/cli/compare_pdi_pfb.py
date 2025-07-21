# ______________________________________________________________________________
#
# This script compares the pdi and pfb files
# ______________________________________________________________________________

import h5py as h5py
import numpy as np
import struct
import glob
import argparse
import traceback


def compare_files(name):
    """
    Compares PDI (HDF5) files with PFB binary files for a given base name.

    This function processes all PDI files in HDF5 format that match the
    given base name, extracts relevant data, and compares it against the
    corresponding PFB binary file.
    The comparison includes checking subvectors and applying optional
    tolerance thresholds.

    Args:
        name (str): The base name of the files (excluding extensions).

    Returns:
        None: Prints the data comparison results, including any detected errors.
    """
    print(
        f"\n_______________________________ \nStart analysis for base name: {name} \n"
    )
    # Find HDF5 files
    hdf5_files = sorted(glob.glob(f"{name}*.h5*"))
    print(f"HDF5 files: {hdf5_files}")
    if not hdf5_files:
        print(f"\nError: No HDF5 files found for '{name}'.\n")
        return

    # Extract metadata
    f = h5py.File(hdf5_files[0], "r")
    vector_data = f["vector_data"]
    subvectors = vector_data["subvectors"]
    subgrids = vector_data["grid"]["subgrids"]
    all_subgrids = vector_data["grid"]["all_subgrids"]
    subregions = vector_data["grid"]["subgrids"]["subregions"]

    X = np.array(f["X"])
    Y = np.array(f["Y"])
    Z = np.array(f["Z"])

    NX = np.array(f["NX"], dtype=int)
    NY = np.array(f["NY"])
    NZ = np.array(f["NZ"])

    DX = np.array(f["DX"])
    DY = np.array(f["DY"])
    DZ = np.array(f["DZ"])

    drop_tolerance = np.array(f["drop_tolerance"])
    with_tolerance = np.array(f["with_tolerance"])

    # Find PFB files
    if with_tolerance <= 0:
        binary_files = sorted(glob.glob(name + "*.pfb"))
    else:
        binary_files = sorted(glob.glob(f"{name}.pfsb"))

    if not binary_files:
        print(f"\nError: No binary PFB/PFSB files found for '{name}'.\n")
        return

    print(f"\nPF Binary files: {binary_files}")

    # ___________________________________________________________________________
    # HDF5 File Processing
    # ___________________________________________________________________________

    print(f"\n_______________________________ \nAnalyzing PDI Files\n")
    print(" Metadata")
    print(" Number of subvectors from PDI: {}".format(len(subvectors)))
    print(" X: {}, Y: {}, Z: {}".format(X, Y, Z))
    print(" NX: {}, NY: {}, NZ: {}".format(NX, NY, NZ))
    print(" DX: {}, DY: {}, DZ: {}".format(DX, DY, DZ))
    print(" drop_tolerance: {}".format(drop_tolerance))
    print(" with_tolerance: {}".format(with_tolerance))
    pdi_subvectors = []
    for file in hdf5_files:
        print("\nFile: {}".format(file))
        f = h5py.File(file, "r")
        for igrid, subvector in enumerate(subvectors):
            d = {}
            d["ix"] = subregions[igrid]["ix"][0]
            d["iy"] = subregions[igrid]["iy"][0]
            d["iz"] = subregions[igrid]["iz"][0]

            d["nx"] = subregions[igrid]["nx"][0]
            d["ny"] = subregions[igrid]["ny"][0]
            d["nz"] = subregions[igrid]["nz"][0]

            d["rx"] = subregions[igrid]["rx"][0]
            d["ry"] = subregions[igrid]["ry"][0]
            d["rz"] = subregions[igrid]["rz"][0]

            d["ix_v"] = subvector["data_space"]["ix"]
            d["iy_v"] = subvector["data_space"]["iy"]
            d["iz_v"] = subvector["data_space"]["iz"]

            d["nx_v"] = subvector["data_space"]["nx"]
            d["ny_v"] = subvector["data_space"]["ny"]
            d["nz_v"] = subvector["data_space"]["nz"]

            yinc = d["nx_v"] - d["nx"]
            zinc = d["nx_v"] * d["ny_v"] - d["ny"] * d["nx_v"]

            if with_tolerance <= 0:
                num_elements = d["nx"] * d["ny"] * d["nz"]
                first_index = (d["ix"] - d["ix_v"]) + (
                    (d["iy"] - d["iy_v"]) + (d["iz"] - d["iz_v"]) * d["ny_v"]
                ) * d["nx_v"]
                d["data"] = np.zeros([d["nx"], d["ny"], d["nz"]])
                k = first_index
                for iz in range(d["iz"], d["iz"] + d["nz"]):
                    for iy in range(d["iy"], d["iy"] + d["ny"]):
                        for ix in range(d["ix"], d["ix"] + d["nx"]):
                            d["data"][ix, iy, iz] = subvector["data"][k]
                            k += 1
                        k += yinc
                    k += zinc
            else:
                num_elements = d["nx"] * d["ny"] * d["nz"]
                first_index = (d["ix"] - d["ix_v"]) + (
                    (d["iy"] - d["iy_v"]) + (d["iz"] - d["iz_v"]) * d["ny_v"]
                ) * d["nx_v"]

                d["data"] = np.zeros([num_elements])
                d["indx"] = np.zeros([num_elements])
                d["indy"] = np.zeros([num_elements])
                d["indz"] = np.zeros([num_elements])

                n = 0
                k = first_index
                for iz in range(d["iz"], d["iz"] + d["nz"]):
                    for iy in range(d["iy"], d["iy"] + d["ny"]):
                        for ix in range(d["ix"], d["ix"] + d["nx"]):
                            if abs(subvector["data"][k]) > drop_tolerance:
                                d["data"][n] = subvector["data"][k]
                                d["indx"][n] = ix
                                d["indy"][n] = iy
                                d["indz"][n] = iz
                                n += 1
                            k += 1
                        k += yinc
                    k += zinc

                d["data"].resize(n)

            data_sum = np.sum(d["data"])
            data_sum_2 = np.sum(subvector["data"])

            print("  > subvector #{}".format(igrid))
            print(
                "    num elements: {} {}".format(num_elements, len(subvector["data"]))
            )
            if with_tolerance:
                print("    num elements tolerance: {}".format(len(d["data"])))
            print("    first index: {}".format(first_index))
            print("    last index: {}".format(k - zinc - yinc - 1 - first_index))
            print("    ix: {}, iy: {}, iz: {}".format(d["ix"], d["iy"], d["iz"]))
            print("    nx: {}, ny: {}, nz: {}".format(d["nx"], d["ny"], d["nz"]))
            print("    rx: {}, ry: {}, rz: {}".format(d["rx"], d["ry"], d["rz"]))
            print(
                "    nx_v: {}, ny_v: {}, nz_v: {}".format(
                    d["nx_v"], d["ny_v"], d["nz_v"]
                )
            )
            print("    yinc: {}, zinc: {}".format(yinc, zinc))
            print("    sum(data): {}".format(data_sum))
            print("    sum(data) 2: {}".format(data_sum_2))
            pdi_subvectors.append(d)

    # ___________________________________________________________________________
    # PFB File Processing
    # ___________________________________________________________________________

    print(f"\n_______________________________ \nAnalyzing PFB Files\n")
    pfb_file = open(binary_files[0], mode="rb")
    content = pfb_file.read()
    position = 0
    (X,) = struct.unpack(">d", content[position : position + 8])
    position += 8
    (Y,) = struct.unpack(">d", content[position : position + 8])
    position += 8
    (Z,) = struct.unpack(">d", content[position : position + 8])
    position += 8

    (NX,) = struct.unpack(">i", content[position : position + 4])
    position += 4
    (NY,) = struct.unpack(">i", content[position : position + 4])
    position += 4
    (NZ,) = struct.unpack(">i", content[position : position + 4])
    position += 4

    (DX,) = struct.unpack(">d", content[position : position + 8])
    position += 8
    (DY,) = struct.unpack(">d", content[position : position + 8])
    position += 8
    (DZ,) = struct.unpack(">d", content[position : position + 8])
    position += 8

    (numgrids,) = struct.unpack(">i", content[position : position + 4])
    position += 4

    print(" X: {}, Y: {}, Z: {}".format(X, Y, Z))
    print(" NX: {}, NY: {}, NZ: {}".format(NX, NY, NZ))
    print(" DX: {}, DY: {}, DZ: {}".format(DX, DY, DZ))

    pfb_subvectors = []

    for file in binary_files:

        pfb_file = open(file, mode="rb")

        print(f"\nFile: {file}")

        position = 3 * (8 + 4 + 8)

        (numgrids,) = struct.unpack(">i", content[position : position + 4])
        position += 4
        numgrids_per_file = int(numgrids / len(binary_files))

        print(" Total number of subgrids: {}".format(numgrids))
        print(" Local number of subgrids: {}".format(numgrids_per_file))

        for igrid in range(numgrids_per_file):
            pfd_d = {}
            (pfd_d["ix"], pfd_d["iy"], pfd_d["iz"]) = struct.unpack(
                ">iii", content[position : position + 4 * 3]
            )
            position += 4 * 3
            (pfd_d["nx"], pfd_d["ny"], pfd_d["nz"]) = struct.unpack(
                ">iii", content[position : position + 4 * 3]
            )
            position += 4 * 3
            (pfd_d["rx"], pfd_d["ry"], pfd_d["rz"]) = struct.unpack(
                ">iii", content[position : position + 4 * 3]
            )
            position += 4 * 3

            if with_tolerance <= 0:
                num_elements = (pfd_d["nx"]) * (pfd_d["ny"]) * (pfd_d["nz"])

                pfd_d["data"] = np.zeros([(pfd_d["nx"]), pfd_d["ny"], (pfd_d["nz"])])
                for iz in range(pfd_d["nz"]):
                    for iy in range(pfd_d["ny"]):
                        for ix in range(pfd_d["nx"]):
                            (pfd_d["data"][ix, iy, iz],) = struct.unpack(
                                ">d", content[position : position + 8]
                            )
                            position += 8
            else:
                (num_elements,) = struct.unpack(">i", content[position : position + 4])
                position += 4

                pfd_d["data"] = np.zeros([num_elements])
                pfd_d["indx"] = np.zeros([num_elements])
                pfd_d["indy"] = np.zeros([num_elements])
                pfd_d["indz"] = np.zeros([num_elements])

                for k in range(num_elements):
                    (pfd_d["indx"][k],) = struct.unpack(
                        ">i", content[position : position + 4]
                    )
                    position += 4
                    (pfd_d["indy"][k],) = struct.unpack(
                        ">i", content[position : position + 4]
                    )
                    position += 4
                    (pfd_d["indz"][k],) = struct.unpack(
                        ">i", content[position : position + 4]
                    )
                    position += 4
                    (pfd_d["data"][k],) = struct.unpack(
                        ">d", content[position : position + 8]
                    )
                    position += 8

            data_sum = np.sum(pfd_d["data"])
            print("  > subvector #{}".format(igrid))
            print("    num elements: {}".format(num_elements))
            print(
                "    ix: {}, iy: {}, iz: {}".format(
                    pfd_d["ix"], pfd_d["iy"], pfd_d["iz"]
                )
            )
            print(
                "    nx: {}, ny: {}, nz: {}".format(
                    pfd_d["nx"], pfd_d["ny"], pfd_d["nz"]
                )
            )
            print(
                "    rx: {}, ry: {}, rz: {}".format(
                    pfd_d["rx"], pfd_d["ry"], pfd_d["rz"]
                )
            )
            print("    sum(data): {}".format(data_sum))
            pfb_subvectors.append(pfd_d)

    # ___________________________________________________________________________
    # Data Comparison
    # ___________________________________________________________________________
    print(f"\n_______________________________ \nData comparison\n")
    error = 0
    for igrid in range(numgrids):
        error += np.sum(
            np.abs(pfb_subvectors[igrid]["data"] - pdi_subvectors[igrid]["data"])
        )

    print(f" - Error: {error}")


# ______________________________________________________________________________
# Main Function
# ______________________________________________________________________________


def main():
    parser = argparse.ArgumentParser(
        description="Compare PDI (HDF5) files with PFB binary files."
    )
    parser.add_argument(
        "names", nargs="+", help="Base name(s) of the files (without extension)."
    )
    args = parser.parse_args()

    for name in args.names:
        try:
            compare_files(name)
        except Exception as e:
            print(f"\nError processing '{name}': {e}\n")
            traceback.print_exc()


if __name__ == "__main__":
    main()
