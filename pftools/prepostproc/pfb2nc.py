####################################################################
## Convert Parflow binary to netCDF
## Author Ketan B. Kulkarni (k.kulkarni@fz-juelich.de)
## SimLab TerrSys Juelich Supercomputing Centre
## Usage: python pfb2nc.py inputFileList outPutFileName varName
## e.g. python pfb2nc.py saturation_file_list.txt parflow_output.nc saturation
## Triple loop(naive implementation) replaced by block input read
## and unpacked and reshaped into a 3-D array. Time dimension loop is added.
## Send the imporvements to k.kulkarni@fz-juelich.de
## Added another time dimension and loop over a list of files for time variable.
####################################################################

# Requires numPy and netcdf python bindings.
import sys
from struct import *
import numpy as np
from netCDF4 import *
from datetime import *

# Read command line input
fileName = sys.argv[1]
outFile = sys.argv[2]
varName = sys.argv[3]

fileList = open(fileName, "r")
lines = fileList.readlines()
fileName = [ii.rstrip("\n") for ii in lines]

# Set the time axis. Change the time incremente suitable to your experiment.
timeArray = [
    datetime(1990, 2, 1) + ii * timedelta(hours=3) for ii in range(0, len(fileName))
]


# Define netCDF dimensions (Adapt for your domain dimensions)
# Following dimensions correspond to Little Washita test case
ncfile = Dataset(outFile, "w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 41)
ncfile.createDimension("lat", 41)
ncfile.createDimension("lev", 50)

# Create a variable
pfVar = ncfile.createVariable(varName, "f8", ("time", "lev", "lat", "lon"))
# Create time variable and add data to it
timeVar = ncfile.createVariable("time", "f8", ("time"))
timeVar.units = "hours since 1990-02-01 00:00:00.0"
timeVar.calendar = "gregorian"
timeVar[:] = date2num(timeArray, units=timeVar.units, calendar=timeVar.calendar)

# Allocate an array to read the data from pfb(Adapt for your domain dimensions)
# Following dimensions correspond to Little Washita test case
press = np.empty([50, 41, 41])

for fname in range(0, len(fileName)):
    file = open(fileName[fname], "rb")
    print(fileName[fname])
    # Read the start index of the domain
    bigEnd = file.read(8)
    x1 = list(unpack(">d", bigEnd))[0]
    # print(x1)
    bigEnd = file.read(8)
    y1 = list(unpack(">d", bigEnd))[0]
    # print(y1)
    bigEnd = file.read(8)
    z1 = list(unpack(">d", bigEnd))[0]
    # print(z1)

    # Read the number of points in x, y and z direction
    bigEnd = file.read(4)
    nx = list(unpack(">i", bigEnd))[0]
    # print(nx)
    bigEnd = file.read(4)
    ny = list(unpack(">i", bigEnd))[0]
    # print(ny)
    bigEnd = file.read(4)
    nz = list(unpack(">i", bigEnd))[0]
    # print(nz)

    # Read dx, dy and dz
    bigEnd = file.read(8)
    dx = list(unpack(">d", bigEnd))[0]
    # print(dx)
    bigEnd = file.read(8)
    dy = list(unpack(">d", bigEnd))[0]
    # print(dy)
    bigEnd = file.read(8)
    dz = list(unpack(">d", bigEnd))[0]
    # print(dz)

    # Allocate array to read the data

    # Read the number of subdomains =  number of procs
    bigEnd = file.read(4)
    nSubGrid = list(unpack(">i", bigEnd))[0]
    # print(nSubGrid)

    for gridCounter in range(0, nSubGrid):

        # Read the subgrid indices and counters
        bigEnd = file.read(4)
        ix = list(unpack(">i", bigEnd))[0]
        # print(ix)
        bigEnd = file.read(4)
        iy = list(unpack(">i", bigEnd))[0]
        # print(iy)
        bigEnd = file.read(4)
        iz = list(unpack(">i", bigEnd))[0]
        # print(iz)

        bigEnd = file.read(4)
        nnx = list(unpack(">i", bigEnd))[0]
        # print(nnx)
        bigEnd = file.read(4)
        nny = list(unpack(">i", bigEnd))[0]
        # print(nny)
        bigEnd = file.read(4)
        nnz = list(unpack(">i", bigEnd))[0]
        # print(nnz)

        bigEnd = file.read(4)
        rx = list(unpack(">i", bigEnd))[0]
        # print(rx)
        bigEnd = file.read(4)
        ry = list(unpack(">i", bigEnd))[0]
        # print(ry)
        bigEnd = file.read(4)
        rz = list(unpack(">i", bigEnd))[0]
        # print(rz)

        bigEnd = file.read(nnx * nny * nnz * 8)
        fmt = ">%dd" % (nnx * nny * nnz)
        temp = list(unpack(fmt, bigEnd))
        press[iz : iz + nnz, iy : iy + nny, ix : ix + nnx] = np.reshape(
            temp, (nnz, nny, nnx)
        )

    # Write into netCDF variable
    pfVar[fname, :, :, :] = press


ncfile.close()
