#!/usr/bin/env python

# usage: compare_nc.py a.nc b.nc [-vague]

# exits with 0 (and no output) if 2 netCDF files contain the same datasets otherwise gives an error.
# 3rd parameter: if -vague is given as command line parameter: exits 0 if datasets of A in B or B in A

import netCDF4 as nc
import numpy as np
import sys

A = nc.Dataset(sys.argv[1])
B = nc.Dataset(sys.argv[2])


if len(A.variables) != len(B.variables):
    print ("they differ in the variables amount!")
    if sys.argv[-1] != "-vague":
        result = 1
    # make sure A is the one with the less variables so the following will work
    if len(A.variables) > len(B.variables):
        A, B = B, A


result = 0
for index in A.variables:
    a = np.array(A.variables[index])
    b = np.array(B.variables[index])
    if not (a==b).all():
        if index == 'time':
            print(a)
            print(b)
        print ("they differ in: " + index)
        print ("difference:")
        print (np.sum(np.abs(a - b)))
        result = 1


exit(result)


