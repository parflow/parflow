#!/usr/bin/env python

# exits with 0 when 2 netCDF files contain the same datasets.

import netCDF4 as nc
import numpy as np
import sys

A = nc.Dataset(sys.argv[1])
B = nc.Dataset(sys.argv[2])


if len(A.variables) != len(B.variables):
    print ("they differ in the variables amount!")
    exit(1)


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


