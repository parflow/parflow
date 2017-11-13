#!/usr/bin/python

import sys, os, stat
def f(fname):
    return os.stat(fname).st_mtime

print (f(sys.argv[1]) - f(sys.argv[2]))
