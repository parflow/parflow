# ---------------------------------------------------------
# Testing copy, remove, and mkdir methods on Python Run object
# ---------------------------------------------------------

import sys

from parflow import Run
from parflow.tools.fs import cp, rm, mkdir, exists

# Update the working directory via the __file__ arg
os_fxn = Run("os_fxn", __file__)


def checkOK(file_path):
    if not exists(file_path):
        sys.exit(1)


def checkKO(file_path):
    if exists(file_path):
        sys.exit(1)


# copying file from adjacent directory
cp("$PF_SRC/test/input/BasicSettings.yaml")
checkOK("BasicSettings.yaml")

cp("$PF_SRC/test/input/BasicSettings.yaml", "TestCopyFile.yaml")
checkOK("TestCopyFile.yaml")

# copying file from adjacent directory with environment variable
cp("$PF_SRC/README.md")
checkOK("README.md")

# removing files
rm("BasicSettings.yaml")
checkKO("BasicSettings.yaml")
rm("README.md")
checkKO("README.md")

# creating directory
checkKO("test_directory")
mkdir("test_directory")
checkOK("test_directory")

# creating same directory - you should get a message that this directory already exists.
mkdir("test_directory")

mkdir("test_directory/test1/test2/")
checkOK("test_directory/test1/test2")

# removing directory
rm("test_directory")
rm("test_directory")
checkKO("test_directory")
