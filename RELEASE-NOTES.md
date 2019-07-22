# ParFlow Release Notes

## Overview of Changes

* Critical bug fixes
* Initial Docker support added

## User Visible Changes

### Critical bug fixes

Two bugs were fixed that caused segmentation faults.  One was in the
new clustering algorithm and the other was in CLM changes to support
setting number of soil levels at run-time.

### Initial Docker support added

Support for Docker is being added to make deployment easier for users
who are running ParFlow on PC and workstation environments.  Docker
images ParFlow will be available at DockerHub and should be
automatically built when ParFlow is updated.  We are still testing
this.

See the https://github.com/parflow/docker repository on GitHub for
instructions on running and a sample problem setup.

## Internal Changes

### Memory leaks

Several memory leaks have been fixed.  Regression tests are running
cleanly with Valgrind.

## Known Issues

