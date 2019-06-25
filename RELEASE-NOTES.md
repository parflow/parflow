# Parflow Release Notes

## Overview of Changes

* Performance improvements of ParFlow looping

## User Visibile Changes

### Performance improvements of ParFlow looping

Previously ParFlow was using an octree structure for looping over
geometries defined on the domain.  This was changed to loop over a set
of boxes in index space.  The new method should be faster for larger
problems but does introduce a performance hit for very small domains
due to the setup costs of running a clustering algorithm.  The
clustering is controllable using the "UseClustering" key.  Setting to
False will revert to the orginal octree loopoing.

```
   pfset  UseClustering  False
```

## Internal Changes


## Known Issues

