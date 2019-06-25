# Parflow Release Notes

## Overview of Changes

* Performance improvements in ParFlow looping

## User Visible Changes

### Added option to pfdist to enable distribution of datasets with different NZ points

The pfdist utility in pftools now has an option to distribute data
that has a different number points in Z than the default grid NZ
value.  Previously one had to set the ComputationalGrid.NZ value
before the pfdist when distributing 1D data.  The '-nz' option enables
setting NZ for the pfdist call.

For example, the kludge:

```
pfset ComputationalGrid.NX 40
pfset ComputationalGrid.NY 40
pfset ComputationalGrid.NZ 1

pfdist file.pfb

pfset ComputationalGrid.NZ 40
```

can be replaced with:

```
pfset ComputationalGrid.NX 40
pfset ComputationalGrid.NY 40
pfset ComputationalGrid.NZ 40

pfdist -nz 1 file.pfb
```

### Performance improvements in ParFlow looping

Previously ParFlow was using an octree structure for looping over
geometries defined on the domain.  This was changed to loop over a set
of boxes in index space.  The new method should be faster for larger
problems but does introduce a performance hit for very small domains
due to the setup costs of running a clustering algorithm.  The
clustering is controllable using the "UseClustering" key.  Setting to
False will revert to the original octree looping.

```
   pfset  UseClustering  False
```

## Internal Changes


## Known Issues

