! 1. Spatial structure:
!    Select one of the following options.
#define GRIDBASED
#undef CATCHMENT
#undef UNSTRUCTURED
#undef SinglePoint

! 2. Land subgrid type classification:
!    Select one of the following options.
#undef LULC_USGS
#define LULC_IGBP
#undef LULC_IGBP_PFT
#undef LULC_IGBP_PC

! 2.1 3D Urban model (put it temporarily here):
#undef URBAN_MODEL

! 3. If defined, debug information is output.
#undef CoLMDEBUG
! 3.1 If defined, range of variables is checked.
#undef RangeCheck
! 3.1 If defined, surface data in vector is mapped to gridded data for checking.
#undef SrfdataDiag

! 4. If defined, MPI parallelization is enabled.
#undef USEMPI
!    Conflict: not used when defined SingPoint.
#if (defined SinglePoint)
#undef USEMPI
#endif

! 5. Hydrological process options.
! 5.1 Two soil hydraulic models can be used.
#undef Campbell_SOIL_MODEL
#define vanGenuchten_Mualem_SOIL_MODEL
! 5.2 If defined, lateral flow is modeled.
#undef CatchLateralFlow
!    Conflicts :
#ifndef CATCHMENT
#undef CatchLateralFlow
#endif

! 6. If defined, CaMa-Flood model will be used.
#undef CaMa_Flood

! 7. If defined, BGC model is used.
#undef BGC

!    Conflicts :  only used when LULC_IGBP_PFT is defined.
#ifndef LULC_IGBP_PFT
#undef BGC
#endif
! 7.1 If defined, CROP model is used
#define CROP
!    Conflicts : only used when BGC is defined
#ifndef BGC
#undef CROP
#endif

! 8. If defined, open Land use and land cover change mode.
#undef LULCC

! 9. If defined, data assimilation is used.
#undef DataAssimilation

! 10. Vector write model.
!     1) "VectorInOneFileP" : write vector data in one file in parallel mode;  
!     2) "VectorInOneFileS" : write vector data in one file in serial mode;  
!     3) Neither "VectorInOneFileS" nor "VectorInOneFileP" is defined : 
!        write vector data in separate files.  
#undef VectorInOneFileP
!     Conflict
#ifdef VectorInOneFileP
#undef VectorInOneFileS
#endif

#define DYN_PHENOLOGY
