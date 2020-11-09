include(FindPackageHandleStandardArgs)

#OAS3 DEPENDENCIES
find_path(MCT_Fortran_LIBDIR libmct.a HINTS ${OAS3_ROOT}/lib)
find_path(MPEU_Fortran_LIBDIR libmpeu.a HINTS ${OAS3_ROOT}/lib)
find_path(SCRIP_Fortran_LIBDIR libscrip.a HINTS ${OAS3_ROOT}/lib)

#OAS3 LIB
if(MCT_Fortran_LIBDIR AND MPEU_Fortran_LIBDIR AND SCRIP_Fortran_LIBDIR)
  find_path(OASIS_Fortran_INCLUDES mod_oasis.mod HINTS ${OAS3_ROOT}/build/lib/psmile.MPI1)
  find_path(OASIS_Fortran_LIBDIR libpsmile.MPI1.a HINTS ${OAS3_ROOT}/lib)
endif(MCT_Fortran_LIBDIR AND MPEU_Fortran_LIBDIR AND SCRIP_Fortran_LIBDIR)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OASIS DEFAULT_MSG OASIS_Fortran_LIBDIR OASIS_Fortran_INCLUDES)
