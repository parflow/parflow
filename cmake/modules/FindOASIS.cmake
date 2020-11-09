include(FindPackageHandleStandardArgs)

#OAS3 DEPENDENCIES
find_library(MCT_Fortran_LIBRARY mct HINTS ${OAS3_ROOT}/lib)
find_library(MPEU_Fortran_LIBRARY mpeu HINTS ${OAS3_ROOT}/lib)
find_library(SCRIP_Fortran_LIBRARY scrip HINTS ${OAS3_ROOT}/lib)

#OAS3 LIB
if(MCT_Fortran_LIBRARY AND MPEU_Fortran_LIBRARY AND SCRIP_Fortran_LIBRARY)
  find_path(OASIS_Fortran_INCLUDES mod_oasis.mod HINTS ${OAS3_ROOT}/build/lib/psmile.MPI1)
  find_library(OASIS_Fortran_LIBRARY psmile.MPI1 HINTS ${OAS3_ROOT}/lib)
endif(MCT_Fortran_LIBRARY AND MPEU_Fortran_LIBRARY AND SCRIP_Fortran_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OASIS DEFAULT_MSG OASIS_Fortran_LIBRARY OASIS_Fortran_INCLUDES)
