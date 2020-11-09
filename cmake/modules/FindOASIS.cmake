#OAS3
find_path(OASIS_Fortran_INCLUDES mod_oasis.mod HINTS ${OAS3_ROOT}/build/lib/psmile.MPI1)
find_library(OASIS_Fortran_LIBRARIES psmile.MPI1 HINTS ${OAS3_ROOT}/lib)

#NETCDF-Fortran
find_path(NETCDF_Fortran_INCLUDES netcdf.inc HINTS ${NETCDF_Fortran_ROOT}/include)
find_library(NETCDF_Fortran_LIBRARIES netcdff HINTS ${NETCDF_Fortran_ROOT}/lib)