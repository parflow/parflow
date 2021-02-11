include(FindPackageHandleStandardArgs)

#NETCDF-Fortran
find_path(NETCDF_Fortran_INCLUDES netcdf.inc HINTS ${NETCDF_Fortran_ROOT}/include)
find_library(NETCDF_Fortran_LIBRARY netcdff HINTS ${NETCDF_Fortran_ROOT}/lib)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(NetCDF-Fortran DEFAULT_MSG NETCDF_Fortran_LIBRARY NETCDF_Fortran_INCLUDES)
