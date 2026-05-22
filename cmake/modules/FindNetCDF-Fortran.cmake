include(FindPackageHandleStandardArgs)

#NETCDF-Fortran
find_path(
    NETCDF_Fortran_INCLUDES
    netcdf.inc
    HINTS ${NETCDF_Fortran_ROOT}/include
)
find_library(NETCDF_Fortran_LIBRARY netcdff HINTS ${NETCDF_Fortran_ROOT}/lib)

find_package_handle_standard_args(
    NetCDF-Fortran
    DEFAULT_MSG
    NETCDF_Fortran_LIBRARY
    NETCDF_Fortran_INCLUDES
)
