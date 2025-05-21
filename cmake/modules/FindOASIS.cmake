include(FindPackageHandleStandardArgs)

#OAS3 DEPENDENCIES
find_library(MCT_Fortran_LIBRARY mct HINTS ${OAS3_ROOT}/lib)
find_library(MPEU_Fortran_LIBRARY mpeu HINTS ${OAS3_ROOT}/lib)
find_library(SCRIP_Fortran_LIBRARY scrip HINTS ${OAS3_ROOT}/lib)

#OAS3 LIB
if(MCT_Fortran_LIBRARY AND MPEU_Fortran_LIBRARY AND SCRIP_Fortran_LIBRARY)
  find_path(OASIS_Fortran_INCLUDES mod_oasis.mod HINTS ${OAS3_ROOT}/build/lib/psmile.MPI1 ${OAS3_ROOT}/include)
  find_library(OASIS_Fortran_LIBRARY psmile.MPI1 HINTS ${OAS3_ROOT}/lib)
endif(MCT_Fortran_LIBRARY AND MPEU_Fortran_LIBRARY AND SCRIP_Fortran_LIBRARY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(OASIS DEFAULT_MSG OASIS_Fortran_LIBRARY OASIS_Fortran_INCLUDES)

if(OASIS_FOUND AND NOT TARGET OASIS3MCT::OASIS3MCT)
   add_library(OASIS3MCT::OASIS3MCT INTERFACE IMPORTED)
   target_include_directories(OASIS3MCT::OASIS3MCT INTERFACE ${OASIS_Fortran_INCLUDES})
   target_link_libraries(OASIS3MCT::OASIS3MCT INTERFACE ${OASIS_Fortran_LIBRARY} ${MCT_Fortran_LIBRARY} ${MPEU_Fortran_LIBRARY} ${SCRIP_Fortran_LIBRARY})

   # TODO: This assumes OASIS3-MCT was built with OpenMP.
   find_package(OpenMP)
   if (OpenMP_FOUND AND TARGET OpenMP::OpenMP_Fortran)
     target_link_libraries(OASIS3MCT::OASIS3MCT INTERFACE OpenMP::OpenMP_Fortran)
     message(STATUS "FindOASIS: OpenMP::OpenMP_Fortran found")
   endif()
endif()