#-----------------------------------------------------------------------------
# - Find Sundials includes and libraries.
#
# This module finds if Sundials is installed and determines where the
# include files and libraries are.  This code sets the following variables:
#  SUNDIALS_FOUND         = Sundials was found
#  SUNDIALS_INCLUDE_DIR   = path to where header files can be found
#  SUNDIALS_LIBRARIES     = link libraries for Sundials
#-----------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)

find_path (SUNDIALS_DIR include/sundials/sundials_config.h HINTS ${SUNDIALS_ROOT} 
  DOC "Sundials Directory")

if (SUNDIALS_DIR)

  set(SUNDIALS_FOUND YES)

  set(SUNDIALS_INCLUDE_DIR ${SUNDIALS_DIR}/include)
  set(SUNDIALS_LIBRARY_DIR ${SUNDIALS_DIR}/lib)

  if(SUNDIALS_FIND_COMPONENTS)
    
    foreach(comp ${SUNDIALS_FIND_COMPONENTS})

      # Need to make sure variable to search for isn't set
      unset(SUNDIALS_LIB CACHE)

      find_library(SUNDIALS_LIB
        NAMES ${comp}
	HINTS ${SUNDIALS_LIBRARY_DIR}
	NO_DEFAULT_PATH)

      if(SUNDIALS_LIB)
        list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_LIB})
      else(SUNDIALS_LIB)	    
        message(FATAL_ERROR "Could not find required Sundials library : ${comp}")
      endif(SUNDIALS_LIB)
    
    endforeach(comp)

  endif(SUNDIALS_FIND_COMPONENTS)

else(SUNDIALS_DIR)
  set(SUNDIALS_FOUND NO)
endif(SUNDIALS_DIR)

find_package_handle_standard_args(SUNDIALS DEFAULT_MSG SUNDIALS_LIBRARIES SUNDIALS_INCLUDE_DIR)
