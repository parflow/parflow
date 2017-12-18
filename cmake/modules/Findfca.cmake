#
# Find fca  contrib (part of FLOWVR but not necessarily compiled and installed)
#
# fca_INCLUDE_DIRECTORY
# fca_LIBRARY        
# fca_FOUND         


FIND_PACKAGE(FlowVR)

find_path(fca_INCLUDE_DIRECTORY
   NAMES     fca.h
   PATHS  ${FLOWVR_base_INCLUDE_DIR}/fca
   NO_DEFAULT_PATH
)

find_library(fca_LIBRARY
  NAMES fca
  PATHS
  ENV LD_LIBRARY_PATH
  ENV LIBRARY_PATH
  ${FlowVR_DIR}/../../../lib
  NO_DEFAULT_PATH
)

if ( fca_INCLUDE_DIRECTORY AND fca_LIBRARY )
  set( fca_FOUND TRUE )
else ( fca_INCLUDE_DIRECTORY AND fca_LIBRARY )
  set( fca_FOUND FALSE )
endif ( fca_INCLUDE_DIRECTORY AND fca_LIBRARY )

