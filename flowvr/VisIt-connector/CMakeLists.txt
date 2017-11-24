ADD_EXECUTABLE(visit-connector main.c)

FIND_LIBRARY(FCA_LIBRARY fca)

set (VISIT_LIBRARY_LIBSIM "" CACHE FILEPATH "Path to libsim (V2)")
set (VISIT_INCLUDE_DIR "" CACHE PATH "Path to visit include directory")

TARGET_INCLUDE_DIRECTORIES(visit-connector PUBLIC ${VISIT_INCLUDE_DIR})
TARGET_INCLUDE_DIRECTORIES(visit-connector PUBLIC "../../pfcommon")

TARGET_LINK_LIBRARIES(visit-connector ${FCA_LIBRARY})
TARGET_LINK_LIBRARIES(visit-connector ${VISIT_LIBRARY_LIBSIM})
TARGET_LINK_LIBRARIES(visit-connector dl)
TARGET_LINK_LIBRARIES(visit-connector pfcommon)

INSTALL(TARGETS visit-connector DESTINATION bin)
