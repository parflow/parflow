#ADD_EXECUTABLE(analyzer random_pressure_steer.c)
ADD_EXECUTABLE(analyzer find_hillslope_K.c)

FIND_LIBRARY(FCA_LIBRARY fca)

TARGET_INCLUDE_DIRECTORIES(analyzer PUBLIC "../../pfcommon")

TARGET_LINK_LIBRARIES(analyzer ${FCA_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer pfcommon)
TARGET_LINK_LIBRARIES(analyzer m)  # we need math

INSTALL(TARGETS analyzer DESTINATION bin)
