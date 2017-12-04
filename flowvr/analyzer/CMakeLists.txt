ADD_EXECUTABLE(analyzer random_pressure_steer.c)

FIND_LIBRARY(FCA_LIBRARY fca)

TARGET_INCLUDE_DIRECTORIES(analyzer PUBLIC "../../pfcommon")

TARGET_LINK_LIBRARIES(analyzer ${FCA_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer pfcommon)

INSTALL(TARGETS analyzer DESTINATION bin)
