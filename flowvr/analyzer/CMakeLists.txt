ADD_EXECUTABLE(analyzer random_pressure_steer.c)

TARGET_INCLUDE_DIRECTORIES(analyzer PUBLIC "../../pfcommon")
TARGET_INCLUDE_DIRECTORIES(pfcommon PUBLIC "${fca_INCLUDE_DIRECTORY}/..")

TARGET_LINK_LIBRARIES(analyzer ${fca_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer pfcommon)
TARGET_LINK_LIBRARIES(analyzer m)  # we need math

INSTALL(TARGETS analyzer DESTINATION bin/parflowvr)
