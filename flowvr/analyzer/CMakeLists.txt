ADD_EXECUTABLE(analyzer random_pressure_steer.c)

FIND_PACKAGE(fca REQUIRED)
TARGET_INCLUDE_DIRECTORIES(analyzer PUBLIC "../pfanalyzer")
TARGET_INCLUDE_DIRECTORIES(analyzer PUBLIC "${fca_INCLUDE_DIRECTORY}")

TARGET_LINK_LIBRARIES(analyzer ${fca_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer pfanalyzer)
TARGET_LINK_LIBRARIES(analyzer m)  # we need math

INSTALL(TARGETS analyzer DESTINATION bin/parflowvr)
