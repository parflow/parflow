ADD_EXECUTABLE(analyzer_find_hillslope_K find_hillslope_K.c)

FIND_PACKAGE(fca REQUIRED)

TARGET_INCLUDE_DIRECTORIES(analyzer_find_hillslope_K PUBLIC "../../pfanalyzer")
TARGET_INCLUDE_DIRECTORIES(analyzer_find_hillslope_K PUBLIC "${fca_INCLUDE_DIRECTORY}")

TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K ${fca_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K pfanalyzer)
TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K m)  # we need math

INSTALL(TARGETS analyzer_find_hillslope_K DESTINATION bin/parflowvr)
