ADD_EXECUTABLE(analyzer_find_hillslope_K find_hillslope_K.c)

TARGET_INCLUDE_DIRECTORIES(analyzer_find_hillslope_K PUBLIC "../../../pfcommon")
TARGET_INCLUDE_DIRECTORIES(pfcommon PUBLIC "${fca_INCLUDE_DIRECTORY}/..")

TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K ${fca_LIBRARY})
TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K pfcommon)
TARGET_LINK_LIBRARIES(analyzer_find_hillslope_K m)  # we need math

INSTALL(TARGETS analyzer_find_hillslope_K DESTINATION bin/parflowvr)
