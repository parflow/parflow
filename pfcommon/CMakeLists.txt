set (SRC_FILES messages.c)

add_library(pfcommon ${SRC_FILES})

install (TARGETS pfcommon DESTINATION lib)

target_include_directories (pfcommon PUBLIC "${FLOWVR_INCLUDE_DIRS}")



