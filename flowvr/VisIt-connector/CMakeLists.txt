ADD_EXECUTABLE(visit-connector main.c)

FIND_PACKAGE(fca REQUIRED)

set (VISIT_LIBRARY_LIBSIM "" CACHE FILEPATH "Path to libsim (V2)")
set (VISIT_INCLUDE_DIR "" CACHE PATH "Path to visit include directory")

TARGET_INCLUDE_DIRECTORIES(visit-connector PUBLIC ${VISIT_INCLUDE_DIR})
TARGET_INCLUDE_DIRECTORIES(visit-connector PUBLIC "../pfanalyzer")
TARGET_INCLUDE_DIRECTORIES(visit-connector PUBLIC "${fca_INCLUDE_DIRECTORY}")

TARGET_LINK_LIBRARIES(visit-connector ${fca_LIBRARY})
TARGET_LINK_LIBRARIES(visit-connector ${VISIT_LIBRARY_LIBSIM})
TARGET_LINK_LIBRARIES(visit-connector dl)
TARGET_LINK_LIBRARIES(visit-connector pfanalyzer)


INSTALL(TARGETS visit-connector DESTINATION bin)

configure_file(visit-connector-starter.sh.in ${CMAKE_BINARY_DIR}/visit-connector-starter.sh @ONLY)

INSTALL(FILES ${CMAKE_BINARY_DIR}/visit-connector-starter.sh DESTINATION bin
	PERMISSIONS OWNER_WRITE
	OWNER_READ
	OWNER_EXECUTE
	GROUP_READ
	GROUP_EXECUTE
	WORLD_READ
	WORLD_EXECUTE)
