
file(GLOB_RECURSE COMMON_SRC_FILES common/*.c)

# TODO Need to be amps impl specific
file(GLOB_RECURSE AMPS_SRC_FILES ${PARFLOW_AMPS_TYPE}/*.c)

add_library(amps ${COMMON_SRC_FILES} ${AMPS_SRC_FILES})

install(TARGETS amps DESTINATION lib)

set(AMPS_SCRIPTS batchmc bootmc freemc getmc killmc peekmc run)

string(REGEX REPLACE "([^;]+)" "${PARFLOW_AMPS_TYPE}/\\1" AMPS_SCRIPTS "${AMPS_SCRIPTS}")

install(FILES ${AMPS_SCRIPTS} DESTINATION bin)

