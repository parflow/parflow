
set (SRC_FILES iterativ.c kinsol.c kinspgmr.c llnlmath.c spgmr.c)

add_library(pfkinsol ${SRC_FILES})

install (TARGETS pfkinsol DESTINATION lib)

