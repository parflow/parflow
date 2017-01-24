
# TODO many little tools not included yet.  See Makefile.in

set (TOOLS_SRC_FILES pftappinit.c printdatabox.c readdatabox.c databox.c
  error.c velocity.c head.c flux.c diff.c stats.c tools_io.c axpy.c
  getsubbox.c enlargebox.c load.c usergrid.c grid.c region.c file.c
  pftools.c top.c compute_domain.c water_balance.c water_table.c
  toposlopes.c sum.c
  )


add_library(pftools SHARED ${TOOLS_SRC_FILES})

target_include_directories(pftools PUBLIC ${TCL_INCLUDE_PATH})
target_link_libraries(pftools ${TCL_LIBRARY})

install(TARGETS pftools DESTINATION bin)

file (GLOB TCL_SRC *.tcl)

install(FILES ${TCL_SRC} DESTINATION bin)
