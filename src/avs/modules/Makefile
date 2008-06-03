#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

###############################################################################
#
# Makefile to make ParFlow AVS modules.
#
###############################################################################

AVS_LIB_PATH = $(AVS_PATH)/lib
AVS_INC_PATH = $(AVS_PATH)/include

include $(AVS_PATH)/include/Makeinclude

#---------------------------------------------------
# the following are defined in Makeinclude above:
#   ACFLAGS, LASTLIBS
#---------------------------------------------------

CFLAGS = $(PFFLAGS) $(ACFLAGS) -I. -I$(AVS_INC_PATH) $(PFAVS_IMAGE_TOOLS_INCLUDE_PATH)

BASE_LIBS = -lgeom -lutil -lm
CSUB_LIBS = -L$(AVS_LIB_PATH) -lflow_c $(BASE_LIBS) $(LASTLIBS)
FSUB_LIBS = -L$(AVS_LIB_PATH) -lflow_f $(BASE_LIBS) $(LASTLIBS)
CCOR_LIBS = -L$(AVS_LIB_PATH) -lsim_c $(BASE_LIBS) $(LASTLIBS)
FCOR_LIBS = -L$(AVS_LIB_PATH) -lsim_f $(BASE_LIBS) $(LASTLIBS)


#------------------------------------------------------------------------------
# defines
#------------------------------------------------------------------------------

SRC = $(PFAVS_IMAGE_TOOLS_DEPEND_SRC) \
   read_parflow.c \
   read_slim.c \
   filter_particles.c \
   cell_velocity.c \
   cell_conductivity.c \
   integer_to_file.c \
   scatter_to_3D.c \
   select_ijk.c \
   well.c \
   collage.c \
   zero_boundary.c \
   vert_sum.c \
   analog_clock.c \
   print_real.c \
   real_to_integer.c \
   integer_to_real.c \
   real_math.c \
   geom_parent.c \
   polygon.c \
   texture_mesh.1.c \
   Zero_Outer.c \
   Intersection.c \
   PF_Slicer.c \
   PF_Surface.c \
   PFsol_to_Geom.c \
   Downsize_Intsec.c \
   Brick_cont.c

COROUTINES = \
   rotate_geometry \
   perspective_zoom


OBJ = $(SRC:.c=.o)

MOD_EXE = pf_modules

MOD_LIB = ParFlow

#------------------------------------------------------------------------------
# compile rules
#------------------------------------------------------------------------------

all: $(MOD_LIB)

$(MOD_LIB): $(MOD_EXE) $(COROUTINES)
	@(echo "Building ParFlow module library file")
	@(avs -compile_library $(MOD_LIB)_source $(MOD_LIB))

$(MOD_EXE): $(MOD_EXE).o $(OBJ)
	@$(PFAVS_CC) -o $(MOD_EXE) $(MOD_EXE).o $(OBJ) $(CSUB_LIBS) $(CFLAGS) $(PFAVS_IMAGE_TOOLS_LIB_PATH) $(PFAVS_IMAGE_TOOLS_LIBS)

rotate_geometry: rotate_geometry.o
	@$(PFAVS_CC) -o rotate_geometry rotate_geometry.o $(CCOR_LIBS) $(CFLAGS)

perspective_zoom: perspective_zoom.o
	@$(PFAVS_CC) -o perspective_zoom perspective_zoom.o $(CCOR_LIBS) $(CFLAGS)

.c.o:
	@$(PFAVS_CC) $(CFLAGS) -c $<

#------------------------------------------------------------------------------
# miscellaneous
#------------------------------------------------------------------------------

clean:
	@(rm -f *.o)

veryclean: clean
	@(rm -f $(MOD_EXE) $(COROUTINES) $(MOD_LIB))

