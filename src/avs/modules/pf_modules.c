/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * AVSinit_modules
 *
 * Main routine linking ParFlow AVS modules.
 *
 *****************************************************************************/

#include <avs.h>


/*--------------------------------------------------------------------------
 * AVSinit_modules
 *--------------------------------------------------------------------------*/

AVSinit_modules()
{
   int ReadParFlow();
   int ReadSlim();
   int FilterParticles();
   int CellConductivity();
   int CellVelocity();
   int IntegerToFile();
   int ScatterTo3D();
   int SelectIJK();
   int WellGeom();
   int collage();
   int write_ANY_image_desc();
   int AC_Analog_Clock();
   int ZeroBoundary();
   int VertSum();
   int printReal();
   int realMath();
   int realToInt();
   int intToReal();
   int geom_parent_desc();
   int polygon_to_geom();
   int mesh_block();
   int Zero_Outer_desc();
   int Intersection_desc();
   int PF_Slicer_desc();
   int PF_Surface_desc();
   int PFsol_to_Geom_desc();
   int Downsize_Intsec_desc();
   int Brick_cont_desc();


   AVSmodule_from_desc(ReadParFlow);
   AVSmodule_from_desc(ReadSlim);
   AVSmodule_from_desc(FilterParticles);
   AVSmodule_from_desc(CellConductivity);
   AVSmodule_from_desc(CellVelocity);
   AVSmodule_from_desc(IntegerToFile);
   AVSmodule_from_desc(ScatterTo3D);
   AVSmodule_from_desc(SelectIJK);
   AVSmodule_from_desc(WellGeom);
   AVSmodule_from_desc(ZeroBoundary);
   AVSmodule_from_desc(VertSum);
   AVSmodule_from_desc(Zero_Outer_desc);
   AVSmodule_from_desc(Intersection_desc);
   AVSmodule_from_desc(PF_Slicer_desc);
   AVSmodule_from_desc(PF_Surface_desc);
   AVSmodule_from_desc(PFsol_to_Geom_desc);
   AVSmodule_from_desc(Downsize_Intsec_desc);
   AVSmodule_from_desc(Brick_cont_desc);
   /* From avs */
   AVSmodule_from_desc(polygon_to_geom);

   /* The following modules come from the avs.ncsc.org site */
   AVSmodule_from_desc(collage);
   AVSmodule_from_desc(printReal);
   AVSmodule_from_desc(realMath);
   AVSmodule_from_desc(realToInt);
   AVSmodule_from_desc(intToReal);
   AVSmodule_from_desc(geom_parent_desc);
   AVSmodule_from_desc(mesh_block);

   /* The follow depends on IMTOOLS from SDSC */
#ifdef IMAGE_TOOLS
   AVSmodule_from_desc(write_ANY_image_desc);
#endif

   AVSmodule_from_desc(AC_Analog_Clock);
}
