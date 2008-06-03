/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * This routine returns the elevations on a patch of a
 * solid at each (x,y) coordinate of an array of subgrid.  The result is
 * returned in an array of 2D real arrays.  
 *
 * This routine is called by the pressure boundary condition routine and 
 * by the pressure initial condition routine which calculate hydrostatic
 * conditions relative to a reference patch on a reference solid.
 *--------------------------------------------------------------------------*/

double         **CalcElevations(geom_solid, ref_patch, subgrids)
GeomSolid       *geom_solid;
int              ref_patch;
SubgridArray    *subgrids;
{
   GrGeomSolid        *grgeom_solid;

   GrGeomExtentArray  *extent_array;

   Background         *bg = GlobalsBackground;

   Subgrid            *subgrid;
		  
   double            **elevation_arrays;
   double             *elevation_array;
   double              z, dz2, zupper, zlower, zinit;
		  
   int	               ix, iy, iz;
   int	               nx, ny, nz;
   int	               rz;
		  
   int                *fdir;
		  
   int	               is, i,  j,  k, iel;
	           

   /*-----------------------------------------------------
    * Convert the Geom solid to a GrGeom solid, making
    * sure that the extent_array extends all the way to
    * the top and bottom of the background.
    *
    * Also set some other miscellaneous values.
    *-----------------------------------------------------*/

   zlower = BackgroundZLower(bg);
   zupper = BackgroundZUpper(bg);
   extent_array = GrGeomCreateExtentArray(subgrids, 0, 0, 0, 0, -1, -1);
   zinit = 0.0;

   GrGeomSolidFromGeom(&grgeom_solid, geom_solid, extent_array);

   GrGeomFreeExtentArray(extent_array);
	 
   /*-----------------------------------------------------
    * For each (x,y) point, determine the elevation
    * and construct the elevation_arrays.
    *-----------------------------------------------------*/

   elevation_arrays = ctalloc(double *, SubgridArraySize(subgrids));

   ForSubgridI(is, subgrids)
   {
      subgrid = SubgridArraySubgrid(subgrids, is);

      /* RDF: assume resolutions are the same in all 3 directions */
      rz = SubgridRZ(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = IndexSpaceZ(zlower, rz);
      
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = IndexSpaceZ(zupper, rz) - iz + 1;

      dz2 = RealSpaceDZ(rz)/2.0;
   
      elevation_array = ctalloc(double, (nx*ny));

      /* Initialize the elevation_array */
      for (iel = 0; iel < (nx*ny); iel++)
	 elevation_array[iel] = zinit;

      /* Construct elevation_array */
      GrGeomPatchLoop(i, j, k, fdir, grgeom_solid, ref_patch, 
		      rz, ix, iy, iz, nx, ny, nz,
      {
	 if (fdir[2] != 0)
	 {
	    iel = (j-iy)*nx + (i-ix);
	    z   = RealSpaceZ(k, rz) + fdir[2]*dz2;
	    
	    elevation_array[iel] = z;
	 }

      });

      elevation_arrays[is] = elevation_array;
   }

   GrGeomFreeSolid(grgeom_solid);

   return elevation_arrays;
}


