/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for the GrGeometry (Grid dependent Geometry) class.
 *
 *****************************************************************************/

#include "parflow.h"
#include "grgeometry.h"


/*--------------------------------------------------------------------------
 * GrGeomGetOctreeInfo:
 *   This routine returns the spatial information needed to relate
 *   an octree with the background.
 *--------------------------------------------------------------------------*/

int      GrGeomGetOctreeInfo(xlp, ylp, zlp, xup, yup, zup,
			     ixp, iyp, izp)
double  *xlp;
double  *ylp;
double  *zlp;
double  *xup;
double  *yup;
double  *zup;
int     *ixp;
int     *iyp;
int     *izp;
{
   Background  *bg = GlobalsBackground;
   double       dtmp;
   int          background_level, n;


   *xlp = BackgroundXLower(bg);
   *ylp = BackgroundYLower(bg);
   *zlp = BackgroundZLower(bg);

   dtmp = ceil( log(BackgroundNX(bg)) / log(2) );
   background_level = (int)dtmp;
   dtmp = ceil( log(BackgroundNY(bg)) / log(2) );
   background_level = max(background_level, (int)dtmp);
   dtmp = ceil( log(BackgroundNZ(bg)) / log(2) );
   background_level = max(background_level, (int)dtmp);

   n = (int)pow(2.0, background_level);

   *xup = BackgroundXLower(bg) + n*BackgroundDX(bg);
   *yup = BackgroundYLower(bg) + n*BackgroundDY(bg);
   *zup = BackgroundZLower(bg) + n*BackgroundDZ(bg);

   *ixp = BackgroundIX(bg);
   *iyp = BackgroundIY(bg);
   *izp = BackgroundIZ(bg);

   return background_level;
}


/*--------------------------------------------------------------------------
 * GrGeomNewExtentArray
 *--------------------------------------------------------------------------*/

GrGeomExtentArray  *GrGeomNewExtentArray(extents, size)
GrGeomExtents      *extents;
int                 size;
{
    GrGeomExtentArray   *new;


    new = talloc(GrGeomExtentArray, 1);

    (new -> extents) = extents;
    (new -> size)    = size;

    return new;
}


/*--------------------------------------------------------------------------
 * GrGeomFreeExtentArray
 *--------------------------------------------------------------------------*/

void                GrGeomFreeExtentArray(extent_array)
GrGeomExtentArray  *extent_array;
{
   tfree(GrGeomExtentArrayExtents(extent_array));

   tfree(extent_array);
}


/*--------------------------------------------------------------------------
 * GrGeomCreateExtentArray:
 *   The arguments [xyz][lu]_ghost indicate the number of ghost layers
 *   in each of the directions.  A negative value is an indication to
 *   extend the layers out to the edge of the grid background.
 *
 *   Important note: the indices returned in the extent_array are in
 *   *octree* coordinates.  The lower corner background index is index
 *   (0, 0, 0) in the octree.
 *
 *   Another important note: the routine GrGeomOctreeFromTIN requires
 *   at least one ghost layer of geometry info in order to construct
 *   the octree correctly.  This routine insures this.
 *--------------------------------------------------------------------------*/

GrGeomExtentArray  *GrGeomCreateExtentArray(subgrids,
					    xl_ghost, xu_ghost,
					    yl_ghost, yu_ghost,
					    zl_ghost, zu_ghost)
SubgridArray       *subgrids;
int                 xl_ghost;
int                 xu_ghost;
int                 yl_ghost;
int                 yu_ghost;
int                 zl_ghost;
int                 zu_ghost;
{
   Background         *bg = GlobalsBackground;

   GrGeomExtentArray  *extent_array;
   GrGeomExtents      *extents;
   int                 size;

   Subgrid            *subgrid;

   int                 ref;
   int                 bg_ix, bg_iy, bg_iz;
   int                 bg_nx, bg_ny, bg_nz;
   int                 is;


   size = SubgridArraySize(subgrids);
   extents = ctalloc(GrGeomExtents, size);

   ForSubgridI(is, subgrids)
   {
      subgrid = SubgridArraySubgrid(subgrids, is);

      /* compute background grid extents on MaxRefLevel index space */
      ref = (int)pow(2.0, GlobalsMaxRefLevel);
      bg_ix = BackgroundIX(bg)*ref;
      bg_iy = BackgroundIY(bg)*ref;
      bg_iz = BackgroundIZ(bg)*ref;
      bg_nx = BackgroundNX(bg)*ref;
      bg_ny = BackgroundNY(bg)*ref;
      bg_nz = BackgroundNZ(bg)*ref;

      ref = Pow2(GlobalsMaxRefLevel);

      /*------------------------------------------
       * set the lower extent values
       *------------------------------------------*/

      if (xl_ghost > -1)
      {
	 xl_ghost = max(xl_ghost, 1);
	 GrGeomExtentsIXLower(extents[is]) =
	    (SubgridIX(subgrid) - xl_ghost) * ref;
      }
      else
      {
	 GrGeomExtentsIXLower(extents[is]) = bg_ix;
      }

      if (yl_ghost > -1)
      {
	 yl_ghost = max(yl_ghost, 1);
	 GrGeomExtentsIYLower(extents[is]) =
	    (SubgridIY(subgrid) - yl_ghost) * ref;
      }
      else
      {
	 GrGeomExtentsIYLower(extents[is]) = bg_iy;
      }

      if (zl_ghost > -1)
      {
	 zl_ghost = max(zl_ghost, 1);
	 GrGeomExtentsIZLower(extents[is]) =
	    (SubgridIZ(subgrid) - zl_ghost) * ref;
      }
      else
      {
	 GrGeomExtentsIZLower(extents[is]) = bg_iz;
      }

      /*------------------------------------------
       * set the upper extent values
       *------------------------------------------*/

      if (xu_ghost > -1)
      {
	 xu_ghost = max(xu_ghost, 1);
	 GrGeomExtentsIXUpper(extents[is]) =
	    (SubgridIX(subgrid) + SubgridNX(subgrid) + xu_ghost) * ref - 1;
      }
      else
      {
	 GrGeomExtentsIXUpper(extents[is]) = bg_ix + bg_nx - 1;
      }

      if (yu_ghost > -1)
      {
	 yu_ghost = max(yu_ghost, 1);
	 GrGeomExtentsIYUpper(extents[is]) =
	    (SubgridIY(subgrid) + SubgridNY(subgrid) + yu_ghost) * ref - 1;
      }
      else
      {
	 GrGeomExtentsIYUpper(extents[is]) = bg_iy + bg_ny - 1;
      }

      if (zu_ghost > -1)
      {
	 zu_ghost = max(zu_ghost, 1);
	 GrGeomExtentsIZUpper(extents[is]) =
	    (SubgridIZ(subgrid) + SubgridNZ(subgrid) + zu_ghost) * ref - 1;
      }
      else
      {
	 GrGeomExtentsIZUpper(extents[is]) = bg_iz + bg_nz - 1;
      }

      /*------------------------------------------
       * convert to "octree coordinates"
       *------------------------------------------*/

      /* Moved into the loop by SGS 7/8/98, was lying outside the is
	 loop which was an error (accessing invalid array elements)
	 */
      
      GrGeomExtentsIXLower(extents[is]) -= bg_ix;
      GrGeomExtentsIYLower(extents[is]) -= bg_iy;
      GrGeomExtentsIZLower(extents[is]) -= bg_iz;
      GrGeomExtentsIXUpper(extents[is]) -= bg_ix;
      GrGeomExtentsIYUpper(extents[is]) -= bg_iy;
      GrGeomExtentsIZUpper(extents[is]) -= bg_iz;

   }


   extent_array = GrGeomNewExtentArray(extents, size);

   return extent_array;
}


/*--------------------------------------------------------------------------
 * GrGeomNewSolid
 *--------------------------------------------------------------------------*/

GrGeomSolid   *GrGeomNewSolid(data, patches, num_patches,
			      octree_bg_level, octree_ix, octree_iy, octree_iz)
GrGeomOctree  *data;
GrGeomOctree **patches;
int            num_patches;
int            octree_bg_level;
int            octree_ix;
int            octree_iy;
int            octree_iz;
{
    GrGeomSolid   *new;


    new = talloc(GrGeomSolid, 1);

    (new -> data)            = data;
    (new -> patches)         = patches;
    (new -> num_patches)     = num_patches;
    (new -> octree_bg_level) = octree_bg_level;
    (new -> octree_ix)       = octree_ix;
    (new -> octree_iy)       = octree_iy;
    (new -> octree_iz)       = octree_iz;

    return new;
}


/*--------------------------------------------------------------------------
 * GrGeomFreeSolid
 *--------------------------------------------------------------------------*/

void          GrGeomFreeSolid(solid)
GrGeomSolid  *solid;
{
   int  i;

   GrGeomFreeOctree(GrGeomSolidData(solid));
   for (i = 0; i < GrGeomSolidNumPatches(solid); i++)
      GrGeomFreeOctree(GrGeomSolidPatch(solid, i));
   tfree(GrGeomSolidPatches(solid));

   tfree(solid);
}


/*--------------------------------------------------------------------------
 * GrGeomSolidFromInd
 *--------------------------------------------------------------------------*/

void             GrGeomSolidFromInd(solid_ptr, indicator_field, indicator)
GrGeomSolid    **solid_ptr;
Vector          *indicator_field;
int              indicator;
{
   GrGeomOctree *solid_octree;

   int           octree_bg_level;
   int           ix, iy, iz;
   double        xl, yl, zl, xu, yu, zu;

   /*------------------------------------------------------
    * Create the GrGeom solids, converting only the first
    * `nsolids' indicator solids
    *------------------------------------------------------*/

   octree_bg_level = GrGeomGetOctreeInfo(&xl, &yl, &zl, &xu, &yu, &zu,
					 &ix, &iy, &iz);

   GrGeomOctreeFromInd(&solid_octree, indicator_field, indicator,
                        xl, yl, zl, xu, yu, zu,
                        octree_bg_level, ix, iy, iz);

   *solid_ptr = GrGeomNewSolid(solid_octree, NULL, 0, octree_bg_level, ix, iy, iz);
}


/*--------------------------------------------------------------------------
 * GrGeomSolidFromGeom
 *--------------------------------------------------------------------------*/

void                GrGeomSolidFromGeom(solid_ptr, geom_solid, extent_array)
GrGeomSolid       **solid_ptr;
GeomSolid          *geom_solid;
GrGeomExtentArray  *extent_array;
{
   GrGeomSolid    *solid;

   GrGeomOctree   *solid_octree;
   GrGeomOctree  **patch_octrees;
   int             num_patches;
   int             octree_bg_level, ix, iy, iz;


   /*------------------------------------------------------
    * Convert to GrGeomOctree format
    *------------------------------------------------------*/

   switch(GeomSolidType(geom_solid))
   {

   case GeomTSolidType:
   {
      GeomTSolid  *solid_data = GeomSolidData(geom_solid);

      GeomTIN     *surface;
      int        **patches;
      int         *num_patch_triangles;

      double       xl, yl, zl, xu, yu, zu;


      surface             = (solid_data -> surface);
      patches             = (solid_data -> patches);            
      num_patches	  = (solid_data -> num_patches);        
      num_patch_triangles = (solid_data -> num_patch_triangles);

      octree_bg_level = GrGeomGetOctreeInfo(&xl, &yl, &zl, &xu, &yu, &zu,
					    &ix, &iy, &iz);

      GrGeomOctreeFromTIN(&solid_octree, &patch_octrees,
			  surface, patches, num_patches, num_patch_triangles,
			  extent_array, xl, yl, zl, xu, yu, zu,
			  octree_bg_level,
			  octree_bg_level + GlobalsMaxRefLevel);

      break;
   }

   }

   solid = GrGeomNewSolid(solid_octree, patch_octrees, num_patches,
			  octree_bg_level, ix, iy, iz);

   *solid_ptr = solid;
}


