/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"
#ifdef HAVE_P4EST
#include <p4est.h>
#include <p8est.h>
#include "../p4est_wrap/parflow_p4est.h"


/*--------------------------------------------------------------------------
 * This routine returns the elevations on a patch of a
 * solid at each (x,y) coordinate of an array of subgrid.  The result is
 * returned in an array of 2D real arrays.
 *
 * This routine is called by the pressure boundary condition routine and
 * by the pressure initial condition routine which calculate hydrostatic
 * conditions relative to a reference patch on a reference solid.
 *--------------------------------------------------------------------------*/

double         **CalcElevations_with_p4est(
   GeomSolid       *geom_solid,
   int              ref_patch,
   SubgridArray    *subgrids,
   ProblemData  *problem_data)
{
   GrGeomSolid        *grgeom_solid;

   GrGeomExtentArray  *extent_array;

   Background         *bg = GlobalsBackground;

   Subgrid            *subgrid;

   Vector      *z_mult            = ProblemDataZmult(problem_data);
   Vector      *rsz               = ProblemDataRealSpaceZ(problem_data);
   Subvector   *z_mult_sub;
   Subvector   *rsz_sub;
   double      *z_mult_dat;
   double      *rsz_dat;

   double            **elevation_arrays;
   double             *elevation_array;
   double              dz2, zupper, zlower, zinit;

   int	               ix, iy, iz;
   int	               nx, ny, nz;
   int	               rz;

   int                *fdir;

   int	               is, i,  j, k, iel,ival;

   int                 num, srcRank, dstRank;

   sc_list_t          *send_buffer;
   sc_array_t         *send_requests;
   sc_array_t         *recv_requests;
   sc_link_t          *link;
   double             *temp_array;
   double             *ebuf;
   sc_MPI_Request     *sreq;
   sc_MPI_Request     *rreq;
   int                 tag, mpiret;
   int                 sidx, info[2];
   int                 R;
   parflow_p4est_grid_t *pfgrid = globals->grid3d->pfgrid;

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

   send_buffer    = sc_list_new (NULL);
   send_requests  = sc_array_new (sizeof (sc_MPI_Request));
   recv_requests  = sc_array_new (sizeof (sc_MPI_Request));

   /* First loop: - All subgrids compute elevation array
    *             - Except zlevel 0 subgrids, post all the Isends to
    *               lowest subgrid in our Z-Column */
   ForSubgridI(is, subgrids)
   {
      subgrid    = SubgridArraySubgrid(subgrids, is);
      z_mult_sub = VectorSubvector(z_mult, is);
      rsz_sub    = VectorSubvector(rsz, is);
      z_mult_dat = SubvectorData(z_mult_sub);
      rsz_dat    = SubvectorData(rsz_sub);

      /* RDF: assume resolutions are the same in all 3 directions */
      rz = SubgridRZ(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = IndexSpaceZ(zlower, rz);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = IndexSpaceZ(zupper, rz) - iz + 1;
      num = nx*ny;

      dz2 = RealSpaceDZ(rz)/2.0;

      elevation_array = ctalloc(double, num);

      /* Initialize the elevation_array */
      for (iel = 0; iel < (nx*ny); iel++)
         elevation_array[iel] = FLT_MAX;

      /* Construct elevation_array */
      GrGeomPatchLoop(i, j, k, fdir, grgeom_solid, ref_patch,
                      rz, ix, iy, iz, nx, ny, nz,
      {
         if (fdir[2] != 0)
         {
            iel = (j-iy)*nx + (i-ix);
            ival = SubvectorEltIndex(z_mult_sub, i,j,k);

	    if( ( (i >= SubgridIX(subgrid)) && (i < (SubgridIX(subgrid) + SubgridNX(subgrid)))) &&
		( (j >= SubgridIY(subgrid)) && (j < (SubgridIY(subgrid) + SubgridNY(subgrid)))) &&
		( (k >= SubgridIZ(subgrid)) && (k < (SubgridIZ(subgrid) + SubgridNZ(subgrid)))) )
	    {
	       elevation_array[iel] = rsz_dat[ival]+ fdir[2]*dz2*z_mult_dat[ival];
	    }
	 }
      });

      sidx = SubgridMinusZneigh(subgrid);

      /** We are not the lowest subgrid in this Z column. */
      if( sidx  >= 0 )
      {
          /* Send elevation array to lowest subgrid in this Z column. */
          parflow_p4est_get_projection_info (subgrid, 0, pfgrid, info);
          tag     = info[1];
          dstRank = info[0];
          sreq    = (sc_MPI_Request *) sc_array_push (send_requests);
          link    = sc_list_append (send_buffer, (void *) elevation_array);
          ebuf    = (double *) link->data;
          mpiret  = sc_MPI_Isend (ebuf, num, sc_MPI_DOUBLE, dstRank,
                                  tag, amps_CommWorld, sreq);
          SC_CHECK_MPI (mpiret);

      }

      elevation_arrays[is] = elevation_array;
   }

   /* Reduction and Column "Broadcast" */
   ForSubgridI(is, subgrids)
   {
     subgrid = SubgridArraySubgrid(subgrids, is);
     nx      = SubgridNX(subgrid);
     ny      = SubgridNY(subgrid);
     sidx    = SubgridMinusZneigh(subgrid);
     num     = nx*ny;

     /** We are the lowest subgrid in this Z column. */
     if( sidx  < 0 )
     {
         elevation_array = elevation_arrays[is];
         parflow_p4est_get_projection_info (subgrid, 0, pfgrid, info);
         tag = info[1];
         temp_array = ctalloc(double, num);

         for(R = 1; R < GlobalsNumProcsZ; R++)
         {
             /** Receive and reduce elevation array from top subgrids in
               * in our column */
             parflow_p4est_get_projection_info (subgrid, R, pfgrid, info);
             srcRank = info[0];
             mpiret  = sc_MPI_Recv (temp_array, num, sc_MPI_DOUBLE, srcRank,
                                    tag, amps_CommWorld, sc_MPI_STATUS_IGNORE);
             SC_CHECK_MPI (mpiret);

             for (iel = 0; iel < num; iel++)
             {
                 elevation_array[iel] = SC_MIN(elevation_array[iel], temp_array[iel]);
             }
         }

         free(temp_array);

	 /* Original algorithm had default value of 0.0.
	  * This forces unset values to 0.0  after reduction */
	 for (iel = 0; iel < num; iel++)
	 {
	    if(elevation_array[iel] == FLT_MAX)
	    {
	       elevation_array[iel] = zinit;
	    }
	 }

	 /** Send reduced array to top subgrids in our column */
	 for(R = 1; R < GlobalsNumProcsZ; R++)
	 {
	     sreq    = (sc_MPI_Request *) sc_array_push (send_requests);
	     parflow_p4est_get_projection_info(subgrid, R, pfgrid, info);
	     dstRank = info[0];
	     link    = sc_list_append (send_buffer, (void *) elevation_array);
	     ebuf    = (double *) link->data;
	     mpiret  = sc_MPI_Isend (ebuf, num, sc_MPI_DOUBLE, dstRank,
				     tag, amps_CommWorld, sreq);
	     SC_CHECK_MPI (mpiret);
	 }
      }else{

         /** We are not the lowest subgrid in this Z column:
           * Ireceive reduced array from bottom subgrid     */
          parflow_p4est_get_projection_info(subgrid, 0, pfgrid, info);
          srcRank = info[0];
          rreq  = (sc_MPI_Request *) sc_array_push (recv_requests);
          tag     = info[1];
          ebuf    = elevation_arrays[is];
          mpiret  = sc_MPI_Irecv (ebuf, num, sc_MPI_DOUBLE, srcRank,
                                  tag, amps_CommWorld, rreq);
          SC_CHECK_MPI (mpiret);
      }
   }

   /** Complete receive operations */
   mpiret = sc_MPI_Waitall ((int) recv_requests->elem_count,
                            (sc_MPI_Request *) recv_requests->array,
                            sc_MPI_STATUSES_IGNORE);
   SC_CHECK_MPI (mpiret);

   /** Complete send operations */
   mpiret = sc_MPI_Waitall ((int) send_requests->elem_count,
                            (sc_MPI_Request *) send_requests->array,
                            sc_MPI_STATUSES_IGNORE);
   SC_CHECK_MPI (mpiret);

   /** Free requests and buffer arrays */
   sc_array_destroy (send_requests);
   sc_array_destroy (recv_requests);
   sc_list_destroy (send_buffer);

   GrGeomFreeSolid(grgeom_solid);

   return elevation_arrays;
}

#endif /* !HAVE_P4EST */
