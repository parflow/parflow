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
#include "toposlopes.h"
#include <math.h>


/*-----------------------------------------------------------------------
 * ComputeSlopeXUpwind:
 *
 * Calculate the topographic slope at [i,j] in the x-direction using a first-
 * order upwind finite difference scheme. 
 *
 * If cell is a local maximum in x, largest downward slope to neightbor is used.
 * If cell is a local minimum in x, slope is set to zero (no drainage in x).
 * Otherwise, upwind slope is used (slope from parent to [i,j]).
 *
 *-----------------------------------------------------------------------*/
void ComputeSlopeXUpwind(
   Databox *dem,
   double   dx, 
   Databox *sx)
{
   int             i,  j;
   int             nx, ny;
   double          s1, s2;

   nx = DataboxNx(dem);
   ny = DataboxNy(dem);

   // Loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // Skip cells with nodata/missing values
         // NOTE: nodata value is hard-wired to -9999.0!
         if (*DataboxCoeff(dem,i,j,0)==-9999.0)
         { 
            *DataboxCoeff(sx,i,j,0) = -9999.0;
         }
        
         // Deal with corners
         // SW corner [0,0]
         else if ((i==0) && (j==0))
         { 
            if (*DataboxCoeff(dem,i+1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0) - *DataboxCoeff(dem,i,j,0))/dx;
            }
         }

         // SE corner [nx-1,0]
         else if ((i==nx-1) && (j==0))
         {
            if (*DataboxCoeff(dem,i-1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
            }
         }

         // NE corner [nx-1,ny-1]
         else if ((i==nx-1) && (j==ny-1))
         {
            if (*DataboxCoeff(dem,i-1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
            }
         }

         // NW corner [0,ny-1]
         else if ((i==0) && (j==ny-1))
         {
            if (*DataboxCoeff(dem,i+1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
            }
         }

         // Eastern edge, not corner [nx-1,1:ny-1]
         else if (i==nx-1)
         {
            if (*DataboxCoeff(dem,i-1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
            }
         }

         // Western edge, not corner [0,1:ny-1]
         else if (i==0)
         {
            if (*DataboxCoeff(dem,i+1,j,0)==-9999.0)
            {
               *DataboxCoeff(sx,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dx;
            }
            else
            {
               *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
            }
         }

         // All other cells...
         else 
         {

            // compute slope in negative x-direction
            if (*DataboxCoeff(dem,i-1,j,0)==-9999.0)
            {
               s1     = (*DataboxCoeff(dem,i,j,0)-0.0)/dx;
            }
            else
            {
               s1     = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
            }

            // compute slope in positive x-direction
            if (*DataboxCoeff(dem,i+1,j,0)==-9999.0)
            {
               s2     = (0.0-*DataboxCoeff(dem,i,j,0))/dx;                       
            }
            else
            {
               s2     = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
            }

            // determine slope (upwind, max down grad, or zero)
            if ((s1>0.) && (s2<0.))                     // LOCAL MAXIMUM -- use max down grad
            {
               if (fabs(s1)>fabs(s2))
               {
                  *DataboxCoeff(sx,i,j,0) = s1;
               } else { 
                  *DataboxCoeff(sx,i,j,0) = s2; 
               }
            } 
            else if ((s1<0.) && (s2>0.))              // LOCAL MINIMUM -- set slope to zero
            {
               *DataboxCoeff(sx,i,j,0)    = 0.0; 
            }
            else if ((s1<0.) && (s2<0.))              // PASS THROUGH (from left)
            {
               *DataboxCoeff(sx,i,j,0)    = s1;
            }
            else if ((s1>0.) && (s2>0.))              // PASS THROUGH (from right)
            {
               *DataboxCoeff(sx,i,j,0)    = s2;
            }
            else                                   // ZERO SLOPE (s1==s2==0.0)
            {
               *DataboxCoeff(sx,i,j,0)    = 0.0;
            }
         }

      }  // end loop over i

   } // end loop over j
          
}


/*-----------------------------------------------------------------------
 * ComputeSlopeYUpwind:
 *
 * Calculate the topographic slope at [i,j] in the y-direction using a first-
 * order upwind finite difference scheme.
 *
 * If cell is a local maximum in y, largest downward slope to neightbor is used.
 * If cell is a local minimum in y, slope is set to zero (no drainage in y).
 * Otherwise, upwind slope is used (slope from parent to [i,j]).
 *
 *-----------------------------------------------------------------------*/
void ComputeSlopeYUpwind(
   Databox *dem,
   double   dy,
   Databox *sy)
{
   int             i,  j;
   int             nx, ny;
   double          s1, s2;

   nx = DataboxNx(dem);
   ny = DataboxNy(dem);

   // Loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // Skip cells with nodata/missing/ocean values
         // NOTE: nodata value is hard-wired to -9999.0!
         if (*DataboxCoeff(dem,i,j,0)==-9999.0)
         {
            *DataboxCoeff(sy,i,j,0) = -9999.0;
         }

         // Deal with corners
         // SW corner [0,0]
         else if ((i==0) && (j==0))
         {
            if (*DataboxCoeff(dem,i,j+1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dy;
            }
            else
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
            }
         }

         // SE corner [nx-1,0]
         else if ((i==nx-1) && (j==0))
         {
            if (*DataboxCoeff(dem,i,j+1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dy;
            }
            else
            { 
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
            }
         }

         // NE corner [nx-1,ny-1]
         else if ((i==nx-1) && (j==ny-1))
         {
            if (*DataboxCoeff(dem,i,j-1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dy;
            }
            else
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
            }
         }

         // NW corner [0,ny-1]
         else if ((i==0) && (j==ny-1))
         {
            if (*DataboxCoeff(dem,i,j-1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dy;
            }
            else
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
            }
         }

         // Southern edge, not corner [1:nx-1,0]
         else if (j==0)
         {
            if (*DataboxCoeff(dem,i,j+1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (0.0-*DataboxCoeff(dem,i,j,0))/dy;
            }
            else
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
            }
         }

         // Northern edge, not corner [1:nx,ny-1]
         else if (j==ny-1)
         {
            if (*DataboxCoeff(dem,i,j-1,0)==-9999.0)
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-0.0)/dy;
            }
            else
            {
               *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
            }
         }

         // All other cells...
         else
         {
 
            // compute slope in negative y-direction
            if (*DataboxCoeff(dem,i,j-1,0)==-9999.0)
            {
               s1     = (*DataboxCoeff(dem,i,j,0)-0.0)/dy;
            }
            else
            {            
               s1     = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
            }

            // compute slope in the positive y-direction
            if (*DataboxCoeff(dem,i,j+1,0)==-9999.0)
            {
               s2     = (0.0-*DataboxCoeff(dem,i,j,0))/dy;
            }
            else
            {
               s2     = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
            }

            // determine slope (upwind, max down grad, or zero)
            if ((s1>0.) && (s2<0.))                     // LOCAL MAXIMUM -- use max down grad
            {
               if (fabs(s1)>fabs(s2))
               {
                  *DataboxCoeff(sy,i,j,0) = s1;
               } else {
                  *DataboxCoeff(sy,i,j,0) = s2;
               }
            }
            else if ((s1<0.) && (s2>0.))              // LOCAL MINIMUM -- set slope to zero
            {
               *DataboxCoeff(sy,i,j,0)    = 0.0;
            }
            else if ((s1<0.) && (s2<0.))              // PASS THROUGH (from left)
            {
               *DataboxCoeff(sy,i,j,0)    = s1;
            }
            else if ((s1>0.) && (s2>0.))              // PASS THROUGH (from right)
            {
               *DataboxCoeff(sy,i,j,0)    = s2;
            }
            else                                   // ZERO SLOPE (s1==s2==0.0)
            {
               *DataboxCoeff(sy,i,j,0)    = 0.0;
            }
         }

      }  // end loop over i

   } // end loop over j

}


/*-----------------------------------------------------------------------
 * ComputeTestParent:
 *
 * Returns 1 if ii,jj is parent of i,j based on sx and sy.
 * Returns 0 otherwise.
 *
 * Otherwise, upwind slope is used (slope from parent to [i,j]).
 *
 *-----------------------------------------------------------------------*/

int ComputeTestParent( 
   int i,
   int j, 
   int ii,
   int jj, 
   Databox *dem,
   Databox *sx, 
   Databox *sy)
{ 

   int test = -999;

   // Make sure [i,j] and [ii,jj] are adjacent
   if ( (fabs(i-ii)+fabs(j-jj)) == 1.0 )
   {
      if ( *DataboxCoeff(dem,ii,jj,0)==-9999.0 )
      {
         test  = 0;
      }
      else if ( (ii==i-1) && (jj==j) && (*DataboxCoeff(sx,ii,jj,0)<0.) )
      {
         test  = 1;
      }
      else if ( (ii==i+1) && (jj==j) && (*DataboxCoeff(sx,ii,jj,0)>0.) )
      {
         test  = 1;
      }
      else if ( (ii==i) && (jj==j-1) && (*DataboxCoeff(sy,ii,jj,0)<0.) )
      {
         test  = 1;
      }
      else if ( (ii==i) && (jj==j+1) && (*DataboxCoeff(sy,ii,jj,0)>0.) )
      {
         test  = 1;
      }
      else
      {
         test  = 0;
      }
   }
   else
   { 
      printf("Error: TestParent(i,j,ii,jj,sx,sy) \n");
      printf("       [i,j] and [ii,jj] are not adjacent cells! \n");
   }
   
   return test; 

}


/*-----------------------------------------------------------------------
 * ComputeParentMap:
 *
 * Computes upstream area for the given cell [i,j] by recursively looping 
 * over neighbors and summing area of all parent cells (moving from cell [i,j]
 * to parents, to their parents, etc. until reaches upper end of basin).
 * 
 * Area returned as NUMBER OF CELLS 
 * To get actual area, multiply area_ij*dx*dy
 *
 *-----------------------------------------------------------------------*/
 
void ComputeParentMap( int i, 
                       int j, 
                       Databox *dem,
                       Databox *sx, 
                       Databox *sy,
                       Databox *parentmap)
{

   int      ii, jj;
   int      parent;

   int nx  = DataboxNx(sx);
   int ny  = DataboxNy(sx);

   // skip if self is nodata cell (dem=-9999.0)
   if (*DataboxCoeff(dem,i,j,0)==-9999.0)
   {
      ;
   }

   // otherwise, loop over neighbors...
   else
   {

      // Add self to parent map
      *DataboxCoeff(parentmap,i,j,0) = 1.0;

      // Loop over neighbors
      for (jj = j-1; jj <= j+1; jj++)
      {
         for (ii = i-1; ii <= i+1; ii++)
         {

            // skip self
            if ((ii==i) && (jj==j))
            {
               ;
            }

            // skip diagonals
            else if ((ii!=i) && (jj!=j))
            {
               ;
            }

            // skip off-grid cells
            else if (ii<0 || jj<0 || ii>nx-1 || jj>ny-1)
            {
               ;
            }

            // skip if NEIGHBOR is nodata cell (dem=-9999.0)
            else if (*DataboxCoeff(dem,ii,jj,0)==-9999.0)
            {
               ;
            }

            // otherwise, test if [ii,jj] is parent of [i,j]...
            else 
            {
      
               parent  = ComputeTestParent( i,j,ii,jj,dem,sx,sy );
             
               // if parent, loop recursively
               if ( (parent==1) && (*DataboxCoeff(parentmap,ii,jj,0)!=1.0) )
               { 
                  ComputeParentMap(ii,jj,dem,sx,sy,parentmap);
               }
            }

         } // end loop over i

      } // end loop over j

   } // end if self==nodata

}
    

/*-----------------------------------------------------------------------
 * ComputeUpstreamArea:
 *
 * Computes upstream area for all cells by looping over grid and calling 
 * ComputeUpstreamAreaIJ for each cell.
 *
 * Area returned as NUMBER OF CELLS
 * To get actual area, multiply area_ij*dx*dy
 *
 *-----------------------------------------------------------------------*/
void ComputeUpstreamArea( 
   Databox *dem, 
   Databox *sx, 
   Databox *sy, 
   Databox *area )
{

   int             i,  j;
   int             ii, jj;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;
   Databox        *parentmap;

   // create new databox for parentmap
   nx        = DataboxNx(sx);
   ny        = DataboxNy(sx);
   nz        = DataboxNz(sx);
   x         = DataboxX(sx);
   y         = DataboxY(sx);
   z         = DataboxZ(sx);
   dx        = DataboxDx(sx);
   dy        = DataboxDy(sx);
   dz        = DataboxDz(sx);
   parentmap = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);

   // loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // zero out parentmap
         for (jj = 0; jj < ny; jj++)
         { 
            for (ii = 0; ii < nx; ii++)
            {
               *DataboxCoeff(parentmap,ii,jj,0) = 0.0;
            } 
         }

         // generate parent map for [i,j]
         // (skip nodata cells...if not nodata, recursive loop)
         if ( *DataboxCoeff(dem,i,j,0)!=-9999.0 )
         {
            ComputeParentMap(i,j,dem,sx,sy,parentmap);
         }

         // calculate area as sum over parentmap
         for (jj = 0; jj < ny; jj++)
         {
            for (ii = 0; ii < nx; ii++)
            {
               *DataboxCoeff(area,i,j,0) = *DataboxCoeff(area,i,j,0) + *DataboxCoeff(parentmap,ii,jj,0);
            }
         }

      } // end loop over i

   } // end loop over j

   FreeDatabox(parentmap);

}


/*-----------------------------------------------------------------------
 * ComputeFlatMap:
 *
 * Computes map of contiguous flat region starting from the given [i,j]. 
 * Recursively loops over neighbors to map extent of flat region
 *
 *-----------------------------------------------------------------------*/

void ComputeFlatMap( 
   int i,
   int j,
   Databox *dem,
   Databox *flatmap)
{

   int      ii, jj;
   int nx  = DataboxNx(dem);
   int ny  = DataboxNy(dem);

   // skip if self is nodata cell (dem=-9999.0)
   if (*DataboxCoeff(dem,i,j,0)==-9999.0)
   {
      ;
   }

   // otherwise, loop over neighbors...
   else
   {

      // Add self to flat map
      *DataboxCoeff(flatmap,i,j,0) = 1.0;

      // Loop over neighbors
      for (jj = j-1; jj <= j+1; jj++)
      {
         for (ii = i-1; ii <= i+1; ii++)
         {

            // skip self
            if ((ii==i) && (jj==j))
            {
               ;
            }

            // skip diagonals
            else if ((ii!=i) && (jj!=j))
            {
               ;
            }

            // skip off-grid cells
            else if (ii<0 || jj<0 || ii>nx-1 || jj>ny-1)
            {
               ;
            }

            // skip if NEIGHBOR is nodata cell (dem=-9999.0)
            else if (*DataboxCoeff(dem,ii,jj,0)==-9999.0)
            {
               ;
            }

            // otherwise, test if [ii,jj] has same elevation as [i,j]...
            else
            {
               // if same elevation, loop recursively
               if (*DataboxCoeff(dem,i,j,0)==*DataboxCoeff(dem,ii,jj,0))
               {
                  // only make recursive call if cell isn't already mapped...
                  if (*DataboxCoeff(flatmap,ii,jj,0)!=1.0) 
                  {
                     *DataboxCoeff(flatmap,ii,jj,0) = 1.0;
                     ComputeFlatMap(ii,jj,dem,flatmap);
                  }
               }
            }

         } // end loop over i

      } // end loop over j

   } // end if self==nodata

}


/*-----------------------------------------------------------------------
 * ComputeFillFlats:
 *
 * Scans DEM for flat areas, defined as contiguous areas where all cells
 * have exactly the same elevation. Areas must have at least one cell with
 * identical elevations at all adjacent neighbors (i.e., one cell with zero
 * slope in both x and y). 
 *
 * Once the extent of a flat area is identified, elevations are adjusted 
 * by bilinearly interpolating across the flat region. This imposes a small
 * arbitrary slope so that areas are no longer perfectly flat.
 *
 *-----------------------------------------------------------------------*/
void ComputeFillFlats(
    Databox *dem,
    Databox *newdem)
{

   int             flat;
   int             i,  j,  ii,  jj;
   int             iscan, jscan;
   double          zleft, dleft;
   double          zright, dright;
   double          zup, dup;
   double          zdown, ddown;
   double          wt_sum, sum_wt;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;
   Databox        *flatmap;
 
   // Read grid info from input vars
   nx      = DataboxNx(dem);
   ny      = DataboxNy(dem);
   nz      = DataboxNz(dem);
   x       = DataboxX(dem);
   y       = DataboxY(dem);
   z       = DataboxZ(dem);
   dx      = DataboxDx(dem);
   dy      = DataboxDy(dem);
   dz      = DataboxDz(dem);

   // Create empty databox for flatmap 
   flatmap = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);

   // Set initial values of newdem to dem
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {
         *DataboxCoeff(newdem,i,j,0) = *DataboxCoeff(dem,i,j,0);
      }
   }

   // Loop over full grid...
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // skip nodata cells
         if ( *DataboxCoeff(newdem,i,j,0) == -9999.0 )
         {
            ;
         }
 
         // else, loop over neighbors, test if cell is a local flat spot
         else
         {

            // assume flat until tests otherwise
            flat    = 1;
            for (jj = j-1; jj <= j+1; jj++)
            { 
               for (ii = i-1; ii <= i+1; ii++)
               { 
                  // make sure [i,j] and [ii,jj] are adjacent
                  if ( (fabs(i-ii)+fabs(j-jj)) == 1.0 )
                  { 
                     // skip off-grid cells
                     if ( (ii<0) || (jj<0) || (ii>=nx) || (jj>=ny) )
                     { 
                        ;
                     }
                     // otherwise, if [i,j] and [ii,jj] are NOT at same elevation, set flat to zero
                     else if (*DataboxCoeff(newdem,i,j,0) != *DataboxCoeff(newdem,ii,jj,0))
                     { 
                           flat = 0;
                     } // end if/else on off-grid cells
                  } // end if on adjacent
               } // end loop over ii
            } // end loop over jj

            // if [i,j] is part of a flat region...
            // -- map extent of flat region
            // -- interpolate to eliminate flat region
            if (flat==1)
            {

               // reset flatmap to all zeros
               for (jj = 0; jj < ny; jj++)
               { 
                  for (ii = 0; ii < nx; ii++)
                  {
                     *DataboxCoeff(flatmap,ii,jj,0) = 0.0;
                  }
               }

               // call ComputeFlatMap
               ComputeFlatMap(i,j,newdem,flatmap);

               // modify newdem values to eliminate flat region
               for (jj = 0; jj < ny; jj++)
               {
                  for (ii = 0; ii < nx; ii++)
                  {

                     // don't change edge cells
                     if ( (ii==0) || (ii==nx-1) || (jj==0) || (jj==ny-1) )
                     {
                        ;
                     }

                     // interpolate to set flat cells
                     else if (*DataboxCoeff(flatmap,ii,jj,0)==1.0)
                     {

                        zleft    = 0.0;
                        dleft    = 0.0;
                        zright   = 0.0;
                        dright   = 0.0;
                        zup      = 0.0;
                        dup      = 0.0;
                        zdown    = 0.0;
                        ddown    = 0.0;
                           
                        // scan left...
                        iscan    = ii;
                        while ( (iscan>0) && (*DataboxCoeff(flatmap,iscan,jj,0)==1.0) )
                        {
                           iscan = iscan - 1;
                        }
                        zleft    = *DataboxCoeff(newdem,iscan,jj,0);
                        dleft    = fabs( (double)ii - (double)iscan );
                        
                        // scan right...
                        iscan    = ii;
                        while ( (iscan<nx-1) && (*DataboxCoeff(flatmap,iscan,jj,0)==1.0) )
                        { 
                           iscan = iscan + 1;
                        }
                        zright   = *DataboxCoeff(newdem,iscan,jj,0);
                        dright   = fabs( (double)ii - (double)iscan );
              
                        // scan up...
                        jscan    = jj;
                        while ( (jscan<ny-1) && (*DataboxCoeff(flatmap,ii,jscan,0)==1.0) )
                        {
                           jscan = jscan + 1;
                        }
                        zup      = *DataboxCoeff(newdem,ii,jscan,0);
                        dup      = fabs( (double)jj - (double)jscan );
                        
                        // scan down...
                        jscan    = jj;
                        while ( (jscan>0) && (*DataboxCoeff(flatmap,ii,jscan,0)==1.0) )
                        { 
                           jscan = jscan - 1;
                        }
                        zdown    = *DataboxCoeff(newdem,ii,jscan,0);
                        ddown    = fabs( (double)jj - (double)jscan );
                        
                        // compute weighted sum and sum of weights (linear inverse distance weighting)
                        wt_sum   = (zleft/dleft) + (zright/dright) + (zup/dup) + (zdown/ddown);
                        sum_wt   = (1./dleft)    + (1./dright)     + (1./dup)  + (1./ddown);

                        // Inverse distance weighted avg...
                        // *DataboxCoeff(newdem,ii,jj,0) = wt_sum/sum_wt;
                        // Inverse distance weighted avg WITH SELF TERM!
                        *DataboxCoeff(newdem,ii,jj,0) = (*DataboxCoeff(newdem,ii,jj,0) + (wt_sum/sum_wt)) / 2.0;

                     } // end else
                  } // end loop over ii
               } // end loop over jj
            } // end if flat==1
         } // end if/else nodata
      } // end loop over i
   } // end loop over j

   FreeDatabox(flatmap);

}


/*-----------------------------------------------------------------------
 * ComputePitFill:
 *
 * Computes sink locations based on 1st order upwind slopes; adds dpit to 
 * sinks to iteratively fill according to traditional "pit fill" strategy. 
 *
 * Note that this routine operates ONCE -- Iterations are handled through
 * parent function (pftools -> PitFillCommand)
 *
 * Inputs is the DEM to be processed and the value of dpit.
 * Outputs is the revised DEM and number of remaining sinks.
 * Uses ComputeSlopeXUpwind and ComputeSlopeYUpwind to determine slopes
 * Considers cells sinks (pits) if sx[i,j]==sy[i,j]==0 *OR* if lower than all adjacent neighbors.
 *
 *-----------------------------------------------------------------------*/
int ComputePitFill( 
    Databox *dem, 
    double   dpit)
{

   int             nsink;
   int             lmin;
   int             i,  j;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;
   double          smag;
 
   Databox        *sx;
   Databox        *sy;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);
 
   // Compute slopes
   sx    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   sy    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);

   // Find Sinks + Pit Fill
   for (j = 0; j < ny; j++)
   { 
      for (i=0; i < nx; i++)
      {

         // skip if nodata cell
         if ( *DataboxCoeff(dem,i,j,0)!=-9999.0 )
         {

            // calculate slope magnitude
            smag   = sqrt( (*DataboxCoeff(sx,i,j,0))*(*DataboxCoeff(sx,i,j,0)) + 
                           (*DataboxCoeff(sy,i,j,0))*(*DataboxCoeff(sy,i,j,0)) );

            // test if local minimum
            lmin   = 0;
            if ( (i>0) && (j>0) && (i<nx-1) && (j<ny-1) )
            {
               if ( (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i-1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i+1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j-1,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j+1,0)) )
               {
                  lmin   = 1;    
               } 
               else
               {
                  lmin   = 0;
               }
            }

            // if smag==0 or lmin==1 -> pitfill
            if ( (smag==0.0) || (lmin==1) )
            {
               *DataboxCoeff(dem,i,j,0) = *DataboxCoeff(dem,i,j,0) + dpit;
            }

         } // end if nodata   
      } // end loop over i
   } // end loop over j

   // Recompute slopes
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);

   // Count remaining sinks
   nsink  = 0;
   for (j = 0; j < ny; j++)
   {
      for (i=0; i < nx; i++)
      {

         // skip nodata cells
         if ( *DataboxCoeff(dem,i,j,0)!=-9999.0 )
         { 
       
            // re-calculate slope magnitude from new DEM
            smag   = sqrt( (*DataboxCoeff(sx,i,j,0))*(*DataboxCoeff(sx,i,j,0)) +
                           (*DataboxCoeff(sy,i,j,0))*(*DataboxCoeff(sy,i,j,0)) );
         
            // test if local minimum
            lmin   = 0;
            if ( (i>0) && (j>0) && (i<nx-1) && (j<ny-1) )
            { 
               if ( (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i-1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i+1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j-1,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j+1,0)) )
               {
                  lmin   = 1;                     
               } 
               else
               {
                  lmin   = 0;
               }
            }

            // if smag==0 or lmin==1 -> count sinks 
            if ( (smag==0.0) || (lmin==1) )
            {
               nsink = nsink + 1;
            }

         } // end if nodata
      } // end loop over i
   } // end loop over j

   FreeDatabox(sx);
   FreeDatabox(sy);
   return nsink;

}


/*-----------------------------------------------------------------------
 * ComputeMovingAvg:
 *
 * Computes sink locations based on 1st order upwind slopes; fills sinks 
 * by taking average over adjacent cells ([i+wsize,j],[i-wsize,j],[i,j+wsize],[i,j-wsize]).
 *
 * Note that this routine operates ONCE -- Iterations are handled through
 * parent function (pftools -> MovingAvgCommand)
 *
 * Inputs is the DEM to be processed and the moving average window (wsize).
 * Outputs is the revised DEM and number of remaining sinks.
 * Uses ComputeSlopeXUpwind and ComputeSlopeYUpwind to determine slopes
 * Considers cells sinks if sx[i,j]==sy[i,j]==0 *OR* if cell is lower than all adjacent neighbors.
 *
 *-----------------------------------------------------------------------*/
int ComputeMovingAvg(
    Databox *dem,
    double   wsize)
{

   int             nsink;
   int             lmin;
   int             i,  j,  ii, jj;
   int             li, ri, lj, rj;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;
   double          smag;
   double          mavg, counter;

   Databox        *sx;
   Databox        *sy;
   // char            sx_hashkey;
   // char            sy_hashkey;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);

   // Compute slopes
   sx    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   sy    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);

   // Run moving average routine
   for (j = 0; j < ny; j++)
   {
      for (i=0; i < nx; i++)
      {

         // skip nodata cells
         if ( *DataboxCoeff(dem,i,j,0)!=-9999.0 )
         {
  
            // calculate slope magnitude
            smag   = sqrt( (*DataboxCoeff(sx,i,j,0))*(*DataboxCoeff(sx,i,j,0)) +
                           (*DataboxCoeff(sy,i,j,0))*(*DataboxCoeff(sy,i,j,0)) );

            // test if local minimum
            lmin   = 0;
            if ( (i>0) && (j>0) && (i<nx-1) && (j<ny-1) )
            {
               if ( (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i-1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i+1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j-1,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j+1,0)) )
               {
                  lmin   = 1;
               }
               else
               {
                  lmin   = 0;
               }
            }

            // if smag==0 or lmin==1 -> moving avg
            if ( (smag==0.0) || (lmin==1) )
            {
               mavg      = 0.0;
               counter   = 0.0;
               li        = i - wsize;
               ri        = i + wsize;
               lj        = j - wsize;
               rj        = j + wsize;
            
               // edges in i
               if (i <= wsize)
               {
                  li     = 0;
               }
               else if (i >= nx-wsize)
               {
                  ri     = nx-1;
               }

               // edges in j
               if (j <= wsize)
               {
                  lj     = 0;
               }
               else if (j >= ny-wsize)
               {
                  rj     = ny-1;
               }

               // calculate average
               for (jj = lj; jj <= rj; jj++)
               { 
                  for (ii = li; ii <= ri; ii++)
                  { 
                     if ( (ii!=i) || (jj!=j) )
                     {
                        mavg    = mavg + *DataboxCoeff(dem,i,j,0);
                        counter = counter + 1.0;
                     } // end if
                  } // end loop over ii
               } // end loop over jj
   
               // add dz/100. to eliminate nagging sinks in flat areas
               *DataboxCoeff(dem,i,j,0) = (mavg + (dz/100.0)) / counter; 

            } // end if smag==0 or lmin==1

         } // end if nodata
      } // end loop over i
   } // end loop over j

   // Recompute slopes
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);

   // Count remaining sinks
   nsink  = 0;
   for (j = 0; j < ny; j++)
   {
      for (i=0; i < nx; i++)
      {

         // skip if nodata
         if ( *DataboxCoeff(dem,i,j,0)!=-9999.0 )
         {

            // re-calculate slope magnitude from new DEM
            smag   = sqrt( (*DataboxCoeff(sx,i,j,0))*(*DataboxCoeff(sx,i,j,0)) +
                           (*DataboxCoeff(sy,i,j,0))*(*DataboxCoeff(sy,i,j,0)) );

            // test if local minimum
            lmin   = 0;
            if ( (i>0) && (j>0) && (i<nx-1) && (j<ny-1) )
            {
               if ( (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i-1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i+1,j,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j-1,0)) &&
                    (*DataboxCoeff(dem,i,j,0) < *DataboxCoeff(dem,i,j+1,0)) )
               {
                  lmin   = 1;
               }
               else
               {
                  lmin   = 0;
               }
            }

            // if smag==0 or lmin==1 -> count sinks
            if ( (smag==0.0) || (lmin==1) )
            {
               nsink = nsink + 1;
            }

         } // end if nodata
      } // end loop over i
   } // end loop over j

   FreeDatabox(sx);
   FreeDatabox(sy);
   return nsink;

}


/*-----------------------------------------------------------------------
 * ComputeSlopeD8:
 *
 * Calculate the topographic slope at [i,j] based on a simple D8 scheme.
 * Drainage direction is first identifed as towards lowest adjacent or diagonal 
 * neighbor. Slope is then calculated as DOWNWIND slope (from [i,j] to child).
 *
 * If cell is a local minimum, slope is set to zero (no drainage).
 *
 *-----------------------------------------------------------------------*/
void ComputeSlopeD8(
   Databox *dem,
   Databox *slope)
{

   int             i,  j,  ii, jj;
   int             imin,   jmin;
   int             nx, ny, nz;
   int             nodata;
   double          x,  y,  z;
   double          dx, dy, dz;
   double          dxy, zmin;
   double          s1, s2, s3;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);

   dxy   = sqrt( dx*dx + dy*dy );

   s1    = 0.;
   s2    = 0.;
   s3    = 0.;

   // Loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // skip nodata cells
         if ( *DataboxCoeff(dem,i,j,0)==-9999.0 )
         {
            *DataboxCoeff(slope,i,j,0) = -9999.0;
         }
       
         else 
         {

            // Loop over neighbors (adjacent and diagonal)
            // ** Find elevation and indices of lowest neighbor
            // ** Exclude off-grid cells
            imin   = -9999;        
            jmin   = -9999;         
            zmin   = 100000000.0;
            nodata = 0;
            for (jj = j-1; jj <= j+1; jj++)
            {
               for (ii = i-1; ii <= i+1; ii++)
               { 
            
                  // skip if off grid
                  if ((ii<0) || (jj<0) || (ii>nx-1) || (jj>ny-1))
                  {
                     ;
                  }
               
                  // find lowest neighbor
                  else
                  {
                     // count number of nodata neighbors
                     if ( *DataboxCoeff(dem,ii,jj,0) == -9999.0 )
                     { 
                        nodata = nodata + 1;
                     }
                     // set zmin, imin, jmin if lower than previous zmin
                     if ( *DataboxCoeff(dem,ii,jj,0) < zmin )
                     {
                        zmin   = *DataboxCoeff(dem,ii,jj,0);
                        imin   = ii; 
                        jmin   = jj;
                     }
                  } // end find lowest neighbor
               } // end loop over ii
            } // end loop over jj

            // Calculate slope towards lowest neighbor

            // If no neighbors are nodata cells:
            if ( nodata==0 )
            {
               // ** If edge cell and local minimum, drain directly off-grid at upwind slope
               // ... SW corner, local minimum ...
               if ( (i==0) && (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               { 
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i+1,j+1,0) - *DataboxCoeff(dem,i,j,0)) / dxy;
               }
 
               // ... SE corner, local minimum ...
               else if ( (i==nx-1) && (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i-1,j+1,0) - *DataboxCoeff(dem,i,j,0)) / dxy;
               }

               // ... NE corner, local minimum ...
               else if ( (i==nx-1) && (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i-1,j-1,0) - *DataboxCoeff(dem,i,j,0)) / dxy;
               }
 
               // ... NW corner, local minimum ...
               else if ( (i==0) && (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i+1,j-1,0) - *DataboxCoeff(dem,i,j,0)) / dxy;
               }

               // ... West edge, not corner, local minimum ...
               else if ( (i==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               { 
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i+1,j,0) - *DataboxCoeff(dem,i,j,0)) / dx;
               }
        
               // ... East edge, not corner, local minimum ...
               else if ( (i==nx-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - *DataboxCoeff(dem,i-1,j,0)) / dx;
               }

               // ... South edge, not corner, local minimum ...
               else if ( (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j+1,0) - *DataboxCoeff(dem,i,j,0)) / dy;
               }
 
               // ... North edge, not corner, local minimum ...
               else if ( (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - *DataboxCoeff(dem,i,j-1,0)) / dy;
               }
 
               // ... All other cells...
               else
               {
         
                  // Local minimum --> set slope to zero
                  if ( zmin==*DataboxCoeff(dem,i,j,0) )
                  {
                     *DataboxCoeff(slope,i,j,0) = 0.0;
                  }
     
                  // Else, calculate slope...
                  else
                  {
                     if ( i==imin )      // adjacent in y
                     {
                        *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) -
                                                          *DataboxCoeff(dem,imin,jmin,0)) / dy;
                     }
                     else if ( j==jmin ) // adjacent in x
                     {
                        *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - 
                                                          *DataboxCoeff(dem,imin,jmin,0)) / dx;
                     }
                     else
                     {
                        *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - 
                                                          *DataboxCoeff(dem,imin,jmin,0)) / dxy;
                     }
                  }
               }
            
            } // end if nodata==0

            // if nodata==1...
            // cell has single nodata neighbor...e.g., dead-end of sharp inlet
            // assume nodata cell is ocean/estuary...z[min] = 0.0 (not -9999)
            // [i,j] drains to nodata cell...
            // (slope calculations same as above, but substitutes 0.0 for local min instead of -9999.0)
            else if ( nodata==1 )
            { 
               // If cell is nodata (-9999.0), it will never be it's own local minimum
               // If not nodata cell and nodata>1, corners/edges will never be local minimum 
               // --> no special conditions needed for corners/edges if nodata>0!
               // If not nodata cell and nodata>1, self will never be local minimum
               // --> no special conditions needed for self=local min!

               if ( i==imin )      // nodata cell is adjacent in y -- assume z[nodata]==0.0 (ocean!)
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - 0.0) / dy;
               }
               else if ( j==jmin ) // nodata cell is adjacent in x -- assume z[nodata]==0.0 (ocean!)
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - 0.0) / dx;
               }
               else                // nodata cell is diagonal -- assume z[nodata]==0 (ocean!)
               {
                  *DataboxCoeff(slope,i,j,0) = fabs(*DataboxCoeff(dem,i,j,0) - 0.0) / dxy;
               }

            } // end if nodata==1

            // if nodata>1...
            // cell has multiple nodata neighbors...e.g. straight coastline, peninsula
            // requires testing special cases...
            // NOTE: the setup below only works because direction doesn't matter...only need max down grad!
            else if ( nodata>1 )
            { 
               s1     = -9999.0;
               s2     = -9999.0;
               s3     = -9999.0;
               // if adjacent left/right is nodata, compute slope over dy
               if ( (*DataboxCoeff(dem,i-1,j,0)==-9999.0) || (*DataboxCoeff(dem,i+1,j,0)==-9999.0) )
               { 
                  s1  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dx;
               }
               if ( (*DataboxCoeff(dem,i,j-1,0)==-9999.0) || (*DataboxCoeff(dem,i,j+1,0)==-9999.0) )
               {
                  s2  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dy;
               }
               if ( (*DataboxCoeff(dem,i-1,j-1,0)==-9999.0) || 
                    (*DataboxCoeff(dem,i-1,j+1,0)==-9999.0) ||
                    (*DataboxCoeff(dem,i+1,j-1,0)==-9999.0) ||
                    (*DataboxCoeff(dem,i+1,j+1,0)==-9999.0) )
               {
                  s3  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dxy;
               }
               
               *DataboxCoeff(slope,i,j,0) = fmax( s1, fmax(s2,s3) );

            } // end if nodata>1

         } // end if nodata

      }  // end loop over i

   } // end loop over j

}


/*-----------------------------------------------------------------------
 * ComputeSegmentD8:
 *
 * Compute the downstream slope segment lenth at [i,j] for D8 slopes.
 * D8 drainage directions are defined towards lowest adjacent or diagonal
 * neighbor. Segment length is then given as the distance from [i,j] to child 
 * (at cell centers).
 *
 * If child is adjacent in x --> ds = dx
 * If child is adjacent in y --> ds = dy
 * If child is diagonal --> ds = dxy = sqrt( dx*dx + dy*dy )
 *
 * If cell is a local minimum --> ds = 0.0
 * 
 *-----------------------------------------------------------------------*/
void ComputeSegmentD8(
   Databox *dem,
   Databox *ds)
{

   int             i,  j,  ii, jj;
   int             imin,   jmin;
   int             nx, ny, nz;
   int             nodata;
   double          x,  y,  z;
   double          dx, dy, dz;
   double          dxy, zmin;
   double          s1, s2, s3, smax;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);

   dxy   = sqrt( dx*dx + dy*dy );

   s1    = 0.;
   s2    = 0.;
   s3    = 0.;
   smax  = 0.;

   // Loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // skip nodata cells
         if ( *DataboxCoeff(dem,i,j,0)==-9999.0 )
         {
            *DataboxCoeff(ds,i,j,0) = -9999.0;
         }

         else
         {

            // Loop over neighbors (adjacent and diagonal)
            // ** Find elevation and indices of lowest neighbor
            // ** Exclude self and off-grid cells
            imin   = -9999;
            jmin   = -9999;
            zmin   = 100000000000.0;
            nodata = 0;
            for (jj = j-1; jj <= j+1; jj++)
            {
               for (ii = i-1; ii <= i+1; ii++)
               {
   
                  // skip if off grid
                  if ((ii<0) || (jj<0) || (ii>nx-1) || (jj>ny-1))
                  {
                    ;
                  }

                  // find lowest neighbor
                  else
                  {
                     // count number of nodata neighbors
                     if ( *DataboxCoeff(dem,ii,jj,0) == -9999.0 )
                     {
                        nodata = nodata + 1;
                     }
                     // set zmin, imin, jmin if lower than previous zmin
                     if ( *DataboxCoeff(dem,ii,jj,0) < zmin )
                     {
                        zmin   = *DataboxCoeff(dem,ii,jj,0);
                        imin   = ii;
                        jmin   = jj;
                     }
                  } // end find lowest neighbor
               }
            }

            // Calculate slope towards lowest neighbor

            // If no neighbors are nodata cells:
            if ( nodata==0 )
            {
               // ** If edge cell and local minimum, drain directly off-grid 
               // ... SW corner, local minimum ...
               if ( (i==0) && (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }
   
               // ... SE corner, local minimum ...
               else if ( (i==nx-1) && (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }

               // ... NE corner, local minimum ...
               else if ( (i==nx-1) && (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }

               // ... NW corner, local minimum ...
               else if ( (i==0) && (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }

               // ... West edge, not corner, local minimum ...
               else if ( (i==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dx;
               }   

               // ... East edge, not corner, local minimum ...
               else if ( (i==nx-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dx;
               }
   
               // ... South edge, not corner, local minimum ...
               else if ( (j==0) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dy;
               }

               // ... North edge, not corner, local minimum ...
               else if ( (j==ny-1) && (zmin==*DataboxCoeff(dem,i,j,0)) )
               {
                  *DataboxCoeff(ds,i,j,0) = dy;
               }

               // ... All other cells...
               else
               {

                  // Local minimum --> set slope to zero
                  if ( zmin==*DataboxCoeff(dem,i,j,0) )
                  {
                     *DataboxCoeff(ds,i,j,0) = 0.0;
                  }

                  // Else, calculate segment length...
                  else
                  {
                     if ( i==imin )      // adjacent in y
                     {
                        *DataboxCoeff(ds,i,j,0) = dy;
                     }
                     else if ( j==jmin ) // adjacent in x
                     {
                        *DataboxCoeff(ds,i,j,0) = dx;
                     }
                     else
                     {
                        *DataboxCoeff(ds,i,j,0) = dxy;
                     }
                  }
               }

            } // end if nodata==0

            // if nodata==1...
            // cell has single nodata neighbor...e.g., dead-end of sharp inlet
            // assume nodata cell is ocean/estuary...z[min] = 0.0 (not -9999)
            // [i,j] drains to nodata cell...
            else if ( nodata==1 )
            {
               // If cell is nodata (-9999.0), it will never be it's own local minimum
               // If not nodata cell and nodata>1, corners/edges will never be local minimum
               // --> no special conditions needed for corners/edges if nodata>0!
               // If not nodata cell and nodata>1, self will never be local minimum
               // --> no special conditions needed for self=local min!

               if ( i==imin )      // nodata cell is adjacent in y -- assume z[nodata]==0.0 (ocean!)
               {
                  *DataboxCoeff(ds,i,j,0) = dy;
               }
               else if ( j==jmin ) // nodata cell is adjacent in x -- assume z[nodata]==0.0 (ocean!)
               {
                  *DataboxCoeff(ds,i,j,0) = dx;
               }
               else                // nodata cell is diagonal -- assume z[nodata]==0 (ocean!)
               {
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }

            } // end if nodata==1

            // if nodata>1...
            // cell has multiple nodata neighbors...e.g. straight coastline, peninsula
            // requires testing special cases...
            // NOTE: the setup below only works because direction doesn't matter
            //       ...only need max down grad to determine segment length!
            else if ( nodata>1 )
            {
               s1     = -9999.0;
               s2     = -9999.0;
               s3     = -9999.0;
               // if adjacent left/right is nodata, compute slope over dy
               if ( (*DataboxCoeff(dem,i-1,j,0)==-9999.0) || (*DataboxCoeff(dem,i+1,j,0)==-9999.0) )
               {
                  s1  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dx;
               }
               if ( (*DataboxCoeff(dem,i,j-1,0)==-9999.0) || (*DataboxCoeff(dem,i,j+1,0)==-9999.0) )
               {
                  s2  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dy;
               }
               if ( (*DataboxCoeff(dem,i-1,j-1,0)==-9999.0) ||
                    (*DataboxCoeff(dem,i-1,j+1,0)==-9999.0) ||
                    (*DataboxCoeff(dem,i+1,j-1,0)==-9999.0) ||
                    (*DataboxCoeff(dem,i+1,j+1,0)==-9999.0) )
               {
                  s3  = fabs(*DataboxCoeff(dem,i,j,0)-0.0) / dxy;
               }

               smax   = fmax( s1, fmax(s1,s3) );
               if ( smax==s1 )
               { 
                  *DataboxCoeff(ds,i,j,0) = dx;
               }
               else if ( smax==s2 )
               {
                  *DataboxCoeff(ds,i,j,0) = dy;
               }
               else
               { 
                  *DataboxCoeff(ds,i,j,0) = dxy;
               }

            } // end if nodata>1

         } // end if nodata cell

      }  // end loop over i

   } // end loop over j

}


/*-----------------------------------------------------------------------
 * ComputeChildD8:
 *
 * Compute elevation of the downstream child of each cell using the D8 method.
 * The value returned for child[i,j] is the elevation of the cell to which [i,j] 
 * drains.
 *
 * If cell is a local minimum --> child = -9999.0
 *
 *-----------------------------------------------------------------------*/
void ComputeChildD8(
   Databox *dem,
   Databox *child)
{

   int             i,  j,  ii, jj;
   int             imin,   jmin;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;
   double          zmin;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);

   // Loop over all [i,j]
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // Loop over neighbors (adjacent and diagonal)
         // ** Find elevation and indices of lowest neighbor
         // ** Exclude off-grid cells
         // ** Exclude nodata cells (treat like offgrid cells @ edges)

         // NOTE:
         // All cells neighboring nodata cells are effectively local minima
         // because these cells will ALWAYS drain off grid to the nodata area.
         // This is because nodata cells are assumed to be ocean, and neighboring 
         // cells are assumed to drain to the ocean. Ignoring nodata cells is 
         // OK here, though, because we only use the D8 child elevations to determine
         // local minima from which to start our Flint's Law recursion...
         // Parent-child relationships are computed by ComputeTestParentD8...

         imin = -9999;
         jmin = -9999;
         zmin = 100000000000.0;
         for (jj = j-1; jj <= j+1; jj++)
         {
            for (ii = i-1; ii <= i+1; ii++)
            {

               // skip if off grid
               if ((ii<0) || (jj<0) || (ii>nx-1) || (jj>ny-1))
               {
                  ;
               }

               // find lowest neighbor
               else
               {
                   if ( (*DataboxCoeff(dem,ii,jj,0) < zmin) )
                   {
                      zmin = *DataboxCoeff(dem,ii,jj,0);
                      imin = ii;
                      jmin = jj;
                   }
               }
            }
         }

         // Determine elevation lowest neighbor -- lowest neighbor is D8 child!!
         // ** If cell is a local minimum (edge or otherwise), set value to -9999.0 (local min)
         if ( zmin==*DataboxCoeff(dem,i,j,0) )
         {
            *DataboxCoeff(child,i,j,0) = -9999.0;
         }

         // Else, set child elevation as lowest neighbor...
         // (if lowest neighbor is nodata, assumes nodata is child...
         //  ... gives child=-9999.0
         //  ... results in local min...
         //  ... works b/c nodata are assumed to be ocean, coasts assumed to drain to ocean)
         else
         {
            *DataboxCoeff(child,i,j,0) = *DataboxCoeff(dem,imin,jmin,0);
         }

      }  // end loop over i

   } // end loop over j

}


/*-----------------------------------------------------------------------
 * ComputeTestParentD8:
 *
 * Returns 1 if ii,jj is the unique D8 parent of i,j (based on elevations).
 * Returns 0 otherwise.
 * 
 * If [i,j] and [ii,jj] are not neighbors, returns -999
 *
 *-----------------------------------------------------------------------*/
int ComputeTestParentD8(
   int i,
   int j,
   int ii,
   int jj,
   Databox *dem)
{

   int           test = -999;
   int           itest, jtest;
   int           imin,   jmin;
   int           nx, ny;
   double        zmin;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   itest = 0;
   jtest = 0;

   // skip nodata cells
   // (nodata cell can't be child because it has no elevation, ignored by code)
   // (nodata cell can't be parent -- everything drains TO nodata/ocean cells, not vice versa)
   if ( (*DataboxCoeff(dem,i,j,0)==-9999.0) || (*DataboxCoeff(dem,ii,jj,0)==-9999.0) )
   { 
      test = 0;
   }

   // not neighbors
   else if ( (fabs(i-ii)>1.0) || (fabs(j-jj)>1.0) )
   {
      test = 0; 
   }

   // neighbors
   else
   {

      // find D8 child of [ii,jj]
      // ** loop over neighbors of [ii,jj] (adjacent and diagonal)
      // ** find elevation and indices of lowest neighbor (including self)
      // ** exclude off-grid cells
      imin = -9999;
      jmin = -9999;
      zmin = 100000000000.0;
      for (jtest = jj-1; jtest <= jj+1; jtest++)
      {
         for (itest = ii-1; itest <= ii+1; itest++)
         {

            // skip if off grid...
            if ((itest<0) || (jtest<0) || (itest>nx-1) || (jtest>ny-1) )
            {
               ;
            }
            
            // skip self...
            else if ( (itest==ii) && (jtest==jj) )
            {
               ;
            }

            // otherwise, find lowest neighbor
            else
            {
               if ( (*DataboxCoeff(dem,itest,jtest,0) < zmin) )
               {
                  zmin = *DataboxCoeff(dem,itest,jtest,0);
                  imin = itest;
                  jmin = jtest;
               }
            }
         } // end loop over itest
      } // end loop over jtest

      // Determine if [ii,jj] is parent of [i,j]
      // ** if zmin==-9999.0, cell drains to a nodata cell (i.e., to ocean)...
      // ** nodata cells are ignored, can't be child...
      // ** fails test
      if ( zmin==-9999.0 )
      {
         test = 0;
      }

      // ** if [imin,jmin] == [i,j] ... then [i,j] is lowest neighbor of [ii,jj]
      // ** if [i,j] is also at lower elevation than [ii,jj], then [ii,jj] is D8 parent of [i,j]
      //    (child must be lower than parent, but need not be unique elevation -- only need lowest elevation!)
      else if ( (imin==i) && (jmin==j)                                         // [imin,jmin] == [i,j] of potential child
            && (*DataboxCoeff(dem,i,j,0)<*DataboxCoeff(dem,ii,jj,0)) )         // child elevation lower than parent 
      { 
         test = 1;
      } 

      // if [i,j] isn't lowest neighbor of [ii,jj], test fails.
      else
      { 
         test = 0;
      }
 
   } // end if/else 

   return test;
 
}     


/*-----------------------------------------------------------------------
 * ComputeFlintsLawRec:
 * 
 * Recursively computes Flint's law up a given drainage network
 *
 * 
 *---------------------------------------------------------------------*/
void ComputeFlintsLawRec(
   int      i, 
   int      j, 
   Databox *dem,
   Databox *demflint,
   Databox *child,
   Databox *area,
   Databox *ds,
   double   c,
   double   p)
{

   int      ii, jj;
   int      nx, ny;
   int      test = -999;

   nx       = DataboxNx(demflint);
   ny       = DataboxNy(demflint);

   // if [i,j] is nodata cell --> skip
   if ( *DataboxCoeff(dem,i,j,0)==-9999.0 )
   { 
      *DataboxCoeff(demflint,i,j,0) = -9999.0;
      test = 0;
   }

   // if i or j is off grid --> skip 
   else if ( (i<0) || (i>=nx) || (j<0) || (j>=ny) )
   {
      test = 0; 
   }

   // else --> loop over neighbors of [i,j]
   else 
   {

      for (jj = j-1; jj <= j+1; jj++ )
      {
         for (ii = i-1; ii <= i+1; ii++ )
         { 

            // skip nodata neighbors...
            if ( *DataboxCoeff(dem,ii,jj,0)==-9999.0 )
            {
               test = 0;
            }

            // skip off-grid neighbors...
            else if ( (ii<0) || (jj<0) || (ii>nx-1) || (jj>ny-1) )
            { 
               test = 0;
            }

            // skip self...
            else if ( (ii==i) && (jj==j) )
            {
               test = 0;
            }

            // if on grid, not nodata, and not self, test if [ii,jj] is D8 parent of [i,j]
            else 
            {  
               test  = ComputeTestParentD8(i,j,ii,jj,dem);
            }

            // if [ii,jj] is a parent 
            // ...and Flint's Law DEM not already computed for [ii,jj] -- i.e., demflint(ii,jj)==-1111. (avoid infinite loop)
            // ...then compute DEM and move upstream (RECURSIVE!)
            if ((test==1) && (*DataboxCoeff(demflint,ii,jj,0)==-1111.0))
            { 

               // compute DEM...   
               *DataboxCoeff(demflint,ii,jj,0) = *DataboxCoeff(demflint,i,j,0) + 
                                                  c * pow(*DataboxCoeff(area,ii,jj,0),p) * *DataboxCoeff(ds,ii,jj,0);

               // recursive loop...
               ComputeFlintsLawRec(ii,jj,dem,demflint,child,area,ds,c,p);

            } // end recursive if statement
 
         } // end loop over ii

      } // end loop over jj

   } // end if re: off grid

}


/*-----------------------------------------------------------------------
 * ComputeFlintsLaw:
 *
 * Compute elevations at all [i,j] using Flint's Law:
 *
 * Flint's law gives slope as a function of upstream area:
 *     S'[i,j] = c*(A[i,j]**p)
 *
 * Using the definition of slope as S = dz/ds = fabs(z[i,j]-z[ii,jj])/ds, where [ii,jj]
 * is the D8 child of [i,j], the elevation at [i,j] is given by:
 *     z[i,j]  = z[ii,jj] + S[i,j]*ds[i,j]
 *
 * We can then estimate the elevation z[i,j] using Flints Law:
 *     z'[i,j] = z[ii,jj] + c*(A[i,j]**p)*ds[i,j]
 *
 * For cells without D8 child (local minima or drains off grid),
 * value is set to value of original DEM.
 *
 * NOTE: This routine loops over all cells to find local minima, then 
 *       recursively loops upstream from child to parent, calculating 
 *       each successive parent's elevation based on the child elevation
 *       and Flint's law. Every drainage path therefore satisfies Flint's
 *       law perfectly. 
 *
 *-----------------------------------------------------------------------*/
void ComputeFlintsLaw(
   Databox *dem,
   double   c,
   double   p,
   Databox *demflint)
{

   int             i,  j;
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;

   Databox        *sx;
   Databox        *sy;
   Databox        *area;
   Databox        *ds;
   Databox        *child;

   nx    = DataboxNx(dem);
   ny    = DataboxNy(dem);
   nz    = DataboxNz(dem);
   x     = DataboxX(dem);
   y     = DataboxY(dem);
   z     = DataboxZ(dem);
   dx    = DataboxDx(dem);
   dy    = DataboxDy(dem);
   dz    = DataboxDz(dem);

   // compute upwind slopes, upstream area
   sx    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   sy    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   area  = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);
   ComputeUpstreamArea(dem,sx,sy,area);
   
   // convert areas from number of cells to units squared
   for (j = 0; j < ny; j++ ) {
    for (i = 0; i < nx; i++ ) {
     *DataboxCoeff(area,i,j,0) = *DataboxCoeff(area,i,j,0) * dx * dy;
    }
   }

   // compute segment lengths and child elevations for D8 grid
   ds    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   child = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSegmentD8(dem,ds);
   ComputeChildD8(dem,child);

   // initialize all cells of computed DEM to -1111.0
   for (j = 0; j < ny; j++) { 
    for (i = 0; i < nx; i++) { 
     *DataboxCoeff(demflint,i,j,0) = -1111.0;
    }
   }

   // compute elevations using Flint's law
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // If original DEM elevation is -9999.0 
         // -- nodata cell (assumed to be ocean/estuary cell)
         // -- set demflint to -9999.0
         // -- continue loop
         if (*DataboxCoeff(dem,i,j,0) == -9999.0 )
         {
            *DataboxCoeff(demflint,i,j,0) = -9999.0;
         }

         // If child elevation is set to -9999.0...
         // -- cell is a local minimum (no child)
         // -- set demflint elevation to original DEM value
         // -- then loop upstream (recursively) to calculate parent elevations
         else if (*DataboxCoeff(child,i,j,0) == -9999.0)
         {

            // set value at [i,j]
            *DataboxCoeff(demflint,i,j,0) = *DataboxCoeff(dem,i,j,0);

            // call recursive function
            ComputeFlintsLawRec(i,j,dem,demflint,child,area,ds,c,p);

         }

      } // end loop over i

   } // end loop over j

   FreeDatabox(sx);
   FreeDatabox(sy);
   FreeDatabox(area);
   FreeDatabox(ds);
   FreeDatabox(child);

}


/*-----------------------------------------------------------------------
 * ComputeFlintsLawFit:
 *
 * Compute parameters of Flint's Law (c and p) based on DEM and area
 * using a least squares fit. Residuals are calculated at each cell with 
 * respect to initial DEM values as:
 *
 *   r[i,j]  =  (z[i,j] - z'[i,j])
 *
 * where z'[i,j] is the elevation estimated via Flint's law:
 * 
 *   z'[i,j] =  z_child[i,j] + c*(A[i,j]**p)*ds[i,j]
 *
 * where z_child is the elevation of the CHILD of [i,j] and ds is the segment
 * length of the slope between [i,j] and it's child. 
 *
 * Least squares minimization is carried out by the Levenberg-Marquardt 
 * method as detailed in Numerical Recipes in C, similar to used in MINPACK 
 * function lmdif. 
 * 
 * NOTES: 
 * - All parent/child relationships, slopes, and segment lengths are 
 *   calculated based on a simple D8 scheme. 
 * - Areas are calculated based on two-direction model used in ParFlow
 *   (i.e., no unique paren-child...any cell that drains into [i,j] is part of area)
 * - Areas must be converted from number of cells to m2 before fitting
 *   (area = area*dx*dy)
 *
 *-----------------------------------------------------------------------*/
void ComputeFlintsLawFit(
   Databox *dem,
   double   c0,
   double   p0,
   int      maxiter,    
   Databox *demflint) 
{

   // grid vars
   int             i,  j,  n1, n2;
   int             nx, ny, nz, iter;
   double          x,  y,  z;
   double          dx, dy, dz;

   // calculated vars
   Databox        *sx;
   Databox        *sy;
   Databox        *area;
   Databox        *ds;
   Databox        *child;

   // get grid info
   nx       = DataboxNx(dem);
   ny       = DataboxNy(dem);
   nz       = DataboxNz(dem);
   x        = DataboxX(dem);
   y        = DataboxY(dem);
   z        = DataboxZ(dem);
   dx       = DataboxDx(dem);
   dy       = DataboxDy(dem);
   dz       = DataboxDz(dem);

   // compute upwind slopes, upstream area
   // NOTE: these values don't change during iteration -- c,p adjusted to fit these values!
   sx       = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   sy       = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   area     = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);
   ComputeUpstreamArea(dem,sx,sy,area);

   // convert areas from number of cells to m^2
   for (j=0; j<ny; j++)
   {
      for (i=0; i<nx; i++)
      {
         *DataboxCoeff(area,i,j,0) = *DataboxCoeff(area,i,j,0) * dx * dy; 
      }
   }

   // compute segment lengths and child elevations for D8 grid
   // NOTE: these values don't change during iteration -- c,p adjusted to fit these values!
   ds       = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   child    = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSegmentD8(dem,ds);
   ComputeChildD8(dem,child);

   // INITIALIZE L-M VARS
   const int  ma = 2;
   double c      = c0;
   double ctry   = c0; 
   double p      = p0;
   double ptry   = p0; 
   double beta[ma];
   double da[ma];
   double alpha[ma][ma];
   double covar[ma][ma];
   double oneda[ma][1];
   double chisq  = 1000.0;
   double ochisq = 0.1;
   double dchisq = 100.0 * (chisq-ochisq)/(ochisq);
   double alamda = 0.00001;

   // initialize coefficient arrays
   for (n1=0; n1<ma; n1++)
   { 
      beta[n1]     = 0.0;
      da[n1]       = 0.0;
      oneda[n1][1] = 0.0;
      for (n2=0; n2<ma; n2++)
      { 
         alpha[n1][n2] = 0.0;
         covar[n1][n2] = 0.0;
      }
   }

   // initialize all cells of computed DEM (demflint) to -1111.0
   for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
         *DataboxCoeff(demflint,i,j,0) = -1111.0;
      }
   }

   // calculate chisq at initial parameter values
   chisq  = ComputeLMCoeff(demflint,dem,area,child,ds,c0,p0,alpha,beta,chisq);
   ochisq = chisq;

   // L-M ITERATION
   // iterates until maxiter or convergence, 
   // where convergence is considered a percent change in chisq less than .1%
   iter          = 0; 
   while ( (iter<maxiter) && (dchisq>0.00001) )
   { 

      // copy fitting matrix, alter by augmenting diagonals
      for (n1=0; n1<ma; (n1)++)
      { 

         // copy all values
         oneda[n1][0]     = beta[n1];
         for (n2=0; n2<ma; (n2)++)
         { 
            covar[n1][n2] = alpha[n1][n2];
         }

         // augment diagonals
         covar[n1][n1]    = alpha[n1][n1] * (1.0+alamda);

      }

      // matrix solution (Gauss-Jordan elimination)
      ComputeGaussJordan(covar,ma,oneda,1);

      // copy oneda -> da
      for (j=0; j<ma; j++) { da[j] = oneda[j][0] ; }

      // adjust parameters for next test
      ctry       = c + da[0];
      ptry       = p + da[1];

      // reset all cells of computed DEM (demflint) to -1111.0
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            *DataboxCoeff(demflint,i,j,0) = -1111.0;
         }
      }

      // Call ComputeLMCoeff
      chisq      = ComputeLMCoeff(demflint,dem,area,child,ds,ctry,ptry,covar,da,chisq);

      // compute convergence criteria
      dchisq     = 100.0 * fabs(chisq-ochisq)/(ochisq);

      // update alamda...
      // if new chisq is smaller than old, moving in right direction...
      if (chisq < ochisq)
      {
         // printf( "ITERATION SUCCESS:\t %d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n", 
         //           iter, ctry, ptry, alamda, ochisq, chisq, dchisq, da[0], da[1] );
         alamda  = alamda * 0.1;               // cut parameter shift
         ochisq  = chisq;                      // reset ochisq
         c       = ctry;                       // set parameter to latest iteration value
         p       = ptry;                       // set parameter to latest iteration value
         for (n1=0; n1<ma; n1++)               // reset alpha and beta values
         { 
            beta[n1] = da[n1];
            for (n2=0; n2<ma; n2++)
            { 
               alpha[n1][n2] = covar[n1][n2];
            }
         }
         
      } 
      
      // else new chisq is larger than old, moving in wrong direction...
      else
      {
         // printf( "ITERATION FAIL:   \t %d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n", 
         //         iter, ctry, ptry, alamda, ochisq, chisq, dchisq, da[0], da[1] );
         alamda  = alamda * 10.0;
         chisq   = ochisq; 
      }

      // iteration and convergence criteria
      iter       = iter + 1;

   } // end of while loop

   // compute elevations using Flint's law
   // with final parameter values...
   // if no iterations succeeded, final values are reset to initial values.
   // -- start by resetting demflint to -1111.0
   for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
         *DataboxCoeff(demflint,i,j,0) = -1111.0;
      }
   }
   // -- then call recursive loop
   for (j = 0; j < ny; j++)
   {
      for (i = 0; i < nx; i++)
      {

         // If original DEM elevation is -9999.0
         // -- nodata cell (assumed to be ocean/estuary cell)
         // -- set demflint to -9999.0
         // -- continue loop
         if (*DataboxCoeff(dem,i,j,0) == -9999.0 )
         {
            *DataboxCoeff(demflint,i,j,0) = -9999.0;
         }

         // If child elevation is set to -9999.0...
         // -- cell is a local minimum (no child)
         // -- set demflint elevation to original DEM value
         // -- then loop upstream (recursively) to calculate parent elevations
         else if (*DataboxCoeff(child,i,j,0) == -9999.0)
         {

            // set value at [i,j]
            *DataboxCoeff(demflint,i,j,0) = *DataboxCoeff(dem,i,j,0);

            // call recursive function
            ComputeFlintsLawRec(i,j,dem,demflint,child,area,ds,ctry,ptry);

         }

      } // end loop over i

   } // end loop over j

   printf( "-------------------------------------------------------------\n" );
   printf( "Flints Law Fit: \n" );
   printf( "iterations: %d \n", iter );
   printf( "[c0, p0] = [%f, %f] \n", c0, p0 );
   printf( "[ c,  p] = [%f, %f] \n", ctry, ptry );

   FreeDatabox(sx);
   FreeDatabox(sy);
   FreeDatabox(area);
   FreeDatabox(ds);
   FreeDatabox(child);

}

// Additional subroutines for ComputeFlintsLawFit
// ComputeFlintLM: Computes slopes and derivatives needed by LM fitting routine
void ComputeFlintLM(
   Databox *dem,             // original dem values
   Databox *demflint,        // estimated dem values
   Databox *area,            // upstream areas  -- independent var
   Databox *child,           // child dem value -- constant
   Databox *ds,              // segment length  -- constant
   double   c,               // c parameter value            (not changed in this routine)
   double   p,               // p (exponent) parameter value (not changed in this routine)
   Databox *dzdc,            // derivative of Flint's law wrt. c
   Databox *dzdp)            // derivative of Flint's law wrt. p     
{

   int      i,  j;
   int      nx, ny;
   
   nx       = DataboxNx(dem);
   ny       = DataboxNy(dem);

   // Calculate Flints Law DEM using current parameter estimates
   for (j=0; j<ny; j++)
   { 
      for (i=0; i<nx; i++)
      { 

         // If original DEM elevation is -9999.0
         // -- nodata cell (assumed to be ocean/estuary cell)
         // -- set demflint to -9999.0
         // -- continue loop
         //    (THI IS THE SAME AS IN ComputeFlintsLaw)
         if (*DataboxCoeff(dem,i,j,0) == -9999.0 )
         {
            *DataboxCoeff(demflint,i,j,0) = -9999.0;
         }

         // If child elevation is set to -9999.0...
         // -- cell is a local minimum (no child)
         // -- set demflint elevation to original DEM value
         // -- then loop upstream (recursively) to calculate parent elevations
         //    (THIS IS THE SAME AS IN ComputeFlintsLaw)
         else if (*DataboxCoeff(child,i,j,0) == -9999.0)
         {

            // set value at [i,j]
            *DataboxCoeff(demflint,i,j,0) = *DataboxCoeff(dem,i,j,0);

            // call recursive function
            ComputeFlintsLawRec(i,j,dem,demflint,child,area,ds,c,p);

         } 

      } // end loop over i
   
   } // end loop over j
   
   // Loop again to calculate derivatives
   for (j=0; j<ny; j++)
   {
      for (i=0; i<nx; i++)
      {

         // skip nodata cells...
         if (*DataboxCoeff(dem,i,j,0) == -9999.0)
         {  
            *DataboxCoeff(dzdc,i,j,0) = 0.0;
            *DataboxCoeff(dzdc,i,j,0) = 0.0;
         }

         // else, compute derivative WRT c, p
         else 
         {
            *DataboxCoeff(dzdc,i,j,0)     = pow(*DataboxCoeff(area,i,j,0), p) * (*DataboxCoeff(ds,i,j,0));

            *DataboxCoeff(dzdp,i,j,0)     = c * pow(*DataboxCoeff(area,i,j,0), p)
                                              * log(*DataboxCoeff(area,i,j,0))
                                              * (*DataboxCoeff(ds,i,j,0));
         } // end if nodata

      } // end loop over i

   } // end loop over j

} 

double ComputeLMCoeff( 
   Databox *demflint,        // estimated elevations -- dependent var
   Databox *dem,             // actual elevations -- constant
   Databox *area,            // upstream areas    -- independent var
   Databox *child,           // child dem value   -- constant
   Databox *ds,              // segment length    -- constant
   double   c,               // c parameter value
   double   p,               // p parameter value
   double   alpha[][2],      // working space     -- [2x2] array
   double   beta[],          // working space     -- [2] array
   double   chisq)           // chisq value
{ 

   int      i, j; 
   int      ma = 2;
   int      nx, ny, nz;
   double   x,  y,  z;
   double   dx, dy, dz;
   double   df; 
   Databox *dfdc;
   Databox *dfdp;

   // get grid info
   nx     = DataboxNx(dem);
   ny     = DataboxNy(dem);
   nz     = DataboxNz(dem);
   x      = DataboxX(dem);
   y      = DataboxY(dem);
   z      = DataboxZ(dem);
   dx     = DataboxDx(dem);
   dy     = DataboxDy(dem);
   dz     = DataboxDz(dem);

   // initialize derivatives
   dfdc   = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   dfdp   = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);

   // set alpha and beta values to zero
   for (i=0; i<ma; i++)
   { 
      beta[i] = 0.0; 
      for (j=0; j<ma; j++)
      { 
         alpha[i][j] = 0.0;  
      }
   }

   // resetting demflint to -1111.0
   for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
         *DataboxCoeff(demflint,i,j,0) = -1111.0;
      }
   }

   // calculate function values and derivatives at all [i,j] for current parameter values [c,p]
   ComputeFlintLM(dem,demflint,area,child,ds,c,p,dfdc,dfdp);

   // loop to calculate chisq, alpha, beta
   chisq   = 0.0;
   for (j=0; j<ny; j++)
   { 
      for (i=0; i<nx; i++)
      { 

         // skip nodata cells
         // (don't fit to nodata cells -- ignored by all calculations)
         if (*DataboxCoeff(dem,i,j,0) == -9999.0)
         {
            ;
         }
      
         // skip local minima 
         // (don't fit to cells w/o downstream slopes)
         else if (*DataboxCoeff(child,i,j,0) == -9999.0)
         { 
            ;
         }

         // compute coefficients based on all other cells
         else
         { 

            // calculate residual at [i,j]
            df          = *DataboxCoeff(dem,i,j,0) - *DataboxCoeff(demflint,i,j,0);

            // add to chisq sum...
            // NOTE: we assume all local variances are unity, so chisq reduces to the sum of squared residuals
            chisq       = chisq + (df * df);

            // add to alphas
            alpha[0][0] = alpha[0][0] + (*DataboxCoeff(dfdc,i,j,0)) * (*DataboxCoeff(dfdc,i,j,0));
            alpha[0][1] = alpha[0][1] + (*DataboxCoeff(dfdc,i,j,0)) * (*DataboxCoeff(dfdp,i,j,0));
            alpha[1][0] = alpha[1][0] + (*DataboxCoeff(dfdp,i,j,0)) * (*DataboxCoeff(dfdc,i,j,0));
            alpha[1][1] = alpha[1][1] + (*DataboxCoeff(dfdp,i,j,0)) * (*DataboxCoeff(dfdp,i,j,0));

            // add to betas
            beta[0]     = beta[0] + (df)*(*DataboxCoeff(dfdc,i,j,0));
            beta[1]     = beta[1] + (df)*(*DataboxCoeff(dfdp,i,j,0));

         } // end if child

      } // end loop over i

   } // end loop over j

   FreeDatabox(dfdc);
   FreeDatabox(dfdp);
   return chisq;

}

void ComputeGaussJordan( 
   double  a[][2], 
   int     n, 
   double  b[][1], 
   int     m)
{
   
   int     i, icol, irow, j, k, l, ll;
   double  big, dum, pivinv, temp;

   // book keeping arrays
   int     indxc[n];
   int     indxr[n];
   int     ipiv[n];

   irow  = 0;
   icol  = 0;

   // set pivot indices to zero
   for (j=0; j<n; j++) ipiv[j]=0;
   
   // loop over columns to be reduced
   for (i=0; i<n; i++) 
   { 
      big  = 0.0;
      for (j=0; j<n; j++)
      { 
         if (ipiv[j] != 1)
         {
            for (k=0; k<n; k++)
            { 
               if (ipiv[k] == 0)
               {
                  if (fabs(a[j][k]) >= big)
                  {
                     big  = fabs(a[j][k]);
                     irow = j;  
                     icol = k;
                  }
               }
               else if (ipiv[k] > 1)
               {
                  printf( "ComputeGaussJordan -- ERROR: Singular Matrix - 1 \n" );
               }
            }
         }   
      }
      ++(ipiv[icol]);

      // interchange rows if necessary
      if (irow != icol)
      {
         for (l=0; l<n; l++) { temp = a[irow][l] ; a[irow][l] = a[icol][l] ; a[icol][l] = temp ; }
         for (l=0; l<m; l++) { temp = b[irow][l] ; b[irow][l] = b[icol][l] ; b[icol][l] = temp ; }
      }

      // divide pivot row by pivot element
      indxr[i]  = irow;
      indxc[i]  = icol;
      if (a[icol][icol] == 0.0) 
      { 
         printf( "ComputeGaussJordan -- ERROR: Singular Matrix - 2 \n" );
         printf( "                      NUDGING a[%d][%d]: 0.0 -> 0.0001 \n", icol, icol );
         a[icol][icol] = 0.0001;
      }
      pivinv        = 1.0 / a[icol][icol];
      a[icol][icol] = 1.0;
      for (l=0; l<n; l++) a[icol][l] *= pivinv;
      for (l=0; l<m; l++) b[icol][l] *= pivinv;
      for (ll=0; ll<n; ll++)
      { 
         if (ll != icol) 
         {
            dum         = a[ll][icol];
            a[ll][icol] = 0.0;    
            for (l=0; l<n; l++) a[ll][l] -= a[icol][l]*dum;
            for (l=0; l<m; l++) b[ll][l] -= b[icol][l]*dum;
         }
      }
   } // end loop over i (column reduce loop)

   // reverse any column interchanges
   for (l=n-1; l>=0; l--)
   { 
      if (indxr[l] != indxc[l])
      { 
         for (k=0; k<n; k++) { temp = a[k][indxr[l]] ; a[k][indxr[l]] = a[k][indxc[l]] ; a[k][indxc[l]] = temp ; }
      }
   }

}    


/*-----------------------------------------------------------------------
 * ComputeFlintsLawByBasin:
 *
 * Compute parameters of Flint's Law (c and p) based on DEM and area
 * using a least squares fit. Function is identical to ComputeFlintsLawFit
 * EXCEPT that Flint's Law is fit for each drainage network separately, 
 * where individual networks are defined by local minima (within or on edge
 * of domain).
 *
 * Residuals are calculated at each cell with respect to initial DEM values as:
 *
 *   r[i,j]  =  (z[i,j] - z'[i,j])
 *
 * where z'[i,j] is the elevation estimated via Flint's law:
 *
 *   z'[i,j] =  z_child[i,j] + c*(A[i,j]**p)*ds[i,j]
 *
 * where z_child is the elevation of the CHILD of [i,j] and ds is the segment
 * length of the slope between [i,j] and it's child.
 *
 * Least squares minimization is carried out by the Levenberg-Marquardt
 * method as detailed in Numerical Recipes in C, similar to used in MINPACK
 * function lmdif.
 *
 * NOTES:
 * - All parent/child relationships, slopes, and segment lengths are
 *   calculated based on a simple D8 scheme.
 * - Areas are calculated based on two-direction model used in ParFlow
 *   (i.e., no unique paren-child...any cell that drains into [i,j] is part of area)
 * - Areas must be converted from number of cells to m2 before fitting
 *   (area = area*dx*dy)
 *
 *-----------------------------------------------------------------------*/
void ComputeFlintsLawByBasin(
   Databox *dem,
   double   c0,
   double   p0,
   int      maxiter,
   Databox *demflint)
{

   // grid vars
   int             i,  j,  n1, n2;
   int             i_grid, j_grid;
   int             nx, ny, nz, iter;
   double          x,  y,  z;
   double          dx, dy, dz;

   // calculated vars
   Databox        *sx;
   Databox        *sy;
   Databox        *area;
   Databox        *ds;
   Databox        *child;
   Databox        *parentmap;
   Databox        *basin_in;
   Databox        *basin_out;
   
   // get grid info
   nx        = DataboxNx(dem);
   ny        = DataboxNy(dem);
   nz        = DataboxNz(dem);
   x         = DataboxX(dem);
   y         = DataboxY(dem);
   z         = DataboxZ(dem);
   dx        = DataboxDx(dem);
   dy        = DataboxDy(dem);
   dz        = DataboxDz(dem);

   // compute upwind slopes, upstream area
   // NOTE: these values don't change during iteration -- c,p adjusted to fit these values!
   sx        = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   sy        = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   area      = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSlopeXUpwind(dem,dx,sx);
   ComputeSlopeYUpwind(dem,dy,sy);
   ComputeUpstreamArea(dem,sx,sy,area);

   // convert areas from number of cells to m^2
   for (j=0; j<ny; j++)
   {
      for (i=0; i<nx; i++)
      {
         *DataboxCoeff(area,i,j,0) = *DataboxCoeff(area,i,j,0) * dx * dy;
      }
   }

   // compute segment lengths and child elevations for D8 grid
   // NOTE: these values don't change during iteration -- c,p adjusted to fit these values!
   ds        = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   child     = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   ComputeSegmentD8(dem,ds);
   ComputeChildD8(dem,child);

   // initialize vars for computing basin values
   parentmap = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   basin_out = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);
   basin_in  = NewDatabox(nx,ny,nz,x,y,z,dx,dy,dz);

   // find individual basins...
   // (1) loop over all cells
   // (2) if local minimum, use ComputeParentMapD8 to identify contributing basin
   // (3) fit + apply Flint's Law over basin, excluding outside cells
   for (j_grid=0; j_grid<ny; j_grid++)
   { 
      for (i_grid=0; i_grid<nx; i_grid++)
      {
         
         // skip nodata cells
         if (*DataboxCoeff(dem,i_grid,j_grid,0)==-9999.0)
         {
            *DataboxCoeff(demflint,i_grid,j_grid,0) = -9999.0;
         }

         // else, test [i_grid,j_grid] is local minimum
         // (remember that child==-9999.0 at local mins)
         else if (*DataboxCoeff(child,i_grid,j_grid,0)==-9999.0)
         {

            // cell is a local minimum...
            // fit and apply Flint's Law over contributing basin

            // compute parentmap to define basin
            // create basin-only DEM (equal to original DEM inside basin, -9999.0 outside)
            ComputeParentMap(i_grid,j_grid,dem,sx,sy,parentmap);
            for (j=0; j<ny; j++)
            {
               for (i=0; j<nx; j++)
               {

                  *DataboxCoeff(basin_out,i,j,0) = -1111.0;

                  if (*DataboxCoeff(parentmap,i,j,0)==1.0)
                  {
                     *DataboxCoeff(basin_in,i,j,0) = *DataboxCoeff(dem,i,j,0);
                  }

                  else
                  {
                     *DataboxCoeff(basin_in,i,j,0) = -9999.0;
                  }

               }
            }
               
            // run fitting routine for basindem
            // (since routine ignores nodata cells, will fit only to basin area) 
            // -- initialize L-M vars
            const int ma     = 2;
            double c      = c0;
            double ctry   = c0;
            double p      = p0;
            double ptry   = p0;
            double beta[ma];
            double da[ma];
            double alpha[ma][ma];
            double covar[ma][ma];
            double oneda[ma][1];
            double chisq  = 1000.0;
            double ochisq = 0.1;
            double dchisq = 100.0 * (chisq-ochisq)/(ochisq);
            double alamda = 0.00001;

            // -- initialize coefficient arrays
            for (n1=0; n1<ma; n1++)
            {
               beta[n1]     = 0.0;
               da[n1]       = 0.0;
               oneda[n1][1] = 0.0;
               for (n2=0; n2<ma; n2++)
               {
                  alpha[n1][n2] = 0.0;
                  covar[n1][n2] = 0.0;
               }
            }

            // -- calculate chisq at initial parameter values
            chisq  = ComputeLMCoeff(basin_out,basin_in,area,child,ds,c0,p0,alpha,beta,chisq);
            ochisq = chisq;

            // iterates until maxiter or convergence,
            // where convergence is considered a percent change in chisq less than .1%
            iter          = 0;
            while ( (iter<maxiter) && (dchisq>0.00001) )
            {

               // copy fitting matrix, alter by augmenting diagonals
               for (n1=0; n1<ma; (n1)++)
               {

                  // copy all values
                  oneda[n1][0]     = beta[n1];
                  for (n2=0; n2<ma; (n2)++)
                  {
                     covar[n1][n2] = alpha[n1][n2];
                  }

                  // augment diagonals
                  covar[n1][n1]    = alpha[n1][n1] * (1.0+alamda);

               }

               // matrix solution (Gauss-Jordan elimination)
               ComputeGaussJordan(covar,ma,oneda,1);
 
               // copy oneda -> da
               for (j=0; j<ma; j++) { da[j] = oneda[j][0] ; }
   
               // adjust parameters for next test
               ctry       = c + da[0];
               ptry       = p + da[1];

               // reset all cells of computed DEM (basin_out) to -1111.0
               for (j = 0; j < ny; j++) 
               {
                  for (i = 0; i < nx; i++) 
                  {
                     *DataboxCoeff(basin_out,i,j,0) = -1111.0;
                  }
               }

               // Call ComputeLMCoeff
               chisq      = ComputeLMCoeff(basin_out,basin_in,area,child,ds,ctry,ptry,covar,da,chisq);

               // compute convergence criteria
               dchisq     = 100.0 * fabs(chisq-ochisq)/(ochisq);

               // update alamda...
               // if new chisq is smaller than old, moving in right direction...
               if (chisq < ochisq)
               {
                  // printf( "ITERATION SUCCESS:\t %d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n",
                  //           iter, ctry, ptry, alamda, ochisq, chisq, dchisq, da[0], da[1] );
                  alamda  = alamda * 0.1;               // cut parameter shift
                  ochisq  = chisq;                      // reset ochisq
                  c       = ctry;                       // set parameter to latest iteration value
                  p       = ptry;                       // set parameter to latest iteration value
                  for (n1=0; n1<ma; n1++)               // reset alpha and beta values
                  {
                     beta[n1] = da[n1];
                     for (n2=0; n2<ma; n2++)
                     {
                        alpha[n1][n2] = covar[n1][n2];
                     }
                  }
               }

               // else new chisq is larger than old, moving in wrong direction...
               else
               {
                  // printf( "ITERATION FAIL:   \t %d \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \n",
                  //         iter, ctry, ptry, alamda, ochisq, chisq, dchisq, da[0], da[1] );
                  alamda  = alamda * 10.0;
                  chisq   = ochisq;
               }

               // iteration and convergence criteria
               iter       = iter + 1;

            } // end of while loop

            // compute elevations using Flint's law with final parameter values...
            // if no iterations succeeded, final values are set to initial values.
            // -- start by resetting basin_out to -1111.0
            for (j = 0; j < ny; j++) 
            {
               for (i = 0; i < nx; i++) 
               {
                  *DataboxCoeff(basin_out,i,j,0) = -1111.0;
               }
            }

            // -- then call recursive loop
            for (j = 0; j < ny; j++)
            {
               for (i = 0; i < nx; i++)
               {

                  // If original DEM elevation is -9999.0
                  // -- nodata cell (assumed to be ocean/estuary cell)
                  // -- set demflint to -9999.0
                  // -- continue loop
                  if (*DataboxCoeff(basin_in,i,j,0) == -9999.0 )
                  {
                     *DataboxCoeff(basin_out,i,j,0) = -9999.0;
                  }

                  // If child elevation is set to -9999.0...
                  // -- cell is a local minimum (no child)
                  // -- set demflint elevation to original DEM value
                  // -- then loop upstream (recursively) to calculate parent elevations
                  else if (*DataboxCoeff(child,i,j,0) == -9999.0)
                  {

                     // set value at [i,j]
                     *DataboxCoeff(basin_out,i,j,0) = *DataboxCoeff(basin_in,i,j,0);

                     // call recursive function
                     ComputeFlintsLawRec(i,j,basin_in,basin_out,child,area,ds,ctry,ptry);

                  }

               } // end loop over i
  
            } // end loop over j

            // copy fitted values from basin_out to demflint
            for (j=0; j<ny; j++)
            {
               for (i=0; i<nx; i++)
               {
                  if (*DataboxCoeff(parentmap,i,j,0)==1.0)
                  {
                     *DataboxCoeff(demflint,i,j,0) = *DataboxCoeff(basin_out,i,j,0);
                  } 
               }
            }

            printf( "-------------------------------------------------------------\n" );
            printf( "BASIN OUTLET / MINIMUM = [%d,%d] \n", i_grid, j_grid );
            printf( "iterations: %d \n", iter );
            printf( "c = %f \n", ctry );
            printf( "p = %f \n", ptry );
            printf( "-------------------------------------------------------------\n" );
            printf( " \n \n \n " );

         } // end if/else child==-9999.0

      } // end loop over i_grid

   } // end loop over j_grid

   FreeDatabox(sx);
   FreeDatabox(sy);
   FreeDatabox(area);
   FreeDatabox(ds);
   FreeDatabox(child);
   FreeDatabox(parentmap);
   FreeDatabox(basin_in);
   FreeDatabox(basin_out);

}


