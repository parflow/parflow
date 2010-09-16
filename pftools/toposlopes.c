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
        
         // Deal with corners
         // SW corner [0,0]
         if ((i==0) && (j==0))
         { 
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
         }
         // SE corner [nx-1,0]
         else if ((i==nx-1) && (j==0))
         {
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
         }
         // NE corner [nx-1,ny-1]
         else if ((i==nx-1) && (j==ny-1))
         {
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
         }
         // NW corner [0,ny-1]
         else if ((i==0) && (j==ny-1))
         {
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
         }

         // Eastern edge, not corner [nx-1,1:ny-1]
         else if (i==nx-1)
         {
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
         }

         // Western edge, not corner [0,1:ny-1]
         else if (i==0)
         {
            *DataboxCoeff(sx,i,j,0) = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
         }

         // All other cells...
         else 
         {
            s1        = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i-1,j,0))/dx;
            s2        = (*DataboxCoeff(dem,i+1,j,0)-*DataboxCoeff(dem,i,j,0))/dx;
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

         // Deal with corners
         // SW corner [0,0]
         if ((i==0) && (j==0))
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
         }
         // SE corner [nx-1,0]
         else if ((i==nx-1) && (j==0))
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
         }
         // NE corner [nx-1,ny-1]
         else if ((i==nx-1) && (j==ny-1))
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
         }
         // NW corner [0,ny-1]
         else if ((i==0) && (j==ny-1))
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
         }

         // Southern edge, not corner [1:nx-1,0]
         else if (j==0)
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
         }

         // Northern edge, not corner [1:nx,ny-1]
         else if (j==ny-1)
         {
            *DataboxCoeff(sy,i,j,0) = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
         }

         // All other cells...
         else
         {
            s1        = (*DataboxCoeff(dem,i,j,0)-*DataboxCoeff(dem,i,j-1,0))/dy;
            s2        = (*DataboxCoeff(dem,i,j+1,0)-*DataboxCoeff(dem,i,j,0))/dy;
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
                 Databox *sx, 
                 Databox *sy)
{ 
   int test = -999;

   // Make sure [i,j] and [ii,jj] are adjacent
   if ( (fabs(i-ii)+fabs(j-jj)) == 1.0 )
   {
      if ( (ii==i-1) && (jj==j) && (*DataboxCoeff(sx,ii,jj,0)<0.) )
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
                       Databox *sx, 
                       Databox *sy,
                       Databox *parentmap)
{

   int      ii, jj;
   int      parent;

   int nx  = DataboxNx(sx);
   int ny  = DataboxNy(sx);

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

         // otherwise, test if [ii,jj] is parent of [i,j]...
         else 
         {
      
            parent  = ComputeTestParent( i,j,ii,jj,sx,sy );
             
            // if parent, loop recursively
            if (parent==1) 
            { 
               ComputeParentMap(ii,jj,sx,sy,parentmap);
            }
         }

      } // end loop over i

   } // end loop over j

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
void ComputeUpstreamArea( Databox *sx, Databox *sy, Databox *area )
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
         ComputeParentMap(i,j,sx,sy,parentmap);

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

   // Find Sinks + Pit Fill
   for (j = 0; j < ny; j++)
   { 
      for (i=0; i < nx; i++)
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

      } // end loop over i
   } // end loop over j

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
   
            *DataboxCoeff(dem,i,j,0) = mavg / counter; 

         } // end if smag==0 or lmin==1

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

      } // end loop over i
   } // end loop over j

   return nsink;

}



