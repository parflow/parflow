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

/* 
 * Setup array for storing the top of the domain.
 *
 * Computes and array that is NX * NY that contains the
 * k-index into the supplied vector that is at the top 
 * of the geometry.
 *
 * Only works with 1 subgrid per task.
 *
 * This assumes number of processors is 1 in Z; assumes
 * that the entire Z column is on a single task.
 * 
 */

#include "parflow.h"

void ComputeTop(Problem     *problem,      /* General problem information */
		ProblemData *problem_data  /* Contains geometry information for the problem */
   ) 
{
   GrGeomSolid   *gr_solid = ProblemDataGrDomain(problem_data);
   Vector        *top      = ProblemDataIndexOfDomainTop(problem_data);
   Vector        *perm_x   = ProblemDataPermeabilityX(problem_data);

   /* use perm grid as top is 2D and want to loop over Z */
   Grid          *grid     = VectorGrid(perm_x);

   SubgridArray  *subgrids = GridSubgrids(grid);

   int      ix, iy, iz;
   int      nx, ny, nz;
   int      r;
   
   int      is, i, j, k;

   double *top_data;
   int index;

   (void) problem;

   PFVConstInit(-1, top);
      
   ForSubgridI(is, subgrids)
   {
      Subgrid       *subgrid          = SubgridArraySubgrid(subgrids, is);

      Subvector     *top_subvector    = VectorSubvector(top, is);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
      
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
      
      r = SubgridRX(subgrid);

      top_data = SubvectorData(top_subvector);
      
      GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
      {
	 index = SubvectorEltIndex(top_subvector, i, j, 0);
	 
	 if( top_data[index] < k ) {
	    top_data[index] = k;
	 }
      });
   }     /* End of subgrid loop */
}
