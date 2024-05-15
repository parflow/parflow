/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

/*
 * Setup array for storing the patch indices for the top of the domain.
 *
 * Computes and array that is NX * NY that contains the
 * patch index for each cell that is on the surface/top of the domain.
 *
 * Only works with 1 subgrid per task.
 *
 * This method currently assumes number of processors is 1 in Z;
 * assumes that the entire Z column is on a single task.
 */

#include "parflow.h"

void ComputePatchTop(Problem *    problem,      /* General problem information */
		     ProblemData *problem_data  /* Contains geometry information for the problem */
		     )
{

  Vector        *index_top = ProblemDataIndexOfDomainTop(problem_data);
  Vector        *patch_top = ProblemDataPatchIndexOfDomainTop(problem_data);
  Vector        *perm_x = ProblemDataPermeabilityX(problem_data);

  Grid          *grid2d = VectorGrid(patch_top);
  SubgridArray  *grid2d_subgrids = GridSubgrids(grid2d);

  /* use perm grid as top is 2D and want to loop over Z */
  Grid          *grid3d = VectorGrid(perm_x);
  SubgridArray  *grid3d_subgrids = GridSubgrids(grid3d);

  GrGeomSolid    *gr_domain = ProblemDataGrDomain(problem_data);

  double *patch_top_data;
  double *index_top_data;

  int ipatch, ival=0;
  PF_UNUSED(ival);

  VectorUpdateCommHandle   *handle;

  (void)problem;

  InitVectorAll(patch_top, -1);

  BCStruct    *bc_struct;

  BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);
  int num_patches = BCPressureDataNumPatches(bc_pressure_data);

  bc_struct = NewBCStruct(GridSubgrids(grid3d),
                          gr_domain,
                          num_patches,
                          BCPressureDataPatchIndexes(bc_pressure_data),
                          BCPressureDataBCTypes(bc_pressure_data),
                          NULL);
  if (num_patches > 0)
  {
    int is;
    ForSubgridI(is, grid3d_subgrids)
    {
      Subgrid       *grid2d_subgrid = SubgridArraySubgrid(grid2d_subgrids, is);
      
      Subvector     *patch_top_subvector = VectorSubvector(patch_top, is);
      Subvector     *index_top_subvector = VectorSubvector(index_top, is);
      
      int grid2d_iz = SubgridIZ(grid2d_subgrid);
      
      patch_top_data = SubvectorData(patch_top_subvector);
      index_top_data = SubvectorData(index_top_subvector);
      
      int i, j, k;
      
      ForBCStructNumPatches(ipatch, bc_struct)
      {
	ForPatchCellsPerFace(BC_ALL,
			     BeforeAllCells(DoNothing),
			     LoopVars(i, j, k, ival, bc_struct, ipatch, is),
			     Locals(int current_patch_index, index2d;),
			     CellSetup(
                             {
			       current_patch_index = -1;
			       index2d = SubvectorEltIndex(patch_top_subvector, i, j, grid2d_iz);
			     }),
			     FACE(LeftFace, DoNothing),
			     FACE(RightFace, DoNothing),
			     FACE(DownFace, DoNothing),
			     FACE(UpFace, DoNothing),
			     FACE(BackFace, DoNothing),
			     FACE(FrontFace,
			     {
			       if( index_top_data[index2d] > 0 )
			       {
				 current_patch_index = ipatch;
			       }
			     }),
			     CellFinalize(
			     {
			       /* If we detected an UpperZFace that
				  was on the top of the domain then
				  set the patch index to the patch.
				*/

			       if (current_patch_index > -1)
			       {
				 //printf("Setting patch index %d, (%d,%d,%d)\n", ipatch, i, j, k);
				 patch_top_data[index2d] = current_patch_index;
			       }
			     }),
			     AfterAllCells(DoNothing)
			     ); /* End BC_ALL */
      } /* End ipatch loop */
    } /* End subgrid loop */
  } /* End num_patches > 0 */

  FreeBCStruct(bc_struct);
    
  /* Pass top values to neighbors.  */
  handle = InitVectorUpdate(patch_top, VectorUpdateAll);
  FinalizeVectorUpdate(handle);
}
