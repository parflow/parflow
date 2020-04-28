/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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

#include "parflow.h"

#include <math.h>

void ComputeOverlandFlowRunningSums(Vector* overlandflow_face_flow[],
				    Vector* overlandflow_cell_outflow,
				    Vector *KE, Vector *KW, Vector *KN, Vector *KS, BCStruct *bc_struct)
{
  Grid        *grid = VectorGrid(KE);
  int is;

  ForSubgridI(is, GridSubgrids(grid))
  {
    Subgrid *subgrid = GridSubgrid(grid, is);

    Subvector *overlandflow_cell_outflow_sub;
    double *overlandflow_cell_outflow_dat;
	
    if(overlandflow_cell_outflow)
    {
      overlandflow_cell_outflow_sub = VectorSubvector(overlandflow_cell_outflow, is);
      overlandflow_cell_outflow_dat = SubvectorData(overlandflow_cell_outflow_sub);
    }


    Subvector *overlandflow_face_flow_sub[OverlandFlowKMax];
    double *overlandflow_face_flow_dat[OverlandFlowKMax];
    
    if (overlandflow_face_flow[0])
    {
      for(int face = 0; face < OverlandFlowKMax; face++)
      {
	overlandflow_face_flow_sub[face] = VectorSubvector(overlandflow_face_flow[face], is);
	overlandflow_face_flow_dat[face] = SubvectorData(overlandflow_face_flow_sub[face]);
      }
    }

    Subvector *kw_sub = VectorSubvector(KW, is);
    Subvector *ke_sub = VectorSubvector(KE, is);
    Subvector *kn_sub = VectorSubvector(KN, is);
    Subvector *ks_sub = VectorSubvector(KS, is);

    double *kw_dat = SubvectorData(kw_sub);
    double *ke_dat = SubvectorData(ke_sub);
    double *kn_dat = SubvectorData(kn_sub);
    double *ks_dat = SubvectorData(ks_sub);

    if(overlandflow_cell_outflow && overlandflow_face_flow[0])
    {
      for (int ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	switch (BCStructBCType(bc_struct, ipatch))
	{
	  case OverlandBC:
	  {
	    int i, j, k;
	    int *fdir;
	    int ival;
	    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
            {
	      if (fdir[2] == 1)
	      {
		int io = SubvectorEltIndex(kw_sub, i, j, 0);
		
		// Calculate total outflow of each cell
		overlandflow_cell_outflow_dat[io] = pfmax(kn_dat[io], 0) + pfmax(-ks_dat[io], 0) + pfmax(-ke_dat[io], 0) + pfmax(kw_dat[io], 0) ;
		// printf("i=%d j=%d k=%d ke_dat=%f kw_dat=%f kn_dat=%f ks_dat=%f\n",i,j,k,ke_dat[io],kw_dat[io],kn_dat[io],ks_dat[io]);

		overlandflow_face_flow_dat[OverlandFlowKE][io] += ke_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKW][io] += kw_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKN][io] += kn_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKS][io] += ks_dat[io];
	      }
	    })
          }
	}
      }
    }
    else if(overlandflow_face_flow[0])
    {
      for (int ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	switch (BCStructBCType(bc_struct, ipatch))
	{
	  case OverlandBC:
	  {
	    int i, j, k;
	    int *fdir;
	    int ival;
	    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
            {
	      if (fdir[2] == 1)
	      {
		int io = SubvectorEltIndex(kw_sub, i, j, 0);
		
		overlandflow_face_flow_dat[OverlandFlowKE][io] += ke_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKW][io] += kw_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKN][io] += kn_dat[io];
		overlandflow_face_flow_dat[OverlandFlowKS][io] += ks_dat[io];
	      }
	    })
          }
	}
      }
    }
    else if (overlandflow_cell_outflow)
    {
      for (int ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	switch (BCStructBCType(bc_struct, ipatch))
	{
	  case OverlandBC:
	  {
	    int i, j, k;
	    int *fdir;
	    int ival;
	    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
            {
	      if (fdir[2] == 1)
	      {
		int io = SubvectorEltIndex(kw_sub, i, j, 0);
		
		// Calculate total outflow of each cell
		overlandflow_cell_outflow_dat[io] = pfmax(kn_dat[io], 0) + pfmax(-ks_dat[io], 0) + pfmax(-ke_dat[io], 0) + pfmax(kw_dat[io], 0) ;
	      }
	    })
          }
	}
      }
    }
  }
}



