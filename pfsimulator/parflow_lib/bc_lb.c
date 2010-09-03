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

#include "parflow.h"

/*--------------------------------------------------------------------------
 * BCPressure:
 *   This routine returns a BCStruct structure which describes where
 *   and what the boundary conditions are.
 *--------------------------------------------------------------------------*/

void LBInitializeBC(
   Lattice *lattice,
   Problem  *problem,
   ProblemData  *problem_data)
{

   /*------------------------------------------------------------*
    * Local variables 
    *------------------------------------------------------------*/

   /* Lattice variables */
   Grid  *grid		= (lattice->grid);
   Vector *pressure	= (lattice->pressure);
   Vector *perm		= (lattice->perm);
   CharVector *cellType = (lattice->cellType);
   double time		= (lattice->t);

   /* Structures */
   BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);
   TimeCycleData   *time_cycle_data;
   SubgridArray   *subgrids      = GridSubgrids(grid);
   GrGeomSolid    *gr_domain;

   /* Patch variables */
   double       ***values;
   double         *patch_values=NULL;
   int            *fdir;

   /* Grid parameters */
   Subgrid   *subgrid;
   int       nx, ny, nz;
   int       ix, iy, iz;
   int       nx_v, ny_v, nz_v;

   /* Indices and counters */
   int       num_patches;
   int       num_phases;
   int       ipatch, is, i, j, k, ival;
   int       cycle_number, interval_number;
   int       r;

   /* Physical variables and coefficients */
   Subvector *sub_p;
   double    *pp;
   Subvector *sub_perm;
   double    *permp;
   Subcharvector *sub_cellType;
   char      *cellTypep;
   double    rho_g;

   /* Communications */
   VectorUpdateCommHandle *handle;
	         
   /*--------------------------
    *  Initializations
    *--------------------------*/
   rho_g = ProblemGravity(problem) * RHO;
   num_patches = BCPressureDataNumPatches(bc_pressure_data);
   gr_domain = ProblemDataGrDomain(problem_data);
   num_phases = BCPressureDataNumPhases(bc_pressure_data);
   if (num_patches > 0)
   {
      time_cycle_data = BCPressureDataTimeCycleData(bc_pressure_data);
      values = ctalloc(double **, num_patches);

      for (ipatch = 0; ipatch < num_patches; ipatch++)
      {
         values[ipatch] = ctalloc(double *, SubgridArraySize(subgrids));

         cycle_number = BCPressureDataCycleNumber(bc_pressure_data,ipatch);
         interval_number = TimeCycleDataComputeIntervalNumber(problem, time, 
					     time_cycle_data, cycle_number);
        
	 switch(BCPressureDataType(bc_pressure_data,ipatch))
	 {
	 case 0:
	 {
	    BCPressureType0 *bc_pressure_type0;

	    GeomSolid       *ref_solid;

	    double           z, dz2;
	    double         **elevations;
	    int              ref_patch, iel;

	    bc_pressure_type0 = (BCPressureType0 *)BCPressureDataIntervalValue(
                                   bc_pressure_data,ipatch,interval_number);
	    ref_solid = ProblemDataSolid(problem_data, 
			     BCPressureType0RefSolid(bc_pressure_type0));
	    ref_patch = BCPressureType0RefPatch(bc_pressure_type0);

	    /* Calculate elevations at (x,y) points on reference patch. */
	    elevations = CalcElevations(ref_solid, ref_patch, subgrids);

	    ForSubgridI(is, subgrids)
	    {
               /* subgrid = GridSubgrid(grid, is); */
               subgrid = SubgridArraySubgrid(subgrids, is);
	       sub_p = VectorSubvector(pressure, is);
	       sub_perm = VectorSubvector(perm, is);
	       sub_cellType = CharVectorSubcharvector(cellType, is);

	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);

	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);

	       /* RDF: assume resolution is the same in all 3 directions */
	       r  = SubgridRX(subgrid);

	       pp = SubvectorData(sub_p);
	       permp = SubvectorData(sub_perm);
	       cellTypep = SubcharvectorData(sub_cellType);

	       nx_v = SubvectorNX(sub_p);
	       ny_v = SubvectorNY(sub_p);
	       nz_v = SubvectorNZ(sub_p);

	       values[ipatch][is] = patch_values;

               dz2  = RealSpaceDZ(0) / 2.0;

	       GrGeomPatchLoop(i, j, k, fdir, gr_domain, ipatch,
		   	       r, ix, iy, iz, nx, ny, nz,
   	       {
                  ival = SubvectorEltIndex(sub_p, i,j,k);
		  iel  = (i-ix) + (j-iy)*nx;
                  z = RealSpaceZ(k, 0) + fdir[2]*dz2;

                  pp[ival] = BCPressureType0Value(bc_pressure_type0)
		             - rho_g*(z - elevations[is][iel]);

                  cellTypep[ival] = 0;
               });

	       tfree(elevations[is]);

	    }     /* End subgrid loop */

	    tfree(elevations);
	    break;
	 }
	 case 1:
	 {
            BCPressureType1 *bc_pressure_type1;
	    int              num_points;
	    double           x, y, z, dx2, dy2, dz2;
	    double           unitx, unity, line_min, line_length, xy, slope;
	    int              ip;

	    bc_pressure_type1 = (BCPressureType1 *)BCPressureDataIntervalValue(bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       /* subgrid = GridSubgrid(grid, is); */
	       subgrid = SubgridArraySubgrid(subgrids, is);
	       sub_p = VectorSubvector(pressure, is);
	       sub_perm = VectorSubvector(perm, is);
	       sub_cellType = CharVectorSubcharvector(cellType, is);
	       
	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);
	       
	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);

	       /* RDF: assume resolution is the same in all 3 directions */
	       r  = SubgridRX(subgrid);
	       
	       pp = SubvectorData(sub_p);
	       permp = SubvectorData(sub_perm);
	       cellTypep = SubcharvectorData(sub_cellType);

	       nx_v = SubvectorNX(sub_p);
	       ny_v = SubvectorNY(sub_p);
	       nz_v = SubvectorNZ(sub_p);

	       values[ipatch][is] = patch_values;
	       
	       dx2  = RealSpaceDX(0) / 2.0;
	       dy2  = RealSpaceDY(0) / 2.0;
	       dz2  = RealSpaceDZ(0) / 2.0;

	       /* compute unit direction vector for piecewise linear line */
               unitx = BCPressureType1XUpper(bc_pressure_type1) - BCPressureType1XLower(bc_pressure_type1);
               unity = BCPressureType1YUpper(bc_pressure_type1) - BCPressureType1YLower(bc_pressure_type1);
               line_length = sqrt(unitx*unitx + unity*unity);
               unitx /= line_length;
               unity /= line_length;
               line_min = BCPressureType1XLower(bc_pressure_type1)*unitx
                        + BCPressureType1YLower(bc_pressure_type1)*unity;

	       GrGeomPatchLoop(i, j, k, fdir, gr_domain, ipatch,
		   	       r, ix, iy, iz, nx, ny, nz,
   	       {
                  ival = SubvectorEltIndex(sub_p, i,j,k);

                  x = RealSpaceX(i, 0) + fdir[0]*dx2;
                  y = RealSpaceY(j, 0) + fdir[1]*dy2;
                  z = RealSpaceZ(k, 0) + fdir[2]*dz2;
	       
                  /* project center of BC face onto piecewise line */
                  xy = (x*unitx + y*unity - line_min) / line_length;

                  /* find two neighboring points */
                  ip = 1;
		  /* Kludge; this needs to be fixed. */
		  num_points = 2;
                  for (; ip < (num_points - 1); ip++)
                  {
                     if (xy < BCPressureType1Point(bc_pressure_type1,ip))
                        break;
                  }
	       
                  /* compute the slope */
                  slope = ((BCPressureType1Value(bc_pressure_type1,ip) - BCPressureType1Value(bc_pressure_type1,(ip-1)))
                        / (BCPressureType1Point(bc_pressure_type1,ip) - BCPressureType1Point(bc_pressure_type1,(ip-1))));
	       
                  pp[ival] = BCPressureType1Value(bc_pressure_type1,ip-1)
                                  + slope*(xy - BCPressureType1Point(
                                                       bc_pressure_type1,ip-1))
		                  - rho_g*z;

                  cellTypep[ival] = 0;
               });

	    }    /* End subgrid loop */

	    break;
	 }
	 case 2:
         {
            BCPressureType2 *bc_pressure_type2;
	    bc_pressure_type2 = (BCPressureType2 *)BCPressureDataIntervalValue(bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
               /* subgrid = GridSubgrid(grid, is); */
               subgrid = SubgridArraySubgrid(subgrids, is);
	       sub_p = VectorSubvector(pressure, is);
	       sub_perm = VectorSubvector(perm, is);
	       sub_cellType = CharVectorSubcharvector(cellType, is);

	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);

	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);

	       /* RDF: assume resolution is the same in all 3 directions */
	       r  = SubgridRX(subgrid);

	       pp = SubvectorData(sub_p);
	       permp = SubvectorData(sub_perm);
	       cellTypep = SubcharvectorData(sub_cellType);

	       nx_v = SubvectorNX(sub_p);
	       ny_v = SubvectorNY(sub_p);
	       nz_v = SubvectorNZ(sub_p);

	       values[ipatch][is] = patch_values;

	       GrGeomPatchLoop(i, j, k, fdir, gr_domain, ipatch,
		   	       r, ix, iy, iz, nx, ny, nz,
   	       {
                  ival = SubvectorEltIndex(sub_p, i,j,k);
                  if (cellTypep[ival])
                  {
                     /* pp[ival] = BCPressureType2Value(bc_pressure_type2); */
                     /* pp[ival] = 0.0; */
                     cellTypep[ival] = 1;
                  }
               });
	    }       /* End subgrid loop */
               break;
	 }
	 }
      }

      for (ipatch = 0; ipatch < num_patches; ipatch++)
      {
         free (values[ipatch]);
      }
      free (values);
   }

   /* Update the boundary layers */
   handle = InitVectorUpdate(pressure, VectorUpdateAll);
   FinalizeVectorUpdate(handle);

   /* Deallocate arrays */
}

