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
#include "llnlmath.h"
#include "llnltyps.h"

/*---------------------------------------------------------------------
 * Define module structures
 *---------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct
{
   Problem      *problem;

   PFModule     *density_module;
   PFModule     *saturation_module;
   PFModule     *rel_perm_module;
   PFModule     *bc_pressure;
   PFModule     *bc_internal;

   Matrix       *J;

   Grid         *grid;
   double       *temp_data;

} InstanceXtra;

/*--------------------------------------------------------------------------
 * Static stencil-shape definition
 *--------------------------------------------------------------------------*/

int           jacobian_stencil_shape[7][3] = {{ 0,  0,  0},
					      {-1,  0,  0},
					      { 1,  0,  0},
					      { 0, -1,  0},
					      { 0,  1,  0},
					      { 0,  0, -1},
					      { 0,  0,  1}};


/*---------------------------------------------------------------------
 * Define macros for jacobian evaluation
 *---------------------------------------------------------------------*/

#define PMean(a, b, c, d)    HarmonicMean(c, d)
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*  This routine provides the interface between KINSOL and ParFlow
    for richards' equation jacobian evaluations and matrix-vector multiplies.*/

int       KINSolMatVec(current_state, x, y, recompute, pressure)
void     *current_state;
N_Vector  x;
N_Vector  y;
int      *recompute;
N_Vector  pressure;
{
   PFModule    *richards_jacobian_eval = StateJacEval(((State*)current_state));
   Matrix      *J                = StateJac(         ((State*)current_state) );
   Vector      *saturation       = StateSaturation(  ((State*)current_state) );
   Vector      *density          = StateDensity(     ((State*)current_state) );
   ProblemData *problem_data     = StateProblemData( ((State*)current_state) );
   double       dt               = StateDt(          ((State*)current_state) );
   double       time             = StateTime(        ((State*)current_state) );

   if ( *recompute )
   { 
      PFModuleInvoke(void, richards_jacobian_eval, 
      (pressure, &J, saturation, density, problem_data,
      dt, time, 0));
   }

   Matvec(1.0, J, x, 0.0, y);

   return(0);
}


/*  This routine evaluates the Richards jacobian based on the current 
    pressure values.  */

void    RichardsJacobianEval(pressure, ptr_to_J, saturation, density, 
problem_data, dt, time, symm_part)
Vector       *pressure;       /* Current pressure values */
Matrix      **ptr_to_J;       /* Pointer to the J pointer - this will be set
		                 to instance_xtra pointer at end */
Vector       *saturation;     /* Saturation / work vector */
Vector       *density;        /* Density vector */
ProblemData  *problem_data;   /* Geometry data for problem */
double        dt;             /* Time step size */
double        time;           /* New time value */
int           symm_part;      /* Specifies whether to compute just the
                                 symmetric part of the Jacobian (1), or the
				 full Jacobian */
{
   PFModule      *this_module     = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   Problem     *problem           = (instance_xtra -> problem);

   PFModule    *density_module    = (instance_xtra -> density_module);
   PFModule    *saturation_module = (instance_xtra -> saturation_module);
   PFModule    *rel_perm_module   = (instance_xtra -> rel_perm_module);
   PFModule    *bc_pressure       = (instance_xtra -> bc_pressure);
   PFModule    *bc_internal       = (instance_xtra -> bc_internal);

   Matrix      *J                 = (instance_xtra -> J);

   Vector      *density_der       = NULL;
   Vector      *saturation_der    = NULL;

   /* Re-use vectors to save memory */
   Vector      *rel_perm          = NULL;
   Vector      *rel_perm_der      = NULL;

   Vector      *porosity          = ProblemDataPorosity(problem_data);
   Vector      *permeability_x    = ProblemDataPermeabilityX(problem_data);
   Vector      *permeability_y    = ProblemDataPermeabilityY(problem_data);
   Vector      *permeability_z    = ProblemDataPermeabilityZ(problem_data);
   Vector      *sstorage          = ProblemDataSpecificStorage(problem_data); //sk

   double       gravity           = ProblemGravity(problem);
   double       viscosity         = ProblemPhaseViscosity(problem, 0);

   Subgrid     *subgrid;

   Subvector   *p_sub, *d_sub, *s_sub, *po_sub, *rp_sub, *ss_sub;
   Subvector   *permx_sub, *permy_sub, *permz_sub, *dd_sub, *sd_sub, *rpd_sub;
   Submatrix   *J_sub;

   Grid        *grid              = VectorGrid(pressure);

   double      *pp, *sp, *sdp, *pop, *dp, *ddp, *rpp, *rpdp;
   double      *permxp, *permyp, *permzp;
   double      *cp, *wp, *ep, *sop, *np, *lp, *up, *op, *ss;

   int          i, j, k, r, is;
   int          ix, iy, iz;
   int          nx, ny, nz;
   int          nx_v, ny_v, nz_v;
   int          nx_m, ny_m, nz_m;
   int          nx_po, ny_po, nz_po;
   int          sy_v, sz_v;
   int          sy_m, sz_m;
   int          ip, ipo, im, iv, io;

   double       dtmp, dx, dy, dz, vol, ffx, ffy, ffz;
   double       diff, coeff, x_coeff, y_coeff, z_coeff;
   double       prod, prod_rt, prod_no, prod_up, prod_val, prod_lo;
   double       prod_der, prod_rt_der, prod_no_der, prod_up_der;
   double       west_temp, east_temp, north_temp, south_temp;
   double       lower_temp, upper_temp, o_temp;
   double       sym_west_temp, sym_east_temp, sym_south_temp, sym_north_temp;
   double       sym_lower_temp, sym_upper_temp;
   double       lower_cond, upper_cond;

   BCStruct    *bc_struct;
   GrGeomSolid *gr_domain         = ProblemDataGrDomain(problem_data);
   double      *bc_patch_values;
   double       value, den_d, dend_d;
   int         *fdir;
   int          ipatch, ival;
   
   CommHandle  *handle;


   /*-----------------------------------------------------------------------
    * Free temp vectors
    *-----------------------------------------------------------------------*/
   density_der     = NewVector(grid, 1, 1);
   saturation_der  = NewVector(grid, 1, 1);

   /*-----------------------------------------------------------------------
    * reuse the temp vectors for both saturation and rel_perm calculations.
    *-----------------------------------------------------------------------*/
   rel_perm          = saturation;
   rel_perm_der      = saturation_der;

   /* Pass pressure values to neighbors.  */
   handle = InitVectorUpdate(pressure, VectorUpdateAll);
   FinalizeVectorUpdate(handle);

   /* Pass permeability values */
   /*
     handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
     FinalizeVectorUpdate(handle);

     handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
     FinalizeVectorUpdate(handle);*/

   /* Initialize matrix values to zero. */
   InitMatrix(J, 0.0);

   /* Calculate time term contributions. */

   PFModuleInvoke(void, density_module, (0, pressure, density, &dtmp, &dtmp, 
					 CALCFCN));
   PFModuleInvoke(void, density_module, (0, pressure, density_der, &dtmp, 
					 &dtmp, CALCDER));
   PFModuleInvoke(void, saturation_module, (saturation, pressure, 
					    density, gravity, problem_data, 
					    CALCFCN));
   PFModuleInvoke(void, saturation_module, (saturation_der, pressure, 
					    density, gravity, problem_data,
					    CALCDER));

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);

      J_sub  = MatrixSubmatrix(J, is);
      cp     = SubmatrixStencilData(J_sub, 0);
	
      p_sub   = VectorSubvector(pressure, is);
      d_sub  = VectorSubvector(density, is);
      s_sub  = VectorSubvector(saturation, is);
      dd_sub = VectorSubvector(density_der, is);
      sd_sub = VectorSubvector(saturation_der, is);
      po_sub = VectorSubvector(porosity, is);
      ss_sub = VectorSubvector(sstorage, is);

      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);
	 
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
	 
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
	 
      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
	 
      vol = dx*dy*dz;
	 
      nx_v  = SubvectorNX(d_sub);
      ny_v  = SubvectorNY(d_sub);
      nz_v  = SubvectorNZ(d_sub);

      nx_po = SubvectorNX(po_sub);
      ny_po = SubvectorNY(po_sub);
      nz_po = SubvectorNZ(po_sub);

      nx_m  = SubmatrixNX(J_sub);
      ny_m  = SubmatrixNY(J_sub);
      nz_m  = SubmatrixNZ(J_sub);

      pp  = SubvectorData(p_sub);
      dp  = SubvectorData(d_sub);
      sp  = SubvectorData(s_sub);
      ddp = SubvectorData(dd_sub);
      sdp = SubvectorData(sd_sub);
      pop = SubvectorData(po_sub);
      ss  = SubvectorData(ss_sub);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {

	 im  = SubmatrixEltIndex(J_sub, i, j, k);
	 ipo = SubvectorEltIndex(po_sub, i, j, k);
	 iv  = SubvectorEltIndex(d_sub,  i, j, k);

	 cp[im] += (sdp[iv]*dp[iv] + sp[iv]*ddp[iv])
	    *pop[ipo]*vol + ss[iv]*vol*(sdp[iv]*dp[iv]*pp[iv]+sp[iv]*ddp[iv]*pp[iv]+sp[iv]*dp[iv]); //sk start
      });

   }   /* End subgrid loop */

   bc_struct = PFModuleInvoke(BCStruct *, bc_pressure, 
   (problem_data, grid, gr_domain, time));

   /* Get boundary pressure values for Dirichlet boundaries.   */
   /* These are needed for upstream weighting in mobilities - need boundary */
   /* values for rel perms and densities. */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      p_sub = VectorSubvector(pressure, is);

      nx_v = SubvectorNX(p_sub);
      ny_v = SubvectorNY(p_sub);
      nz_v = SubvectorNZ(p_sub);
	 
      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      pp = SubvectorData(p_sub);

      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	 bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

	 switch(BCStructBCType(bc_struct, ipatch))
	 {

	    case DirichletBC:
	    {
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  ip   = SubvectorEltIndex(p_sub, i, j, k);
		  value =  bc_patch_values[ival];
		  pp[ip + fdir[0]*1 + fdir[1]*sy_v + fdir[2]*sz_v] = value;
	       
	       });
	       break;
	    }

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */

   /* Calculate rel_perm and rel_perm_der */

   PFModuleInvoke(void, rel_perm_module, 
   (rel_perm, pressure, density, gravity, problem_data, 
   CALCFCN));

   PFModuleInvoke(void, rel_perm_module, 
   (rel_perm_der, pressure, density, gravity, problem_data, 
   CALCDER));

   /* Calculate contributions from second order derivatives and gravity */
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      p_sub    = VectorSubvector(pressure, is);
      d_sub    = VectorSubvector(density, is);
      rp_sub   = VectorSubvector(rel_perm, is);
      dd_sub   = VectorSubvector(density_der, is);
      rpd_sub  = VectorSubvector(rel_perm_der, is);
      permx_sub = VectorSubvector(permeability_x, is);
      permy_sub = VectorSubvector(permeability_y, is);
      permz_sub = VectorSubvector(permeability_z, is);
      J_sub    = MatrixSubmatrix(J, is);
	 
      ix = SubgridIX(subgrid) - 1;
      iy = SubgridIY(subgrid) - 1;
      iz = SubgridIZ(subgrid) - 1;

      nx = SubgridNX(subgrid) + 1;
      ny = SubgridNY(subgrid) + 1;
      nz = SubgridNZ(subgrid) + 1;
	 
      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
	 
      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;

      nx_v = SubvectorNX(p_sub);
      ny_v = SubvectorNY(p_sub);
      nz_v = SubvectorNZ(p_sub);
	 
      nx_m = SubmatrixNX(J_sub);
      ny_m = SubmatrixNY(J_sub);
      nz_m = SubmatrixNZ(J_sub);

      sy_v = nx_v;
      sz_v = ny_v * nx_v;
      sy_m = nx_m;
      sz_m = ny_m * nx_m;

      cp    = SubmatrixStencilData(J_sub, 0);
      wp    = SubmatrixStencilData(J_sub, 1);
      ep    = SubmatrixStencilData(J_sub, 2);
      sop   = SubmatrixStencilData(J_sub, 3);
      np    = SubmatrixStencilData(J_sub, 4);
      lp    = SubmatrixStencilData(J_sub, 5);
      up    = SubmatrixStencilData(J_sub, 6);

      pp     = SubvectorData(p_sub);
      dp     = SubvectorData(d_sub);
      rpp    = SubvectorData(rp_sub);
      ddp    = SubvectorData(dd_sub);
      rpdp   = SubvectorData(rpd_sub);
      permxp = SubvectorData(permx_sub);
      permyp = SubvectorData(permy_sub);
      permzp = SubvectorData(permz_sub);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {

	 ip = SubvectorEltIndex(p_sub, i, j, k);
	 im = SubmatrixEltIndex(J_sub, i, j, k);

	 prod        = rpp[ip] * dp[ip];
	 prod_der    = rpdp[ip] * dp[ip] + rpp[ip] * ddp[ip];

	 prod_rt     = rpp[ip+1] * dp[ip+1];
	 prod_rt_der = rpdp[ip+1] * dp[ip+1] + rpp[ip+1] * ddp[ip+1];

	 prod_no     = rpp[ip+sy_v] * dp[ip+sy_v];
	 prod_no_der = rpdp[ip+sy_v] * dp[ip+sy_v] 
	    + rpp[ip+sy_v] * ddp[ip+sy_v];

	 prod_up     = rpp[ip+sz_v] * dp[ip+sz_v];
	 prod_up_der = rpdp[ip+sz_v] * dp[ip+sz_v] 
	    + rpp[ip+sz_v] * ddp[ip+sz_v];

	 /* diff >= 0 implies flow goes left to right */
	 diff = pp[ip] - pp[ip+1];

	 x_coeff = dt * ffx * (1.0/dx) 
	    * PMean(pp[ip], pp[ip+1], permxp[ip], permxp[ip+1]) 
	    / viscosity;

	 sym_west_temp = - x_coeff 
	    * RPMean(pp[ip], pp[ip+1], prod, prod_rt);

	 west_temp = - x_coeff * diff 
	    * RPMean(pp[ip], pp[ip+1], prod_der, 0.0)
	    + sym_west_temp;

	 sym_east_temp = x_coeff
	    * -RPMean(pp[ip], pp[ip+1], prod, prod_rt);

	 east_temp = x_coeff * diff
	    * RPMean(pp[ip], pp[ip+1], 0.0, prod_rt_der)
	    + sym_east_temp;

	 /* diff >= 0 implies flow goes south to north */
	 diff = pp[ip] - pp[ip+sy_v];

	 y_coeff = dt * ffy * (1.0/dy) 
	    * PMean(pp[ip], pp[ip+sy_v], permyp[ip], permyp[ip+sy_v]) 
	    / viscosity;

	 sym_south_temp = - y_coeff
	    *RPMean(pp[ip], pp[ip+sy_v], prod, prod_no);

	 south_temp = - y_coeff * diff
	    * RPMean(pp[ip], pp[ip+sy_v], prod_der, 0.0)
	    + sym_south_temp;

	 sym_north_temp = y_coeff
	    * -RPMean(pp[ip], pp[ip+sy_v], prod, prod_no);

	 north_temp = y_coeff * diff
	    *  RPMean(pp[ip], pp[ip+sy_v], 0.0, 
	    prod_no_der)
	    + sym_north_temp;

	 /* diff >= 0 implies flow goes lower to upper */
	 lower_cond = pp[ip]      - 0.5 * dz * dp[ip]      * gravity;
	 upper_cond = pp[ip+sz_v] + 0.5 * dz * dp[ip+sz_v] * gravity;
	 diff = lower_cond - upper_cond;

	 z_coeff = dt * ffz * (1.0 / dz) 
	    * PMean(lower_cond, upper_cond, 
	    permzp[ip], permzp[ip+sz_v]) 
	    / viscosity;

	 sym_lower_temp = - z_coeff
	    * RPMean(lower_cond, upper_cond, prod, 
	    prod_up);
		
	 lower_temp = - z_coeff 
	    * ( diff * RPMean(lower_cond, upper_cond, prod_der, 0.0) 
	    + ( - gravity * 0.5 * dz * ddp[ip]
	    * RPMean(lower_cond, upper_cond, prod, 
	    prod_up) ) )
	    + sym_lower_temp;

	 sym_upper_temp = z_coeff
	    * -RPMean(lower_cond, upper_cond, prod, 
	    prod_up);

	 upper_temp = z_coeff
	    * ( diff * RPMean(lower_cond, upper_cond, 0.0, 
	    prod_up_der) 
	    + ( - gravity * 0.5 * dz * ddp[ip+sz_v]
	    *RPMean(lower_cond, upper_cond, prod, 
	    prod_up) ) )
	    + sym_upper_temp;

	 cp[im]      -= west_temp + south_temp + lower_temp;
	 cp[im+1]    -= east_temp;
	 cp[im+sy_m] -= north_temp;
	 cp[im+sz_m] -= upper_temp;

	 if (! symm_part)
	 {
	    ep[im] += east_temp;
	    np[im] += north_temp;
	    up[im] += upper_temp;

	    wp[im+1] += west_temp;
	    sop[im+sy_m] += south_temp;
	    lp[im+sz_m] += lower_temp;
	 }
	 else  /* Symmetric matrix: just update upper coeffs */
	 {
	    ep[im] += sym_east_temp;
	    np[im] += sym_north_temp;
	    up[im] += sym_upper_temp;
	 }

      });

  }  // 

   /*  Calculate correction for boundary conditions */

   if (symm_part)
   {
      /*  For symmetric part only, we first adjust coefficients of normal */
      /*  direction boundary pressure by adding in the nonsymmetric part. */
      /*  The entire coefficicent will be subtracted from the diagonal    */
      /*  and set to zero in the subsequent section - no matter what type */
      /*  of BC is involved.  Without this correction, only the symmetric */
      /*  part would be removed, incorrectly leaving the nonsymmetric     */
      /*  contribution on the diagonal.                                   */
     
      ForSubgridI(is, GridSubgrids(grid))
      {
	 subgrid = GridSubgrid(grid, is);
	 
	 p_sub     = VectorSubvector(pressure, is);
	 dd_sub    = VectorSubvector(density_der, is);
	 rpd_sub   = VectorSubvector(rel_perm_der, is);
	 d_sub     = VectorSubvector(density, is);
	 rp_sub    = VectorSubvector(rel_perm, is);
	 permx_sub = VectorSubvector(permeability_x, is);
	 permy_sub = VectorSubvector(permeability_y, is);
	 permz_sub = VectorSubvector(permeability_z, is);
	 J_sub     = MatrixSubmatrix(J, is);

	 dx = SubgridDX(subgrid);
	 dy = SubgridDY(subgrid);
	 dz = SubgridDZ(subgrid);
      
	 ffx = dy * dz;
	 ffy = dx * dz;
	 ffz = dx * dy;
	 
	 nx_v = SubvectorNX(p_sub);
	 ny_v = SubvectorNY(p_sub);
	 nz_v = SubvectorNZ(p_sub);
	 
	 sy_v = nx_v;
	 sz_v = ny_v * nx_v;

	 cp    = SubmatrixStencilData(J_sub, 0);
	 wp    = SubmatrixStencilData(J_sub, 1);
	 ep    = SubmatrixStencilData(J_sub, 2);
	 sop   = SubmatrixStencilData(J_sub, 3);
	 np    = SubmatrixStencilData(J_sub, 4);
	 lp    = SubmatrixStencilData(J_sub, 5);
	 up    = SubmatrixStencilData(J_sub, 6);

	 pp     = SubvectorData(p_sub);
	 ddp    = SubvectorData(dd_sub);
	 rpdp   = SubvectorData(rpd_sub);
	 dp     = SubvectorData(d_sub);
	 rpp    = SubvectorData(rp_sub);
	 permxp = SubvectorData(permx_sub);
	 permyp = SubvectorData(permy_sub);
	 permzp = SubvectorData(permz_sub);

	 for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
	 {
	    BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
	       ip = SubvectorEltIndex(p_sub, i, j, k);
	       im = SubmatrixEltIndex(J_sub, i, j, k);

	       if (fdir[0])
	       {
		  switch(fdir[0])
		  {
		     case -1:
		     {
		        diff = pp[ip-1] - pp[ip];
			prod_der = rpdp[ip-1]*dp[ip-1] + rpp[ip-1]*ddp[ip-1];
		        coeff = dt * ffx * (1.0/dx) 
			   * PMean(pp[ip-1], pp[ip], permxp[ip-1], permxp[ip]) 
			   / viscosity;
		        wp[im] = - coeff * diff
			   * RPMean(pp[ip-1], pp[ip], prod_der, 0.0);      
			break;
		     }
		     case 1:
		     {
		        diff = pp[ip] - pp[ip+1];
			prod_der = rpdp[ip+1]*dp[ip+1] + rpp[ip+1]*ddp[ip+1];
		        coeff = dt * ffx * (1.0/dx) 
			   * PMean(pp[ip], pp[ip+1], permxp[ip], permxp[ip+1]) 
			   / viscosity;
		        ep[im] = coeff * diff
			   * RPMean(pp[ip], pp[ip+1], 0.0, prod_der);      
			break;
		     }
		  }   /* End switch on fdir[0] */

	       }      /* End if (fdir[0]) */

	       else if (fdir[1])
	       {
		  switch(fdir[1])
		  {
		     case -1:
		     {
		        diff = pp[ip-sy_v] - pp[ip];
			prod_der = rpdp[ip-sy_v] * dp[ip-sy_v] 
			   + rpp[ip-sy_v] * ddp[ip-sy_v];
		        coeff = dt * ffy * (1.0/dy) 
			   * PMean(pp[ip-sy_v], pp[ip], 
			   permyp[ip-sy_v], permyp[ip]) 
			   / viscosity;
		        sop[im] = - coeff * diff
			   * RPMean(pp[ip-sy_v], pp[ip], prod_der, 0.0);      
			break;
		     }
		     case 1:
		     {
		        diff = pp[ip] - pp[ip+sy_v];
			prod_der = rpdp[ip+sy_v] * dp[ip+sy_v] 
			   + rpp[ip+sy_v] * ddp[ip+sy_v];
		        coeff = dt * ffy * (1.0/dy) 
			   * PMean(pp[ip], pp[ip+sy_v], 
			   permyp[ip], permyp[ip+sy_v]) 
			   / viscosity;
		        np[im] = - coeff * diff
			   * RPMean(pp[ip], pp[ip+sy_v], 0.0, prod_der);      
			break;
		     }
		  }   /* End switch on fdir[1] */

	       }      /* End if (fdir[1]) */

	       else if (fdir[2])
	       {
		  switch(fdir[2])
		  {
		     case -1:
		     {
			lower_cond = (pp[ip-sz_v]) 
			   - 0.5 * dz * dp[ip-sz_v] * gravity;
			upper_cond = (pp[ip] ) + 0.5 * dz * dp[ip] * gravity;
		        diff = lower_cond - upper_cond;
			prod_der = rpdp[ip-sz_v] * dp[ip-sz_v] 
			   + rpp[ip-sz_v] * ddp[ip-sz_v];
			prod_lo = rpp[ip-sz_v] * dp[ip-sz_v];
		        coeff = dt * ffz * (1.0/dz) 
			   * PMean(pp[ip-sz_v], pp[ip], 
			   permzp[ip-sz_v], permzp[ip]) 
			   / viscosity;
		        lp[im] = - coeff * 
			   ( diff * RPMean(lower_cond, upper_cond, 
			   prod_der, 0.0)
			   - gravity * 0.5 * dz * ddp[ip]
			   * RPMean(lower_cond, upper_cond, prod_lo, prod));
			break;
		     }
		     case 1:
		     {
			lower_cond = (pp[ip]) - 0.5 * dz * dp[ip] * gravity;
			upper_cond = (pp[ip+sz_v] ) 
			   + 0.5 * dz * dp[ip+sz_v] * gravity;
		        diff = lower_cond - upper_cond;
			prod_der = rpdp[ip+sz_v] * dp[ip+sz_v] 
			   + rpp[ip+sz_v] * ddp[ip+sz_v];
			prod_up = rpp[ip+sz_v] * dp[ip+sz_v];
		        coeff = dt * ffz * (1.0/dz) 
			   * PMean(lower_cond, upper_cond, 
			   permzp[ip], permzp[ip+sz_v]) 
			   / viscosity;
		        up[im] = - coeff * 
			   ( diff * RPMean(lower_cond, upper_cond, 
			   0.0, prod_der)
			   - gravity * 0.5 * dz * ddp[ip]
			   * RPMean(lower_cond, upper_cond, prod, prod_up));
			break;
		     }
		  }   /* End switch on fdir[2] */

	       }   /* End if (fdir[2]) */
		     
	    });   /* End Patch Loop */

	 }        /* End ipatch loop */
      }           /* End subgrid loop */
   }                 /* End if symm_part */


   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      p_sub     = VectorSubvector(pressure, is);
      s_sub     = VectorSubvector(saturation, is);
      dd_sub    = VectorSubvector(density_der, is);
      rpd_sub   = VectorSubvector(rel_perm_der, is);
      d_sub     = VectorSubvector(density, is);
      rp_sub    = VectorSubvector(rel_perm, is);
      permx_sub = VectorSubvector(permeability_x, is);
      permy_sub = VectorSubvector(permeability_y, is);
      permz_sub = VectorSubvector(permeability_z, is);
      J_sub     = MatrixSubmatrix(J, is);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
      
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      
      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;
	 
      nx_v = SubvectorNX(p_sub);
      ny_v = SubvectorNY(p_sub);
      nz_v = SubvectorNZ(p_sub);
	 
      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      cp    = SubmatrixStencilData(J_sub, 0);
      wp    = SubmatrixStencilData(J_sub, 1);
      ep    = SubmatrixStencilData(J_sub, 2);
      sop    = SubmatrixStencilData(J_sub, 3);
      np    = SubmatrixStencilData(J_sub, 4);
      lp    = SubmatrixStencilData(J_sub, 5);
      up    = SubmatrixStencilData(J_sub, 6);

      pp     = SubvectorData(p_sub);
      sp     = SubvectorData(s_sub);
      ddp    = SubvectorData(dd_sub);
      rpdp   = SubvectorData(rpd_sub);
      dp     = SubvectorData(d_sub);
      rpp    = SubvectorData(rp_sub);
      permxp = SubvectorData(permx_sub);
      permyp = SubvectorData(permy_sub);
      permzp = SubvectorData(permz_sub);

      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	 bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

	 switch(BCStructBCType(bc_struct, ipatch))
	 {

	    case DirichletBC:
	    {
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  ip   = SubvectorEltIndex(p_sub, i, j, k);

		  value =  bc_patch_values[ival];

		  PFModuleInvoke( void, density_module, 
		  (0, NULL, NULL, &value, &den_d, CALCFCN));
		  PFModuleInvoke( void, density_module, 
		  (0, NULL, NULL, &value, &dend_d, CALCDER));

		  ip = SubvectorEltIndex(p_sub, i, j, k);
		  im = SubmatrixEltIndex(J_sub, i, j, k);

		  prod        = rpp[ip] * dp[ip];
		  prod_der    = rpdp[ip] * dp[ip] + rpp[ip] * ddp[ip];

		  if (fdir[0])
		  {
		     coeff = dt * ffx * (2.0/dx) * permxp[ip] / viscosity;

		     switch(fdir[0])
		     {
			case -1:
			{
			   op = wp;
			   prod_val = rpp[ip-1] * den_d;
			   diff = value - pp[ip];
			   o_temp =  coeff 
			      * ( diff * RPMean(value, pp[ip], 0.0, prod_der) 
			      - RPMean(value, pp[ip], prod_val, prod) );
			   break;
			}
			case 1:
			{
			   op = ep;
			   prod_val = rpp[ip+1] * den_d;
			   diff = pp[ip] - value;
			   o_temp = - coeff 
			      * ( diff * RPMean(pp[ip], value, prod_der, 0.0) 
			      + RPMean(pp[ip], value, prod, prod_val) );
			   break;
			}
		     }   /* End switch on fdir[0] */

		  }      /* End if (fdir[0]) */

		  else if (fdir[1])
		  {
		     coeff = dt * ffy * (2.0/dy) * permyp[ip] / viscosity;

		     switch(fdir[1])
		     {
			case -1:
			{
			   op = sop;
			   prod_val = rpp[ip-sy_v] * den_d;
			   diff = value - pp[ip];
			   o_temp =  coeff 
			      * ( diff * RPMean(value, pp[ip], 0.0, prod_der) 
			      - RPMean(value, pp[ip], prod_val, prod) );
			   break;
			}
			case 1:
			{
			   op = np;
			   prod_val = rpp[ip+sy_v] * den_d;
			   diff = pp[ip] - value;
			   o_temp = - coeff 
			      * ( diff * RPMean(pp[ip], value, prod_der, 0.0) 
			      + RPMean(pp[ip], value, prod, prod_val) );
			   break;
			}
		     }   /* End switch on fdir[1] */

		  }      /* End if (fdir[1]) */

		  else if (fdir[2])
		  {
		     coeff = dt * ffz * (2.0/dz) * permzp[ip] / viscosity;

		     switch(fdir[2])
		     {
			case -1:
			{
			   op = lp;
			   prod_val = rpp[ip-sz_v] * den_d;

			   lower_cond = (value ) - 0.5 * dz * den_d * gravity;
			   upper_cond = (pp[ip]) + 0.5 * dz * dp[ip] * gravity;
			   diff = lower_cond - upper_cond;

			   o_temp =  coeff 
			      * ( diff * RPMean(lower_cond, upper_cond, 0.0, prod_der) 
			      + ( (-1.0 - gravity * 0.5 * dz * ddp[ip])
			      * RPMean(lower_cond, upper_cond, prod_val, prod)));
			   break;
			}
			case 1:
			{
			   op = up;
			   prod_val = rpp[ip+sz_v] * den_d;

			   lower_cond = (pp[ip]) - 0.5 * dz * dp[ip] * gravity;
			   upper_cond = (value ) + 0.5 * dz * den_d * gravity;
			   diff = lower_cond - upper_cond;

			   o_temp = - coeff 
			      * ( diff * RPMean(lower_cond, upper_cond, prod_der, 0.0) 
			      + ( (1.0 - gravity * 0.5 * dz * ddp[ip])
			      * RPMean(lower_cond, upper_cond, prod, prod_val)));
			   break;
			}
		     }   /* End switch on fdir[2] */

		  }   /* End if (fdir[2]) */

		  cp[im] += op[im];
		  cp[im] -= o_temp;
		  op[im] = 0.0;

	       });

	       break;
	    }           /* End DirichletBC case */

	    case FluxBC:
	    {
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  im = SubmatrixEltIndex(J_sub, i, j, k);

		  if (fdir[0] == -1)  op = wp;
		  if (fdir[0] ==  1)  op = ep;
		  if (fdir[1] == -1)  op = sop;
		  if (fdir[1] ==  1)  op = np;
		  if (fdir[2] == -1)  op = lp;
		  if (fdir[2] ==  1)  op = up;

		  cp[im] += op[im];
		  op[im] = 0.0;
	       });

	       break;
	    }     /* End fluxbc case */

	    case OverlandBC: //sk
	    {
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  im = SubmatrixEltIndex(J_sub, i, j, k);

		  if (fdir[0] == -1)  op = wp;
		  if (fdir[0] ==  1)  op = ep;
		  if (fdir[1] == -1)  op = sop;
		  if (fdir[1] ==  1)  op = np;
		  if (fdir[2] == -1)  op = lp;
		  if (fdir[2] ==  1)  op = up;

		  cp[im] += op[im];
		  op[im] = 0.0;
	       });

	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  if (fdir[2])
		  {
		     switch(fdir[2])
		     {
			case 1:

			   ip   = SubvectorEltIndex(p_sub, i, j, k);
			   io   = SubvectorEltIndex(p_sub, i, j, 0);
			   im = SubmatrixEltIndex(J_sub, i, j, k);
         
			   if ((pp[ip]) > 0.0) 
			   {
			      cp[im] += vol /dz*(dt+1);
			   }	
   			   break;
		     }
		  }

	       });

	       break;
	    }     /* End overland flow case */

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */

   FreeBCStruct(bc_struct);

   PFModuleInvoke( void, bc_internal, (problem, problem_data, NULL, J, time,
   pressure, CALCDER));


   /* Set pressures outside domain to zero.  
    * Recall: equation to solve is f = 0, so components of f outside 
    * domain are set to the respective pressure value.
    *
    * Should change this to set pressures to scaling value.
    * CSW: Should I set this to pressure * vol * dt ??? */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      J_sub = MatrixSubmatrix(J, is);

      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
	 
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
	 
      cp = SubmatrixStencilData(J_sub, 0);
      wp = SubmatrixStencilData(J_sub, 1);
      ep = SubmatrixStencilData(J_sub, 2);
      sop = SubmatrixStencilData(J_sub, 3);
      np = SubmatrixStencilData(J_sub, 4);
      lp = SubmatrixStencilData(J_sub, 5);
      up = SubmatrixStencilData(J_sub, 6);

      GrGeomOutLoop(i, j, k, gr_domain,
      r, ix, iy, iz, nx, ny, nz,
      {
	 im   = SubmatrixEltIndex(J_sub, i, j, k);
	 cp[im] = 1.0;
	 wp[im] = 0.0;
	 ep[im] = 0.0;
	 sop[im] = 0.0;
	 np[im] = 0.0;
	 lp[im] = 0.0;
	 up[im] = 0.0;
      });
   }

   /*-----------------------------------------------------------------------
    * Update matrix ghost points
    *-----------------------------------------------------------------------*/

   if (MatrixCommPkg(J))
   {
      handle = InitMatrixUpdate(J);
      FinalizeMatrixUpdate(handle);
   }

   *ptr_to_J = J;

   /*-----------------------------------------------------------------------
    * Free temp vectors
    *-----------------------------------------------------------------------*/
   FreeVector(density_der);
   FreeVector(saturation_der);

   return;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *RichardsJacobianEvalInitInstanceXtra(
   Problem     *problem,
   Grid        *grid,
   double      *temp_data,
   int          symmetric_jac)
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

   Stencil       *stencil;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
	 FreeMatrix(instance_xtra -> J);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      /* set up jacobian matrix */
      stencil = NewStencil(jacobian_stencil_shape, 7);

      if (symmetric_jac)
	 (instance_xtra -> J) = NewMatrix(grid, NULL, stencil, ON, stencil);
      else
	 (instance_xtra -> J) = NewMatrix(grid, NULL, stencil, OFF, stencil);

   }

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;
   }

   if ( problem != NULL)
   {
      (instance_xtra -> problem) = problem;
   }

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> density_module) =
         PFModuleNewInstance(ProblemPhaseDensity(problem), () );
      (instance_xtra -> bc_pressure) =
	 PFModuleNewInstance(ProblemBCPressure(problem), (problem) );
      (instance_xtra -> saturation_module) =
         PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL) );
      (instance_xtra -> rel_perm_module) =
         PFModuleNewInstance(ProblemPhaseRelPerm(problem), (NULL, NULL) );
      (instance_xtra -> bc_internal) =
         PFModuleNewInstance(ProblemBCInternal(problem), () );

   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> density_module), ());
      PFModuleReNewInstance((instance_xtra -> bc_pressure), (problem));
      PFModuleReNewInstance((instance_xtra -> saturation_module), 
      (NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> rel_perm_module), 
      (NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> bc_internal), ());

   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  RichardsJacobianEvalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if(instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> density_module);
      PFModuleFreeInstance(instance_xtra -> bc_pressure);
      PFModuleFreeInstance(instance_xtra -> saturation_module);
      PFModuleFreeInstance(instance_xtra -> rel_perm_module);
      PFModuleFreeInstance(instance_xtra -> bc_internal);

      FreeMatrix(instance_xtra -> J);

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *RichardsJacobianEvalNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;


   public_xtra = NULL;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  RichardsJacobianEvalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if (public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RichardsJacobianEvalSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   int  sz = 0;

   return sz;
}
