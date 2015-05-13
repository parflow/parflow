/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Static stencil-shape definition
 *--------------------------------------------------------------------------*/
 
int           jacobian_stencil_shape_temperature[7][3] = {{ 0,  0,  0},
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
#define AMean(a, b)          ArithmeticMean(a,b)
 

/*---------------------------------------------------------------------
 * Define module structures
 *---------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct
{
   Problem      *problem;

   PFModule     *density_module;
   PFModule     *heat_capacity_module;
   PFModule     *saturation_module;
   PFModule     *thermal_conductivity;
   PFModule     *bc_temperature;
   PFModule     *bc_internal;

   Vector       *density_der;
   Vector       *saturation_der;
   Vector       *thermal_cond;
   Vector       *thermal_cond_der;

   Matrix       *J;

   Grid         *grid;
   double       *temp_data;

} InstanceXtra;

/*  This routine evaluates the Richards jacobian based on the current 
    temperature values.  */

void    TemperatureJacobianEval( 
Vector       *temperature,       /* Current temperature values */
Matrix      **ptr_to_J,       /* Pointer to the J pointer - this will be set
		                 to instance_xtra pointer at end */
Vector       *pressure,    
Vector       *saturation,     /* Saturation / work vector */
Vector       *density,        /* Density vector */
Vector       *heat_capacity_water, 
Vector       *heat_capacity_rock, 
Vector       *x_velocity,
Vector       *y_velocity,
Vector       *z_velocity,
ProblemData  *problem_data,   /* Geometry data for problem */
double        dt,             /* Time step size */
double        time,           /* New time value */
int           symm_part)      /* Specifies whether to compute just the
                                 symmetric part of the Jacobian (1), or the
				 full Jacobian */
{
   PFModule      *this_module     = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   Problem     *problem           = (instance_xtra -> problem);

   PFModule    *density_module       = (instance_xtra -> density_module);
   PFModule    *heat_capacity_module = (instance_xtra -> heat_capacity_module);
   PFModule    *saturation_module    = (instance_xtra -> saturation_module);
   PFModule    *thermal_conductivity = (instance_xtra -> thermal_conductivity);
   PFModule    *bc_temperature       = (instance_xtra -> bc_temperature);
   PFModule    *bc_internal          = (instance_xtra -> bc_internal);

   Matrix      *J                 = (instance_xtra -> J);

   Vector      *density_der       = (instance_xtra -> density_der);
   Vector      *saturation_der    = (instance_xtra -> saturation_der);

   /* Re-use vectors to save memory */
   Vector      *thermal_cond          = (instance_xtra -> thermal_cond);
   Vector      *thermal_cond_der      = (instance_xtra -> thermal_cond_der);

  //double      press[58][30][360],pressbc[58][30],xslope[58][30],yslope[58][30],mans[58][30];

   Vector      *porosity          = ProblemDataPorosity(problem_data);
   Vector      *permeability_x    = ProblemDataPermeabilityX(problem_data);
   Vector      *permeability_y    = ProblemDataPermeabilityY(problem_data);
   Vector      *permeability_z    = ProblemDataPermeabilityZ(problem_data);
   Vector      *sstorage          = ProblemDataSpecificStorage(problem_data);

   double       gravity           = ProblemGravity(problem);

   Subgrid     *subgrid;

   Subvector   *p_sub, *t_sub, *d_sub, *s_sub, *po_sub, *tcp_sub, *hcw_sub, *hcr_sub, *ss_sub;
   Subvector   *dd_sub, *sd_sub, *tcdp_sub;
   Submatrix   *J_sub;

   Grid        *grid              = VectorGrid(temperature);

   double      *pp, *tp, *sp, *pop, *dp, *ddp, *tcp, *tcdp, *ss;
   double      *cp, *wp, *ep, *sop, *np, *lp, *up, *op, *hcwp, *hcrp;

   /* Fluid flow velcocities */ 
   Subvector   *xv_sub, *yv_sub, *zv_sub;
   double      *xvp,*yvp,*zvp;
 
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
   double       lower_temp, upper_temp, o_temp, o_new;
   double       sym_west_temp, sym_east_temp, sym_south_temp, sym_north_temp;
   double       sym_lower_temp, sym_upper_temp;
   double       lower_cond, upper_cond;
   double       temp_conv;

   BCStruct    *bc_struct;
   GrGeomSolid *gr_domain         = ProblemDataGrDomain(problem_data);
   double      *bc_patch_values;
   double       value, den_d, dend_d;
   int         *fdir;
   int          ipatch, ival;
   CommHandle  *handle;
   VectorUpdateCommHandle  *vector_update_handle;



//TODO: #FG This has to be outsourced to a module
Vector *constThermalCond = NewVectorType( grid, 1, 1, vector_cell_centered  );
double *ctcp;
Subvector *ctc_sub;
InitVectorAll(constThermalCond, 0.0);
ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
      ctc_sub  = VectorSubvector(constThermalCond, is);
      ctcp     = SubvectorData(ctc_sub);
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
      r = SubgridRX(subgrid);

      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
      {
        ip  = SubvectorEltIndex(ctc_sub,   i,j,k);
        ctcp[ip]=1.0;
      });

   }

//ENDTODO:



   /* Pass temperature values to neighbors.  */
   vector_update_handle = InitVectorUpdate(temperature, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);

   /* Pass permeability values */
   /*
   vector_update_handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);

   vector_update_handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);

   vector_update_handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);*/


   /* Initialize matrix values to zero. */
   InitMatrix(J, 0.0);

   /* Calculate time term contributions. */

//FG   PFModuleInvoke(void, density_module, (0, pressure, temperature, density, &dtmp, &dtmp, 
//FG					 CALCFCN));
//FG   PFModuleInvoke(void, density_module, (0, pressure, temperature, density_der, &dtmp, 
//FG					 &dtmp, CALCDER_T));
   PFModuleInvoke(void, saturation_module, (saturation, pressure, 
					    density, gravity, problem_data, 
					    CALCFCN));
   PFModuleInvoke(void, saturation_module, (saturation_der, pressure, 
					    density, gravity, problem_data,
					    CALCDER));

#if 1

      ForSubgridI(is, GridSubgrids(grid))
      {
	 subgrid = GridSubgrid(grid, is);

	 J_sub  = MatrixSubmatrix(J, is);
	 cp     = SubmatrixStencilData(J_sub, 0);
	
	 t_sub  = VectorSubvector(temperature, is);
         p_sub  = VectorSubvector(pressure, is);
	 d_sub  = VectorSubvector(density, is);
	 s_sub  = VectorSubvector(saturation, is);
	 dd_sub = VectorSubvector(density_der, is);
	 sd_sub = VectorSubvector(saturation_der, is);
	 po_sub = VectorSubvector(porosity, is);
	 hcw_sub = VectorSubvector(heat_capacity_water, is);
	 hcr_sub = VectorSubvector(heat_capacity_rock, is);
	 ss_sub = VectorSubvector(sstorage, is);
	 
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
         r = SubgridRX(subgrid);

	 nx_v   = SubvectorNX(d_sub);
	 ny_v  = SubvectorNY(d_sub);
	 nz_v  = SubvectorNZ(d_sub);

	 nx_po = SubvectorNX(po_sub);
	 ny_po = SubvectorNY(po_sub);
	 nz_po = SubvectorNZ(po_sub);

	 nx_m  = SubmatrixNX(J_sub);
	 ny_m  = SubmatrixNY(J_sub);
	 nz_m  = SubmatrixNZ(J_sub);

	 tp  = SubvectorData(t_sub);
         pp  = SubvectorData(p_sub);
	 dp  = SubvectorData(d_sub);
	 sp  = SubvectorData(s_sub);
	 ddp = SubvectorData(dd_sub);
	 pop = SubvectorData(po_sub);
	 hcwp = SubvectorData(hcw_sub);
	 hcrp = SubvectorData(hcr_sub);
	 ss = SubvectorData(ss_sub);

  //       im  = SubmatrixEltIndex(J_sub,  ix, iy, iz);
  //       ipo = SubvectorEltIndex(po_sub, ix, iy, iz);
  //       iv  = SubvectorEltIndex(d_sub,  ix, iy, iz);

  //       BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
  //                 im,  nx_m,  ny_m,  nz_m,  1, 1, 1,
  //                 ipo, nx_po, ny_po, nz_po, 1, 1, 1,
   //                iv,  nx_v,  ny_v,  nz_v,  1, 1, 1,
            GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
		   {
			   im  = SubmatrixEltIndex(J_sub,  i,j,k);
         		   ipo = SubvectorEltIndex(po_sub, i,j,k);
		           iv  = SubvectorEltIndex(d_sub,  i,j,k);
					//FGtest dp und sp =1 	
			   cp[im] += pop[ipo]* /*hcwp[iv]*/ vol* 1.0 ;//( sp[iv]*dp[iv] + tp[iv]*sp[iv]*ddp[iv] );
			 //cp[im] += (pp[iv]*sp[iv])*hcwp[iv]*(ss[iv]/gravity)*vol;
                           cp[im] += vol * (1.0 - pop[ipo])  /*hcrp[iv]*/;

		   });
      }   /* End subgrid loop */
#endif


   bc_struct = PFModuleInvoke(BCStruct *, bc_temperature, 
			      (problem_data, grid, gr_domain, time));

   /* Get boundary temperature values for Dirichlet boundaries.   */
   /* These are needed for upstream weighting in mobilities - need boundary */
   /* values for rel perms and densities. */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      t_sub = VectorSubvector(temperature, is);

      nx_v = SubvectorNX(t_sub);
      ny_v = SubvectorNY(t_sub);
      nz_v = SubvectorNZ(t_sub);
	 
      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      tp = SubvectorData(t_sub);

      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	 bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

	 switch(BCStructBCType(bc_struct, ipatch))
	 {

	 case DirichletBC:
	 {
            BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
	       ip   = SubvectorEltIndex(t_sub, i, j, k);
	       value =  bc_patch_values[ival];
	       tp[ip + fdir[0]*1 + fdir[1]*sy_v + fdir[2]*sz_v] = value;
	       
	    });
	    break;
	 }

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */

   /* Calculate thermal_cond and thermal_cond_der */
//FG

   PFModuleInvoke(void, thermal_conductivity,
                  (thermal_cond, temperature, saturation, 1.0, problem_data,
                   CALCFCN));

   PFModuleInvoke(void, thermal_conductivity,
                  (thermal_cond_der, temperature, saturation, 1.0, problem_data,
                   CALCDER));

#if 1
   /* Calculate contributions from second order derivatives and gravity */
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      t_sub    = VectorSubvector(temperature, is);
      tcp_sub   = VectorSubvector(thermal_cond, is);
      tcdp_sub  = VectorSubvector(thermal_cond_der, is);
      J_sub    = MatrixSubmatrix(J, is);
      ctc_sub  = VectorSubvector(constThermalCond, is);
      ctcp     = SubvectorData(ctc_sub);	 
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

      nx_v = SubvectorNX(t_sub);
      ny_v = SubvectorNY(t_sub);
      nz_v = SubvectorNZ(t_sub);
	 
      nx_m = SubmatrixNX(J_sub);
      ny_m = SubmatrixNY(J_sub);
      nz_m = SubmatrixNZ(J_sub);

      r = SubgridRX(subgrid);

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

      tp     = SubvectorData(t_sub);
      tcp    = SubvectorData(tcp_sub);
      tcdp   = SubvectorData(tcdp_sub);
      
      ip = SubvectorEltIndex(p_sub, ix, iy, iz);
      im = SubmatrixEltIndex(J_sub, ix, iy, iz);

//printf("##########\n");
//      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
//                ip, nx_v, ny_v, nz_v, 1, 1, 1,
//                im, nx_m, ny_m, nz_m, 1, 1, 1,
             GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	     {
                ip = SubvectorEltIndex(t_sub, i,j,k);
                im = SubmatrixEltIndex(J_sub, i,j,k);
 //printf("##FG ip %d ; tp: %e;  \n",ip,tp[ip]);              

	        prod        = tcp[ip];
		prod_der    = tcdp[ip];

	        prod_rt     = tcp[ip+1];
		prod_rt_der = tcdp[ip+1];

	        prod_no     = tcp[ip+sy_v];
		prod_no_der = tcdp[ip+sy_v];

	        prod_up     = tcp[ip+sz_v];
		prod_up_der = tcdp[ip+sz_v]; 

	        /* diff >= 0 implies flow goes left to right */
	        diff = tp[ip] - tp[ip+1];

		x_coeff = dt * ffx * (1.0/dx); 

		sym_west_temp = - x_coeff 
		                  * RPMean(tp[ip], tp[ip+1], prod, prod_rt)* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1])
		                  ;

		west_temp = - x_coeff * diff 
		                 * RPMean(tp[ip], tp[ip+1], prod_der, 0.0)* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1])
		              + sym_west_temp;

		sym_east_temp = x_coeff
		                * -RPMean(tp[ip], tp[ip+1], prod, prod_rt)* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1])
		                ;

		east_temp = x_coeff * diff
		               * RPMean(tp[ip], tp[ip+1], 0.0, prod_rt_der)* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1])
		            + sym_east_temp;





	        /* diff >= 0 implies flow goes south to north */
	        diff = tp[ip] - tp[ip+sy_v];

		y_coeff = dt * ffy * (1.0/dy); 

		sym_south_temp = - y_coeff
                                   *RPMean(tp[ip], tp[ip+sy_v], prod, prod_no)* PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
				  ;

		south_temp = - y_coeff * diff
                                  * RPMean(tp[ip], tp[ip+sy_v], prod_der, 0.0)* PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
                               + sym_south_temp;

		sym_north_temp = y_coeff
                                 * -RPMean(tp[ip], tp[ip+sy_v], prod, prod_no)* PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
                                 ;

		north_temp = y_coeff * diff
                                *  RPMean(tp[ip], tp[ip+sy_v], 0.0, 
					  prod_no_der)* PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
		             + sym_north_temp;

	        /* diff >= 0 implies flow goes lower to upper */
		lower_cond = tp[ip];
		upper_cond = tp[ip+sz_v];
		diff = lower_cond - upper_cond;

		z_coeff = dt * ffz * (1.0 / dz); 

		sym_lower_temp = - z_coeff
                                   * RPMean(lower_cond, upper_cond, prod, 
					    prod_up)*PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
				;
		
		lower_temp = - z_coeff 
		      * ( diff * RPMean(lower_cond, upper_cond, prod_der, 0.0)* PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v]) 
//FG why?
//			  + RPMean(lower_cond, upper_cond, prod, 
//				       prod_up) * PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
                          )   + sym_lower_temp;

		sym_upper_temp = z_coeff
                                 * -RPMean(lower_cond, upper_cond, prod, 
					   prod_up)* PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
                              ;

		upper_temp = z_coeff
                       * ( diff * RPMean(lower_cond, upper_cond, 0.0, 
					 prod_up_der) * PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
//FG why?
//			  + RPMean(lower_cond, upper_cond, prod, 
//				      prod_up) * PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v]) 
                              ) + sym_upper_temp;
//printf("##FG im %d ; we %e ; s %e ; l %e ; n %e ; u %e ; e %e \n",im,west_temp,south_temp,lower_temp,north_temp,upper_temp,east_temp);
	
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

		   /*
		   wp[im+1] += sym_west_temp;
		   sop[im+sy_m] += sym_south_temp;
		   lp[im+sz_m] += sym_lower_temp;
		   */
		}


	     });
   }



#endif

#if 1
   /* Calculate contributions from convection*/
   ForSubgridI(is, GridSubgrids(grid)) 
   {

      subgrid = GridSubgrid(grid, is); 

      r = SubgridRX(subgrid);
                 
      d_sub    = VectorSubvector(density, is);
      dd_sub   = VectorSubvector(density_der, is); 
      hcw_sub = VectorSubvector(heat_capacity_water, is);
      zv_sub    = VectorSubvector(z_velocity, is);
      yv_sub    = VectorSubvector(y_velocity, is);
      xv_sub    = VectorSubvector(x_velocity, is);
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
 
      nx_v = SubvectorNX(t_sub);
      ny_v = SubvectorNY(t_sub);
      nz_v = SubvectorNZ(t_sub);
 
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
 
      dp     = SubvectorData(d_sub);
      ddp    = SubvectorData(dd_sub);
      hcwp   = SubvectorData(hcw_sub);
      xvp    = SubvectorData(xv_sub);
      yvp    = SubvectorData(yv_sub);
      zvp    = SubvectorData(zv_sub);
 
      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz, 
             {
                ip = SubvectorEltIndex(d_sub, i,j,k);
                im = SubmatrixEltIndex(J_sub, i,j,k);

                prod        = dp[ip];
                prod_der    = ddp[ip]; 
                                            
                prod_rt     = dp[ip+1];
                prod_rt_der = ddp[ip+1];
                        
                prod_no     = dp[ip+sy_v];
                prod_no_der = ddp[ip+sy_v]; 
                                       
                prod_up     = dp[ip+sz_v];
                prod_up_der = ddp[ip+sz_v]; 
                 
                x_coeff = ffx * xvp[ip];   
                east_temp = x_coeff * RPMean(0.0, xvp[ip], 0.0, dp[ip]*hcwp[ip]+ddp[ip]*hcwp[ip]);
                 
                y_coeff = ffy * yvp[ip];
                north_temp = y_coeff * RPMean(0.0, yvp[ip], 0.0, dp[ip]*hcwp[ip]+ddp[ip]*hcwp[ip]);
 
                z_coeff = ffz * zvp[ip];
                upper_temp = z_coeff * RPMean(0.0, zvp[ip], 0.0, dp[ip]*hcwp[ip]+ddp[ip]*hcwp[ip]); 
 
                /*cp[im]      += dt * (east_temp + north_temp + upper_temp);
                cp[im+1]    -= east_temp;
                cp[im+sy_m] -= north_temp;
                cp[im+sz_m] -= upper_temp;*/
       
                //ep[im] += east_temp;
                //np[im] += north_temp;
                //up[im] += upper_temp;
    
                //wp[im+1] += west_temp;
                //sop[im+sy_m] += south_temp;
                //lp[im+sz_m] += lower_temp;
       
             });
   }   
       
                 
              
#endif           
                 

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
	 
	    t_sub     = VectorSubvector(temperature, is);
	    tcdp_sub   = VectorSubvector(thermal_cond_der, is);
	    tcp_sub    = VectorSubvector(thermal_cond, is);
	    J_sub     = MatrixSubmatrix(J, is);


	    dx = SubgridDX(subgrid);
	    dy = SubgridDY(subgrid);
	    dz = SubgridDZ(subgrid);
      
	    ffx = dy * dz;
	    ffy = dx * dz;
	    ffz = dx * dy;
	 
	    nx_v = SubvectorNX(t_sub);
	    ny_v = SubvectorNY(t_sub);
	    nz_v = SubvectorNZ(t_sub);
	 
	    sy_v = nx_v;
	    sz_v = ny_v * nx_v;

	    cp    = SubmatrixStencilData(J_sub, 0);
	    wp    = SubmatrixStencilData(J_sub, 1);
	    ep    = SubmatrixStencilData(J_sub, 2);
	    sop   = SubmatrixStencilData(J_sub, 3);
	    np    = SubmatrixStencilData(J_sub, 4);
	    lp    = SubmatrixStencilData(J_sub, 5);
	    up    = SubmatrixStencilData(J_sub, 6);

	    tp     = SubvectorData(t_sub);
	    tcdp   = SubvectorData(tcdp_sub);
	    tcp    = SubvectorData(tcp_sub);

	    for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
	    {
               BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  ip = SubvectorEltIndex(t_sub, i, j, k);
		  im = SubmatrixEltIndex(J_sub, i, j, k);

		  if (fdir[0])
	          {
		     switch(fdir[0])
		     {
		     case -1:
		     {
		        diff = tp[ip-1] - tp[ip];
			prod_der = tcdp[ip-1];
		        coeff = dt * ffx * (1.0/dx); 
		        wp[im] = - coeff * diff
			   * RPMean(tp[ip-1], tp[ip], prod_der, 0.0);      
			break;
		     }
		     case 1:
		     {
		        diff = tp[ip] - tp[ip+1];
			prod_der = tcdp[ip+1];
		        coeff = dt * ffx * (1.0/dx); 
		        ep[im] = coeff * diff
			   * RPMean(tp[ip], tp[ip+1], 0.0, prod_der);      
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
		        diff = tp[ip-sy_v] - tp[ip];
			prod_der = tcdp[ip-sy_v]; 
		        coeff = dt * ffy * (1.0/dy); 
		        sop[im] = - coeff * diff
			   * RPMean(tp[ip-sy_v], tp[ip], prod_der, 0.0);      
			break;
		     }
		     case 1:
		     {
		        diff = tp[ip] - tp[ip+sy_v];
			prod_der = tcdp[ip+sy_v];
		        coeff = dt * ffy * (1.0/dy); 
		        np[im] = - coeff * diff
			   * RPMean(tp[ip], tp[ip+sy_v], 0.0, prod_der);      
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
			lower_cond = (tp[ip-sz_v]); 
			upper_cond = (tp[ip] );
		        diff = lower_cond - upper_cond;
			prod_der = tcdp[ip-sz_v];
			prod_lo = tcp[ip-sz_v];
		        coeff = dt * ffz * (1.0/dz); 
		        lp[im] = - coeff * diff * RPMean(lower_cond, upper_cond, prod_der, 0.0);
                       break;
		     }
		     case 1:
		     {
			lower_cond = (tp[ip]);
			upper_cond = (tp[ip+sz_v] ); 
		        diff = lower_cond - upper_cond;
			prod_der = tcdp[ip+sz_v];
			prod_up = tcp[ip+sz_v];
		        coeff = dt * ffz * (1.0/dz); 
		        up[im] = - coeff * diff * RPMean(lower_cond, upper_cond, 0.0, prod_der);
			break;
		     }
		     }   /* End switch on fdir[2] */

		  }   /* End if (fdir[2]) */
		     
	       });   /* End Patch Loop */

	    }        /* End ipatch loop */
	 }           /* End subgrid loop */
   }                 /* End if symm_part */


#if 1   
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      t_sub     = VectorSubvector(temperature, is);
      tcdp_sub   = VectorSubvector(thermal_cond_der, is);
      tcp_sub    = VectorSubvector(thermal_cond, is);
      J_sub     = MatrixSubmatrix(J, is);
      ctc_sub  = VectorSubvector(constThermalCond, is);
      ctcp     = SubvectorData(ctc_sub);
      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
      
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      
      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;
	 
      nx_v = SubvectorNX(t_sub);
      ny_v = SubvectorNY(t_sub);
      nz_v = SubvectorNZ(t_sub);
	 
      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      cp    = SubmatrixStencilData(J_sub, 0);
      wp    = SubmatrixStencilData(J_sub, 1);
      ep    = SubmatrixStencilData(J_sub, 2);
      sop    = SubmatrixStencilData(J_sub, 3);
      np    = SubmatrixStencilData(J_sub, 4);
      lp    = SubmatrixStencilData(J_sub, 5);
      up    = SubmatrixStencilData(J_sub, 6);

      tp     = SubvectorData(t_sub);
      tcdp   = SubvectorData(tcdp_sub);
      tcp    = SubvectorData(tcp_sub);


      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	 bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

	 switch(BCStructBCType(bc_struct, ipatch))
	 {

	 case DirichletBC:
	 {
            BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
	       ip   = SubvectorEltIndex(t_sub, i, j, k);

	       value =  bc_patch_values[ival];

	       ip = SubvectorEltIndex(t_sub, i, j, k);
	       im = SubmatrixEltIndex(J_sub, i, j, k);

	       prod        = tcp[ip];
	       prod_der    = tcdp[ip];

	       if (fdir[0])
	       {
		  coeff = dt * ffx * (2.0/dx);

		  switch(fdir[0])
		  {
		  case -1:
		  {
		     op = wp;
		     prod_val = tcp[ip-1];
		     diff = value - tp[ip];
		     o_temp =  coeff 
		       * ( diff * RPMean(value, tp[ip], 0.0, prod_der)  * PMean(tp[ip-1], tp[ip], ctcp[ip-1], ctcp[ip])
			   - RPMean(value, tp[ip], prod_val, prod) * PMean(tp[ip], tp[ip+sz_v], ctcp[ip-1], ctcp[ip])
			);
		     o_new = coeff * (diff * tcdp[ip] - tcp[ip]);
		     break;
		  }
		  case 1:
		  {
//printf("##FG in\n");
		     op = ep;
		     prod_val = tcp[ip+1];
		     diff = tp[ip] - value;
		     o_temp = - coeff 
		       * ( diff * RPMean(tp[ip], value, prod_der, 0.0) * PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1]) 
			+    RPMean(tp[ip], value, prod, prod_val) * PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1])
 			);
		      o_new = - coeff * (diff * tcdp[ip] + tcp[ip]);
		     break;
		  }
		  }   /* End switch on fdir[0] */

	       }      /* End if (fdir[0]) */

	       else if (fdir[1])
	       {
		  coeff = dt * ffy * (2.0/dy);

		  switch(fdir[1])
		  {
		  case -1:
		  {
		     op = sop;
		     prod_val = tcp[ip-sy_v];
		     diff = value - tp[ip];
		     o_temp =  coeff 
		       * ( diff * RPMean(value, tp[ip], 0.0, prod_der) * PMean(tp[ip-sy_v], tp[ip], ctcp[ip-sy_v], ctcp[ip])
			   - RPMean(value, tp[ip], prod_val, prod) * PMean(tp[ip-sy_v], tp[ip], ctcp[ip-sy_v], ctcp[ip])
			);
                     o_new = coeff * (diff * tcdp[ip] - tcp[ip]);
		     break;
		  }
		  case 1:
		  {
		     op = np;
		     prod_val = tcp[ip+sy_v];
		     diff = tp[ip] - value;
		     o_temp = - coeff 
		       * ( diff * RPMean(tp[ip], value, prod_der, 0.0) * PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
			   + RPMean(tp[ip], value, prod, prod_val) * PMean(tp[ip], tp[ip+sy_v], ctcp[ip], ctcp[ip+sy_v])
			 );
                     o_new = - coeff * (diff * tcdp[ip] + tcp[ip]);
		     break;
		  }
		  }   /* End switch on fdir[1] */

	       }      /* End if (fdir[1]) */

	       else if (fdir[2])
	       {
		  coeff = dt * ffz * (2.0/dz);

		  switch(fdir[2])
		  {
		  case -1:
		  {
		     op = lp;
		     prod_val = tcp[ip-sz_v];

		     lower_cond = (value );
		     upper_cond = (tp[ip]);
		     diff = lower_cond - upper_cond;

		     o_temp =  coeff 
		      * ( diff * RPMean(lower_cond, upper_cond, 0.0, prod_der) * PMean(tp[ip-sz_v], tp[ip], ctcp[ip-sz_v], ctcp[ip])
			  + ( (-1.0)
			    * RPMean(lower_cond, upper_cond, prod_val, prod) * PMean(tp[ip-sz_v], tp[ip], ctcp[ip-sz_v], ctcp[ip])
			     ));
		     o_new = coeff * (diff * tcdp[ip] - tcp[ip]);
		  break;
		  }
		  case 1:
		  {
		     op = up;
		     prod_val = tcp[ip+sz_v] * den_d;

		     lower_cond = (tp[ip]);
		     upper_cond = (value );
		     diff = lower_cond - upper_cond;

		     o_temp = - coeff 
		      * ( diff * RPMean(lower_cond, upper_cond, prod_der, 0.0) * PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
			  + ( (1.0)
			    * RPMean(lower_cond, upper_cond, prod, prod_val)* PMean(tp[ip], tp[ip+sz_v], ctcp[ip], ctcp[ip+sz_v])
			     ));
                     o_new = - coeff * (diff * tcdp[ip] + tcp[ip]);
		     break;
		  }
		  }   /* End switch on fdir[2] */

		  }   /* End if (fdir[2]) */
//printf("##FG 1: im %d ; cp[im] %e \n",im,cp[im]);
               cp[im] -= o_temp;
//printf("##FG 2: im %d ; cp[im] %e ; o_temp %e \n",im,cp[im],o_temp);
	       cp[im] -= o_new;  //op[im];
//printf("##FG 3: im %d ; cp[im] %e ; op[im] %e \n",im,cp[im],op[im]);
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
//printf("##FG im %d ; op[im] %e \n",im,op[im]);
	       cp[im] += op[im];
	       op[im] = 0.0;

               /*Correction for contribution from convection*/ 
               if (fdir[0]) 
               { 
 
                  switch(fdir[0])
                  {  
                  case -1:
                  { 
                     break;
                  } 
                  case 1: 
                  { 
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
                     break; 
                  }
                  case 1: 
                  {
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
                  break;
                  }
                  case 1:
                  {
                     break;
                  }
                  }   /* End switch on fdir[2] */
 
                  }   /* End if (fdir[2]) */

	    });

	    break;
	 }     /* End fluxbc case */

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */
#endif


   FreeBCStruct(bc_struct);
#if 1
   PFModuleInvoke( void, bc_internal, (problem, problem_data, NULL, J, time,
				       temperature, CALCDER));
#endif
#if 1


   /* Set temperatures outside domain to zero.  
    * Recall: equation to solve is f = 0, so components of f outside 
    * domain are set to the respective temperature value.
    *
    * Should change this to set temperature to scaling value.
    * CSW: Should I set this to temperature * vol * dt ??? */

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
#endif

   /*-----------------------------------------------------------------------
    * Update matrix ghost points
    *-----------------------------------------------------------------------*/

   if (MatrixCommPkg(J))
   {
      handle = InitMatrixUpdate(J);
      FinalizeMatrixUpdate(handle);
   }

   *ptr_to_J = J;


   return;
}


/*--------------------------------------------------------------------------
 * TemperatureJacobianEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *TemperatureJacobianEvalInitInstanceXtra(
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
      instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* free old data */
      if ( (instance_xtra -> grid) != NULL )
      {
	 FreeMatrix(instance_xtra -> J);
         FreeVector(instance_xtra -> density_der);
         FreeVector(instance_xtra -> saturation_der);
         FreeVector(instance_xtra -> thermal_cond);
         FreeVector(instance_xtra -> thermal_cond_der);
      }

      /* set new data */
      (instance_xtra -> grid) = grid;

      /* set up jacobian matrix */
      stencil = NewStencil(jacobian_stencil_shape_temperature, 7);

      if (symmetric_jac)
	 (instance_xtra -> J) = NewMatrix(grid, NULL, stencil, ON, stencil);
      else
	 (instance_xtra -> J) = NewMatrix(grid, NULL, stencil, OFF, stencil);

      (instance_xtra -> density_der)     = NewVector(grid, 1, 1);
      (instance_xtra -> saturation_der)  = NewVector(grid, 1, 1);
      (instance_xtra -> thermal_cond)        = NewVector(grid, 1, 1);
      (instance_xtra -> thermal_cond_der)        = NewVector(grid, 1, 1);
   }

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;
//      SetTempVectorData((instance_xtra -> density_der), temp_data);
//      temp_data += SizeOfVector(instance_xtra -> density_der);
//      SetTempVectorData((instance_xtra -> saturation_der), temp_data);
//      temp_data += SizeOfVector(instance_xtra -> saturation_der);
   }

   if ( problem != NULL)
   {
      (instance_xtra -> problem) = problem;
   }

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> density_module) =
         PFModuleNewInstance(ProblemPhaseDensity(problem), () );
      (instance_xtra -> heat_capacity_module) =
         PFModuleNewInstance(ProblemPhaseHeatCapacity(problem), () );
      (instance_xtra -> bc_temperature) =
        PFModuleNewInstance(ProblemBCTemperature(problem), (problem) );
      (instance_xtra -> saturation_module) =
         PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL) );
      (instance_xtra -> thermal_conductivity) =
         PFModuleNewInstance(ProblemThermalConductivity(problem), (NULL) );
      (instance_xtra -> bc_internal) =
         PFModuleNewInstance(ProblemBCInternal(problem), () );

   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> density_module), ());
      PFModuleReNewInstance((instance_xtra -> heat_capacity_module), ());
      PFModuleReNewInstance((instance_xtra -> bc_temperature), (problem));
      PFModuleReNewInstance((instance_xtra -> saturation_module),(NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> thermal_conductivity),(NULL));
      PFModuleReNewInstance((instance_xtra -> bc_internal), ());

   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * TemperatureJacobianEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  TemperatureJacobianEvalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   if(instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> density_module);
      PFModuleFreeInstance(instance_xtra -> heat_capacity_module);
      PFModuleFreeInstance(instance_xtra -> bc_temperature);
      PFModuleFreeInstance(instance_xtra -> saturation_module);
      PFModuleFreeInstance(instance_xtra -> thermal_conductivity);
      PFModuleFreeInstance(instance_xtra -> bc_internal);

      FreeVector(instance_xtra -> density_der);
      FreeVector(instance_xtra -> saturation_der);
      FreeVector(instance_xtra -> thermal_cond);
      FreeVector(instance_xtra -> thermal_cond_der);
      
      FreeMatrix(instance_xtra -> J);

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TemperatureJacobianEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *TemperatureJacobianEvalNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;


   public_xtra = NULL;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * TemperatureJacobianEvalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  TemperatureJacobianEvalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);

   if (public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TemperatureJacobianEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  TemperatureJacobianEvalSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   int  sz = 0;

   /* add local TempData size to `sz' */
   sz += SizeOfVector(instance_xtra -> density_der);
   sz += SizeOfVector(instance_xtra -> saturation_der);

   return sz;
}
