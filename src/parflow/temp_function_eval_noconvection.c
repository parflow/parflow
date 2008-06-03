/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"



/*---------------------------------------------------------------------
 * Define module structures
 *---------------------------------------------------------------------*/

typedef struct
{
   int       time_index;

} PublicXtra;

typedef struct
{
   Problem      *problem;

   PFModule     *density_module;
   PFModule     *heat_capacity_module;
   PFModule     *viscosity_module;
   PFModule     *saturation_module;
   PFModule     *rel_perm_module;
   PFModule     *thermal_conductivity;
   PFModule     *phase_source;
   PFModule     *bc_pressure;
   PFModule     *bc_internal;
} InstanceXtra;

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/

#define PMean(a, b, c, d)    HarmonicMean(c, d)
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*  This routine evaluates the nonlinear function based on the current 
    temperature values.  This evaluation is basically an application
    of the stencil to the temperature array. */

void    TempFunctionEval(temperature, fval, problem_data, pressure, saturation, 
		       old_saturation, density, old_density, heat_capacity_water, heat_capacity_rock, dt, 
                       time, old_temperature, evap_trans, x_velocity, y_velocity, z_velocity)

Vector      *temperature;       /* Current temperature values */
Vector      *fval;           /* Return values of the nonlinear function */
ProblemData *problem_data;   /* Geometry data for problem */
Vector      *pressure;
Vector      *saturation;     /* Saturation / work vector */
Vector      *old_saturation; /* Saturation values at previous time step */
Vector      *density;        /* Density vector */
Vector      *old_density;    /* Density values at previous time step */
Vector      *heat_capacity_water;
Vector      *heat_capacity_rock;
double       dt;             /* Time step size */
double       time;           /* New time value */
Vector      *old_temperature;
Vector      *evap_trans;     /*sk sink term from land surface model*/
Vector     *x_velocity;
Vector     *y_velocity;
Vector     *z_velocity;

{
   PFModule      *this_module     = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);
   PublicXtra    *public_xtra     = PFModulePublicXtra(this_module);

   Problem     *problem           = (instance_xtra -> problem);

   PFModule    *density_module       = (instance_xtra -> density_module);
   PFModule    *heat_capacity_module = (instance_xtra -> heat_capacity_module);
   PFModule    *viscosity_module     = (instance_xtra -> viscosity_module);
   PFModule    *saturation_module    = (instance_xtra -> saturation_module);
   PFModule    *thermal_conductivity = (instance_xtra -> thermal_conductivity);
   PFModule    *rel_perm_module      = (instance_xtra -> rel_perm_module);
   PFModule    *phase_source         = (instance_xtra -> phase_source);
   PFModule    *bc_pressure          = (instance_xtra -> bc_pressure);
   PFModule    *bc_internal          = (instance_xtra -> bc_internal);

   /* Re-use saturation vector to save memory */
   Vector      *rel_perm          = saturation;
   Vector      *source            = saturation;

 //  double      press[12][12][10],pressbc[12][12],xslope[12][12],yslope[12][12];
//   double      press[2][1][390],pressbc[400][1],xslope[58][30],yslope[58][30];

   Vector      *porosity          = ProblemDataPorosity(problem_data);
   Vector      *permeability_x    = ProblemDataPermeabilityX(problem_data);
   Vector      *permeability_y    = ProblemDataPermeabilityY(problem_data);
   Vector      *permeability_z    = ProblemDataPermeabilityZ(problem_data);

   double       gravity           = 0.0;
   //double       gravity           = ProblemGravity(problem);

   Subgrid     *subgrid;

   Subvector   *p_sub, *t_sub, *d_sub, *v_sub, *od_sub, *s_sub, *ot_sub, *os_sub, *po_sub, *op_sub, *ss_sub, *et_sub, *hcw_sub, *hcr_sub;

   Subvector   *f_sub, *rp_sub, *permx_sub, *permy_sub, *permz_sub;

   Grid        *grid              = VectorGrid(temperature);

   double      *pp, *tp, *odp, *otp, *sp, *osp, *pop, *fp, *dp, *vp, *rpp, *opp, *ss, *et, *hcwp, *hcrp;
   double      *permxp, *permyp, *permzp;

   int          i, j, k, r, is;
   int          ix, iy, iz;
   int          nx, ny, nz,gnx,gny;
   int          nx_f, ny_f, nz_f;
   int          nx_p, ny_p, nz_p;
   int          nx_po, ny_po, nz_po;
   int          sy_p, sz_p;
   int          ip, ipo,io;

   int          water = 0;
   int          rock  = 1;

   double       dtmp, dx, dy, dz, vol, ffx, ffy, ffz;
   double       u_right, u_front, u_upper;
   double       diff = 0.0e0;
   double       lower_cond, upper_cond;
   
   BCStruct    *bc_struct;
   GrGeomSolid *gr_domain         = ProblemDataGrDomain(problem_data);
   double      *bc_patch_values;
   double       u_old = 0.0e0;
   double       u_new = 0.0e0;
   double       value;
   int         *fdir;
   int          ipatch, ival;
   int          dir = 0;
   
   CommHandle  *handle;

   BeginTiming(public_xtra -> time_index);

   /* Initialize function values to zero. */
   PFVConstInit(0.0, fval);

   /* Pass temperature values to neighbors.  */
   handle = InitVectorUpdate(temperature, VectorUpdateAll);
   FinalizeVectorUpdate(handle);
 

   /* Pass permeability values */
   /*
   handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
   FinalizeVectorUpdate(handle);

   handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
   FinalizeVectorUpdate(handle);

   handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
   FinalizeVectorUpdate(handle); */

   /* Calculate temperatrue dependent properties: density and saturation */

   PFModuleInvoke(void, density_module, (0, temperature, density, &dtmp, &dtmp, 
					 CALCFCN));

   PFModuleInvoke(void, saturation_module, (saturation, pressure, density, 
					    1.0 , problem_data, CALCFCN));

   /* bc_struct = PFModuleInvoke(BCStruct *, bc_pressure, 
			      (problem_data, grid, gr_domain, time));*/

   /*@ Why are the above things calculated here again; they were allready
       calculated in the driver solver_richards and passed further @*/

#if 0
   printf("Check 1 - before accumulation term.\n");
  fflush(NULL);
   malloc_verify(NULL);
#endif

#if 1

   /* Calculate accumulation terms for the function values */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      d_sub  = VectorSubvector(density, is);
      hcw_sub = VectorSubvector(heat_capacity_water, is);
      od_sub = VectorSubvector(old_density, is);
      t_sub = VectorSubvector(temperature, is);
      ot_sub = VectorSubvector(old_temperature, is);
      s_sub  = VectorSubvector(saturation, is);
      os_sub = VectorSubvector(old_saturation, is);
      po_sub = VectorSubvector(porosity, is);
      f_sub  = VectorSubvector(fval, is);
	 
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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      nx_po = SubvectorNX(po_sub);
      ny_po = SubvectorNY(po_sub);
      nz_po = SubvectorNZ(po_sub);

      dp  = SubvectorData(d_sub);
      hcwp = SubvectorData(hcw_sub);
      odp = SubvectorData(od_sub);
      sp  = SubvectorData(s_sub);
      tp = SubvectorData(t_sub);
      otp = SubvectorData(ot_sub);
      osp = SubvectorData(os_sub);
      pop = SubvectorData(po_sub);
      fp  = SubvectorData(f_sub);
      
      ip  = SubvectorEltIndex(f_sub,   ix, iy, iz);
      ipo = SubvectorEltIndex(po_sub,  ix, iy, iz);

      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
		ip, nx_f, ny_f, nz_f, 1, 1, 1,
		ipo, nx_po, ny_po, nz_po, 1, 1, 1,
	     { 
             fp[ip] = (tp[ip]*dp[ip]*sp[ip] - otp[ip]*odp[ip]*osp[ip])*hcwp[ip]*pop[ipo]*vol;;
             
	     });
   }
#endif

   /*@ Add in contributions from rock*/

#if 1

      ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      t_sub = VectorSubvector(temperature, is);
      ot_sub = VectorSubvector(old_temperature, is);
      hcr_sub = VectorSubvector(heat_capacity_rock, is);
      po_sub = VectorSubvector(porosity, is);
      f_sub  = VectorSubvector(fval, is);
	 
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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      tp = SubvectorData(t_sub);
      otp = SubvectorData(ot_sub);
      hcrp = SubvectorData(hcr_sub);
      pop = SubvectorData(po_sub);
      fp  = SubvectorData(f_sub);
      
      ip = SubvectorEltIndex(f_sub, ix, iy, iz);
 
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		ip, nx_f, ny_f, nz_f, 1, 1, 1,
	     {
			 fp[ip] += vol * hcrp[ip] * (1.0 - pop[ip]) * (tp[ip] - otp[ip]);
	     });
   }
#endif


   /* Add in contributions from source terms - user specified sources and
      flux wells.  Calculate phase source values overwriting current 
      saturation vector */
#if 1
   PFModuleInvoke(void, phase_source, (source, problem, problem_data,
				       time));

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      s_sub  = VectorSubvector(source, is);
      f_sub  = VectorSubvector(fval, is);
      et_sub = VectorSubvector(evap_trans, is);
	 
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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      sp = SubvectorData(s_sub);
      fp = SubvectorData(f_sub);
      et = SubvectorData(et_sub);
      
      ip = SubvectorEltIndex(f_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		ip, nx_f, ny_f, nz_f, 1, 1, 1,
	     {
	        fp[ip] -= vol * dt * sp[ip];
	     });
   }
#endif

   bc_struct = PFModuleInvoke(BCStruct *, bc_pressure, 
			      (problem_data, grid, gr_domain, time));


   /* Get boundary temperature values for Dirichlet boundaries.   */
   /* These are needed for upstream weighting in mobilities - need boundary */
   /* values for rel perms and densities. */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      p_sub   = VectorSubvector(temperature, is);

      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;

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
	       pp[ip + fdir[0]*1 + fdir[1]*sy_p + fdir[2]*sz_p] = value;
	       
	    });
	    break;
	 }

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */

   /* Calculate relative permeability values overwriting current 
      phase source values */

   /*PFModuleInvoke(void, rel_perm_module, 
		  (rel_perm, temperature, density, 1.0, problem_data, 
		   CALCFCN));*/

   PFModuleInvoke(void, thermal_conductivity,
                  (rel_perm, pressure, saturation, gravity, problem_data,
                   CALCFCN));


#if 1
   /* Calculate contributions from second order derivatives and gravity */
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      p_sub     = VectorSubvector(temperature, is);
      d_sub     = VectorSubvector(density, is);
      rp_sub    = VectorSubvector(rel_perm, is);
      f_sub     = VectorSubvector(fval, is);
      permx_sub = VectorSubvector(permeability_x, is);
      permy_sub = VectorSubvector(permeability_y, is);
      permz_sub = VectorSubvector(permeability_z, is);
	 
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

      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;

      pp    = SubvectorData(p_sub);
      dp    = SubvectorData(d_sub);
      rpp   = SubvectorData(rp_sub);
      fp    = SubvectorData(f_sub);
      permxp = SubvectorData(permx_sub);
      permyp = SubvectorData(permy_sub);
      permzp = SubvectorData(permz_sub);
      
      ip = SubvectorEltIndex(p_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		ip, nx_p, ny_p, nz_p, 1, 1, 1,
	     {
//	   if(k==0) printf("i, j, t, io, qy %d %d %d %d %e\n",i,j,t,io,pp[ip]);
	        /* Calculate right face velocity.
		   diff >= 0 implies flow goes left to right */
	        diff    = pp[ip] - pp[ip+1];
		u_right = ffx 
		              * (diff / dx )
		              * RPMean(pp[ip], pp[ip+1], rpp[ip]*dp[ip],
				       rpp[ip+1]*dp[ip+1]);

	        /* Calculate front face velocity.
		   diff >= 0 implies flow goes back to front */
	        diff    = pp[ip] - pp[ip+sy_p];
		u_front = ffy 
		              * (diff / dy )
		              * RPMean(pp[ip], pp[ip+sy_p], rpp[ip]*dp[ip],
				       rpp[ip+sy_p]*dp[ip+sy_p]);
		
	        /* Calculate upper face velocity.
		   diff >= 0 implies flow goes lower to upper */
		lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
		upper_cond = (pp[ip+sz_p] / dz) + 0.5 * dp[ip+sz_p] * gravity;
		diff = lower_cond - upper_cond;
		u_upper = ffz 
		              * diff
		              * RPMean(lower_cond, upper_cond, rpp[ip]*dp[ip], 
				       rpp[ip+sz_p]*dp[ip+sz_p]);

	        fp[ip]      += dt * ( u_right + u_front + u_upper );
		fp[ip+1]    -= dt * u_right;
		fp[ip+sy_p] -= dt * u_front;
		fp[ip+sz_p] -= dt * u_upper;
		/*
		if ((k == 0) && (i == 7) && (j == 0)) 
 		  printf("Update stencil contribution: fp[ip] %12.8f \n" 
 		  "  u_upper %14.10f u_right %14.10f u_front %14.10f\n"
 		  "  pp[ip] %12.8f \n"
		  "  pp[ip+1] %12.8f pp[ip+sy_p] %12.8f pp[ip+sz_p] %12.8f\n" 
		  "  pp[ip-1] %12.8f pp[ip-sy_p] %12.8f pp[ip-sz_p] %12.8f\n" 
 		  "   Rel perms:  ip  %f ip+1 %f ip+sy_p %f ip+sz_p %f \n"
		  "   Densities:  ip  %f ip+1 %f ip+sy_p %f ip+sz_p %f \n", 
 		  fp[ip], u_upper, u_right, u_front,  
 		  pp[ip], pp[ip+1], pp[ip+sy_p], pp[ip+sz_p],
 		  pp[ip-1], pp[ip-sy_p], pp[ip-sz_p],
 		  rpp[ip], rpp[ip+1], rpp[ip+sy_p], rpp[ip+sz_p],
		  dp[ip], dp[ip+1], dp[ip+sy_p], dp[ip+sz_p]);
		  */
	     });
   }

#endif

   /*  Calculate correction for boundary conditions */
#if 1   
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      d_sub     = VectorSubvector(density, is);
      rp_sub    = VectorSubvector(rel_perm, is);
      f_sub     = VectorSubvector(fval, is);
      permx_sub = VectorSubvector(permeability_x, is);
      permy_sub = VectorSubvector(permeability_y, is);
      permz_sub = VectorSubvector(permeability_z, is);

      p_sub     = VectorSubvector(temperature, is);
      op_sub = VectorSubvector(temperature, is);
      os_sub = VectorSubvector(old_saturation, is);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
      
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      
      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid); 
      
      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;

	  vol = dx * dy * dz;
	 
      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;

      dp     = SubvectorData(d_sub);
      rpp    = SubvectorData(rp_sub);
      fp     = SubvectorData(f_sub);
      permxp = SubvectorData(permx_sub);
      permyp = SubvectorData(permy_sub);
      permzp = SubvectorData(permz_sub);

      pp = SubvectorData(p_sub);
      opp = SubvectorData(op_sub);
      osp = SubvectorData(os_sub);

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

	       /* Don't currently do upstream weighting on boundaries */

	       if (fdir[0])
	       {
		  switch(fdir[0])
		  {
		  case -1:
		     dir = -1;
   		     diff  = pp[ip-1] - pp[ip];
		     u_old = ffx 
		       * (diff / dx )
		       * RPMean(pp[ip-1], pp[ip], 
				rpp[ip-1]*dp[ip-1], rpp[ip]*dp[ip]); 
		     diff = value - pp[ip];
		     u_new = RPMean(value, pp[ip], 
				    rpp[ip-1]*dp[ip-1], rpp[ip]*dp[ip]);
		     break;
		  case  1:
		     dir = 1;
   		     diff  = pp[ip] - pp[ip+1];
		     u_old = ffx 
		       * (diff / dx )
		       * RPMean(pp[ip], pp[ip+1],
				rpp[ip]*dp[ip], rpp[ip+1]*dp[ip+1]); 
		     diff = pp[ip] - value;
		     u_new = RPMean(pp[ip], value,
				    rpp[ip]*dp[ip], rpp[ip+1]*dp[ip+1]);
		     break;
		  }
		  u_new = u_new * ffx  
		               * 2.0 * (diff/dx);
		  /*
	       if ((k == 0) && (i == 7) && (j == 0))
		 printf("Right BC u_new %12.8f u_old %12.8f value %12.8f "
			"dir %i RPVal %f\n",
			u_new, u_old, value, dir,(RPMean(pp[ip], value,
				    rpp[ip]*dp[ip], rpp[ip+1]*dp[ip+1])));
				    */
	       }
	       else if (fdir[1])
	       {
		  switch(fdir[1])
		  {
		  case -1:
		     dir = -1;
   		     diff  = pp[ip-sy_p] - pp[ip];
		     u_old = ffy 
		       * (diff / dy )
		       * RPMean(pp[ip-sy_p], pp[ip], 
				rpp[ip-sy_p]*dp[ip-sy_p], rpp[ip]*dp[ip]); 
		     diff =  value - pp[ip];
		     u_new = RPMean(value, pp[ip], 
				    rpp[ip-sy_p]*dp[ip-sy_p], rpp[ip]*dp[ip]);
		     break;
		  case  1:
		     dir = 1;
   		     diff  = pp[ip] - pp[ip+sy_p];
		     u_old = ffy 
		       * (diff / dy )
		       * RPMean(pp[ip], pp[ip+sy_p], 
				rpp[ip]*dp[ip], rpp[ip+sy_p]*dp[ip+sy_p]);
		     diff = pp[ip] - value;
		     u_new = RPMean(pp[ip], value,
				    rpp[ip]*dp[ip], rpp[ip+sy_p]*dp[ip+sy_p]);
		     break;
		  }
		  u_new = u_new * ffy 
                               * 2.0 * (diff/dy);
		  /*
	       if ((k == 0) && (i == 0) && (j == 0))
		 printf("Front BC u_new %12.8f u_old %12.8f value %12.8f "
			"dir %i\n",
			u_new, u_old, value, dir);
			*/
	       }
	       else if (fdir[2])
	       {
		  switch(fdir[2])
		  {
		  case -1:
		     {
		     dir = -1;
		     lower_cond = (pp[ip-sz_p] / dz) 
                                    - 0.5 * dp[ip-sz_p] * gravity;
		     upper_cond = (pp[ip] / dz) + 0.5 * dp[ip] * gravity;
		     diff = lower_cond - upper_cond;

		     u_old = ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond, 
				rpp[ip-sz_p]*dp[ip-sz_p], rpp[ip]*dp[ip]); 

		     lower_cond = (value / dz) - 0.25 * dp[ip] * gravity;
		     upper_cond = (pp[ip] / dz) + 0.25 * dp[ip] * gravity;
		     diff = lower_cond - upper_cond;
		     u_new = RPMean(lower_cond, upper_cond, 
				    rpp[ip-sz_p]*dp[ip-sz_p], rpp[ip]*dp[ip]);
		     break;
		     }   /* End case -1 */
		  case  1:
		     {
		     dir = 1;
		     lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
		     upper_cond = (pp[ip+sz_p] / dz) 
		                    - 0.5 * dp[ip+sz_p] * gravity;
		     diff = lower_cond - upper_cond;
		     u_old = ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond, 
				rpp[ip]*dp[ip], rpp[ip+sz_p]*dp[ip+sz_p]);
		     lower_cond = (pp[ip] / dz) - 0.25 * dp[ip] * gravity;
		     upper_cond = (value / dz) + 0.25 * dp[ip] * gravity;
		     diff = lower_cond - upper_cond;
		     u_new = RPMean(lower_cond, upper_cond,
				    rpp[ip]*dp[ip], rpp[ip+sz_p]*dp[ip+sz_p]);
		     break;
		     }   /* End case 1 */
		  }
		  u_new = u_new * ffz 
                               * 2.0 * diff;
		  /*
	       if ((k == 25) && (i==1) && (j == 0) )
		 printf("Upper BC u_new %12.8f u_old %12.8f value %12.8f\n"
			"   rpp[ip] %12.8f rpp[ip+sz_p] %12.8f "
                        "dp[ip] %12.8f\n"
			"   dp[ip+sz_p] %12.8f diff %12.8f permp[ip] %12.8f\n",
			u_new, u_old, value, rpp[ip], rpp[ip+sz_p], 
			dp[ip], dp[ip+sz_p], diff, permp[ip]);
			*/
	       }
	       /*
	       if ( (k == 25) && (i == 1) && (j == 0))
		  printf("f before BC additions: %12.8f \n", fp[ip]);
		  */

	       /* Remove the boundary term computed above */
	       fp[ip] -= dt * dir * u_old;

	       /* Add the correct boundary term */
	       fp[ip] += dt * dir * u_new;
	       /*
	       if ( (k == 25) && (i == 1) && (j == 0))
		  printf("f after BC additions: %12.8f \n\n",
			 fp[ip]);
			 */
	    });

	    break;
	 }

	 case FluxBC:
	 {
            BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
               ip   = SubvectorEltIndex(p_sub, i, j, k);

	       if (fdir[0])
	       {
		  switch(fdir[0])
		  {
		  case -1:
		     dir = -1;
   		     diff  = pp[ip-1] - pp[ip];
		     u_old = ffx 
		       * (diff / dx )
		       * RPMean(pp[ip-1], pp[ip], 
				rpp[ip-1]*dp[ip-1], rpp[ip]*dp[ip]); 
		     break;
		  case  1:
		     dir = 1;
   		     diff  = pp[ip] - pp[ip+1];
		     u_old = ffx 
		       * (diff / dx )
		       * RPMean(pp[ip], pp[ip+1], 
				rpp[ip]*dp[ip], rpp[ip+1]*dp[ip+1]); 
		     break;
		  }
		  u_new = ffx;
	       }
	       else if (fdir[1])
	       {
		  switch(fdir[1])
		  {
		  case -1:
		     dir = -1;
   		     diff  = pp[ip-sy_p] - pp[ip];
		     u_old = ffy 
		       * (diff / dy )
		       * RPMean(pp[ip-sy_p], pp[ip], 
				rpp[ip-sy_p]*dp[ip-sy_p], rpp[ip]*dp[ip]); 
		     break;
		  case  1:
		     dir = 1;
   		     diff  = pp[ip] - pp[ip+sy_p];
		     u_old = ffy 
		       * (diff / dy )
		       * RPMean(pp[ip], pp[ip+sy_p], 
				rpp[ip]*dp[ip], rpp[ip+sy_p]*dp[ip+sy_p]);
		     break;
		  }
		  u_new = ffy;
	       }
	       else if (fdir[2])
	       {
		  switch(fdir[2])
		  {
		  case -1:
		     dir = -1;
		     lower_cond = (pp[ip-sz_p] / dz) 
		                    - 0.5 * dp[ip-sz_p] * gravity;
		     upper_cond = (pp[ip] / dz) + 0.5 * dp[ip] * gravity;
		     diff = lower_cond - upper_cond;
		     u_old = ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond, 
				rpp[ip-sz_p]*dp[ip-sz_p], rpp[ip]*dp[ip]); 
		     break;
		  case  1:
		     dir = 1;
		     lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
		     upper_cond = (pp[ip+sz_p] / dz)
		                    + 0.5 * dp[ip+sz_p] * gravity;
		     diff = lower_cond - upper_cond;
		     u_old = ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond,
				rpp[ip]*dp[ip], rpp[ip+sz_p]*dp[ip+sz_p]);
		     break;
		  }
		  u_new = ffz;
	       }

	       /* Remove the boundary term computed above */
	       fp[ip] -= dt * dir * u_old;
	       /*
	       if ((fdir[2] > 0) && (i == 0) && (j == 0))
		  printf("f before flux BC additions: %12.8f \n", fp[ip]);
		  */
	       /* Add the correct boundary term */
	       u_new = u_new * bc_patch_values[ival];
	       fp[ip] += dt * dir * u_new;
	       /*
	       if ((fdir[2] < 0) && (i == 0) && (j == 0))
		  printf("f after flux BC additions: %12.8f \n\n",
			 fp[ip]);
			 */
	    });

	    break;
	 }     /* End fluxbc case */


	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */



#endif
   FreeBCStruct(bc_struct);
#if 1
   PFModuleInvoke( void, bc_internal, (problem, problem_data, fval, NULL, 
				       time, temperature, CALCFCN));
#endif

#if 1

   /* Set temperatures outside domain to zero.  
    * Recall: equation to solve is f = 0, so components of f outside 
    * domain are set to the respective temperature value.
    *
    * Should change this to set temperatures to scaling value.
    * CSW: Should I set this to temperature * vol * dt ??? */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      p_sub = VectorSubvector(temperature, is);
      f_sub = VectorSubvector(fval, is);

      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
	 
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
	 
      pp = SubvectorData(p_sub);
      fp = SubvectorData(f_sub);

      GrGeomOutLoop(i, j, k, gr_domain,
		    r, ix, iy, iz, nx, ny, nz,
		    {
		      ip   = SubvectorEltIndex(f_sub, i, j, k);
		      fp[ip] = pp[ip];
		    });
   }
#endif

   EndTiming(public_xtra -> time_index);

   return;
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *TempFunctionEvalInitInstanceXtra(problem, grid, temp_data)
Problem     *problem;
Grid        *grid;
double      *temp_data;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

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
      (instance_xtra -> saturation_module) =
         PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL) );
      (instance_xtra -> thermal_conductivity) =
         PFModuleNewInstance(ProblemThermalConductivity(problem), (NULL) );
      (instance_xtra -> rel_perm_module) =
         PFModuleNewInstance(ProblemPhaseRelPerm(problem), (NULL, NULL) );
      (instance_xtra -> phase_source) =
         PFModuleNewInstance(ProblemPhaseSource(problem), (grid));
      (instance_xtra -> bc_pressure) =
         PFModuleNewInstance(ProblemBCPressure(problem), (problem));
      (instance_xtra -> bc_internal) =
         PFModuleNewInstance(ProblemBCInternal(problem), () );

   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> density_module), ());
      PFModuleReNewInstance((instance_xtra -> heat_capacity_module), ());
      PFModuleReNewInstance((instance_xtra -> saturation_module),(NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> thermal_conductivity), (NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> rel_perm_module),(NULL, NULL));
      PFModuleReNewInstance((instance_xtra -> phase_source), (NULL));
      PFModuleReNewInstance((instance_xtra -> bc_pressure), (problem));
      PFModuleReNewInstance((instance_xtra -> bc_internal), ());
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  TempFunctionEvalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if(instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> density_module);
      PFModuleFreeInstance(instance_xtra -> saturation_module);
    //  PFModuleFreeInstance(instance_xtra -> thermal_conductivity);
      PFModuleFreeInstance(instance_xtra -> rel_perm_module);
      PFModuleFreeInstance(instance_xtra -> phase_source);
      PFModuleFreeInstance(instance_xtra -> bc_pressure);
      PFModuleFreeInstance(instance_xtra -> bc_internal);
      
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *TempFunctionEvalNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;


   public_xtra = ctalloc(PublicXtra, 1);

   (public_xtra -> time_index) = RegisterTiming("NL_F_Eval");

   PFModulePublicXtra(this_module) = public_xtra;

   return this_module;
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  TempFunctionEvalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if (public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  TempFunctionEvalSizeOfTempData()
{
   return 0;
}



