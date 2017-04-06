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
   PFModule     *temp_source;
   PFModule     *bc_temperature;
   PFModule     *bc_internal;

   Vector       *therm_cond;
   Vector       *source;
   Vector       *qsource;
   Vector       *tsource;
 
   Grid         *grid;
} InstanceXtra;

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/

#define PMean(a, b, c, d)    HarmonicMean(c, d)
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)
#define AMean(a,b)           ArithmeticMean(a,b)

/*  This routine evaluates the nonlinear function based on the current 
    temperature values.  This evaluation is basically an application
    of the stencil to the temperature array. */

void    TemperatureFunctionEval( 
	N_Vector      speciesNVector,
	Vector      *fval,           /* Return values of the nonlinear function */
	void	    *current_state
)
{
   PFModule      *this_module     = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);
   Problem     *problem           = (instance_xtra -> problem);

   PFModule    *density_module       = (instance_xtra -> density_module);
   PFModule    *heat_capacity_module = (instance_xtra -> heat_capacity_module);
   PFModule    *viscosity_module     = (instance_xtra -> viscosity_module);
   PFModule    *saturation_module    = (instance_xtra -> saturation_module);
   PFModule    *thermal_conductivity = (instance_xtra -> thermal_conductivity);
   PFModule    *rel_perm_module      = (instance_xtra -> rel_perm_module);
   PFModule    *phase_source         = (instance_xtra -> phase_source);
   PFModule    *temp_source          = (instance_xtra -> temp_source);
   PFModule    *bc_temperature       = (instance_xtra -> bc_temperature);
   PFModule    *bc_internal          = (instance_xtra -> bc_internal);


   //state-vectors
   Vector *  pressure = NV_CONTENT_PF(speciesNVector)->dims[0] ;
   Vector *  temperature = NV_CONTENT_PF(speciesNVector)->dims[1] ; 
   ProblemData *problem_data   = StateProblemData( ((State*)current_state) );
   Vector      *old_pressure   = StateOldPressure(((State*)current_state) );
   Vector      *saturation     = StateSaturation(  ((State*)current_state) );
   Vector      *old_saturation = StateOldSaturation(((State*)current_state) );
   Vector      *density        = StateDensity(     ((State*)current_state) );
   Vector      *old_density    = StateOldDensity(  ((State*)current_state) );
   double       dt             = StateDt(          ((State*)current_state) );
   double       time           = StateTime(        ((State*)current_state) );
   Vector       *evap_trans    = StateEvapTrans(   ((State*)current_state) );
   Vector       *ovrl_bc_flx   = StateOvrlBcFlx(   ((State*)current_state) );
   Vector       *heat_capacity_water  = StateHeatCapacityWater(((State*)current_state) );
   Vector       *heat_capacity_rock   = StateHeatCapacityRock(((State*)current_state) );
   Vector       *viscosity              = StateViscosity(((State*)current_state) );
   Vector       *old_temperature        = StateOldTemperature(((State*)current_state) );
   Vector       *clm_energy_source      = StateClmEnergySource(((State*)current_state) );
   Vector       *forc_t                 = StateForcT(((State*)current_state) );
   Vector       *x_velocity             = StateXvel(((State*)current_state) );
   Vector       *y_velocity             = StateYvel(((State*)current_state) );
   Vector       *z_velocity             = StateZvel(((State*)current_state) );



   Vector      *therm_cond          = (instance_xtra -> therm_cond);
   Vector      *source          = (instance_xtra -> source);
   Vector      *qsource          = (instance_xtra -> qsource);
   Vector      *tsource          = (instance_xtra -> tsource);


   Vector      *porosity          = ProblemDataPorosity(problem_data);
   Vector      *permeability_x    = ProblemDataPermeabilityX(problem_data);
   Vector      *permeability_y    = ProblemDataPermeabilityY(problem_data);
   Vector      *permeability_z    = ProblemDataPermeabilityZ(problem_data);
   Vector      *sstorage          = ProblemDataSpecificStorage(problem_data);

   /* @RMM variable dz multiplier */
   Vector      *z_mult            = ProblemDataZmult(problem_data);  //@RMM
   Subvector   *z_mult_sub;   //@RMM
   double      *z_mult_dat;   //@RMM

   double       gravity           = ProblemGravity(problem);

   Subgrid     *subgrid;

   Subvector   *p_sub, *t_sub, *d_sub, *v_sub, *od_sub, *s_sub, *ot_sub, *os_sub, *po_sub, *op_sub, *ss_sub, *et_sub, *hcw_sub, *hcr_sub;
   Subvector   *f_sub, *tc_sub, *permx_sub, *permy_sub, *permz_sub;
   Subvector   *qs_sub, *ts_sub, *hs_sub, *le_sub, *ft_sub;

   Grid        *grid              = VectorGrid(temperature);

   double      *pp, *tp, *odp, *otp, *sp, *osp, *pop, *fp, *dp, *vp, *tcp, *opp, *ss, *et, *hcwp, *hcrp;
   double      *permxp, *permyp, *permzp;
   double      *tsp, *qsp, *hsp, *lep, *ftp;

   /* Fluid flow velcocities */
   Subvector   *xv_sub, *yv_sub, *zv_sub;
   double      *xvp,*yvp,*zvp;

   int          i, j, k, r, is;
   int          ix, iy, iz;
   int          nx, ny, nz,gnx,gny;
   int          nx_f, ny_f, nz_f;
   int          nx_p, ny_p, nz_p;
   int          nx_po, ny_po, nz_po;
   int          sy_p, sz_p;
   int          ip, ipo,io,ipp1;

   double       dtmp, dx, dy, dz, vol, ffx, ffy, ffz, sep;
   double       u_right, u_front, u_upper;
   double       diff = 0.0;
   double       lower_cond, upper_cond;
   
   BCStruct    *bc_struct;
   GrGeomSolid *gr_domain         = ProblemDataGrDomain(problem_data);
   double      *bc_patch_values;
   double       u_old = 0.0;
   double       u_new = 0.0;
   double       value, schng;
   int         *fdir;
   int          ipatch, ival;
   int          dir = 0;

//TODO: #FG This has to be outsourced to a module with tcl options. set to 1.0 for now 
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



   CommHandle  *handle;
   VectorUpdateCommHandle  *vector_update_handle;

   BeginTiming(public_xtra -> time_index);

   /* Initialize function values to zero. */
   PFVConstInit(0.0, fval);

   /* Pass temperature values to neighbors.  */
   vector_update_handle = InitVectorUpdate(temperature, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);
   vector_update_handle = InitVectorUpdate(pressure, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);
 
   /* Pass permeability values */
   /*
   vector_update_handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);

   vector_update_handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle);

   vector_update_handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
   FinalizeVectorUpdate(vector_update_handle); */

   /* Calculate temperatrue dependent properties: density and saturation */


//not implemented  
//PFModuleInvoke(void, density_module, (0, pressure, temperature, density, &dtmp, &dtmp, 
//					 CALCFCN));

   PFModuleInvoke(void, saturation_module, (saturation, pressure, density, 
					    gravity , problem_data, CALCFCN));



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
      p_sub  = VectorSubvector(pressure, is);
      op_sub  = VectorSubvector(old_pressure, is);
      os_sub = VectorSubvector(old_saturation, is);
      po_sub = VectorSubvector(porosity, is);
      ss_sub = VectorSubvector(sstorage, is);
      f_sub  = VectorSubvector(fval, is);
	 
      xv_sub    = VectorSubvector(x_velocity, is);
      yv_sub    = VectorSubvector(y_velocity, is);
      zv_sub    = VectorSubvector(z_velocity, is);

      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);

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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      nx_po = SubvectorNX(po_sub);
      ny_po = SubvectorNY(po_sub);
      nz_po = SubvectorNZ(po_sub);

      sy_p = nx_f;
      sz_p = ny_f * nx_f;

      dp  = SubvectorData(d_sub);
      hcwp = SubvectorData(hcw_sub);
      odp = SubvectorData(od_sub);
      sp  = SubvectorData(s_sub);
      tp = SubvectorData(t_sub);
      otp = SubvectorData(ot_sub);
      osp = SubvectorData(os_sub);
      pop = SubvectorData(po_sub);
      ss = SubvectorData(ss_sub);
      fp  = SubvectorData(f_sub);
      pp  = SubvectorData(p_sub);
      opp  = SubvectorData(op_sub);
      
      xvp = SubvectorData(xv_sub);
      yvp = SubvectorData(yv_sub);
      zvp = SubvectorData(zv_sub);


      GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	     { 
      ip  = SubvectorEltIndex(f_sub,   i,j,k);
      ipo = SubvectorEltIndex(po_sub,  i,j,k);
          //   fp[ip] = (tp[ip] - otp[ip]) * 1 /*sp[ip]*/  * 1 /*dp[ip]*/ * /* hcwp[ip]*/ 1.0 * pop[ipo] * vol;
		fp[ip] = ((tp[ip]*sp[ip]) - (otp[ip]*osp[ip])) * pop[ipo] * vol * z_mult_dat[ip] *  hcwp[ip];
    
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
	 
      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);

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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      tp = SubvectorData(t_sub);
      otp = SubvectorData(ot_sub);
      hcrp = SubvectorData(hcr_sub);
      pop = SubvectorData(po_sub);
      fp  = SubvectorData(f_sub);
      
       GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	     {
			  ip = SubvectorEltIndex(f_sub, i,j,k);
			 fp[ip] += vol * z_mult_dat[ip] * (1.0 - hcwp[ip])  * (1.0 - pop[ip]) * (tp[ip] - otp[ip]);

	     });

   }
#endif


   /* Add in contributions from source terms - user specified sources and
      flux wells.  Calculate phase source values overwriting current 
      saturation vector */

//   PFModuleInvoke(void, phase_source, (qsource, tsource, problem, problem_data,      time));
//   PFModuleInvoke(void, temp_source, (source, problem, problem_data,       time));


#if 1
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      t_sub = VectorSubvector(temperature, is);
      qs_sub  = VectorSubvector(qsource, is);
      d_sub  = VectorSubvector(density, is);
      ts_sub = VectorSubvector(tsource, is);
      hs_sub = VectorSubvector(source, is);
      le_sub = VectorSubvector(clm_energy_source, is);
      ft_sub = VectorSubvector(forc_t, is);
      hcw_sub = VectorSubvector(heat_capacity_water, is);
      et_sub = VectorSubvector(evap_trans, is);
      f_sub  = VectorSubvector(fval, is);
	 
      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);

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

      nx_f = SubvectorNX(f_sub);
      ny_f = SubvectorNY(f_sub);
      nz_f = SubvectorNZ(f_sub);
	 
      tp  = SubvectorData(t_sub);
      qsp  = SubvectorData(qs_sub);
      tsp = SubvectorData(ts_sub);
      hsp = SubvectorData(hs_sub);
      lep = SubvectorData(le_sub);
      ftp = SubvectorData(ft_sub);
      et  = SubvectorData(et_sub);
      hcwp = SubvectorData(hcw_sub);
      fp  = SubvectorData(f_sub);
      dp  = SubvectorData(d_sub);
      

              GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	     {
                ip = SubvectorEltIndex(f_sub, i,j,k);
	        //fp[ip] -= vol * dt * hsp[ip]; // energy sink source
	        //fp[ip] -= vol * dt * dp[ip] * qsp[ip] * hcwp[ip] * tsp[ip]; // mass-energy sink
	        //if (i==0 && j ==0 && k== 239) fp[ip] -= vol * dt * dp[ip] * 1.0e-6 * hcwp[ip] * (283.0 - 273.15); // mass-energy sink
//FGtest  kein CLM -> fällt weg 
	        fp[ip] -= vol *z_mult_dat[ip] * dt * 0 /* lep[ip]*/ /dz; // energy due to ET from CLM
                 
	     });
   }
#endif

   bc_struct = PFModuleInvoke(BCStruct *, bc_temperature, 
			      (problem_data, grid, gr_domain, time));


   /* Get boundary temperature values for Dirichlet boundaries.   */
   /* These are needed for upstream weighting in mobilities - need boundary */
   /* values for rel perms and densities. */

   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      t_sub   = VectorSubvector(temperature, is);

      nx_p = SubvectorNX(t_sub);
      ny_p = SubvectorNY(t_sub);
      nz_p = SubvectorNZ(t_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;


      tp = SubvectorData(t_sub);


      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
	 bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

	 switch(BCStructBCType(bc_struct, ipatch))
	 {
         printf("%d\n",ipatch);

	 case DirichletBC:
	 {
            BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
	       ip   = SubvectorEltIndex(t_sub, i, j, k);
	       value =  bc_patch_values[ival];
	       tp[ip + fdir[0]*1 + fdir[1]*sy_p + fdir[2]*sz_p] = value;
	       
	    });
	    break;
	 }

	 }     /* End switch BCtype */
      }        /* End ipatch loop */
   }           /* End subgrid loop */

   /* Calculate thermal permeability values*/

   PFModuleInvoke(void, thermal_conductivity,
                  (therm_cond, pressure, saturation, gravity, problem_data,
                   CALCFCN));


#if 1
   /* Calculate contributions from diffusion*/
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	
      t_sub     = VectorSubvector(temperature, is);
      tc_sub    = VectorSubvector(therm_cond, is);
      f_sub     = VectorSubvector(fval, is);
      ctc_sub  = VectorSubvector(constThermalCond, is);

      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);

      ix = SubgridIX(subgrid) -1;
      iy = SubgridIY(subgrid) -1;
      iz = SubgridIZ(subgrid) -1;
	 
      nx = SubgridNX(subgrid) +1;
      ny = SubgridNY(subgrid) +1;
      nz = SubgridNZ(subgrid) +1;
	 
      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);
	 
      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;

      nx_p = SubvectorNX(t_sub);
      ny_p = SubvectorNY(t_sub);
      nz_p = SubvectorNZ(t_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;
      r = SubgridRX(subgrid);

      tp    = SubvectorData(t_sub);
      tcp   = SubvectorData(tc_sub);
      fp    = SubvectorData(f_sub);
      ctcp     = SubvectorData(ctc_sub);


             GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
             {
                ip = SubvectorEltIndex(t_sub, i,j,k);

	        /* Calculate right face velocity.
		   diff >= 0 implies flow goes left to right */
	        diff    = tp[ip] - tp[ip+1];
		u_right = dt * ffx * z_mult_dat[ip]  
		              * (diff / dx )
		              * RPMean(tp[ip], tp[ip+1], tcp[ip], tcp[ip+1])
		              * PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1]);

	        /* Calculate front face velocity.
		   diff >= 0 implies flow goes back to front */
	        diff    = tp[ip] - tp[ip+sy_p];
		u_front = dt * ffy * z_mult_dat[ip]
		              * (diff / dy )
		              * RPMean(tp[ip], tp[ip+sy_p], tcp[ip],
				       tcp[ip+sy_p])* PMean(tp[ip], tp[ip+sy_p], ctcp[ip], ctcp[ip+sy_p]);
	
	
	        /* Calculate upper face velocity.
		   diff >= 0 implies flow goes lower to upper */
                sep = dz*(AMean(z_mult_dat[ip],z_mult_dat[ip+sz_p]));

		lower_cond = (tp[ip] / sep);
		upper_cond = (tp[ip+sz_p] / sep);
		diff = lower_cond - upper_cond;
		u_upper = dt * ffz 
		              * diff
		              * RPMean(lower_cond, upper_cond, tcp[ip], 
				       tcp[ip+sz_p])* PMean(tp[ip], tp[ip+sz_p], ctcp[ip], ctcp[ip+sz_p]);


	        fp[ip]      +=  u_right + u_front + u_upper;
  		fp[ip+1]    -=  u_right;
		fp[ip+sy_p] -=  u_front;
		fp[ip+sz_p] -=  u_upper;

	     });

   }

#endif

#if 1
   /* Calculate contributions from convection*/
   ForSubgridI(is, GridSubgrids(grid))
   {             
      subgrid = GridSubgrid(grid, is); 
                 
      t_sub     = VectorSubvector(temperature, is);
      d_sub     = VectorSubvector(density, is);
      f_sub     = VectorSubvector(fval, is); 
      hcw_sub   = VectorSubvector(heat_capacity_water, is);
      xv_sub    = VectorSubvector(x_velocity, is);
      yv_sub    = VectorSubvector(y_velocity, is);
      zv_sub    = VectorSubvector(z_velocity, is);
      po_sub    = VectorSubvector(porosity, is);
      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);                                  
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
                 
      nx_p = SubvectorNX(t_sub); 
      ny_p = SubvectorNY(t_sub); 
      nz_p = SubvectorNZ(t_sub);
 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;
 
      tp    = SubvectorData(t_sub);
      dp    = SubvectorData(d_sub);
      fp    = SubvectorData(f_sub);
      hcwp  = SubvectorData(hcw_sub);
      pop   = SubvectorData(po_sub);
 
      xvp = SubvectorData(xv_sub);
      yvp = SubvectorData(yv_sub);
      zvp = SubvectorData(zv_sub);
      r = SubgridRX(subgrid);

             GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
             {
                ip = SubvectorEltIndex(t_sub, i,j,k);
                /* Calculate right face temperature at i+1/2.
                   diff >= 0 implies flow goes left to right */
                u_right = ffx* z_mult_dat[ip] * RPMean(0.0, xvp[ip], hcwp[ip+1]*(tp[ip+1]),      hcwp[ip]*(tp[ip])) * xvp[ip]*dt; 
               
                /* Calculate front face velocity.
                   diff >= 0 implies flow goes back to front */
                u_front = ffy* z_mult_dat[ip] * RPMean(0.0, yvp[ip], hcwp[ip+sy_p]*(tp[ip+sy_p]), hcwp[ip]*(tp[ip])) * yvp[ip]*dt;
                /* Calculate upper face velocity.
                   diff >= 0 implies flow goes lower to upper */
                u_upper = ffz * RPMean(0.0, zvp[ip], hcwp[ip+sz_p]*(tp[ip+sz_p]), hcwp[ip]*(tp[ip])) * zvp[ip]*dt;

                fp[ip]      +=  u_upper + u_front + u_right ;

                fp[ip+1]    -=  u_right;
                fp[ip+sy_p] -=  u_front;
                fp[ip+sz_p] -=  u_upper;
                
             });
   }
 
#endif



   /*  Calculate correction for boundary conditions */
#if 1   
   ForSubgridI(is, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, is);
	 
      tc_sub    = VectorSubvector(therm_cond, is);
      f_sub     = VectorSubvector(fval, is);

      t_sub     = VectorSubvector(temperature, is);
      d_sub     = VectorSubvector(density, is);
      ot_sub = VectorSubvector(temperature, is);

      hcw_sub   = VectorSubvector(heat_capacity_water, is);
      xv_sub    = VectorSubvector(x_velocity, is);
      yv_sub    = VectorSubvector(y_velocity, is);
      zv_sub    = VectorSubvector(z_velocity, is);
      po_sub    = VectorSubvector(porosity, is);
      ctc_sub  = VectorSubvector(constThermalCond, is);
      ctcp     = SubvectorData(ctc_sub);
      /* @RMM added to provide access to zmult */
      z_mult_sub = VectorSubvector(z_mult, is);
      /* @RMM added to provide variable dz */
      z_mult_dat = SubvectorData(z_mult_sub);

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
	 
      nx_p = SubvectorNX(t_sub);
      ny_p = SubvectorNY(t_sub);
      nz_p = SubvectorNZ(t_sub);
	 
      sy_p = nx_p;
      sz_p = ny_p * nx_p;

      tcp    = SubvectorData(tc_sub);
      fp     = SubvectorData(f_sub);

      tp = SubvectorData(t_sub);
      dp = SubvectorData(d_sub);
      otp = SubvectorData(ot_sub);
      pop = SubvectorData(po_sub);
      
      hcwp    = SubvectorData(hcw_sub);
      xvp = SubvectorData(xv_sub);
      yvp = SubvectorData(yv_sub);
      zvp = SubvectorData(zv_sub);

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

	       /* Don't currently do upstream weighting on boundaries */

	       if (fdir[0])
	       {
		  switch(fdir[0])
		  {
		  case -1:
		     dir = -1;
   		     diff  = tp[ip-1] - tp[ip];
		     u_old = dt*ffx*z_mult_dat[ip] * (diff / dx ) * RPMean(tp[ip-1], tp[ip], tcp[ip-1], tcp[ip])* PMean(tp[ip-1], tp[ip], ctcp[ip-1], ctcp[ip]); 
                     u_old+= ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip-1], hcwp[ip]*(tp[ip]),hcwp[ip-1]*(tp[ip-1])) * xvp[ip-1]*dt;
 
                     diff = value - tp[ip];
		     u_new = dt*ffx*z_mult_dat[ip] * 2.0 * (diff/dx) * tcp[ip];
                     u_new+= ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip-1], hcwp[ip]*(tp[ip]),hcwp[ip]*(value)) * xvp[ip-1]*dt;
		     break;
		  case  1:
		     dir = 1;
   		     diff  = tp[ip] - tp[ip+1];
		     u_old = dt*ffx*z_mult_dat[ip] * (diff / dx ) * RPMean(tp[ip], tp[ip+1], tcp[ip], tcp[ip+1])* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1]); 
                     u_old+=ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip], hcwp[ip+1]*(tp[ip+1]), hcwp[ip]*(tp[ip])) *xvp[ip]*dt;

		     diff = tp[ip] - value;
		     u_new = dt*ffx*z_mult_dat[ip] * 2.0 * (diff/dx) * tcp[ip];
                     u_new+= ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip], hcwp[ip]*(value), hcwp[ip]*(tp[ip])) * xvp[ip]*dt;
		     break;
		  }
	       }
	       else if (fdir[1])
	       {
		  switch(fdir[1])
		  {
		  case -1:
		     dir = -1;
   		     diff  = tp[ip-sy_p] - tp[ip];
		     u_old = dt*ffy*z_mult_dat[ip] * (diff / dy )* RPMean(tp[ip-sy_p], tp[ip], tcp[ip-sy_p], tcp[ip])* PMean(tp[ip-sy_p], tp[ip], ctcp[ip-sy_p], ctcp[ip]); 
                     u_old+= ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip-sy_p], hcwp[ip]*(tp[ip]),hcwp[ip-sy_p]*(tp[ip-sy_p])) * yvp[ip-sy_p]*dt;

		     diff =  value - tp[ip];
		     u_new = dt*ffy*z_mult_dat[ip] * 2.0 * (diff / dy) * tcp[ip];
                     u_new+= ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip-sy_p], hcwp[ip]*(tp[ip]),hcwp[ip]*(value)) * yvp[ip-sy_p]*dt;
		     break;
		  case  1:
		     dir = 1;
   		     diff  = tp[ip] - tp[ip+sy_p];
		     u_old = dt*ffy*z_mult_dat[ip] * (diff / dy ) * RPMean(tp[ip], tp[ip+sy_p], tcp[ip], tcp[ip+sy_p])* PMean(tp[ip], tp[ip+sy_p], ctcp[ip], ctcp[ip+sy_p]);
                     u_old+=ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip], hcwp[ip+sy_p]*(tp[ip+sy_p]), hcwp[ip]*(tp[ip])) * yvp[ip]*dt;

		     diff = tp[ip] - value;
		     u_new = dt*ffy*z_mult_dat[ip] * 2.0 *(diff/dy) * tcp[ip];
                     u_new+= ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip], hcwp[ip]*(value), hcwp[ip]*(tp[ip])) * yvp[ip]*dt;
		     break;
		  }
	       }
	       else if (fdir[2])
	       {
		  switch(fdir[2])
		  {
		  case -1:
		     {
		     dir = -1;
		     sep = dz*(AMean(z_mult_dat[ip],z_mult_dat[ip+sz_p]));
		     diff = tp[ip-sz_p] - tp[ip];
		     u_old = dt*ffz * (diff/sep) * RPMean(tp[ip-sz_p], tp[ip], tcp[ip-sz_p], tcp[ip])* PMean(tp[ip-sz_p], tp[ip], ctcp[ip-sz_p], ctcp[ip]); 
                     u_old+= ffz * RPMean(0.0,  zvp[ip-sz_p], hcwp[ip]*(tp[ip]),hcwp[ip-sz_p]*(tp[ip-sz_p])) * zvp[ip-sz_p]*dt;

		     diff = value - tp[ip];
		     u_new =dt* ffz * 2.0 * (diff/sep) * tcp[ip];
                     u_new+= ffz * RPMean(0.0,  zvp[ip-sz_p], hcwp[ip]*(tp[ip]),hcwp[ip]*(value)) * zvp[ip-sz_p]*dt;
		     break;
		     }   /* End case -1 */
		  case  1:
		     {
                     sep = dz*(AMean(z_mult_dat[ip],z_mult_dat[ip+sz_p]));
		     dir = 1;
		     diff = tp[ip] - tp[ip+sz_p];
		     u_old =dt* ffz * (diff/sep) * RPMean(tp[ip], tp[ip+sz_p], tcp[ip], tcp[ip+sz_p])* PMean(tp[ip], tp[ip+sz_p], ctcp[ip], ctcp[ip+sz_p]);
                     u_old+=ffz * RPMean(0.0, zvp[ip], hcwp[ip+sz_p]*(tp[ip+sz_p]), hcwp[ip]*(tp[ip])) *  zvp[ip]*dt;

		     diff = tp[ip] - value;
		     u_new =dt* ffz * 2.0 * (diff/sep) * tcp[ip];
                     u_new+= ffz * RPMean(0.0, zvp[ip], hcwp[ip]*(value), hcwp[ip]*(tp[ip])) *  zvp[ip]*dt;
		     break;
		     }   /* End case 1 */
		  }
	       }
	       /* Remove the boundary term computed above */
	       fp[ip] -=  dir * u_old;

	       /* Add the correct boundary term */
	       fp[ip] +=  dir * u_new;
	    });

	    break;
	 }

	 case FluxBC:
	 {
            BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	    {
               ip   = SubvectorEltIndex(t_sub, i, j, k);
	       if (fdir[0])
	       {
		  switch(fdir[0])
		  {
		  case -1:
		     dir = -1;
   		     diff  = tp[ip-1] - tp[ip];
		     u_old = dt*ffx*z_mult_dat[ip] 
		       * (diff / dx )
		       * RPMean(tp[ip-1], tp[ip], 
				tcp[ip-1], tcp[ip])* PMean(tp[ip-1], tp[ip], ctcp[ip-1], ctcp[ip]); 
                     u_old+= ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip-1], hcwp[ip]*(tp[ip]),hcwp[ip-1]*(tp[ip-1])) * xvp[ip-1]*dt;

                     u_new  = ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip-1], hcwp[ip]*(tp[ip]),hcwp[ip]*(bc_patch_values[ival]*dx/2) / tcp[ip] + tp[ip]) * xvp[ip-1]*dt;
                     u_new += dt*ffx*z_mult_dat[ip] *  bc_patch_values[ival];
		     break;
		  case  1:
		     dir = 1;
   		     diff  = tp[ip] - tp[ip+1];
		     u_old = dt*ffx*z_mult_dat[ip] 
		       * (diff / dx )
		       * RPMean(tp[ip], tp[ip+1], 
				tcp[ip], tcp[ip+1])* PMean(tp[ip], tp[ip+1], ctcp[ip], ctcp[ip+1]); 
                     u_old+=ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip], hcwp[ip+1]*(tp[ip+1]), hcwp[ip]*(tp[ip])) * xvp[ip]*dt;

                     u_new = ffx*z_mult_dat[ip] * RPMean(0.0, xvp[ip], hcwp[ip]*(bc_patch_values[ival]*dx/2) / tcp[ip] + tp[ip], hcwp[ip]*tp[ip]) * xvp[ip]*dt;
                     u_new += dt*ffx * bc_patch_values[ival];
		     break;
		  }
	       }
	       else if (fdir[1])
	       {
		  switch(fdir[1])
		  {
		  case -1:
		     dir = -1;
   		     diff  = tp[ip-sy_p] - tp[ip];
		     u_old = dt*ffy*z_mult_dat[ip] 
		       * (diff / dy )
		       * RPMean(tp[ip-sy_p], tp[ip], 
				tcp[ip-sy_p], tcp[ip])* PMean(tp[ip-sy_p], tp[ip], ctcp[ip-sy_p], ctcp[ip]); 
                     u_old+= ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip-sy_p], hcwp[ip]*(tp[ip]),hcwp[ip-sy_p]*(tp[ip-sy_p])) * yvp[ip-sy_p]*dt;
                     u_new  = ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip-sy_p], hcwp[ip]*(tp[ip]),hcwp[ip]*(bc_patch_values[ival]*dy/2) / tcp[ip] + tp[ip]) * yvp[ip-sy_p]*dt;

		     u_new += dt*ffy*z_mult_dat[ip] * bc_patch_values[ival];
		     break;
		  case  1:
		     dir = 1;
   		     diff  = tp[ip] - tp[ip+sy_p];
		     u_old = dt*ffy*z_mult_dat[ip] 
		       * (diff / dy )
		       * RPMean(tp[ip], tp[ip+sy_p], 
				tcp[ip], tcp[ip+sy_p])* PMean(tp[ip], tp[ip+sy_p], ctcp[ip], ctcp[ip+sy_p]);
                     u_old+=ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip], hcwp[ip+sy_p]*(tp[ip+sy_p]), hcwp[ip]*(tp[ip])) * yvp[ip]*dt;
                     u_new = ffy*z_mult_dat[ip] * RPMean(0.0, yvp[ip], hcwp[ip]*(bc_patch_values[ival]*dy/2) / tcp[ip] + tp[ip], hcwp[ip]*tp[ip]) * yvp[ip]*dt;

                     u_new += dt* ffy*z_mult_dat[ip] *  bc_patch_values[ival];
		     break;
		  }
	       }
	       else if (fdir[2])
	       {
		  switch(fdir[2])
		  {
		  case -1:
		     dir = -1;
		     sep = dz*(AMean(z_mult_dat[ip],z_mult_dat[ip+sz_p]));
		     lower_cond = (tp[ip-sz_p] / sep); 
		     upper_cond = (tp[ip] / sep);
		     diff = lower_cond - upper_cond;
		     
		     u_old = dt*ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond, 
				tcp[ip-sz_p], tcp[ip])* PMean(tp[ip-sz_p], tp[ip], ctcp[ip-sz_p], ctcp[ip]); 
                     u_old+= ffz * RPMean(0.0, zvp[ip-sz_p], hcwp[ip]*(tp[ip]),hcwp[ip-sz_p]*(tp[ip-sz_p])) * zvp[ip-sz_p]*dt;
                     u_new  = ffz * RPMean(0.0, zvp[ip-sz_p], hcwp[ip]*(tp[ip]),hcwp[ip]*(bc_patch_values[ival]*dz/2) / tcp[ip] + tp[ip]) * zvp[ip-sz_p]*dt;

	             u_new += dt*ffz * bc_patch_values[ival];
		     break;
		  case  1:
		     dir = 1;
                     sep = dz*(AMean(z_mult_dat[ip],z_mult_dat[ip+sz_p]));
		     lower_cond = (tp[ip] / sep);
		     upper_cond = (tp[ip+sz_p] / sep);
		     diff = lower_cond - upper_cond;
		     u_old = dt*ffz 
		       * diff
		       * RPMean(lower_cond, upper_cond,
				tcp[ip], tcp[ip+sz_p])* PMean(tp[ip], tp[ip+sz_p], ctcp[ip], ctcp[ip+sz_p]);
                     u_old+= ffz * RPMean(0.0, zvp[ip], hcwp[ip+sz_p]*(tp[ip+sz_p]), hcwp[ip]*(tp[ip])) * zvp[ip]*dt;
                     u_new = ffz * RPMean(0.0, zvp[ip], hcwp[ip]*(bc_patch_values[ival]*dz/2) / tcp[ip] + tp[ip], hcwp[ip]*tp[ip]) * zvp[ip]*dt;

		     u_new +=dt* ffz * bc_patch_values[ival];
		     break;
		  }
	       }
	       /* Remove the boundary term computed above */
               fp[ip] -=  dir * u_old;
	       /* Add the correct boundary term */
	       fp[ip] +=  dir * u_new;
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
	 
      t_sub = VectorSubvector(temperature, is);
      f_sub = VectorSubvector(fval, is);

      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);
	 
      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);
	 
      tp = SubvectorData(t_sub);
      fp = SubvectorData(f_sub);

      GrGeomOutLoop(i, j, k, gr_domain,
		    r, ix, iy, iz, nx, ny, nz,
		    {
		      ip   = SubvectorEltIndex(f_sub, i, j, k);
		      fp[ip] = 0.0;
		    });
   }
#endif
#if 0 
  
   /* Debugging Loop  */ 
  
   ForSubgridI(is, GridSubgrids(grid))
   {          
      subgrid = GridSubgrid(grid, is);
       
      t_sub = VectorSubvector(temperature, is);
      ot_sub = VectorSubvector(old_temperature, is);
      d_sub = VectorSubvector(density, is); 
      v_sub = VectorSubvector(viscosity, is); 
      xv_sub = VectorSubvector(x_velocity, is); 
      yv_sub = VectorSubvector(y_velocity, is); 
      zv_sub = VectorSubvector(z_velocity, is); 
      f_sub = VectorSubvector(fval, is); 
       
      /* RDF: assumes resolutions are the same in all 3 directions */
      r = SubgridRX(subgrid);
 
      ix = SubgridIX(subgrid)-1;
      iy = SubgridIY(subgrid)-1;
      iz = SubgridIZ(subgrid)-1;
 
      nx = SubgridNX(subgrid)+1;
      ny = SubgridNY(subgrid)+1;
      nz = SubgridNZ(subgrid)+1;
 
      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);
 
      tp = SubvectorData(t_sub);
      otp = SubvectorData(ot_sub);
      dp = SubvectorData(d_sub);
      vp = SubvectorData(v_sub);
      xvp = SubvectorData(xv_sub);
      yvp = SubvectorData(yv_sub);
      zvp = SubvectorData(zv_sub);
      fp = SubvectorData(f_sub);
 
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                ip, nx_p, ny_p, nz_p, 1, 1, 1,
             {
                      ip   = SubvectorEltIndex(f_sub, i, j, k);
                      printf("#FG %d %d %d %d %e %e %e %e \n",ip,i,j,k,fp[ip],xvp[ip],yvp[ip],zvp[ip]);
             });
   }
#endif
 

   EndTiming(public_xtra -> time_index);
   return;
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *TemperatureFunctionEvalInitInstanceXtra(
Problem     *problem,
Grid        *grid,
double      *temp_data)
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
	instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   if ( grid != NULL )
   {   
      /* free old data */  
      if ( (instance_xtra -> grid) != NULL )
      {   
         FreeVector(instance_xtra -> therm_cond);
         FreeVector(instance_xtra -> source);
         FreeVector(instance_xtra -> qsource);
         FreeVector(instance_xtra -> tsource);
      }
       
      /* set new data */ 
      (instance_xtra -> grid) = grid;
       
      (instance_xtra -> therm_cond)        = NewVector(grid, 1, 1);
       InitVectorAll(instance_xtra -> therm_cond, 0.0);
      (instance_xtra -> source)        = NewVector(grid, 1, 1);
       InitVectorAll(instance_xtra -> source, 0.0);
      (instance_xtra -> qsource)        = NewVector(grid, 1, 1);
       InitVectorAll(instance_xtra -> qsource, 0.0);
      (instance_xtra -> tsource)        = NewVector(grid, 1, 1);
       InitVectorAll(instance_xtra -> tsource, 0.0);
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
      (instance_xtra -> saturation_module) =
         PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL) );
      (instance_xtra -> thermal_conductivity) =
         PFModuleNewInstance(ProblemThermalConductivity(problem), (NULL) );
      (instance_xtra -> rel_perm_module) =
         PFModuleNewInstance(ProblemPhaseRelPerm(problem), (NULL, NULL) );
      (instance_xtra -> phase_source) =
         PFModuleNewInstance(ProblemPhaseSource(problem), (grid));
      (instance_xtra -> temp_source) =
         PFModuleNewInstance(ProblemTempSource(problem), (grid));
      (instance_xtra -> bc_temperature) =
         PFModuleNewInstance(ProblemBCTemperature(problem), (problem));
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
      PFModuleReNewInstance((instance_xtra -> temp_source), (NULL));
      PFModuleReNewInstance((instance_xtra -> bc_temperature), (problem));
      PFModuleReNewInstance((instance_xtra -> bc_internal), ());
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  TemperatureFunctionEvalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
   if(instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> density_module);
      PFModuleFreeInstance(instance_xtra -> saturation_module);
    //  PFModuleFreeInstance(instance_xtra -> thermal_conductivity);
      PFModuleFreeInstance(instance_xtra -> rel_perm_module);
      PFModuleFreeInstance(instance_xtra -> phase_source);
      PFModuleFreeInstance(instance_xtra -> temp_source);
      PFModuleFreeInstance(instance_xtra -> bc_temperature);
      PFModuleFreeInstance(instance_xtra -> bc_internal);
      
      FreeVector(instance_xtra -> therm_cond);
      FreeVector(instance_xtra -> source);
      FreeVector(instance_xtra -> qsource);
      FreeVector(instance_xtra -> tsource);

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *TemperatureFunctionEvalNewPublicXtra()
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

void  TemperatureFunctionEvalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);

   if (public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * TempFunctionEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  TemperatureFunctionEvalSizeOfTempData()
{
   return 0;
}



