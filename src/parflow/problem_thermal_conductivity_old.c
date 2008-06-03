/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *  This module computes thermal conductivities.  
 *
 *
 *  The thermal conductivity  module can be invoked either expecting only a 
 *  double array of densities back - where NULL Vectors are 
 *  sent in for the phase saturation and the thermal conductivity return Vector - or a 
 *  Vector of densities at each grid block.  Note that code using the
 *  Vector thermal conductivity option can also have a constant thermal conductivity.
 *  This "overloading" was provided so that thethermal conductivity module written
 *  for the Richards' solver modules would be backward compatible with
 *  the Impes modules and so that densities can be evaluated for saturations
 *  not necessarily associated with a grid (as in boundary patches).
 *
 *****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int     num_phases;

   int    *type; /* array of size num_phases of input types */
   void  **data; /* array of size num_phases of pointers to Type structures */

} PublicXtra;

typedef void InstanceXtra;

typedef struct
{
   double  constant;
} Type0;

typedef struct
{
   double  conductivity_dry;
   double  conductivity_cwet;
} Type1;                      /*  */


/*-------------------------------------------------------------------------
 * PhaseDensity
 *-------------------------------------------------------------------------*/

void    ThermalConductivity(problem_data,

Vector *phase_saturation;  /* Vector of phase saturations at each block */
Vector *thermconduct_v;       /* Vector of return htermal conductivities at each block */
double *saturation_d;      /* Double array of saturations */
double *thermconduct_d;       /* Double array return thermal conductivity*/
int     fcn;             /* Flag determining what to calculate 
                          * fcn = CALCFCN => calculate the function value
			  * fcn = CALCDER => calculate the function 
			  *                  derivative */

/*  Module returns either a double array or Vector of densities.
 *  If thermconduct_v is NULL, then a double array is returned. 
 *  This "overloading" was provided so that the htermal conductivity module written
 *  for the Richards' solver modules would be backward compatible with
 *  the Impes modules.
 */
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Type0         *dummy0;
   Type1         *dummy1;

   Grid          *grid;

   Subvector     *p_sub;
   Subvector     *d_sub;

   double        *pp; 
   double        *dp; 

   Subgrid       *subgrid;

   int            sg;

   int            ix,   iy,   iz;
   int            nx,   ny,   nz;
   int            nx_p, ny_p, nz_p;
   int            nx_d, ny_d, nz_d;

   int            i, j, k, ip, id;


   switch((public_xtra -> type[phase]))
   {

   case 0:
   {
      int num_regions;
      int *region_indices;
      double *values;

      GrGeomSolid *gr_solid;
      double  constant;
      int ir;

      dummy0 = (Type0 *)(public_xtra -> data[phase]);
      num_regions = (dummy0 -> num_regions);
      region_indices = (dummy0 -> region_indices);
      constant = (dummy0 -> values);

      for (ir = 0; ir < num_regions; ir++)
      {
         gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
         constant = values[ir];

	 ForSubgridI(sg, GridSubgrids(grid))
	 {
	    subgrid = GridSubgrid(grid, sg);

	    d_sub = VectorSubvector(thermconduct_v,  sg);

	    ix = SubgridIX(subgrid);
	    iy = SubgridIY(subgrid);
	    iz = SubgridIZ(subgrid);

	    nx = SubgridNX(subgrid);
	    ny = SubgridNY(subgrid);
	    nz = SubgridNZ(subgrid);

	    nx_d = SubvectorNX(d_sub);
	    ny_d = SubvectorNY(d_sub);
	    nz_d = SubvectorNZ(d_sub);
         
            r = SubgridRX(subgrid);

            data = SubvectorData(ps_sub);
            GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
              ips = SubvectorEltIndex(ps, i, j, k);
              data[ips] = constant;
	 }   /* End subgrid loop */
      break;
   }         /* End case 0 */

   case 1:
   {
      int num_regions;
      int *region_indices;
      double *values;

      GrGeomSolid *gr_solid;
      double  cwet, cdry;
      int ir;

      dummy1 = (Type1 *)(public_xtra -> data);
      wet_values = (dummy1 -> conductivity_wet);
      dry_values = (dummy1 -> conductivity_dry);


      for (ir = 0; ir < num_regions; ir++)
      {
        gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
        cwet = wet_values[ir];
        cdry = dry_values[ir];

        ForSubgridI(sg, GridSubgrids(grid))
	{
          subgrid = GridSubgrid(grid, sg);

          wet_sub = VectorSubvector(conductivity_wet, sg);
	  dry_sub = VectorSubvector(conductivity_dry, sg);

          ix = SubgridIX(subgrid);
          iy = SubgridIY(subgrid);
          iz = SubgridIZ(subgrid);

          nx = SubgridNX(subgrid);
          ny = SubgridNY(subgrid);
          nz = SubgridNZ(subgrid);

          r = SubgridRX(subgrid);

          cwet = SubvectorData(wet_sub);
          cdry = SubvectorData(dry_sub);

          GrGeomInLoop(i, j, k, gr_solid, r, ix, iy,iz, nx, ny, nz,
          {
            ips = SubvectorEltIndex(ps_sub, i, j, k);
           
            cwet
		 if ( fcn == CALCFCN )
		 {
		    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
			      ip, nx_p, ny_p, nz_p, 1, 1, 1,
			      id, nx_d, ny_d, nz_d, 1, 1, 1,
			      {
			        dp[id] = cdry + pp[ip] * (cwet - cdry);
			      });
		 }
		 else   /* fcn = CALCDER */
		 {
		    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
			      ip, nx_p, ny_p, nz_p, 1, 1, 1,
			      id, nx_d, ny_d, nz_d, 1, 1, 1,
			      {
		                dp[id] = 0.0;
			      });
		 }
	      }
	}
      else
	{
	   if ( fcn == CALCFCN )
	   {
	      (*thermconduct_d) =  cdry + (*saturation_d) * (cwet - cdry);
	   }
	   else
	   {
	      (*thermconduct_d) =  0.0;
	   }
	}

      break;
   }         /* End case 1 */

   }         /* End switch */
}

/*--------------------------------------------------------------------------
 * PhaseDensityInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseDensityInitInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

#if 0
   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);
#endif
   instance_xtra = NULL;

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * PhaseDensityFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PhaseThermalConductivityFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      tfree(instance_xtra);
   }

}


/*--------------------------------------------------------------------------
 * PhaseDensityNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *PhaseThermalConductivityNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0            *dummy0;
   Type1            *dummy1;

   char          *switch_name;
   NameArray       switch_na;
   char          key[IDB_MAX_KEY_LEN];

   int               i;

   /*----------------------------------------------------------
    * The name array to map names to switch values 
    *----------------------------------------------------------*/
   switch_na = NA_NewNameArray("Constant EquationOfState1");

   public_xtra = ctalloc(PublicXtra, 1);

      switch_name = GetString("Phase.ThermalConductivity.Type");

      public_xtra -> type = NA_NameToIndex(switch_na, switch_name);

      switch((public_xtra -> type))
      {
	 case 0:
	 {  
            int num_regions, ir;

	    dummy0 = ctalloc(Type0, 1);
	    
            switch_name = GetString("ThermalConductivity.GeoNames");

            dummy0 -> regions = NA_NewNameArray(switch_name);

            dummy0 -> num_regions = NA_SizeOf(regions);

            num_regions = (dummy0 -> num_regions);

            (dummy0 -> region_indices) = ctalloc(int,    num_regions);
            (dummy0 -> values)         = ctalloc(double, num_regions);
           
            for (ir = 0; ir < num_regions; ir++)
            {
              region = NA_IndexToName(dummy0 -> regions, ir);
             
              dummy0 -> region_indices[ir] = 
                  NA_NameToIndex(GlobalsGeomNames, region);

	      sprintf(key, "Geom.%s.ThermalConductivity.Value", region);
	      dummy0 -> values[ir] = GetDouble(key);
	    }

	    (public_xtra -> data) = (void *) dummy0;
	    
	    break;
	 }
	 
	 case 1:
	 {
            int num_regions, ir; 
            
	    dummy1 = ctalloc(Type1, 1);

            switch_name = Getstring("ThermalConductivity.GeoNames");

            dummy1 -> regions = NA_NewNameArray(switch_name);

            dummy -> num_regions = NA_SizeOf(regions);

            num_regions = (dummy1 -> num_regions);

            (dummy1 -> region_indices) = ctalloc(int, num_regions);
            (dummy1 -> values)         = ctalloc(double, num_regions);
 
            for (ir = 0; irr < num_regions; ir++
            {
              region = NA_IndexToName(dummy1 -> regions, ir);

              dummy1 -> region_indices[ir] =
                  NA_NameToIndex(GlobalsGeoNames, region);
 
	    sprintf(key, "Geom.%s.ThermalConductivity.Wet", region);
	    dummy1 -> conductivity_wet = GetDouble(key);

	    sprintf(key, "Phase.%s.ThermalConductivity.Dry", region);
	    dummy1 -> conductivity_dry = GetDouble(key);
	    
	    (public_xtra -> data) = (void *) dummy1;
	    
	    break;
	 }

	 default:
	 {
	    InputError("Error: invalid type <%s> for key <%s>\n",
		       switch_name, key);
	 }
      }

   NA_FreeNameArray(switch_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*-------------------------------------------------------------------------
 * PhaseDensityFreePublicXtra
 *-------------------------------------------------------------------------*/

void  ThermalConductivityFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0        *dummy0;
   Type1        *dummy1;

   int           i;

   if ( public_xtra )
   {
      for (i = 0; i < (public_xtra -> num_phases); i++)
      {
         switch((public_xtra -> type[i]))
	 {
	 case 0:
	    dummy0 = (Type0 *)(public_xtra -> data[i]);
	    tfree(dummy0);
	    break;
	 case 1:
	    dummy1 = (Type1 *)(public_xtra -> data[i]);
	    tfree(dummy1);
            break;
         }
      }

      tfree(public_xtra -> data);
      tfree(public_xtra -> type);

      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * PhaseDensitySizeOfTempData
 *--------------------------------------------------------------------------*/

int  ThermalConductivitySizeOfTempData()
{
   return 0;
}
