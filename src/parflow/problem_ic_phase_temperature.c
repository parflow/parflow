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
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{

   NameArray regions; /* "NameArray" is defined in "input_database.h" and contains info given below 
					  typedef struct NameArray__ 
                      {
                           int num;
                           char **names;
                           char *tok_string;
                           char *string;

                      } NameArrayStruct; */

   int      type;
   void    *data;

} PublicXtra;

typedef struct
{

   Grid     *grid;

   double   *temp_data;

} InstanceXtra;

typedef struct
{
   int      num_regions;
   int     *region_indices;
   double  *values;

} Type0;                       /* constant regions */

typedef struct
{
   int      num_regions;
   int     *region_indices;
   double  *reference_elevations;
   double  *temperature_values;
                               
} Type1;                       /* linear regions with a single
				* reference depth for each region */
				
typedef struct
{
   int      num_regions;
   int     *region_indices;

   /* for the reference patch */
   int     *geom_indices;
   int     *patch_indices;

   double  *temperature_values;
                               
} Type2;                       /* linear regions with a
				* reference patch for each region */
				
typedef struct
{
   char    *filename;

   Vector  *ic_values;

} Type3;                      /* Spatially varying field over entire domain
                                 read from a file */



/*--------------------------------------------------------------------------
 * ICPhaseTemperature:
 *    This routine returns a Vector of temperatures at the initial time.
 *--------------------------------------------------------------------------*/

void         ICPhaseTemperature(ic_temperature, problem_data, problem)

Vector      *ic_temperature;  /* Return values of intial condition */
ProblemData *problem_data; /* Contains geometry information for the problem */
Problem     *problem;      /* General problem information */

{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Grid          *grid = VectorGrid(ic_temperature);

   GrGeomSolid   *gr_solid, *gr_domain;

   Type0         *dummy0;
   Type1         *dummy1;
   Type2         *dummy2;
   Type3         *dummy3;

   SubgridArray  *subgrids = GridSubgrids(grid);

   Subgrid       *subgrid;

   Subvector     *ps_sub;
   Subvector     *tf_sub;
   Subvector     *tnd_sub;
   Subvector     *tndd_sub;
   Subvector     *ic_values_sub;

   double        *data;
   double        *fcn_data;
   double        *psdat, *ic_values_dat;

   double         gravity = -ProblemGravity(problem);

   int	          num_regions;
   int           *region_indices;

   int            ix, iy, iz;
   int            nx, ny, nz;
   int            r;

   int            is, i, j, k, ips, iel, ipicv;


   /*-----------------------------------------------------------------------
    * Initial temperature conditions
    *-----------------------------------------------------------------------*/

   InitVector(ic_temperature, 0.0);

   switch((public_xtra -> type))
   {
   case 0:  /* Assign constant values within regions. */
   {
      double  *values;
      int      ir;

      dummy0 = (Type0 *)(public_xtra -> data);

      num_regions    = (dummy0 -> num_regions);
      region_indices = (dummy0 -> region_indices);
      values         = (dummy0 -> values);

      for (ir = 0; ir < num_regions; ir++)
      {
	 gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);

	 ForSubgridI(is, subgrids)
	 {
            subgrid = SubgridArraySubgrid(subgrids, is);
            ps_sub  = VectorSubvector(ic_temperature, is);
	    
	    ix = SubgridIX(subgrid)-1;
	    iy = SubgridIY(subgrid)-1;
	    iz = SubgridIZ(subgrid)-1;
	    
	    nx = SubgridNX(subgrid)+2;
	    ny = SubgridNY(subgrid)+2;
	    nz = SubgridNZ(subgrid)+2;
	    
	    /* RDF: assume resolution is the same in all 3 directions */
	    r = SubgridRX(subgrid);
	    
	    data = SubvectorData(ps_sub);
	    GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
	       ips = SubvectorEltIndex(ps_sub, i, j, k); /*#define SubvectorEltIndex(subvector, x, y, z) \
														(((x) - SubvectorIX(subvector)) + \
														(((y) - SubvectorIY(subvector)) + \
														(((z) - SubvectorIZ(subvector))) * \
															SubvectorNY(subvector)) * \
															SubvectorNX(subvector))*/

	       data[ips] = values[ir];
	    });
	 }     /* End of subgrid loop */
      }        /* End of region loop */
      break;
   }           /* End of case 0 */

   case 1:  /* Linear regions relative to a single reference depth. */
   {

      /* Calculate linear conditions within region for 
	 elevations different from reference elevation.  
	 Linear condition is:
	       grad p - rho g grad z = 0 */

      int      max_its = 10;
      int      iterations;

      double  *reference_elevations;
      double  *temperature_values;
      double   z;
      int      ir;

      dummy1 = (Type1 *)(public_xtra -> data);

      num_regions          = (dummy1 -> num_regions);
      region_indices       = (dummy1 -> region_indices);
      reference_elevations = (dummy1 -> reference_elevations);
      temperature_values      = (dummy1 -> temperature_values);

         /* Get new temperature values. */
         for (ir = 0; ir < num_regions; ir++)
         {
	    gr_solid    = ProblemDataGrSolid(problem_data, region_indices[ir]);

	    ForSubgridI(is, subgrids)
	    {
               subgrid             = SubgridArraySubgrid(subgrids, is);
	       ps_sub              = VectorSubvector(ic_temperature, is);
	       data    	           = SubvectorData(ps_sub);
	    
	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);
	    
	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);
	    
	       /* RDF: assume resolution is the same in all 3 directions */
	       r = SubgridRX(subgrid);

		  GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	          {
	             ips = SubvectorEltIndex(ps_sub, i, j, k);
		     z = RealSpaceZ(k, SubgridRZ(subgrid));
		     data[ips] = data[ips] - (z-reference_elevations[ir]); 
		  });
	    }     /* End of subgrid loop */
	 }        /* End of region loop */
	 

      break;
   }           /* End of case 1 */

   case 2:  /* Linear regions with a reference surface for each region */
   {

      /* Calculate linear conditions within region for 
         elevations at reference surface.  
         Linear condition is:
               grad p - rho g grad z = 0 */

      GeomSolid *ref_solid;
	    
      int      *patch_indices;
      int      *geom_indices;
      double   *temperature_values;
      double   *ref_den;
      double    z;
      double  ***elevations;
      int       ir;

      dummy2 = (Type2 *)(public_xtra -> data);

      num_regions     = (dummy2 -> num_regions);
      region_indices  = (dummy2 -> region_indices);
      geom_indices    = (dummy2 -> geom_indices);
      patch_indices   = (dummy2 -> patch_indices);
      temperature_values = (dummy2 -> temperature_values);

      elevations      = ctalloc(double**, num_regions);


      /* Calculate array of elevations on reference surface. */
      for (ir = 0; ir < num_regions; ir++)
      {
	 ref_solid = ProblemDataSolid(problem_data, geom_indices[ir]);
	 elevations[ir] = CalcElevations(ref_solid, patch_indices[ir], 
					 subgrids);

      }        /* End of region loop */

         for (ir = 0; ir < num_regions; ir++)
         {
	    gr_solid    = ProblemDataGrSolid(problem_data, region_indices[ir]);

	    ForSubgridI(is, subgrids)
	    {
               subgrid             = SubgridArraySubgrid(subgrids, is);
	       ps_sub              = VectorSubvector(ic_temperature, is);
	       data    	           = SubvectorData(ps_sub);
	    
	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);
	    
	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);
	    
	       /* RDF: assume resolution is the same in all 3 directions */
	       r = SubgridRX(subgrid);

		  GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
	          {
	             ips = SubvectorEltIndex(ps_sub, i, j, k);
                     iel = (i-ix) + (j-iy)*nx;
		     data[ips] = temperature_values[ir] -  ( RealSpaceZ(k, SubgridRZ(subgrid)) -  elevations[ir][is][iel] );
		  });
	    }     /* End of subgrid loop */
	 }        /* End of region loop */
	 

      for (ir = 0; ir < num_regions; ir++)
      {
	 ForSubgridI(is, subgrids)
	 {
	    tfree(elevations[ir][is]);
	 }
	 tfree(elevations[ir]);
      }
      tfree(elevations);

      break;
   }           /* End of case 2 */
   case 3:  /* ParFlow binary file with spatially varying temperature values */
   {
      Vector *ic_values;
      
      dummy3 = (Type3 *)(public_xtra -> data);

      ic_values = dummy3 -> ic_values;

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
	 subgrid = SubgridArraySubgrid(subgrids, is);
 
	 ps_sub = VectorSubvector(ic_temperature, is);
	 ic_values_sub = VectorSubvector(ic_values, is);

	 ix = SubgridIX(subgrid);
	 iy = SubgridIY(subgrid);
	 iz = SubgridIZ(subgrid);

	 nx = SubgridNX(subgrid);
	 ny = SubgridNY(subgrid);
	 nz = SubgridNZ(subgrid);

	 /* RDF: assume resolution is the same in all 3 directions */
	 r = SubgridRX(subgrid);

	 psdat = SubvectorData(ps_sub);
	 ic_values_dat = SubvectorData(ic_values_sub);

	 GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	 {
	    ips = SubvectorEltIndex(ps_sub, i, j, k);
	    ipicv = SubvectorEltIndex(ic_values_sub, i, j, k);

	    psdat[ips] = ic_values_dat[ipicv];
	 });
      }        /* End subgrid loop */
      break;
   }           /* End case 3 */
   }           /* End of switch statement */
}


/*--------------------------------------------------------------------------
 * ICPhaseTemperatureInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *ICPhaseTemperatureInitInstanceXtra(problem, grid, temp_data)

Problem   *problem;
Grid      *grid;
double    *temp_data;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   Type3         *dummy3;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if ( grid != NULL )
   {
      /* set new data */
      (instance_xtra -> grid) = grid;

      /* Uses a spatially varying field */
      if (public_xtra -> type == 3)  
      {
	 dummy3 = (Type3 *)(public_xtra -> data);
         (dummy3 -> ic_values) = NewTempVector(grid, 1, 1);
      }

   }

   if ( temp_data != NULL )
   {
      /* Uses a spatially varying field */
      if (public_xtra -> type == 3)  
      {
	 dummy3 = (Type3 *)(public_xtra -> data);
         SetTempVectorData((dummy3 -> ic_values), temp_data);
         temp_data += SizeOfVector(dummy3 -> ic_values);

         ReadPFBinary((dummy3 -> filename), 
		      (dummy3 -> ic_values));
      }
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * ICPhaseTemperatureFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  ICPhaseTemperatureFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      free(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * ICPhaseTemperatureNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ICPhaseTemperatureNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   int num_regions;
   int ir;

   Type0	 *dummy0;
   Type1	 *dummy1;
   Type2	 *dummy2;
   Type3         *dummy3;

   char *switch_name;
   char *region;
   
   char key[IDB_MAX_KEY_LEN];

   NameArray type_na;

   type_na = NA_NewNameArray(
	      "Constant HydroStaticDepth HydroStaticPatch PFBFile");

   public_xtra = ctalloc(PublicXtra, 1);

   switch_name = GetString("ICTemperature.Type");

   public_xtra -> type = NA_NameToIndex(type_na, switch_name);

   switch_name = GetString("ICTemperature.GeomNames");
   public_xtra -> regions = NA_NewNameArray(switch_name);
   
   num_regions = NA_Sizeof(public_xtra -> regions);

   switch((public_xtra -> type))
   {
      case 0:
      {
	 dummy0 = ctalloc(Type0, 1);

	 dummy0 -> num_regions = num_regions;
      
	 (dummy0 -> region_indices) = ctalloc(int,    num_regions);
	 (dummy0 -> values)         = ctalloc(double, num_regions);

	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy0 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);
	    
	    sprintf(key, "Geom.%s.ICTemperature.Value", region);
	    dummy0 -> values[ir] = GetDouble(key);
	 }

	 (public_xtra -> data) = (void *) dummy0;
	 break;
      }
      case 1:
      {
	 dummy1 = ctalloc(Type1, 1);
	 
	  dummy1 -> num_regions = num_regions;
      
	 (dummy1 -> region_indices)       = ctalloc(int,    num_regions);
	 (dummy1 -> reference_elevations) = ctalloc(double, num_regions);
	 (dummy1 -> temperature_values)      = ctalloc(double, num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy1 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);
	    
	    sprintf(key, "Geom.%s.ICTemperature.RefElevation", region);
	    dummy1 -> reference_elevations[ir] = GetDouble(key);

	    sprintf(key, "Geom.%s.ICTemperature.Value", region);
	    dummy1 -> temperature_values[ir] = GetDouble(key);
	 }
	 
	 (public_xtra -> data) = (void *) dummy1;
	 
      break;
      }
      case 2:
      {
	 dummy2 = ctalloc(Type2, 1);
	 
	 dummy2 -> num_regions = num_regions;
	 
	 (dummy2 -> region_indices)  = ctalloc(int,    num_regions);
	 (dummy2 -> geom_indices)    = ctalloc(int,    num_regions);
	 (dummy2 -> patch_indices)   = ctalloc(int,    num_regions);
	 (dummy2 -> temperature_values) = ctalloc(double, num_regions);
	 
	 for (ir = 0; ir < num_regions; ir++)
	 {
	    
	    region = NA_IndexToName(public_xtra -> regions, ir);
	    
	    dummy2 -> region_indices[ir] = 
	       NA_NameToIndex(GlobalsGeomNames, region);

	    sprintf(key, "Geom.%s.ICTemperature.Value", region);
	    dummy2 -> temperature_values[ir] = GetDouble(key);

	    sprintf(key, "Geom.%s.ICTemperature.RefGeom", region);
	    switch_name = GetString(key);

	    dummy2 -> geom_indices[ir] = NA_NameToIndex(GlobalsGeomNames, 
						       switch_name);

	    if(dummy2 -> geom_indices[ir] < 0)
	    {
	       InputError("Error: invalid geometry name <%s> for key <%s>\n",
			  switch_name, key);
	    }

	    sprintf(key, "Geom.%s.ICTemperature.RefPatch", region);
	    switch_name = GetString(key);

	    dummy2 -> patch_indices[ir] = 
	       NA_NameToIndex(GeomSolidPatches(
		  GlobalsGeometries[dummy2 -> geom_indices[ir]]),
			      switch_name);

	    if (dummy2 -> patch_indices[ir] < 0 )
	    {
	       InputError("Error: invalid patch name <%s> for key <%s>\n",
			  switch_name, key);
	    }
	 }
	 
	 (public_xtra -> data) = (void *) dummy2;
	 
	 break;
      }
      
   case 3:
   {
      dummy3 = ctalloc(Type3, 1);

      sprintf(key, "Geom.%s.ICTemperature.FileName", "domain");
      dummy3 -> filename = GetString(key);

      public_xtra -> data = (void *) dummy3;
      
      break;
   }

   default:
   {
      InputError("Error: invalid type <%s> for key <%s>\n",
		 switch_name, key);
   }
   }
   
   NA_FreeNameArray(type_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * ICPhaseTemperatureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ICPhaseTemperatureFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);
   
   
   Type0       *dummy0;
   Type1       *dummy1;
   Type2       *dummy2;
   
   if ( public_xtra )
   {
      
      NA_FreeNameArray(public_xtra -> regions);
      
      switch((public_xtra -> type))
      {
	 case 0:
	 {
	    dummy0 = (Type0 *)(public_xtra -> data);
	    
	    tfree(dummy0 -> region_indices);
	    tfree(dummy0 -> values);
	    tfree(dummy0);
	    break;
	 }
	 case 1:
	 {
	    dummy1 = (Type1 *)(public_xtra -> data);
	    
	    tfree(dummy1 -> region_indices);
	    tfree(dummy1 -> reference_elevations);
	    tfree(dummy1 -> temperature_values);
	    tfree(dummy1);
	 }
	 case 2:
	 {
	    dummy2 = (Type2 *)(public_xtra -> data);
	    
	    tfree(dummy2 -> region_indices);
	    tfree(dummy2 -> patch_indices);
	    tfree(dummy2 -> geom_indices);
	    tfree(dummy2 -> temperature_values);
	    tfree(dummy2);
	 }
      }
      
      tfree(public_xtra);
   }
   
}

/*--------------------------------------------------------------------------
 * ICPhaseTemperatureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ICPhaseTemperatureSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Type3         *dummy3;

   int  sz = 0;

   if (public_xtra->type == 3)
   {
      dummy3 = (Type3 *)(public_xtra->data);
      sz += SizeOfVector(dummy3->ic_values);
   }

   return sz;
}
