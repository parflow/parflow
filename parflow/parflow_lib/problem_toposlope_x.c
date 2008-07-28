/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
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

   int    type;
   void  *data;

} PublicXtra;

typedef struct
{
	Grid   *grid;
	double *temp_data;
} InstanceXtra;

typedef struct
{
   NameArray regions;
   int      num_regions;


   int     *region_indices;
   double  *values;

} Type0;                       /* constant regions */

typedef struct
{
   int      function_type;
} Type1;                       /* Known forcing term on entire domain */

typedef struct
{
	char  *filename;
	Vector *sx_values;
} Type2;                       /* .pfb file */


/*--------------------------------------------------------------------------
 * XSlope 
 *--------------------------------------------------------------------------*/

void         XSlope(problem_data,x_slope,dummy)
ProblemData *problem_data;
Vector      *x_slope;
Vector      *dummy;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Grid             *grid = VectorGrid(dummy);

   GrGeomSolid      *gr_solid, *gr_domain;

   Type0            *dummy0;
   Type1            *dummy1;
   Type2            *dummy2;

   CommHandle       *handle;

   SubgridArray     *subgrids = GridSubgrids(grid);

   Subgrid          *subgrid;
   Subvector        *ps_sub;
   Subvector        *sx_values_sub;

   double           *data;
   double           *psdat, *sx_values_dat;
   //double            slopex[60][32][392];

   int               ix, iy, iz;
   int               nx, ny, nz;
   int               r;

   int               is, i, j, k, ips, ipicv;
   double            time=0.0;


   /*-----------------------------------------------------------------------
    * Put in any user defined sources for this phase
    *-----------------------------------------------------------------------*/

   InitVectorAll(x_slope, 0.0);

   switch((public_xtra -> type))
   {
   case 0:
   {
      int      num_regions;
      int     *region_indices;
      double  *values;

      double        value;
      int           ir;

	  dummy0 = (Type0 *)(public_xtra -> data);

      num_regions    = (dummy0 -> num_regions);
      region_indices = (dummy0 -> region_indices);
      values         = (dummy0 -> values);

      for (ir = 0; ir < num_regions; ir++)
      {
     gr_solid = ProblemDataGrSolid(problem_data, region_indices[ir]);
     value    = values[ir];

	 ForSubgridI(is, subgrids)
	 {
            subgrid = SubgridArraySubgrid(subgrids, is);
            ps_sub  = VectorSubvector(x_slope, is);
	    
	    ix = SubgridIX(subgrid);
	    iy = SubgridIY(subgrid);
	    iz = SubgridIZ(subgrid);
	    
	    nx = SubgridNX(subgrid);
	    ny = SubgridNY(subgrid);
	    nz = SubgridNZ(subgrid);
	    
	    /* RDF: assume resolution is the same in all 3 directions */
	    r = SubgridRX(subgrid);
	    
	    data = SubvectorData(ps_sub);
	    GrGeomInLoop(i, j, k, gr_solid, r, ix, iy, iz, nx, ny, nz,
            {
	       ips = SubvectorEltIndex(ps_sub, i, j, 0);

	       data[ips] = value;
	    });
	 }
      }
      
      break;
   }   /* End case 0 */

   case 1:
   {
      GrGeomSolid  *gr_domain;
      double        x, y, z;
      int           function_type;

      dummy1 = (Type1 *)(public_xtra -> data);

      function_type    = (dummy1 -> function_type);

      gr_domain = ProblemDataGrDomain(problem_data);

      ForSubgridI(is, subgrids)
      {
	subgrid = SubgridArraySubgrid(subgrids, is);
	ps_sub  = VectorSubvector(x_slope, is);
	    
	ix = SubgridIX(subgrid);
	iy = SubgridIY(subgrid);
	iz = SubgridIZ(subgrid);
	    
	nx = SubgridNX(subgrid);
	ny = SubgridNY(subgrid);
	nz = SubgridNZ(subgrid);
	    
	/* RDF: assume resolution is the same in all 3 directions */
	r = SubgridRX(subgrid);
	    
	data = SubvectorData(ps_sub);

	switch(function_type)
	{	
	case 1:  /* p= x */
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
	      x = RealSpaceX(i, SubgridRX(subgrid));
		/* nonlinear case -div(p grad p) = f */
	      data[ips] = -1.0;
	   });
	   break;

	}   /* End case 1 */

	case 2:  /* p= x+y+z */
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
		/* nonlinear case -div(p grad p) = f */
	      data[ips] = -3.0;
	   });
	   break;

	}   /* End case 2 */

	case 3:  /* p= x^3y^2 + sinxy + 1 */
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
	      x = RealSpaceX(i, SubgridRX(subgrid));
	      y = RealSpaceY(j, SubgridRY(subgrid));
		/* nonlinear case -div(p grad p) = f */
	      data[ips] = -pow((3*x*x*y*y + y*cos(x*y)), 2) - pow((2*x*x*x*y + x*cos(x*y)),2) - (x*x*x*y*y + sin(x*y) + 1)*(6*x*y*y + 2*x*x*x - (x*x + y*y)*sin(x*y)); 
	   });
	   break;

	}   /* End case 3 */

	case 4:  /* f for p = x^3y^4 + x^2 + sinxy cosy + 1 */
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
	      x = RealSpaceX(i, SubgridRX(subgrid));
	      y = RealSpaceY(j, SubgridRY(subgrid));
	      z = RealSpaceZ(k, SubgridRZ(subgrid));
	      
	      data[ips] = -pow(3*x*x*pow(y,4) + 2*x + y*cos(x*y)*cos(y),2) - pow(4*x*x*x*y*y*y + x*cos(x*y)*cos(y) - sin(x*y)*sin(y),2) - (x*x*x*pow(y,4) + x*x + sin(x*y)*cos(y) + 1)*(6*x*pow(y,4) + 2 - (x*x + y*y + 1)*sin(x*y)*cos(y) + 12*x*x*x*y*y - 2*x*cos(x*y)*sin(y));

	   });
	   break;

	}   /* End case 4 */

	case 5:  /* f = xyz-y^2z^2t^2-x^2z^2t^2-x^2y^2t^2 (p=xyzt+1)*/
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
	      x = RealSpaceX(i, SubgridRX(subgrid));
	      y = RealSpaceY(j, SubgridRY(subgrid));
	      z = RealSpaceZ(k, SubgridRZ(subgrid));

	      data[ips] = x*y*z - time*time*(y*y*z*z + x*x*z*z + x*x*y*y);
	   });
	   break;

	}   /* End case 5 */

	case 6:  /* f = xyz-y^2z^2t^2-2x^2z^2t^2-3x^2y^2t^2 (p=xyzt+1, 
		                                             K=(1; 2; 3) )*/
	{
	   GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
	   {
	      ips = SubvectorEltIndex(ps_sub, i, j, k);
	      x = RealSpaceX(i, SubgridRX(subgrid));
	      y = RealSpaceY(j, SubgridRY(subgrid));
	      z = RealSpaceZ(k, SubgridRZ(subgrid));

	      data[ips] = x*y*z 
		          - time*time*(y*y*z*z + x*x*z*z*2.0 + x*x*y*y*3.0);
	   });
	   break;

	}   /* End case 6 */

	}   /* End switch statement on function_types */

      }     /* End subgrid loop */
   
      break;
   }   /* End case 1 for input types */

   case 2:
   {
	   Vector *sx_val;

	   dummy2 = (Type2 *)(public_xtra -> data);

	   sx_val= dummy2 -> sx_values;

	   gr_domain = ProblemDataGrDomain(problem_data);

	   ForSubgridI(is, subgrids)
	   {
		   subgrid = SubgridArraySubgrid(subgrids, is);
		   ps_sub = VectorSubvector(x_slope, is);
		   sx_values_sub = VectorSubvector(sx_val, is);

		   ix = SubgridIX(subgrid);
		   iy = SubgridIY(subgrid);
		   iz = SubgridIZ(subgrid);

		   nx = SubgridNX(subgrid);
		   ny = SubgridNY(subgrid);
		   nz = SubgridNZ(subgrid);

		   r = SubgridRX(subgrid);

		   psdat  = SubvectorData(ps_sub);
		   sx_values_dat = SubvectorData(sx_values_sub);

		   GrGeomInLoop(i,j,k,gr_domain,r,ix,iy,iz,nx,ny,nz,
		   {
			   ips = SubvectorEltIndex(ps_sub,i,j,0);
			   ipicv = SubvectorEltIndex(sx_values_sub,i,j,k);

			   psdat[ips] = sx_values_dat[ipicv];
		   });
	   } /* End subgrid loop */

	   break;
   }

   }   /* End switch statement for input types */
   
   handle = InitVectorUpdate(x_slope, VectorUpdateAll);
   FinalizeVectorUpdate(handle);
 
}


/*--------------------------------------------------------------------------
 * XSlopeInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *XSlopeInitInstanceXtra(grid)
Grid    *grid;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra;

   Type2  *dummy2;
   

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if (grid != NULL)
   {
	   (instance_xtra -> grid) = grid;

   if (public_xtra -> type == 2)
   {
	   dummy2 = (Type2 *)(public_xtra -> data);

	   dummy2 -> sx_values = NewTempVector(grid, 1, 1);
	   (instance_xtra -> temp_data) = talloc(double, SizeOfVector(dummy2 -> sx_values));
	   SetTempVectorData((dummy2 -> sx_values),  (instance_xtra -> temp_data));

	   ReadPFBinary((dummy2 -> filename),(dummy2 -> sx_values));
   
   }
   }

   
   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * XSlopeFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  XSlopeFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   Type2  *dummy2;

    if ( public_xtra -> type ==2)
	{
	             dummy2 = (Type2 *)(public_xtra -> data);
				 FreeTempVector(dummy2 -> sx_values);

				 tfree(instance_xtra -> temp_data);
	}

	 if (instance_xtra)
   {
	   free(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * XSlopeNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *XSlopeNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0         *dummy0;
   Type1         *dummy1;
   Type2         *dummy2;

   int  num_regions, ir;
   
   char key[IDB_MAX_KEY_LEN];

   char *switch_name;

   NameArray type_na;
   NameArray function_type_na;

   type_na = NA_NewNameArray("Constant PredefinedFunction PFBFile");

   function_type_na = NA_NewNameArray("dum0 X XPlusYPlusZ X3Y2PlusSinXYPlus1 \
                                       X3Y4PlusX2PlusSinXYCosYPlus1 \
                                       XYZTPlus1 XYZTPlus1PermTensor");
   public_xtra = ctalloc(PublicXtra, 1);
         
      switch_name = GetString("TopoSlopesX.Type");
      
      public_xtra -> type = NA_NameToIndex(type_na, switch_name);
      
      switch((public_xtra -> type))
      {
	 case 0:
	 {
		 dummy0 = ctalloc(Type0, 1);
	    
	    switch_name = GetString("TopoSlopesX.GeomNames");
	    
	    dummy0 -> regions = NA_NewNameArray(switch_name);
	    
	    num_regions = (dummy0 -> num_regions) = NA_Sizeof(dummy0 -> regions);
	    
	    (dummy0 -> region_indices) = ctalloc(unsigned int, num_regions);
	    (dummy0 -> values)         = ctalloc(double, num_regions);
	    
	    for (ir = 0; ir < num_regions; ir++)
	    {
	       
	       dummy0 -> region_indices[ir] = 
		  NA_NameToIndex(GlobalsGeomNames,
				 NA_IndexToName(dummy0 -> regions, ir));
	       
	       sprintf(key, "TopoSlopesX.Geom.%s.Value", NA_IndexToName(dummy0 -> regions, ir));
	       dummy0 -> values[ir] = GetDouble(key);
	    }
	    
	    (public_xtra -> data) = (void *) dummy0;

	    break;
	 }   /* End case 0 */
	 
	 case 1:
	 {
	    dummy1 = ctalloc(Type1, 1);
	    
	    switch_name = GetString("PhaseSources.PredefinedFunction");
	    
	    dummy1 -> function_type = 
	       NA_NameToIndex(function_type_na, switch_name);

	    if(dummy1 -> function_type < 0)
	    {
	       InputError("Error: invalid function <%s> for key <%s>\n",
			  switch_name, key);
	    }
	    
	    (public_xtra -> data) = (void *) dummy1;
	    
	    break;
	 }   /* End case 1 */

	 case 2:
		 {
			 dummy2 = ctalloc(Type2, 1);
             
			 dummy2 -> filename = GetString("TopoSlopesX.FileName");
			 
			 (public_xtra -> data) = (void *) dummy2;
			 break;
		 }

	 default:
	 {
	    InputError("Error: invalid type <%s> for key <%s>\n",
		       switch_name, key);
	 }
      }   /* End case statement */
   
   
   NA_FreeNameArray(type_na);
   NA_FreeNameArray(function_type_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*-------------------------------------------------------------------------
 * XSlopeFreePublicXtra
 *-------------------------------------------------------------------------*/

void  XSlopeFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0       *dummy0;
   Type1       *dummy1;


   if ( public_xtra )
   {
         switch((public_xtra -> type))
         {
         case 0:
         {
			 dummy0 = (Type0 *)(public_xtra -> data);

	    NA_FreeNameArray(dummy0 -> regions);

	    tfree(dummy0 -> region_indices);
	    tfree(dummy0 -> values);
            tfree(dummy0);
            break;
         }
         case 1:
         {
            dummy1 = (Type1 *)(public_xtra -> data);

            tfree(dummy1);
            break;
         }
		 }
      
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * XSlopeSizeOfTempData
 *--------------------------------------------------------------------------*/

int  XSlopeSizeOfTempData()
{
   return 0;
}
