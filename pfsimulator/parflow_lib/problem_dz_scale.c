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
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{

   int     type;
   int     variable_dz;
   void   *data;

} PublicXtra;

typedef void InstanceXtra;

typedef struct
{
   NameArray regions;
   int      num_regions;
   int     *region_indices;
   double  *values;

} Type0;                       /* constant regions */


/*--------------------------------------------------------------------------
 * dZ Scaling values
 *--------------------------------------------------------------------------*/
void dzScale (ProblemData *problem_data, Vector *dz_mult )
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);

   Grid          *grid = VectorGrid(dz_mult);

   Type0          *dummy0;

   SubgridArray   *subgrids = GridSubgrids(grid);

   Subgrid        *subgrid;
   Subvector      *ps_sub;

   double         *data;

   int             ix, iy, iz;
   int             nx, ny, nz;
   int             r;

   int             is, i, j, k, ips;


   /*-----------------------------------------------------------------------
    * dz Scale
    *-----------------------------------------------------------------------*/

   InitVector(dz_mult, 1.0);

    if (public_xtra -> variable_dz) {
   switch((public_xtra -> type))
   {
   case 0:
   {
      int      num_regions;
      int     *region_indices;
      double  *values;

      GrGeomSolid  *gr_solid;
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
            ps_sub  = VectorSubvector(dz_mult, is);
	    
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
	       ips = SubvectorEltIndex(ps_sub, i, j, k);
	       data[ips] = value;
	    });
	 }
      }

      break;
   }
   }
    }
    }


/*--------------------------------------------------------------------------
 * dzScaleInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *dzScaleInitInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


#if 0
   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);
#endif
   instance_xtra = NULL;

   PFModuleInstanceXtra(this_module) = instance_xtra;

   return this_module;
}

/*-------------------------------------------------------------------------
 * dzScaleFreeInstanceXtra
 *-------------------------------------------------------------------------*/

void  dzScaleFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = (InstanceXtra *)PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * dzScaleNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *dzScaleNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0         *dummy0;

   char *switch_name;
   char *region;
   char key[IDB_MAX_KEY_LEN];
    char *name;
    int *switch_value;
     NameArray      switch_na;

   NameArray type_na;
    
   type_na = NA_NewNameArray("Constant");

   public_xtra = ctalloc(PublicXtra, 1);

    /* @RMM added switch for grid dz multipliers */
    /* RMM set dz multipliers (default=False) */
    printf("flag 1: pre name set \n");
    name = "Solver.Nonlinear.VariableDz";
    switch_na = NA_NewNameArray("False True");
    
    switch_name = GetStringDefault(name, "False");
    switch_value = NA_NameToIndex(switch_na, switch_name);

    if(switch_value < 0)
    {
        InputError("Error: invalid value <%s> for key <%s>\n",
                   switch_name, key );
    }
    

    public_xtra -> variable_dz = switch_value;
    
    
    if (public_xtra -> variable_dz == 1) { 
        
        printf("var dz true \n");
      switch_name = GetString("dzScale.Type");
      
      public_xtra -> type = NA_NameToIndex(type_na, switch_name);


      switch((public_xtra -> type))
      {
	 case 0:
	 {
	    int  num_regions, ir;
	    
	    dummy0 = ctalloc(Type0, 1);

	    switch_name = GetString("dzScale.GeomNames");

	    dummy0 -> regions = NA_NewNameArray(switch_name);

	    dummy0 -> num_regions = NA_Sizeof(dummy0 -> regions);

	    num_regions = (dummy0 -> num_regions);
	    
	    (dummy0 -> region_indices) = ctalloc(int,    num_regions);
	    (dummy0 -> values)         = ctalloc(double, num_regions);
	    
	    for (ir = 0; ir < num_regions; ir++)
	    {
	       region = NA_IndexToName(dummy0 -> regions, ir);

	       dummy0 -> region_indices[ir] = 
		  NA_NameToIndex(GlobalsGeomNames, region);

	       sprintf(key, "Geom.%s.dzScale.Value", region);
	       dummy0 -> values[ir] = GetDouble(key);
	    }
	    
	    (public_xtra -> data) = (void *) dummy0;
	    
	    break;
	 }
	 
	 default:
	 {
	    InputError("Error: invalid type <%s> for key <%s>\n",
		       switch_name, key);
	 }

      }
    }
   
   NA_FreeNameArray(type_na);
   
   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * dzScaleFreePublicXtra
 *--------------------------------------------------------------------------*/

void  dzScaleFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = (PublicXtra *)PFModulePublicXtra(this_module);

   Type0       *dummy0;


   if (public_xtra -> variable_dz) {
           switch((public_xtra -> type))
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
         }
           }
   }else {
        
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * dzScaletorageSizeOfTempData
 *--------------------------------------------------------------------------*/

int  dzScaleSizeOfTempData()
{
   return 0;
}
