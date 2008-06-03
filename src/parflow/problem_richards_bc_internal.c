/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Routine for setting up internal boundary conditions for the nonlinear 
 * function evaluation in the Richards' solver.
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   NameArray internal_bc_names;
   int      num_conditions;

   int     *type;
   void   **data;
} PublicXtra;

typedef struct
{
   double  xlocation;
   double  ylocation;
   double  zlocation;
   double  value;
} Type0;                      /* basic point condition */

typedef void InstanceXtra;


/*--------------------------------------------------------------------------
 * RichardsBCInternal:
 *   Add interior boundary conditions.
 *--------------------------------------------------------------------------*/

void RichardsBCInternal(problem, problem_data, f, A, time, pressure, fcn)
Problem     *problem;
ProblemData *problem_data;
Vector      *f;
Matrix      *A;
double       time;
Vector      *pressure;
int          fcn;
{
   PFModule      *this_module    = ThisPFModule;
   PublicXtra    *public_xtra    = PFModulePublicXtra(this_module);

   WellData         *well_data = ProblemDataWellData(problem_data);
   WellDataPhysical *well_data_physical;
   WellDataValue    *well_data_value;

   TimeCycleData    *time_cycle_data;

   int               num_conditions = (public_xtra -> num_conditions);
   int               num_wells, total_num;

   Type0            *dummy0;

   Grid             *grid = VectorGrid(pressure);

   SubgridArray     *internal_bc_subgrids;

   Subgrid          *subgrid, *subgrid_ind, *new_subgrid;

   Subvector        *f_sub, *p_sub;
   Submatrix        *A_sub;

   double           *cp, *wp, *ep, *sp, *np, *lp, *up;
   double           *fp, *pp;
   double           *internal_bc_conditions;

   double            dx, dy, dz;
   double            value;

   int               ix, iy, iz;
   int               nx, ny, nz;
   int               rx, ry, rz;
   int               process;

   int               i, j, k;
   int               grid_index, well, index;
   int               cycle_number, interval_number;
   int               ip, im;

   /*--------------------------------------------------------------------
    * gridify the internal boundary locations (should be done elsewhere?)
    *--------------------------------------------------------------------*/

   if ( num_conditions > 0 )
   {

      internal_bc_subgrids = NewSubgridArray();
      internal_bc_conditions = ctalloc(double, num_conditions);

      for (i = 0; i < num_conditions;i++)
      {
         switch((public_xtra -> type[i]))
         {

         case 0:
         {
            dummy0 = (Type0 *)(public_xtra -> data[i]);

            ix = IndexSpaceX((dummy0 -> xlocation), 0);
            iy = IndexSpaceY((dummy0 -> ylocation), 0);
            iz = IndexSpaceZ((dummy0 -> zlocation), 0);

            nx = 1;
            ny = 1;
            nz = 1;

            rx = 0;
            ry = 0;
            rz = 0;

            process = amps_Rank(amps_CommWorld);

            new_subgrid = NewSubgrid(ix, iy, iz,
                                     nx, ny, nz,
                                     rx, ry, rz,
                                     process);

            AppendSubgrid(new_subgrid, internal_bc_subgrids);

            internal_bc_conditions[i] = (dummy0 -> value);

            break;
         }
         }
      }
   }

   /*--------------------------------------------------------------------
    * Put in the internal conditions using the subgrids computed above
    * Put in any pressure wells from the well package
    *--------------------------------------------------------------------*/

   num_wells = WellDataNumPressWells(well_data);
   total_num = num_conditions + num_wells;

   if (( num_conditions > 0 ) || ( num_wells > 0 ))
   {
      /* Set explicit pressure assignments*/

      for (grid_index = 0; grid_index < GridNumSubgrids(grid); grid_index++)
      {
         subgrid = GridSubgrid(grid, grid_index);

         f_sub = VectorSubvector(f, grid_index);
         p_sub = VectorSubvector(pressure, grid_index);
         A_sub = MatrixSubmatrix(A, grid_index);

         fp    = SubvectorData(f_sub);
         pp    = SubvectorData(p_sub);

	 cp    = SubmatrixStencilData(A_sub, 0);
	 wp    = SubmatrixStencilData(A_sub, 1);
	 ep    = SubmatrixStencilData(A_sub, 2);
	 sp    = SubmatrixStencilData(A_sub, 3);
	 np    = SubmatrixStencilData(A_sub, 4);
	 lp    = SubmatrixStencilData(A_sub, 5);
	 up    = SubmatrixStencilData(A_sub, 6);

         for ( index=0; index < total_num; index++ )
         {
            if ( index < num_conditions )
            {
               subgrid_ind = SubgridArraySubgrid(internal_bc_subgrids, index);
               value = internal_bc_conditions[index];
            }
            else
            {
               well = index - num_conditions;
               time_cycle_data = WellDataTimeCycleData(well_data);
               well_data_physical = WellDataPressWellPhysical(well_data,well);
               cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
               interval_number = 
                  TimeCycleDataComputeIntervalNumber(problem, time, 
                                                     time_cycle_data, 
                                                     cycle_number);
               well_data_value = 
                 WellDataPressWellIntervalValue(well_data, well, 
                                                interval_number);
               subgrid_ind = WellDataPhysicalSubgrid(well_data_physical);
               value = WellDataValuePhaseValue(well_data_value, 0);
            }

            ix = SubgridIX(subgrid_ind);
            iy = SubgridIY(subgrid_ind);
            iz = SubgridIZ(subgrid_ind);

            nx = SubgridNX(subgrid_ind);
            ny = SubgridNY(subgrid_ind);
            nz = SubgridNZ(subgrid_ind);

            dx = SubgridDX(subgrid_ind);
            dy = SubgridDY(subgrid_ind);
            dz = SubgridDZ(subgrid_ind);

	    if (fcn == CALCFCN)
	    {
               BoxLoopI0(i, j, k,
                         ix, iy, iz, nx, ny, nz,
	       {
                  ip = SubvectorEltIndex(f_sub, i, j, k);
	          fp[ip] = pp[ip] - value;
	       });
	    }
	    else if (fcn == CALCDER)
	    {
                BoxLoopI0(i, j, k,
			  ix, iy, iz, nx, ny, nz,
	        {
                   im = SubmatrixEltIndex(A_sub, i, j, k);
		   cp[im] = 1.0;
		   wp[im] = 0.0;
		   ep[im] = 0.0;
		   sp[im] = 0.0;
		   np[im] = 0.0;
		   lp[im] = 0.0;
		   up[im] = 0.0;
		});
	    }

	 }        /* End loop over conditions */
      }           /* End loop over processor subgrids */

      FreeSubgridArray(internal_bc_subgrids);
      tfree(internal_bc_conditions);

   }              /* End if have well or internal pressure conditions */

}


/*--------------------------------------------------------------------------
 * BCInternalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *RichardsBCInternalInitInstanceXtra()
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
 * BCInternalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void   RichardsBCInternalFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * RichardsBCInternalNewPublicXtra
 *--------------------------------------------------------------------------*/


PFModule  *RichardsBCInternalNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0         *dummy0;

   int            num_conditions, i;

   char *internal_bc_names;

   char key[IDB_MAX_KEY_LEN];

   public_xtra = ctalloc(PublicXtra, 1);
   
   internal_bc_names = GetStringDefault("InternalBC.Names", "");
   public_xtra -> internal_bc_names = NA_NewNameArray(internal_bc_names);
   
   (public_xtra -> num_conditions) = num_conditions = 
      NA_Sizeof(public_xtra -> internal_bc_names);
   
   if ( num_conditions > 0 )
   {
      (public_xtra -> type)      = ctalloc(int,    num_conditions);
      (public_xtra -> data)      = ctalloc(void *, num_conditions);
      
      for (i = 0; i < num_conditions; i++)
      {
         /***************************************************************/
         /*  For the forseeable future we only have one  so just insert */
         /*   that into the type field without asking for input (which  */
         /* would mean that the input file would have to change - YUCK) */
         /***************************************************************/
#if 0
         invoice = amps_NewInvoice("%i", &(public_xtra -> type[i]));
         amps_SFBCast(amps_CommWorld, file, invoice);
         amps_FreeInvoice(invoice);
#endif
         (public_xtra -> type[i]) = 0;
	 
         switch((public_xtra -> type[i]))
         {
	    case 0:
	    {
	       dummy0 = ctalloc(Type0, 1);
	       
	       sprintf(key, "InternalBC.%s.X", 
		       NA_IndexToName(public_xtra -> internal_bc_names, i));
	       dummy0 -> xlocation = GetDouble(key);
	       
	       sprintf(key, "InternalBC.%s.Y", 
		       NA_IndexToName(public_xtra -> internal_bc_names, i));
	       dummy0 -> xlocation = GetDouble(key);
	       
	       sprintf(key, "InternalBC.%s.Z", 
		       NA_IndexToName(public_xtra -> internal_bc_names, i));
	       dummy0 -> xlocation = GetDouble(key);
	       
	       sprintf(key, "InternalBC.%s.Value", 
		       NA_IndexToName(public_xtra -> internal_bc_names, i));
	       dummy0 -> value = GetDouble(key);
	       
	       
	       (public_xtra -> data[i]) = (void *) dummy0;
	       
	       break;
	    }
	 }
      }
   }
   
   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * RichardsBCInternalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  RichardsBCInternalFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0         *dummy0;

   int            num_conditions, i;

   if (public_xtra)
   {

      NA_FreeNameArray(public_xtra -> internal_bc_names);

      num_conditions = (public_xtra -> num_conditions);

      if ( num_conditions > 0 )
      {

         for (i = 0; i < num_conditions; i++)
         {
            switch((public_xtra -> type[i]))
            {
            case 0:
            {
               dummy0 = (Type0 *)(public_xtra -> data[i]);
               tfree(dummy0);

               break;
            }
            }
         }

         tfree(public_xtra -> data);
         tfree(public_xtra -> type);
      }

      tfree(public_xtra);
   }

}

/*--------------------------------------------------------------------------
 * RichardsBCInternalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RichardsBCInternalSizeOfTempData()
{
   return 0;
}
