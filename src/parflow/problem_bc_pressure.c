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
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{
   int     num_phases;

} PublicXtra;

typedef struct
{
   Problem  *problem;
   PFModule *phase_density;
   //PFModule *bc_temperature;

} InstanceXtra;

/*--------------------------------------------------------------------------
 * BCPressure:
 *   This routine returns a BCStruct structure which describes where
 *   and what the boundary conditions are.
 *--------------------------------------------------------------------------*/

BCStruct    *BCPressure(problem_data, grid, gr_domain, time)

ProblemData *problem_data;  /* Contains BC info transferred by the 
			       BCPressurePackage function */
Grid        *grid;          /* Grid data */
GrGeomSolid *gr_domain;     /* Gridded domain solid */
double       time;          /* Current time - needed to determine where on
			       the boundary time cycle we are */
{
   PFModule       *this_module   = ThisPFModule;
   PublicXtra     *public_xtra   = PFModulePublicXtra(this_module);
   InstanceXtra   *instance_xtra = PFModuleInstanceXtra(this_module);

   PFModule       *phase_density = (instance_xtra -> phase_density);

   BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);

   TimeCycleData  *time_cycle_data;

   int             num_phases    = (public_xtra -> num_phases);

   Problem        *problem       = (instance_xtra -> problem);

   SubgridArray   *subgrids      = GridSubgrids(grid);

   Subgrid        *subgrid;

   BCStruct       *bc_struct, *temp_bc_struct;
   double       ***values;

   double         *patch_values, *temp_patch_values;
   int             patch_values_size;

   int            *fdir;

   int             num_patches;
   int             ipatch, is, i, j, k, ival, phase;
   int             cycle_number, interval_number;
	         
   //temp_bc_struct = PFModuleInvoke(BCStruct *, bc_temperature,
   //                           (problem_data, grid, gr_domain, time));

   bc_struct = NULL;

   num_patches = BCPressureDataNumPatches(bc_pressure_data);

   if (num_patches > 0)
   {
      time_cycle_data = BCPressureDataTimeCycleData(bc_pressure_data);

      /*---------------------------------------------------------------------
       * Set up bc_struct with NULL values component
       *---------------------------------------------------------------------*/

      bc_struct = NewBCStruct(subgrids, gr_domain,
                              num_patches,
                              BCPressureDataPatchIndexes(bc_pressure_data),
                              BCPressureDataBCTypes(bc_pressure_data),
                              NULL);

      /*---------------------------------------------------------------------
       * Set up values component of bc_struct
       *---------------------------------------------------------------------*/

      values = ctalloc(double **, num_patches);
      BCStructValues(bc_struct) = values;

      for (ipatch = 0; ipatch < num_patches; ipatch++)
      {
         values[ipatch]  = ctalloc(double *, SubgridArraySize(subgrids));

         cycle_number    = BCPressureDataCycleNumber(bc_pressure_data,ipatch);
         interval_number = TimeCycleDataComputeIntervalNumber(
			       problem, time, time_cycle_data, cycle_number);

	 switch(BCPressureDataType(bc_pressure_data,ipatch))
         {

	 case 0:
         {
	    /* Constant pressure value specified on a reference patch.
	       Calculate hydrostatic conditions along boundary patch for 
	       elevations different from reference patch elevations.
	       Hydrostatic condition is:
	       grad p - rho g grad z = 0 */

	    BCPressureType0 *bc_pressure_type0;

	    GeomSolid       *ref_solid;

	    double         **elevations;
	    double           z, dz2, dtmp;
	    double           offset, interface_press, interface_den;
	    double           ref_den, ref_press, nonlin_resid;
	    double           density_der, density, fcn_val;
	    double           height;
	    double           gravity = -ProblemGravity(problem);
	   
	    int              ref_patch;
	    int              max_its = 10;
	    int              iterations;
	    int	             ix, iy, iz, nx, ny, nz, r, iel;
	   
	    bc_pressure_type0 = BCPressureDataIntervalValue(
		                    bc_pressure_data,ipatch,interval_number);

	    ref_solid = ProblemDataSolid(problem_data, 
			     BCPressureType0RefSolid(bc_pressure_type0));
	    ref_patch = BCPressureType0RefPatch(bc_pressure_type0);

	    /* Calculate elevations at (x,y) points on reference patch. */
	    elevations = CalcElevations(ref_solid, ref_patch, subgrids);

	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

	       ix = SubgridIX(subgrid);
	       iy = SubgridIY(subgrid);
	       iz = SubgridIZ(subgrid);

	       nx = SubgridNX(subgrid);
	       ny = SubgridNY(subgrid);
	       nz = SubgridNZ(subgrid);

	       /* RDF: assume resolution is the same in all 3 directions */
	       r  = SubgridRX(subgrid);

	       dz2 = SubgridDZ(subgrid)*0.5;

	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, 
				 ipatch, is, 
               {
                  ref_press = BCPressureType0Value(bc_pressure_type0);
		  PFModuleInvoke(void, phase_density,
				 (0, NULL, NULL, NULL, &ref_press, &ref_den, 
				  CALCFCN));

		  z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2]*dz2;
		  iel = (i-ix) + (j-iy)*nx;
		  fcn_val = 0.0;
		  nonlin_resid = 1.0;
		  iterations = -1;
		  
		  /* Solve a nonlinear problem for hydrostatic pressure
		     at points on boundary patch given pressure on reference
		     patch.  Note that the problem is only nonlinear if 
		     density depends on pressure. 

		     The nonlinear problem to solve is:
		       F(p) = 0
		       F(p) = P - P_ref 
		              - 0.5*(rho(P) + rho(P_ref))*gravity*(z - z_ref)

		     Newton's method is used to find a solution. */

		  while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
                  {
	             if (iterations > -1)
	             {
                        PFModuleInvoke(void, phase_density, 
				       (0, NULL, NULL, NULL, &patch_values[ival], 
					&density_der, CALCDER));
			dtmp = 1.0 - 0.5*density_der*gravity
			                *(z-elevations[is][iel]);
			patch_values[ival] = patch_values[ival] - fcn_val/dtmp;
		     }
		     else
	             {
		        patch_values[ival] = ref_press;
		     }
		     PFModuleInvoke(void, phase_density, 
				    (0, NULL, NULL, NULL, &patch_values[ival], 
				     &density, CALCFCN));

		     fcn_val = patch_values[ival] - ref_press 
		               - 0.5*(density + ref_den)*gravity
		                    *(z - elevations[is][iel]);
		     nonlin_resid = fabs(fcn_val);

		     iterations++;

		  }        /* End of while loop */


		  /* Iterate over the phases and reset pressures according to 
		     hydrostatic conditions with appropriate densities.
		     At each interface, we have hydrostatic conditions, so 

		     z_inter = (P_inter - P_ref) / 
		                (0.5*(rho(P_inter)+rho(P_ref))*gravity
			       + z_ref

		     Thus, the interface height and pressure are known 
		     and hydrostatic conditions can be determined for 
		     new phase.  

		     NOTE:  This only works for Pc = 0. */

		  for (phase = 1; phase < num_phases; phase++)
	          {
		     interface_press = BCPressureType0ValueAtInterface(
			         		bc_pressure_type0,phase);
		     PFModuleInvoke(void, phase_density,
				    (phase-1, NULL, NULL, NULL, &interface_press, 
				     &interface_den, CALCFCN));
		     offset = (interface_press - ref_press)
		               / (0.5*(interface_den + ref_den)*gravity);
		     ref_press = interface_press;
		     PFModuleInvoke(void, phase_density, 
				    (phase, NULL, NULL, NULL, &ref_press, &ref_den,
				     CALCFCN));

		     /* Only reset pressure value if in another phase.
			The following "if" test determines whether this point
			is in another phase by checking if the computed 
			pressure is less than the interface value.  This
			test ONLY works if the phases are distributed such
			that the lighter phases are above the heavier ones. */

		     if (patch_values[ival] < interface_press)
                     {
		        height = elevations[is][iel];
		        nonlin_resid = 1.0;
			iterations = -1;
			while ((nonlin_resid > 1.0E-6)&&(iterations < max_its))
                        {
	                   if (iterations > -1)
	                   {
                              PFModuleInvoke(void, phase_density, 
				    (phase, NULL, NULL, NULL, &patch_values[ival], 
				     &density_der, CALCDER));

			      dtmp = 1.0 - 0.5*density_der*gravity
			                      *(z-height);
			      patch_values[ival] = patch_values[ival] 
			                           - fcn_val/dtmp;
			   }
		           else
	                   {
		              height = height + offset;
			      patch_values[ival] = ref_press;
			   }
	 
			   PFModuleInvoke(void, phase_density, 
					  (phase, NULL, NULL, NULL, 
					   &patch_values[ival], &density, 
					   CALCFCN));

			   fcn_val = patch_values[ival] - ref_press 
		                     - 0.5*(density + ref_den)
		                          *gravity*(z - height);
			   nonlin_resid = fabs(fcn_val);

			   iterations++;

			}        /* End of while loop */
		     }           /* End if above interface */

		  }              /* End phase loop */

	       });               /* End BCStructPatchLoop body */
	       
	       tfree(elevations[is]);

	    }                    /* End subgrid loop */
	    tfree(elevations);

	    break;    
	 }                       /* End case 0 */

	 case 1:
         {
	    /* Piecewise linear pressure value specified on reference 
	       patch.
	       Calculate hydrostatic conditions along patch for 
	       elevations different from reference patch elevations.  
	       Hydrostatic condition is:
	                   grad p - rho g grad z = 0 */

	    BCPressureType1 *bc_pressure_type1;

	    int              num_points;
	    int              ip;

	    double           x, y, z, dx2, dy2, dz2;
	    double           unitx, unity, line_min, line_length, xy, slope;

	    double           dtmp, offset, interface_press, interface_den;
	    double           ref_den, ref_press, nonlin_resid;
	    double           density_der, density, fcn_val;
	    double           height;
	    double           gravity = -ProblemGravity(problem);
	    
	    int              max_its = 10;
	    int              iterations;

	    bc_pressure_type1 = BCPressureDataIntervalValue(
                                      bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

               dx2  = SubgridDX(subgrid) / 2.0;
               dy2  = SubgridDY(subgrid) / 2.0;
               dz2  = SubgridDZ(subgrid) / 2.0;

               /* compute unit direction vector for piecewise linear line */
               unitx = BCPressureType1XUpper(bc_pressure_type1) 
		       - BCPressureType1XLower(bc_pressure_type1);
               unity = BCPressureType1YUpper(bc_pressure_type1) 
		       - BCPressureType1YLower(bc_pressure_type1);
               line_length = sqrt(unitx*unitx + unity*unity);
               unitx /= line_length;
               unity /= line_length;
               line_min = BCPressureType1XLower(bc_pressure_type1)*unitx
                        + BCPressureType1YLower(bc_pressure_type1)*unity;

               BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
               {
                  x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0]*dx2;
                  y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1]*dy2;
                  z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2]*dz2;
	       
                  /* project center of BC face onto piecewise line */
                  xy = (x*unitx + y*unity - line_min) / line_length;

                  /* find two neighboring points */
                  ip = 1;
				  num_points = BCPressureType1NumPoints(bc_pressure_type1);
                  for (; ip < (num_points - 1); ip++)
                  {
                     if (xy < BCPressureType1Point(bc_pressure_type1,ip))
                        break;
                  }
	       
                  /* compute the slope */
                  slope = ((BCPressureType1Value(bc_pressure_type1,ip) 
			    - BCPressureType1Value(bc_pressure_type1,(ip-1)))
                        / (BCPressureType1Point(bc_pressure_type1,ip) 
			   - BCPressureType1Point(bc_pressure_type1,(ip-1))));

		  ref_press = BCPressureType1Value(bc_pressure_type1,ip-1)
                   + slope*(xy - BCPressureType1Point(bc_pressure_type1,ip-1));
		  PFModuleInvoke(void, phase_density,
				 (0, NULL, NULL, NULL, &ref_press, &ref_den, 
				  CALCFCN));
		  fcn_val = 0.0;
		  nonlin_resid = 1.0;
		  iterations = -1;
		  
		  /* Solve a nonlinear problem for hydrostatic pressure
		     at points on boundary patch given reference pressure.  
		     Note that the problem is only nonlinear if 
		     density depends on pressure. 

                     The nonlinear problem to solve is:
                       F(p) = 0
                       F(p) = P - P_ref 
                              - 0.5*(rho(P) + rho(P_ref))*gravity*z

                     Newton's method is used to find a solution. */

		  while ((nonlin_resid > 1.0E-6) && (iterations < max_its))
                  {
	             if (iterations > -1)
	             {
                        PFModuleInvoke(void, phase_density, 
				       (0, NULL, NULL, NULL, &patch_values[ival], 
					&density_der, CALCDER));
			dtmp = 1.0 - 0.5*density_der*gravity*z;
			patch_values[ival] = patch_values[ival] - fcn_val/dtmp;
		     }
		     else
	             {
		        patch_values[ival] = ref_press;
		     }
		     PFModuleInvoke(void, phase_density, 
				    (0, NULL, NULL, NULL, &patch_values[ival], 
				     &density, CALCFCN));

		     fcn_val = patch_values[ival] - ref_press 
		               - 0.5*(density + ref_den)*gravity*z;
		     nonlin_resid = fabs(fcn_val);

		     iterations++;

		  }        /* End of while loop */

		  /* Iterate over the phases and reset pressures according to 
		     hydrostatic conditions with appropriate densities. 
                     At each interface, we have hydrostatic conditions, so 

                     z_inter = (P_inter - P_ref) / 
                                (0.5*(rho(P_inter)+rho(P_ref))*gravity
                               + z_ref

                     Thus, the interface height and pressure are known 
                     and hydrostatic conditions can be determined for 
                     new phase.  

                     NOTE:  This only works for Pc = 0. */

		  for (phase = 1; phase < num_phases; phase++)
	          {
		     interface_press = BCPressureType1ValueAtInterface(
			         		bc_pressure_type1,phase);
		     PFModuleInvoke(void, phase_density,
				    (phase-1, NULL, NULL, NULL, &interface_press, 
				     &interface_den, CALCFCN));
		     offset = (interface_press - ref_press)
		               / (0.5*(interface_den + ref_den)*gravity);
		     ref_press = interface_press;
		     PFModuleInvoke(void, phase_density, 
				    (phase, NULL, NULL, NULL, &ref_press, &ref_den,
				     CALCFCN));

		     /* Only reset pressure value if in another phase.
			The following "if" test determines whether this point
			is in another phase by checking if the computed 
			pressure is less than the interface value.  This
			test ONLY works if the phases are distributed such
			that the lighter phases are above the heavier ones. */

                     if (patch_values[ival] < interface_press)
                     {
		        height = 0.0;
		        nonlin_resid = 1.0;
			iterations = -1;
			while ((nonlin_resid > 1.0E-6)&&(iterations < max_its))
                        {
	                   if (iterations > -1)
	                   {
                              PFModuleInvoke(void, phase_density, 
				    (phase, NULL, NULL, NULL, &patch_values[ival], 
				     &density_der, CALCDER));

			      dtmp = 1.0 - 0.5*density_der*gravity*(z-height);
			      patch_values[ival] = patch_values[ival] 
			                           - fcn_val/dtmp;
			   }
		           else
	                   {
			      height = height + offset;
			      patch_values[ival] = ref_press;
			   }
	 
			   PFModuleInvoke(void, phase_density, 
					  (phase, NULL, NULL, NULL, 
					   &patch_values[ival], &density, 
					   CALCFCN));

			   fcn_val = patch_values[ival] - ref_press 
		                     - 0.5*(density + ref_den)*gravity
			                  *(z - height);
			   nonlin_resid = fabs(fcn_val);

			   iterations++;

			}        /* End of while loop */
		     }           /* End if above interface */

		  }              /* End phase loop */
	       });               /* End BCStructPatchLoop body */
	       
	    }
	    break;
	 }

	 case 2:
	 {
	    /* Constant flux rate value on patch */
	    BCPressureType2 *bc_pressure_type2;
	    double           flux;

	    bc_pressure_type2 = BCPressureDataIntervalValue(
				    bc_pressure_data,ipatch,interval_number);

	    flux = BCPressureType2Value(bc_pressure_type2);
	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values[ival] = flux;
	       });
	    }     /* End subgrid loop */
	    break;
	 }	

	 case 3:
	 {
	    /* Constant volumetric flux value on patch */
	    BCPressureType3 *bc_pressure_type3;
	    double           dx, dy, dz;
	    double           area, volumetric_flux;

	    bc_pressure_type3 = BCPressureDataIntervalValue(
                                      bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

               dx  = SubgridDX(subgrid);
               dy  = SubgridDY(subgrid);
               dz  = SubgridDZ(subgrid);

               area = 0.0;
               BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
               {
                  /* primary direction x */
                  if (fdir[0])
                  {
                     area += dy * dz;
                  }
                  /* primary direction y */
                  else if (fdir[1])
                  {
                     area += dx * dz;
                  }
                  /* primary direction z */
                  else if (fdir[2])
                  {
                     area += dx * dy;
                  }
               });

               if (area > 0.0)
               {
                  volumetric_flux = BCPressureType3Value(bc_pressure_type3) 
		                       / area;
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
                     patch_values[ival] = volumetric_flux;
                  });
               }
	    }         /* End subgrid loop */
	    break;
	 }

	 case 4:
         {
	    /* Read input pressures from file (temporary).
	       This case assumes hydraulic head input conditions and 
	       a constant density.  */

	    BCPressureType4 *bc_pressure_type4;
	    Vector          *tmp_vector;
	    Subvector       *subvector;
	    double          *data;
	    char            *filename;
	    double          *tmpp;
	    int              itmp;
	    double           z, dz2;
	    double           density, dtmp;
	    
	    double           gravity = ProblemGravity(problem);

	    /* Calculate density using dtmp as dummy argument. */
	    dtmp = 0.0;
	    PFModuleInvoke(void, phase_density,
			   (0, NULL, NULL, NULL, &dtmp, &density, CALCFCN));

	    bc_pressure_type4 = BCPressureDataIntervalValue(
                                   bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       dz2  = SubgridDZ(subgrid) / 2.0;

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

               tmp_vector = NewTempVector(grid, 1, 0);
               data = ctalloc(double, SizeOfVector(tmp_vector));
               SetTempVectorData(tmp_vector, data);

               filename = BCPressureType4FileName(bc_pressure_type4);
               ReadPFBinary(filename, tmp_vector);

               subvector = VectorSubvector(tmp_vector, is);

               tmpp = SubvectorData(subvector);
               BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
               {
                  z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2]*dz2;

                  itmp = SubvectorEltIndex(subvector, i, j, k);

                  patch_values[ival] = tmpp[itmp] - density*gravity*z;

               });

               tfree(VectorData(tmp_vector));
               FreeTempVector(tmp_vector);

	    }           /* End subgrid loop */
	    break;
	 }

	 case 5:
	 {
	    /* Read input fluxes from file (temporary) */
	    BCPressureType5 *bc_pressure_type5;
	    Vector          *tmp_vector;
	    Subvector       *subvector;
	    double          *data;
	    char            *filename;
	    double          *tmpp;
	    int              itmp;
	    
	    bc_pressure_type5 = BCPressureDataIntervalValue(
                                   bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

               tmp_vector = NewTempVector(grid, 1, 0);
               data = ctalloc(double, SizeOfVector(tmp_vector));
               SetTempVectorData(tmp_vector, data);

               filename = BCPressureType5FileName(bc_pressure_type5);
               ReadPFBinary(filename, tmp_vector);

               subvector = VectorSubvector(tmp_vector, is);

               tmpp = SubvectorData(subvector);
               BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
               {
                  itmp = SubvectorEltIndex(subvector, i, j, k);

                  patch_values[ival] = tmpp[itmp];
               });

               tfree(VectorData(tmp_vector));
               FreeTempVector(tmp_vector);
	    }       /* End subgrid loop */
	    break;
	 }

	 case 6:
	 {
	    /* Calculate pressure based on pre-defined functions */
	    BCPressureType6 *bc_pressure_type6;
	    double           x, y, z, dx2, dy2, dz2;
	    int              fcn_type;
	    
	    bc_pressure_type6 = BCPressureDataIntervalValue(
                                   bc_pressure_data,ipatch,interval_number);

	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

               dx2  = SubgridDX(subgrid) / 2.0;
               dy2  = SubgridDY(subgrid) / 2.0;
               dz2  = SubgridDZ(subgrid) / 2.0;

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

	       fcn_type = BCPressureType6FunctionType(bc_pressure_type6);

	       switch(fcn_type)
	       {
	       case 1: /* p = x */
	       {
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;

		     patch_values[ival] = x;
		  });

		  break;

	       }    /* End case 1 */

	       case 2: /* p = x + y + z */
	       {
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;
		     y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1] * dy2;
		     z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2] * dz2;

		     patch_values[ival] = x + y + z;
		  });

		  break;

	       }    /* End case 2 */

	       case 3: /* p = x^3y^2 + sinxy + 1*/
	       {
	         
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;
		     y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1] * dy2;

		     patch_values[ival] = x*x*x*y*y + sin(x*y) + 1;
		  });
		  break;

	       }    /* End case 3 */

	       case 4: /* p = x^3 y^4 + x^2 + sinxy cosy + 1 */
	       {
	         
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;
		     y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1] * dy2;
		     z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2] * dz2;

		     patch_values[ival] = pow(x,3)*pow(y,4) + x*x + sin(x*y)*cos(y) + 1;
		  });
		  break;

	       }    /* End case 4 */

	       case 5: /* p = xyzt +1 */
	       {
	         
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;
		     y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1] * dy2;
		     z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2] * dz2;

		     patch_values[ival] = x * y * z * time + 1;
		  });
		  break;

	       }    /* End case 5 */

	       case 6: /* p = xyzt +1 */
	       {
	         
                  BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
                  {
		     x = RealSpaceX(i, SubgridRX(subgrid)) + fdir[0] * dx2;
		     y = RealSpaceY(j, SubgridRY(subgrid)) + fdir[1] * dy2;
		     z = RealSpaceZ(k, SubgridRZ(subgrid)) + fdir[2] * dz2;

		     patch_values[ival] = x * y * z * time + 1;
		  });
		  break;

	       }    /* End case 5 */

	       }    /* End switch */

	    }       /* End subgrid loop */
	    break;
	 }

	 case 7:
	 {
	    /* Constant "rainfall" rate value on patch */
	    BCPressureType7 *bc_pressure_type7;
	    double           flux;

	    bc_pressure_type7 = BCPressureDataIntervalValue(
				    bc_pressure_data,ipatch,interval_number);

	    flux = BCPressureType7Value(bc_pressure_type7);
	    ForSubgridI(is, subgrids)
	    {
	       subgrid = SubgridArraySubgrid(subgrids, is);

	       /* compute patch_values_size (this isn't really needed yet) */
	       patch_values_size = 0;
	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values_size++;
	       });

	       patch_values = ctalloc(double, patch_values_size);
	       values[ipatch][is] = patch_values;

	       BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
	       {
		  patch_values[ival] = flux;
	       });
	    }     /* End subgrid loop */
	    break;
	 }	


	 }
      }
   }

   return bc_struct;
}


/*--------------------------------------------------------------------------
 * BCPressureInitInstanceXtra 
 *--------------------------------------------------------------------------*/

PFModule *BCPressureInitInstanceXtra(problem)
Problem *problem;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `problem'
    *-----------------------------------------------------------------------*/

   if ( problem != NULL)
   {
      (instance_xtra -> problem) = problem;
   }

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> phase_density) =
        PFModuleNewInstance(ProblemPhaseDensity(problem), ());
   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> phase_density), ());
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * BCPressureFreeInstanceXtra 
 *--------------------------------------------------------------------------*/

void BCPressureFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> phase_density);

      tfree(instance_xtra);
   }
}

/*--------------------------------------------------------------------------
 * BCPressureNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *BCPressureNewPublicXtra(num_phases)
int        num_phases;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   /* allocate space for the public_xtra structure */
   public_xtra = ctalloc(PublicXtra, 1);

   (public_xtra -> num_phases) = num_phases;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*--------------------------------------------------------------------------
 * BCPressureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  BCPressureFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   if ( public_xtra )
   {
      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * BCPressureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  BCPressureSizeOfTempData()
{
   return 0;
}
