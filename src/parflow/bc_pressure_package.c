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
   int     num_phases;

   /* cycling info */
   int     num_cycles;

   int    *interval_divisions;
   int   **intervals;
   int    *repeat_counts;

   /* patch info */
   NameArray patches;
   int     num_patches;

   int    *input_types;   /* num_patches input types */
   int    *patch_indexes; /* num_patches patch indexes */
   int    *cycle_numbers; /* num_patches cycle numbers */
   void  **data;          /* num_patches pointers to Type structures */
} PublicXtra;

typedef struct
{
   Problem *problem;
} InstanceXtra;

typedef struct
{
   int       reference_solid;
   int       reference_patch;
   double   *values;
   double  **value_at_interface;
} Type0;               /* Dirichlet, constant */

typedef struct
{
   double   *xlower;
   double   *ylower;
   double   *xupper;
   double   *yupper;
   int      *num_points;
   double  **points;      /* num_points points */
   double  **values;      /* num_points values */
   double  **value_at_interface;
} Type1;               /* Dirichlet, piece-wise linear */

typedef struct
{
   double   *values;
} Type2;               /* Fixed Flux, constant */

typedef struct
{
   double   *values;
} Type3;               /* Volumetric Flux, constant */

typedef struct
{
   char     **filenames;
} Type4;               /* Pressure, given in "filenames".pfb */

typedef struct
{
   char     **filenames;
} Type5;               /* Flux, given in "filenames".pfb */

typedef struct
{
   int	      function_type;
} Type6;           /* Dir. pressure for testing known solution MATH problems */

typedef struct
{
   double   *values;
} Type7;               /* //sk Overland flow, constant rainfall*/


/*--------------------------------------------------------------------------
 * BCPressurePackage
 *--------------------------------------------------------------------------*/

void         BCPressurePackage(problem_data)
ProblemData *problem_data;
{
   PFModule         *this_module   = ThisPFModule;
   PublicXtra       *public_xtra   = PFModulePublicXtra(this_module);

   BCPressureData   *bc_pressure_data 
                        = ProblemDataBCPressureData(problem_data);

   Type0            *dummy0;
   Type1            *dummy1;
   Type2            *dummy2;
   Type3            *dummy3;
   Type4            *dummy4;
   Type5            *dummy5;
   Type6            *dummy6;
   Type7            *dummy7;  

   TimeCycleData    *time_cycle_data;

   int               num_patches;
   int               i;
   int               cycle_length, cycle_number, interval_division, 
                     interval_number;

   /* Allocate the bc data */
   BCPressureDataNumPhases(bc_pressure_data) = (public_xtra -> num_phases);

   BCPressureDataNumPatches(bc_pressure_data) = (public_xtra -> num_patches);

   if ( (public_xtra -> num_patches) > 0 )
   {
      /* Load the time cycle data */
      time_cycle_data = NewTimeCycleData((public_xtra -> num_cycles), 
					 (public_xtra -> interval_divisions));

      for(cycle_number = 0; cycle_number < (public_xtra -> num_cycles); 
	  cycle_number++)
      {
         TimeCycleDataIntervalDivision(time_cycle_data, cycle_number) 
            = (public_xtra -> interval_divisions[cycle_number]);
         cycle_length = 0;
         for(interval_number = 0; 
	     interval_number < (public_xtra -> 
				interval_divisions[cycle_number]); 
	     interval_number++)
         {
            cycle_length += (public_xtra -> intervals[cycle_number])[interval_number];
            TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number) = (public_xtra -> intervals[cycle_number])[interval_number];
         }
         TimeCycleDataRepeatCount(time_cycle_data, cycle_number) = (public_xtra -> repeat_counts[cycle_number]);
         TimeCycleDataCycleLength(time_cycle_data, cycle_number) = cycle_length;
      }

      BCPressureDataTimeCycleData(bc_pressure_data) = time_cycle_data;

     /* Load the Boundary Condition Data */

      num_patches = BCPressureDataNumPatches(bc_pressure_data);
      BCPressureDataTypes(bc_pressure_data)        = ctalloc(int, num_patches);
      BCPressureDataPatchIndexes(bc_pressure_data) = ctalloc(int, num_patches);
      BCPressureDataCycleNumbers(bc_pressure_data) = ctalloc(int, num_patches);
      BCPressureDataBCTypes(bc_pressure_data)      = ctalloc(int, num_patches);
      BCPressureDataValues(bc_pressure_data)       = ctalloc(void **, 
							     num_patches);

      for (i = 0; i < num_patches; i++)
      {
         BCPressureDataType(bc_pressure_data,i) = (public_xtra -> input_types[i]);
         BCPressureDataPatchIndex(bc_pressure_data,i) = (public_xtra -> patch_indexes[i]);
         BCPressureDataCycleNumber(bc_pressure_data,i) = (public_xtra -> cycle_numbers[i]);

         interval_division = TimeCycleDataIntervalDivision(time_cycle_data, BCPressureDataCycleNumber(bc_pressure_data,i));
         BCPressureDataIntervalValues(bc_pressure_data, i) = ctalloc(void *, interval_division);
         for(interval_number = 0; interval_number < interval_division; interval_number++)
         {
            switch((public_xtra -> input_types[i]))
            {
            /* Setup a fixed pressure condition structure */
            case 0:
            {
               BCPressureType0 *bc_pressure_type0;

               int              phase;
               int num_phases = BCPressureDataNumPhases(bc_pressure_data);

               BCPressureDataBCType(bc_pressure_data,i) = DirichletBC;

               dummy0 = (Type0 *)(public_xtra -> data[i]);

               bc_pressure_type0 = ctalloc(BCPressureType0, 1);

	       BCPressureType0RefSolid(bc_pressure_type0) = 
                                                   (dummy0 -> reference_solid);

	       BCPressureType0RefPatch(bc_pressure_type0) = 
                                                   (dummy0 -> reference_patch);

               BCPressureType0Value(bc_pressure_type0) = (dummy0 -> values[interval_number]);

               if (num_phases > 1)
               {
                  BCPressureType0ValueAtInterfaces(bc_pressure_type0) = ctalloc(double, (num_phases - 1));
                  for(phase = 1; phase < num_phases; phase++)
                  {
                     BCPressureType0ValueAtInterface(bc_pressure_type0,phase) = ((dummy0 -> value_at_interface[interval_number])[phase-1]);
                  }
               }
               else
               {
                  BCPressureType0ValueAtInterfaces(bc_pressure_type0) = NULL;
               }

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) = (void *) bc_pressure_type0;

               break;
            }

            /* Setup a piecewise linear pressure condition structure */
            case 1:
            {
               BCPressureType1   *bc_pressure_type1;
               int                point, phase;
               int                num_points;
               int                num_phases = BCPressureDataNumPhases(bc_pressure_data);

               BCPressureDataBCType(bc_pressure_data,i) = DirichletBC;

               dummy1 = (Type1 *)(public_xtra -> data[i]);

               num_points = (dummy1 -> num_points[interval_number]);

               bc_pressure_type1 = ctalloc(BCPressureType1, 1);

               BCPressureType1XLower(bc_pressure_type1) = (dummy1 -> xlower[interval_number]);
               BCPressureType1YLower(bc_pressure_type1) = (dummy1 -> ylower[interval_number]);
               BCPressureType1XUpper(bc_pressure_type1) = (dummy1 -> xupper[interval_number]);
               BCPressureType1YUpper(bc_pressure_type1) = (dummy1 -> yupper[interval_number]);
               BCPressureType1NumPoints(bc_pressure_type1) = (dummy1 -> num_points[interval_number]);

               BCPressureType1Points(bc_pressure_type1) = ctalloc(double, num_points);
               BCPressureType1Values(bc_pressure_type1) = ctalloc(double, num_points);

               for(point = 0; point < num_points; point++)
               {
                  BCPressureType1Point(bc_pressure_type1,point) = ((dummy1 -> points[interval_number])[point]);
                  BCPressureType1Value(bc_pressure_type1,point) = ((dummy1 -> values[interval_number])[point]);
               }

               if (num_phases > 1)
               {
                  BCPressureType1ValueAtInterfaces(bc_pressure_type1) = ctalloc(double, (num_phases - 1));

                  for(phase = 1; phase < num_phases; phase++)
                  {
                     BCPressureType1ValueAtInterface(bc_pressure_type1,phase) = ((dummy1 -> value_at_interface[interval_number])[phase-1]);
                  }
               }
               else
               {
                  BCPressureType1ValueAtInterfaces(bc_pressure_type1) = NULL;
               }

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) = (void *) bc_pressure_type1;

               break;
            }

            /* Setup a constant flux condition structure */
            case 2:
            {
               BCPressureType2   *bc_pressure_type2;

               BCPressureDataBCType(bc_pressure_data,i) = FluxBC;

               dummy2 = (Type2 *)(public_xtra -> data[i]);

               bc_pressure_type2 = ctalloc(BCPressureType2, 1);

               BCPressureType2Value(bc_pressure_type2) 
                                = (dummy2 -> values[interval_number]);

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
                             = (void *) bc_pressure_type2;

               break;
            }

            /* Setup a volumetric flux condition structure */
            case 3:
            {
               BCPressureType3   *bc_pressure_type3;

               BCPressureDataBCType(bc_pressure_data,i) = FluxBC;

               dummy3 = (Type3 *)(public_xtra -> data[i]);

               bc_pressure_type3 = ctalloc(BCPressureType3, 1);

               BCPressureType3Value(bc_pressure_type3) 
                     = (dummy3 -> values[interval_number]);

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
                     = (void *) bc_pressure_type3;

               break;
            }

            /* Setup a file defined pressure condition structure */
            case 4:
            {
               BCPressureType4   *bc_pressure_type4;

               BCPressureDataBCType(bc_pressure_data,i) = DirichletBC;

               dummy4 = (Type4 *)(public_xtra -> data[i]);

               bc_pressure_type4 = ctalloc(BCPressureType4, 1);

               BCPressureType4FileName(bc_pressure_type4) 
                  = ctalloc(char, strlen((dummy4 -> filenames)[interval_number])+1);

               strcpy(BCPressureType4FileName(bc_pressure_type4),
		      ((dummy4 -> filenames)[interval_number]));

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
                  = (void *) bc_pressure_type4;

               break;
            }

            /* Setup a file defined flux condition structure */
            case 5:
            {
               BCPressureType5   *bc_pressure_type5;

               BCPressureDataBCType(bc_pressure_data,i) = FluxBC;

               dummy5 = (Type5 *)(public_xtra -> data[i]);

               bc_pressure_type5 = ctalloc(BCPressureType5, 1);

               BCPressureType5FileName(bc_pressure_type5) 
		  = ctalloc(char, strlen((dummy5 -> filenames)[interval_number])+1);

               strcpy(BCPressureType5FileName(bc_pressure_type5),
		      ((dummy5 -> filenames)[interval_number]));

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
		  = (void *) bc_pressure_type5;

               break;
            }

            /* Setup a Dir. pressure MATH problem condition */
            case 6:
            {
               BCPressureType6   *bc_pressure_type6; 

               BCPressureDataBCType(bc_pressure_data,i) = DirichletBC;

               dummy6 = (Type6 *)(public_xtra -> data[i]);

               bc_pressure_type6 = ctalloc(BCPressureType6, 1);

               BCPressureType6FunctionType(bc_pressure_type6) 
		  = (dummy6 -> function_type);

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
		  = (void *) bc_pressure_type6;

               break;
            }

			/*//sk  Setup a overland flow condition structure */
            case 7:
            {
               BCPressureType7   *bc_pressure_type7;

               BCPressureDataBCType(bc_pressure_data,i) = OverlandBC;

               dummy7 = (Type7 *)(public_xtra -> data[i]);

               bc_pressure_type7 = ctalloc(BCPressureType7, 1);

               BCPressureType7Value(bc_pressure_type7) 
                                = (dummy7 -> values[interval_number]);

               BCPressureDataIntervalValue(bc_pressure_data,i,interval_number) 
                             = (void *) bc_pressure_type7;

               break;
            }

            }
         }
      }
   }
}


/*--------------------------------------------------------------------------
 * BCPressurePackageInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *BCPressurePackageInitInstanceXtra(problem)
Problem *problem;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;

   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   if (problem != NULL)
   {
      (instance_xtra -> problem) = problem;
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * BCPressurePackageFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  BCPressurePackageFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * BCPressurePackageNewPublicXtra
 *--------------------------------------------------------------------------*/
PFModule  *BCPressurePackageNewPublicXtra(num_phases)
int        num_phases;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   char *patch_names;

   char *patch_name;

   char key[IDB_MAX_KEY_LEN];
   
   char *switch_name;
   
   char *cycle_name;
   
   int global_cycle;

   int domain_index;

   int phase;
   char *interval_name;

   int            num_patches;
   int            num_cycles;

   Type0         *dummy0;
   Type1         *dummy1;
   Type2         *dummy2;
   Type3         *dummy3;
   Type4         *dummy4;
   Type5         *dummy5;
   Type6         *dummy6;
   Type7         *dummy7;

   int             i, interval_number, interval_division;

   NameArray type_na;
   NameArray function_na;

   //sk
   type_na = NA_NewNameArray("DirEquilRefPatch DirEquilPLinear FluxConst FluxVolumetric PressureFile FluxFile ExactSolution OverlandFlow");

   function_na = NA_NewNameArray("dum0 X XPlusYPlusZ X3Y2PlusSinXYPlus1 X3Y4PlusX2PlusSinXYCosYPlus1 XYZTPlus1 XYZTPlus1PermTensor");

   /* allocate space for the public_xtra structure */
   public_xtra = ctalloc(PublicXtra, 1);

   (public_xtra -> num_phases) = num_phases;

   patch_names = GetString("BCPressure.PatchNames");
   
   public_xtra -> patches = NA_NewNameArray(patch_names);
   num_patches = NA_Sizeof(public_xtra -> patches);

   (public_xtra -> num_patches) = num_patches;
   (public_xtra -> num_cycles) = num_patches;

   if ( num_patches > 0)
   {
      (public_xtra -> num_cycles)  = num_cycles = num_patches;

      (public_xtra -> interval_divisions) = ctalloc(int,   num_cycles);
      (public_xtra -> intervals)          = ctalloc(int *, num_cycles);
      (public_xtra -> repeat_counts)      = ctalloc(int,   num_cycles);

      (public_xtra -> input_types)   = ctalloc(int,    num_patches);
      (public_xtra -> patch_indexes) = ctalloc(int,    num_patches);
      (public_xtra -> cycle_numbers) = ctalloc(int,    num_patches);
      (public_xtra -> data)          = ctalloc(void *, num_patches);

      /* Determine the domain geom index from domain name */
      switch_name = GetString("Domain.GeomName");
      domain_index = NA_NameToIndex(GlobalsGeomNames, switch_name);
      if ( domain_index < 0 )
      {
	 InputError("Error: invalid geometry name <%s> for key <%s>\n",
		    switch_name, "Domain.GeomName");
      }

      for (i = 0; i < num_patches; i++)
      {

	 patch_name = NA_IndexToName(public_xtra -> patches, i);
	 
	 public_xtra -> patch_indexes[i] =
	    NA_NameToIndex(GlobalsGeometries[domain_index]->patches, 
			   patch_name);

	 sprintf(key, "Patch.%s.BCPressure.Type", patch_name);
	 switch_name = GetString(key);
	 	 
	 public_xtra -> input_types[i] = NA_NameToIndex(type_na, switch_name);

	 if(public_xtra -> input_types[i] < 0)
	 {
	    InputError("Error: invalid type name <%s> for <%s>\n",
		       switch_name, key);
	 }

	 sprintf(key, "Patch.%s.BCPressure.Cycle", patch_name);
	 cycle_name = GetString(key);

	 public_xtra -> cycle_numbers[i] = i;

	 global_cycle = NA_NameToIndex(GlobalsCycleNames, cycle_name);

	 if(global_cycle < 0)
	 {
	    InputError("Error: invalid cycle name <%s> for <%s>\n",
		       cycle_name, key);
	 }


	 interval_division = public_xtra -> interval_divisions[i] = 
	    GlobalsIntervalDivisions[global_cycle];

	 public_xtra -> repeat_counts[i] = 
	    GlobalsRepeatCounts[global_cycle];

	 (public_xtra -> intervals[i]) = ctalloc(int, interval_division);
	 for(interval_number = 0; interval_number < interval_division;
	     interval_number++)
	 {
	    public_xtra -> intervals[i][interval_number] =
	       GlobalsIntervals[global_cycle][interval_number];
	 }

         switch((public_xtra -> input_types[i]))
         {

         case 0:
         {
            int     size;

            dummy0 = ctalloc(Type0, 1);

            (dummy0 -> values)             = ctalloc(double,   
						     interval_division);
            (dummy0 -> value_at_interface) = ctalloc(double *, 
						     interval_division);

	    sprintf(key, "Patch.%s.BCPressure.RefGeom", patch_name);
	    switch_name = GetString(key);

	    dummy0 -> reference_solid = NA_NameToIndex(GlobalsGeomNames, 
						       switch_name);

	    if(dummy0 -> reference_solid < 0)
	    {
	       InputError("Error: invalid geometry name <%s> for reference solid <%s>\n", switch_name, key);
	    }

	    sprintf(key, "Patch.%s.BCPressure.RefPatch", patch_name);
	    switch_name = GetString(key);

	    dummy0 -> reference_patch = 
	       NA_NameToIndex(GeomSolidPatches(
		  GlobalsGeometries[dummy0 -> reference_solid]),
			      switch_name);

	    if(dummy0 -> reference_patch < 0)
	    {
	       InputError("Error: invalid reference patch name <%s> for key <%s>\n", switch_name, key);
	    }

            for(interval_number = 0; 
		interval_number < interval_division; 
		interval_number++)
            {

	       sprintf(key, "Patch.%s.BCPressure.%s.Value", 
		       patch_name,
		       NA_IndexToName(
			  GlobalsIntervalNames[global_cycle],
			  interval_number));
		  
	       dummy0 -> values[interval_number] = GetDouble(key);

               if (num_phases > 1)
               {
                  size = (num_phases - 1);

                  (dummy0 -> value_at_interface[interval_number]) = 
		                                         ctalloc(double, size);

		  for(phase=1; phase < num_phases; phase++)
		  {
		     sprintf(key, "Patch.%s.BCPressure.%s.%s.IntValue",
			     patch_name,
			     NA_IndexToName(
				GlobalsIntervalNames[global_cycle],
				interval_number),
			     NA_IndexToName(GlobalsPhaseNames, phase));
		     
		     dummy0 -> value_at_interface[interval_number][phase] = 
			GetDouble(key);
		  }
               }
               else
               {
                  (dummy0 -> value_at_interface[interval_number]) = NULL;
               }
            }

            (public_xtra -> data[i]) = (void *) dummy0;
            break;
         }

         case 1:
         {
            int     k, num_points, size;

            dummy1 = ctalloc(Type1, 1);

            (dummy1 -> xlower) = ctalloc(double, interval_division);
            (dummy1 -> ylower) = ctalloc(double, interval_division);
            (dummy1 -> xupper) = ctalloc(double, interval_division);
            (dummy1 -> yupper) = ctalloc(double, interval_division);

            (dummy1 -> num_points) = ctalloc(int, interval_division);

            (dummy1 -> points)             = ctalloc(double *, interval_division);
            (dummy1 -> values)             = ctalloc(double *, interval_division);
            (dummy1 -> value_at_interface) = ctalloc(double *, interval_division);

            for(interval_number = 0; 
		interval_number < interval_division; 
		interval_number++)
            {
	       interval_name = 
		  NA_IndexToName(GlobalsIntervalNames[global_cycle],
				 interval_number);
		  
               /* read in the xy-line */
	       sprintf(key, "Patch.%s.BCPressure.%s.XLower", patch_name,
		       interval_name);
	       dummy1 -> xlower[interval_number] = GetDouble(key);

	       sprintf(key, "Patch.%s.BCPressure.%s.YLower", patch_name,
		       interval_name);
	       dummy1 -> ylower[interval_number] = GetDouble(key);

	       sprintf(key, "Patch.%s.BCPressure.%s.XUpper", patch_name,
		       interval_name);
	       dummy1 -> xupper[interval_number] = GetDouble(key);

	       sprintf(key, "Patch.%s.BCPressure.%s.YUpper", patch_name,
		       interval_name);
	       dummy1 -> yupper[interval_number] = GetDouble(key);

               /* read num_points */
	       sprintf(key, "Patch.%s.BCPressure.%s.NumPoints", patch_name,
		       interval_name);
	       num_points = GetInt(key);

               (dummy1 -> num_points[interval_number]) = num_points;

               (dummy1 -> points[interval_number]) = ctalloc(double, num_points);
               (dummy1 -> values[interval_number]) = ctalloc(double, num_points);

               for (k = 0; k < num_points; k++)
               {
		  sprintf(key, "Patch.%s.BCPressure.%s.%d.Location",
			  patch_name, interval_name, k);		  
		  dummy1 -> points[interval_number][k] = GetDouble(key);

		  sprintf(key, "Patch.%s.BCPressure.%s.%d.Value",
			  patch_name, interval_name, k);		  
		  dummy1 -> values[interval_number][k] = GetDouble(key);
               }

               if (num_phases > 1)
               {
                  size = (num_phases - 1);

                  (dummy1 -> value_at_interface[interval_number]) = 
		     ctalloc(double, size);

		  for(phase=1; phase < num_phases; phase++)
		  {
		     sprintf(key, "Patch.%s.BCPressure.%s.%s.IntValue",
			     patch_name,
			     NA_IndexToName(
				GlobalsIntervalNames[global_cycle],
				interval_number),
			     NA_IndexToName(GlobalsPhaseNames, phase));
		     
		     dummy1 -> value_at_interface[interval_number][phase] = 
			GetDouble(key);
		  }
               }
               else
               {
                  (dummy1 -> value_at_interface[interval_number]) = NULL;
               }
            }

            (public_xtra -> data[i]) = (void *) dummy1;
            break;
         }

         case 2:
         {
            dummy2 = ctalloc(Type2, 1);

            (dummy2 -> values) = ctalloc(double, interval_division);

            for(interval_number = 0; interval_number < interval_division; interval_number++)
	    { 
	       sprintf(key, "Patch.%s.BCPressure.%s.Value", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
			  interval_number));

	       dummy2 -> values[interval_number] = GetDouble(key);
            }

            (public_xtra -> data[i]) = (void *) dummy2;
            break;
         }

         case 3:
         {
            dummy3 = ctalloc(Type3, 1);

            (dummy3 -> values) = ctalloc(double, interval_division);

            for(interval_number = 0; interval_number < interval_division; interval_number++)
            {
	       sprintf(key, "Patch.%s.BCPressure.%s.Value", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
			  interval_number));

	       dummy3 -> values[interval_number] = GetDouble(key);
            }

            (public_xtra -> data[i]) = (void *) dummy3;
            break;
         }

         case 4:
         {
            dummy4 = ctalloc(Type4, 1);

            (dummy4 -> filenames) = ctalloc(char *, interval_division);

            for(interval_number = 0; interval_number < interval_division; interval_number++)
            {

	       sprintf(key, "Patch.%s.BCPressure.%s.FileName", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
			  interval_number));
	       
               dummy4 -> filenames[interval_number] = GetString(key);
            }

            (public_xtra -> data[i]) = (void *) dummy4;
            break;
         }

         case 5:
         {
            dummy5 = ctalloc(Type5, 1);

            (dummy5 -> filenames) = ctalloc(char *, interval_division);

            for(interval_number = 0; interval_number < interval_division; interval_number++)
            {
	       sprintf(key, "Patch.%s.BCPressure.%s.FileName", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
			  interval_number));
	       
               dummy5 -> filenames[interval_number] = GetString(key);
            }

            (public_xtra -> data[i]) = (void *) dummy5;
            break;
         }

         case 6:
         {
            dummy6 = ctalloc(Type6, 1);

            for(interval_number = 0; interval_number < interval_division; 
		interval_number++)
            {
	       sprintf(key, "Patch.%s.BCPressure.%s.PredefinedFunction", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
				      interval_number));
	       switch_name = GetString(key);
	       
	       dummy6 -> function_type = NA_NameToIndex(function_na, 
							switch_name);

	       if(dummy6 -> function_type < 0 )
	       {
	          InputError("Error: invalid function type <%s> for key <%s>\n",
			     switch_name, key);
	       }

	       (public_xtra -> data[i]) = (void *) dummy6;
            }
            break;
         }

		 case 7:
         {
            dummy7 = ctalloc(Type7, 1);

            (dummy7 -> values) = ctalloc(double, interval_division);

            for(interval_number = 0; interval_number < interval_division; interval_number++)
	    { 
	       sprintf(key, "Patch.%s.BCPressure.%s.Value", 
		       patch_name,
		       NA_IndexToName(GlobalsIntervalNames[global_cycle],
			  interval_number));

	       dummy7 -> values[interval_number] = GetDouble(key);
            }

            (public_xtra -> data[i]) = (void *) dummy7;
            break;
         }


         }   /* End switch statement */
      }
   }

   NA_FreeNameArray(type_na);
   NA_FreeNameArray(function_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*-------------------------------------------------------------------------
 * BCPressurePackageFreePublicXtra
 *-------------------------------------------------------------------------*/

void  BCPressurePackageFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0         *dummy0;
   Type1         *dummy1;
   Type2         *dummy2;
   Type3         *dummy3;
   Type4         *dummy4;
   Type5         *dummy5;
   Type6         *dummy6;
   Type7         *dummy7;

   int            num_patches, num_cycles;
   int            i, interval_number, interval_division;

   if (public_xtra)
   {
      /* Free the well information */
      num_patches = (public_xtra -> num_patches);
      NA_FreeNameArray(public_xtra -> patches);

      if ( num_patches > 0 )
      {
         for (i = 0; i < num_patches; i++)
         {
            interval_division = (public_xtra -> interval_divisions[(public_xtra -> cycle_numbers[i])]);
            switch((public_xtra -> input_types[i]))
            {

            case 0:
            {
               dummy0 = (Type0 *)(public_xtra -> data[i]);

               for(interval_number = 0; interval_number < interval_division; interval_number++)
               {
                  tfree((dummy0 -> value_at_interface[interval_number]));
               }

               tfree((dummy0 -> value_at_interface));
               tfree((dummy0 -> values));

               tfree(dummy0);

               break;
            }

            case 1:
            {
               int interval_number;

               dummy1 = (Type1 *)(public_xtra -> data[i]);

               for(interval_number = 0; interval_number < interval_division; interval_number++)
               {
                  tfree((dummy1 -> value_at_interface[interval_number]));
                  tfree((dummy1 -> values[interval_number]));
                  tfree((dummy1 -> points[interval_number]));
               }

               tfree((dummy1 -> value_at_interface));
               tfree((dummy1 -> points));
               tfree((dummy1 -> values));

               tfree((dummy1 -> num_points));

               tfree((dummy1 -> yupper));
               tfree((dummy1 -> xupper));
               tfree((dummy1 -> ylower));
               tfree((dummy1 -> xlower));

               tfree(dummy1);

               break;
            }

            case 2:
            {
               dummy2 = (Type2 *)(public_xtra -> data[i]);

               tfree((dummy2 -> values));
               tfree(dummy2);

               break;
            }

            case 3:
            {
               dummy3 = (Type3 *)(public_xtra -> data[i]);

               tfree((dummy3 -> values));
               tfree(dummy3);

               break;
            }

            case 4:
            {
               int interval_number;

               dummy4 = (Type4 *)(public_xtra -> data[i]);

               for(interval_number = 0; interval_number < interval_division; interval_number++)
               {
                  tfree(((dummy4 -> filenames)[interval_number]));
               }

               tfree((dummy4 -> filenames));

               tfree(dummy4);

               break;
            }

            case 5:
            {
               int interval_number;

               dummy5 = (Type5 *)(public_xtra -> data[i]);

               for(interval_number = 0; interval_number < interval_division; interval_number++)
               {
                  tfree(((dummy5 -> filenames)[interval_number]));
               }

               tfree((dummy5 -> filenames));

               tfree(dummy5);

               break;
            }

            case 6:
            {
               dummy6 = (Type6 *)(public_xtra -> data[i]);

               tfree(dummy6);

               break;
            }

			//sk
			case 7:
            {
               dummy7 = (Type7 *)(public_xtra -> data[i]);

               tfree((dummy7 -> values));
               tfree(dummy7);

               break;
            }


            }
         }

         tfree((public_xtra -> data));
         tfree((public_xtra -> cycle_numbers));
         tfree((public_xtra -> patch_indexes));
         tfree((public_xtra -> input_types));

         /* Free the time cycling information */
         num_cycles = (public_xtra -> num_cycles);

         tfree((public_xtra -> repeat_counts));

         for(i = 0; i < num_cycles; i++)
         {
            tfree((public_xtra -> intervals[i]));
         }
         tfree((public_xtra -> intervals));

         tfree((public_xtra -> interval_divisions));
      }

      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * BCPressurePackageSizeOfTempData
 *--------------------------------------------------------------------------*/

int  BCPressurePackageSizeOfTempData()
{
   return 0;
}
