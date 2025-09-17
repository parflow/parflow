/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#include "parflow.h"

#include <string.h>


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
  int num_phases;

  /* cycling info */
  int num_cycles;

  int    *interval_divisions;
  int   **intervals;
  int    *repeat_counts;

  /* patch info */
  NameArray patches;
  int num_patches;

  int    *input_types;    /* num_patches input types */
  int    *patch_indexes;  /* num_patches patch indexes */
  int    *cycle_numbers;  /* num_patches cycle numbers */
  void  **data;           /* num_patches pointers to Type structures */

  int using_overland_flow;
} PublicXtra;

typedef struct {
  Problem *problem;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * Generate BC Type Structures from header file
 * These are what used to be the Type0 Type1 ... structs
 *--------------------------------------------------------------------------*/
#define BC_TYPE(type, values) typedef struct values Type ## type;
BC_TYPE_TABLE
#undef BC_TYPE


/*--------------------------------------------------------------------------
 * BCPressurePackage
 *--------------------------------------------------------------------------*/

void         BCPressurePackage(
                               ProblemData *problem_data)
{
  PFModule         *this_module = ThisPFModule;
  PublicXtra       *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  BCPressureData   *bc_pressure_data
    = ProblemDataBCPressureData(problem_data);

  TimeCycleData    *time_cycle_data;

  int num_patches;
  int i;
  int cycle_length, cycle_number, interval_division,
      interval_number;

  /* Allocate the bc data */
  BCPressureDataNumPhases(bc_pressure_data) = (public_xtra->num_phases);

  BCPressureDataNumPatches(bc_pressure_data) = (public_xtra->num_patches);

  if ((public_xtra->num_patches) > 0)
  {
    /* Load the time cycle data */
    time_cycle_data = NewTimeCycleData((public_xtra->num_cycles),
                                       (public_xtra->interval_divisions));

    for (cycle_number = 0; cycle_number < (public_xtra->num_cycles);
         cycle_number++)
    {
      TimeCycleDataIntervalDivision(time_cycle_data, cycle_number)
        = (public_xtra->interval_divisions[cycle_number]);
      cycle_length = 0;
      for (interval_number = 0;
           interval_number < (public_xtra->
                              interval_divisions[cycle_number]);
           interval_number++)
      {
        cycle_length += (public_xtra->intervals[cycle_number])[interval_number];
        TimeCycleDataInterval(time_cycle_data, cycle_number, interval_number) = (public_xtra->intervals[cycle_number])[interval_number];
      }
      TimeCycleDataRepeatCount(time_cycle_data, cycle_number) = (public_xtra->repeat_counts[cycle_number]);
      TimeCycleDataCycleLength(time_cycle_data, cycle_number) = cycle_length;
    }

    BCPressureDataTimeCycleData(bc_pressure_data) = time_cycle_data;

    /* Load the Boundary Condition Data */

    num_patches = BCPressureDataNumPatches(bc_pressure_data);
    BCPressureDataTypes(bc_pressure_data) = ctalloc(int, num_patches);
    BCPressureDataPatchIndexes(bc_pressure_data) = ctalloc(int, num_patches);
    BCPressureDataCycleNumbers(bc_pressure_data) = ctalloc(int, num_patches);
    BCPressureDataBCTypes(bc_pressure_data) = ctalloc(int, num_patches);
    BCPressureDataValues(bc_pressure_data) = ctalloc(void **,
                                                     num_patches);

    for (i = 0; i < num_patches; i++)
    {
      BCPressureDataType(bc_pressure_data, i) = (public_xtra->input_types[i]);
      BCPressureDataPatchIndex(bc_pressure_data, i) = (public_xtra->patch_indexes[i]);
      BCPressureDataCycleNumber(bc_pressure_data, i) = (public_xtra->cycle_numbers[i]);

      interval_division = TimeCycleDataIntervalDivision(time_cycle_data, BCPressureDataCycleNumber(bc_pressure_data, i));
      BCPressureDataIntervalValues(bc_pressure_data, i) = ctalloc(void *, interval_division);
      for (interval_number = 0; interval_number < interval_division; interval_number++)
      {
        switch ((public_xtra)->input_types[(i)])
        {
#if 0 /* Do not uncomment this block, it is example code */
          case MyNewPatchType:
          {
            /* Allocate the struct for an interval */
            NewBCPressureTypeStruct(MyNewPatchType, interval_data);

            /* Set the BC type for use in BCStructPatchLoop branching (defined in problem_bc.h) */
            BCPressureDataBCType(bc_pressure_data, i) = MyNewPatchTypeBC;

            /* Retrieve the Type struct allocated in BCPressurePackageNewPublicXtra */
            GetTypeStruct(MyNewPatchType, data, public_xtra, i);

            /* Assign the struct values for the given interval */
            MyNewPatchTypeValue(interval_data)
              = (data->values[interval_number]);

            /* Save the pointer into bc_pressure_data */
            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End MyNewPatchType */
#endif // #if 0

          /* Setup a fixed pressure condition structure */
          case DirEquilRefPatch:
          {
            NewBCPressureTypeStruct(DirEquilRefPatch, interval_data);

            int phase;
            int num_phases = BCPressureDataNumPhases(bc_pressure_data);

            BCPressureDataBCType(bc_pressure_data, i) = DirichletBC;

            GetTypeStruct(DirEquilRefPatch, data, public_xtra, i);

            DirEquilRefPatchRefSolid(interval_data) =
              (data->reference_solid);

            DirEquilRefPatchRefPatch(interval_data) =
              (data->reference_patch);

            DirEquilRefPatchValue(interval_data) = (data->values[interval_number]);

            if (num_phases > 1)
            {
              DirEquilRefPatchValueAtInterfaces(interval_data) = ctalloc(double, (num_phases - 1));
              for (phase = 1; phase < num_phases; phase++)
              {
                DirEquilRefPatchValueAtInterface(interval_data, phase)
                  = ((data->value_at_interface[interval_number])[phase - 1]);
              }
            }
            else
            {
              DirEquilRefPatchValueAtInterfaces(interval_data) = NULL;
            }

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number) = (void*)interval_data;
            break;
          } /* End DirEquilRefPatch */

          /* Setup a piecewise linear pressure condition structure */
          case DirEquilPLinear:
          {
            int point;
            int phase;

            int num_points;
            int num_phases = BCPressureDataNumPhases(bc_pressure_data);

            NewBCPressureTypeStruct(DirEquilPLinear, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = DirichletBC;

            GetTypeStruct(DirEquilPLinear, data, public_xtra, i);

            num_points = (data->num_points[interval_number]);

            DirEquilPLinearXLower(interval_data) = (data->xlower[interval_number]);
            DirEquilPLinearYLower(interval_data) = (data->ylower[interval_number]);
            DirEquilPLinearXUpper(interval_data) = (data->xupper[interval_number]);
            DirEquilPLinearYUpper(interval_data) = (data->yupper[interval_number]);
            DirEquilPLinearNumPoints(interval_data) = (data->num_points[interval_number]);

            DirEquilPLinearPoints(interval_data) = ctalloc(double, num_points);
            DirEquilPLinearValues(interval_data) = ctalloc(double, num_points);

            for (point = 0; point < num_points; point++)
            {
              DirEquilPLinearPoint(interval_data, point) = ((data->points[interval_number])[point]);
              DirEquilPLinearValue(interval_data, point) = ((data->values[interval_number])[point]);
            }

            if (num_phases > 1)
            {
              DirEquilPLinearValueAtInterfaces(interval_data) = ctalloc(double, (num_phases - 1));

              for (phase = 1; phase < num_phases; phase++)
              {
                DirEquilPLinearValueAtInterface(interval_data, phase) = ((data->value_at_interface[interval_number])[phase - 1]);
              }
            }
            else
            {
              DirEquilPLinearValueAtInterfaces(interval_data) = NULL;
            }

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number) = (void*)interval_data;

            break;
          } /* End DirEquilPLinear */

          /* Setup a constant flux condition structure */
          case FluxConst:
          {
            NewBCPressureTypeStruct(FluxConst, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = FluxBC;

            GetTypeStruct(FluxConst, data, public_xtra, i);

            FluxConstValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End FluxConst */

          /* Setup a volumetric flux condition structure */
          case FluxVolumetric:
          {
            NewBCPressureTypeStruct(FluxVolumetric, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = FluxBC;

            GetTypeStruct(FluxVolumetric, data, public_xtra, i);

            FluxVolumetricValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;
            break;
          } /* End FluxVolumetric */

          /* Setup a file defined pressure condition structure */
          case PressureFile:
          {
            NewBCPressureTypeStruct(PressureFile, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = DirichletBC;

            GetTypeStruct(PressureFile, data, public_xtra, i);

            PressureFileName(interval_data)
              = ctalloc(char, strlen((data->filenames)[interval_number]) + 1);

            strcpy(PressureFileName(interval_data),
                   ((data->filenames)[interval_number]));

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;
            break;
          } /* End PressureFile */

          /* Setup a file defined flux condition structure */
          case FluxFile:
          {
            NewBCPressureTypeStruct(FluxFile, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = FluxBC;

            GetTypeStruct(FluxFile, data, public_xtra, i);

            FluxFileName(interval_data)
              = ctalloc(char, strlen((data->filenames)[interval_number]) + 1);

            strcpy(FluxFileName(interval_data),
                   ((data->filenames)[interval_number]));

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End FluxFile */

          /* Setup a Dir. pressure MATH problem condition */
          case ExactSolution:
          {
            NewBCPressureTypeStruct(ExactSolution, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = DirichletBC;

            GetTypeStruct(ExactSolution, data, public_xtra, i);

            ExactSolutionFunctionType(interval_data)
              = (data->function_type);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End ExactSolution */

          /*//sk  Setup a overland flow condition structure */
          case OverlandFlow:
          {
            NewBCPressureTypeStruct(OverlandFlow, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = OverlandBC;

            GetTypeStruct(OverlandFlow, data, public_xtra, i);

            OverlandFlowValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;
            break;
          } /* End OverlandFlow */

          /* Setup a file defined flux condition structure for overland flow BC*/
          case OverlandFlowPFB:
          {
            NewBCPressureTypeStruct(OverlandFlowPFB, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = OverlandBC;

            GetTypeStruct(OverlandFlowPFB, data, public_xtra, i);

            OverlandFlowPFBFileName(interval_data)
              = ctalloc(char, strlen((data->filenames)[interval_number]) + 1);

            strcpy(OverlandFlowPFBFileName(interval_data),
                   ((data->filenames)[interval_number]));

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End OverlandFlowPFB */

          /* Set up a seepage face condition structure */
          case SeepageFace:
          {
            NewBCPressureTypeStruct(SeepageFace, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = SeepageFaceBC;

            GetTypeStruct(SeepageFace, data, public_xtra, i);

            SeepageFaceValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;
            break;
          } /* End SeepageFace */

          /* Set up overland OverlandKinematic condition structure */
          case OverlandKinematic:
          {
            NewBCPressureTypeStruct(OverlandKinematic, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = OverlandKinematicBC;

            GetTypeStruct(OverlandKinematic, data, public_xtra, i);

            OverlandKinematicValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End OverlandKinematic */

          /* Set up overland OverlandDiffusive condition structure */
          case OverlandDiffusive:
          {
            NewBCPressureTypeStruct(OverlandDiffusive, interval_data);

            BCPressureDataBCType(bc_pressure_data, i) = OverlandDiffusiveBC;

            GetTypeStruct(OverlandDiffusive, data, public_xtra, i);

            OverlandDiffusiveValue(interval_data)
              = (data->values[interval_number]);

            BCPressureDataIntervalValue(bc_pressure_data, i, interval_number)
              = (void*)interval_data;

            break;
          } /* End OverlandDiffusive */

          default:
          {
            PARFLOW_ERROR("Invalid BC input type");
          }
        } /* End switch BC type  */
      } /* End for interval */
    } /* End for patch */
  }
}

/*--------------------------------------------------------------------------
 * BCPressurePackageInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule *BCPressurePackageInitInstanceXtra(
                                            Problem *problem)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (problem != NULL)
  {
    (instance_xtra->problem) = problem;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * BCPressurePackageFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  BCPressurePackageFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * BCPressurePackageNewPublicXtra
 *--------------------------------------------------------------------------*/
PFModule  *BCPressurePackageNewPublicXtra(
                                          int num_phases)
{
  PFModule      *this_module = ThisPFModule;
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

  int num_patches;
  int num_cycles;

  int i, interval_number, interval_division;

  NameArray type_na;
  NameArray function_na;

  /* @MCB: Generate the magic string that determines indexing
   * This will stringify the type names and the compiler will concat for us */
#define BC_TYPE(a, b) #a " "
  type_na = NA_NewNameArray(BC_TYPE_TABLE);
#undef BC_TYPE

  function_na = NA_NewNameArray("dum0 X XPlusYPlusZ X3Y2PlusSinXYPlus1 X3Y4PlusX2PlusSinXYCosYPlus1 XYZTPlus1 XYZTPlus1PermTensor");

  /* allocate space for the public_xtra structure */
  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->num_phases) = num_phases;

  patch_names = GetString("BCPressure.PatchNames");

  public_xtra->patches = NA_NewNameArray(patch_names);
  num_patches = NA_Sizeof(public_xtra->patches);

  (public_xtra->num_patches) = num_patches;
  (public_xtra->num_cycles) = num_patches;

  if (num_patches > 0)
  {
    (public_xtra->num_cycles) = num_cycles = num_patches;

    (public_xtra->interval_divisions) = ctalloc(int, num_cycles);
    (public_xtra->intervals) = ctalloc(int *, num_cycles);
    (public_xtra->repeat_counts) = ctalloc(int, num_cycles);

    (public_xtra->input_types) = ctalloc(int, num_patches);
    (public_xtra->patch_indexes) = ctalloc(int, num_patches);
    (public_xtra->cycle_numbers) = ctalloc(int, num_patches);
    (public_xtra->data) = ctalloc(void *, num_patches);
    (public_xtra->using_overland_flow) = FALSE;

    /* Determine the domain geom index from domain name */
    switch_name = GetString("Domain.GeomName");
    domain_index = NA_NameToIndexExitOnError(GlobalsGeomNames, switch_name, "Domain.GeomName");

    ForEachPatch(num_patches, i)
    {
      patch_name = NA_IndexToName(public_xtra->patches, i);

      public_xtra->patch_indexes[i] =
        NA_NameToIndex(GlobalsGeometries[domain_index]->patches,
                       patch_name);

      if (public_xtra->patch_indexes[i] < 0)
      {
        amps_Printf("Invalid patch name <%s>\n", patch_name);
        NA_InputError(GlobalsGeometries[domain_index]->patches, patch_name, "");
      }

      sprintf(key, "Patch.%s.BCPressure.Type", patch_name);
      switch_name = GetString(key);
      public_xtra->input_types[i] = NA_NameToIndexExitOnError(type_na, switch_name, key);

      sprintf(key, "Patch.%s.BCPressure.Cycle", patch_name);
      cycle_name = GetString(key);
      public_xtra->cycle_numbers[i] = i;
      global_cycle = NA_NameToIndexExitOnError(GlobalsCycleNames, cycle_name, key);

      interval_division = public_xtra->interval_divisions[i] =
        GlobalsIntervalDivisions[global_cycle];

      public_xtra->repeat_counts[i] =
        GlobalsRepeatCounts[global_cycle];

      (public_xtra->intervals[i]) = ctalloc(int, interval_division);

      ForEachInterval(interval_division, interval_number)
      {
        public_xtra->intervals[i][interval_number] =
          GlobalsIntervals[global_cycle][interval_number];
      }

      switch ((public_xtra)->input_types[(i)])
      {
#if 0 /* Do not undef this block, it is example code not for actual use */
      /* Example flow for setting up TypeStruct for boundary conditions */
      /*
       * switch NewPatchType:
       * {
       * // Allocate the struct, second parameter is whatever variable name you wish to use in this scope
       * NewTypeStruct(MyNewPatchType, data);
       *
       * // Allocate struct data if necessary
       * (data->values) = ctalloc(double, interval_division);
       *
       * // Populate the data with something, e.g. values from a file
       * ForEachInterval(interval_division, interval_number)
       * {
       *  sprintf(key, "Patch.%s.BCPressure.%s.Value",
       *          patch_name,
       *          NA_IndexToName(
       *                         GlobalsIntervalNames[global_cycle],
       *                         interval_number));
       *
       *  data->values[interval_number] = GetDouble(key);
       * }
       *
       * // Store the allocated and populated Type struct into public_xtra
       * StoreTypeStruct(public_xtra, data, i);
       * break;
       * } // End Example
       */
#endif // #if 0

        case DirEquilRefPatch:
        {
          int size;

          NewTypeStruct(DirEquilRefPatch, data);

          (data->values) = ctalloc(double,
                                   interval_division);
          (data->value_at_interface) = ctalloc(double *,
                                               interval_division);

          sprintf(key, "Patch.%s.BCPressure.RefGeom", patch_name);
          switch_name = GetString(key);

          data->reference_solid = NA_NameToIndex(GlobalsGeomNames,
                                                 switch_name);

          if (data->reference_solid < 0)
          {
            InputError("Error: invalid geometry name <%s> for reference solid <%s>\n", switch_name, key);
          }

          sprintf(key, "Patch.%s.BCPressure.RefPatch", patch_name);
          switch_name = GetString(key);

          data->reference_patch =
            NA_NameToIndexExitOnError(GeomSolidPatches(
                                                       GlobalsGeometries[data->reference_solid]),
                                      switch_name, key);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(
                                   GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);

            if (num_phases > 1)
            {
              size = (num_phases - 1);

              (data->value_at_interface[interval_number]) =
                ctalloc(double, size);

              for (phase = 1; phase < num_phases; phase++)
              {
                sprintf(key, "Patch.%s.BCPressure.%s.%s.IntValue",
                        patch_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                data->value_at_interface[interval_number][phase] =
                  GetDouble(key);
              }
            }
            else
            {
              (data->value_at_interface[interval_number]) = NULL;
            }
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End DirEquilRefPatch */

        case DirEquilPLinear:
        {
          int k;
          int num_points;
          int size;

          NewTypeStruct(DirEquilPLinear, data);

          (data->xlower) = ctalloc(double, interval_division);
          (data->ylower) = ctalloc(double, interval_division);
          (data->xupper) = ctalloc(double, interval_division);
          (data->yupper) = ctalloc(double, interval_division);

          (data->num_points) = ctalloc(int, interval_division);

          (data->points) = ctalloc(double *, interval_division);
          (data->values) = ctalloc(double *, interval_division);
          (data->value_at_interface) = ctalloc(double *, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            interval_name =
              NA_IndexToName(GlobalsIntervalNames[global_cycle],
                             interval_number);

            /* read in the xy-line */
            sprintf(key, "Patch.%s.BCPressure.%s.XLower", patch_name,
                    interval_name);
            data->xlower[interval_number] = GetDouble(key);

            sprintf(key, "Patch.%s.BCPressure.%s.YLower", patch_name,
                    interval_name);
            data->ylower[interval_number] = GetDouble(key);

            sprintf(key, "Patch.%s.BCPressure.%s.XUpper", patch_name,
                    interval_name);
            data->xupper[interval_number] = GetDouble(key);

            sprintf(key, "Patch.%s.BCPressure.%s.YUpper", patch_name,
                    interval_name);
            data->yupper[interval_number] = GetDouble(key);

            /* read num_points */
            sprintf(key, "Patch.%s.BCPressure.%s.NumPoints", patch_name,
                    interval_name);
            num_points = GetInt(key);

            (data->num_points[interval_number]) = num_points;

            (data->points[interval_number]) = ctalloc(double, num_points);
            (data->values[interval_number]) = ctalloc(double, num_points);

            for (k = 0; k < num_points; k++)
            {
              sprintf(key, "Patch.%s.BCPressure.%s.%d.Location",
                      patch_name, interval_name, k);
              data->points[interval_number][k] = GetDouble(key);

              sprintf(key, "Patch.%s.BCPressure.%s.%d.Value",
                      patch_name, interval_name, k);
              data->values[interval_number][k] = GetDouble(key);
            }

            if (num_phases > 1)
            {
              size = (num_phases - 1);

              (data->value_at_interface[interval_number]) =
                ctalloc(double, size);

              for (phase = 1; phase < num_phases; phase++)
              {
                sprintf(key, "Patch.%s.BCPressure.%s.%s.IntValue",
                        patch_name,
                        NA_IndexToName(
                                       GlobalsIntervalNames[global_cycle],
                                       interval_number),
                        NA_IndexToName(GlobalsPhaseNames, phase));

                data->value_at_interface[interval_number][phase] =
                  GetDouble(key);
              }
            }
            else
            {
              (data->value_at_interface[interval_number]) = NULL;
            }
          }

          StoreTypeStruct(public_xtra, data, i);
          break;
        } /* End DirEquilPLinear */

        case FluxConst:
        {
          NewTypeStruct(FluxConst, data);

          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End FluxConst */

        case FluxVolumetric:
        {
          NewTypeStruct(FluxVolumetric, data);

          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End FluxVolumetric */

        case PressureFile:
        {
          NewTypeStruct(PressureFile, data);

          (data->filenames) = ctalloc(char *, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.FileName",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->filenames[interval_number] = GetString(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End PressureFile */

        case FluxFile:
        {
          NewTypeStruct(FluxFile, data);

          (data->filenames) = ctalloc(char *, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.FileName",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->filenames[interval_number] = GetString(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End FluxFile */

        case ExactSolution:
        {
          NewTypeStruct(ExactSolution, data);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.PredefinedFunction",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));
            switch_name = GetString(key);

            data->function_type = NA_NameToIndexExitOnError(function_na,
                                                            switch_name, key);

            // MCB: This is overwriting the data struct inside the for loop
            // Also structured this way in master branch without changes
            // Bug? Intentional?
            StoreTypeStruct(public_xtra, data, i);
          }

          break;
        } /* End ExactSolution */

        case OverlandFlow:
        {
          NewTypeStruct(OverlandFlow, data);

          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End OverlandFlow */

        case OverlandFlowPFB:
        {
          NewTypeStruct(OverlandFlowPFB, data);

          (data->filenames) = ctalloc(char *, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.FileName",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->filenames[interval_number] = GetString(key);
          }

          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End OverlandFlowPFB */

        case SeepageFace:
        {
          /* Constant "rainfall" rate value on patch */
          NewTypeStruct(SeepageFace, data);

          /* MCB: Need to setup values for patch */
          /* Something akin to */

          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }
          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End SeepageFace */

        case OverlandKinematic:
        {
          /* Constant "rainfall" rate value on patch */
          NewTypeStruct(OverlandKinematic, data);
          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }
          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End OverlandKinematic */

        case OverlandDiffusive:
        {
          /* Constant "rainfall" rate value on patch */
          NewTypeStruct(OverlandDiffusive, data);
          (data->values) = ctalloc(double, interval_division);

          ForEachInterval(interval_division, interval_number)
          {
            sprintf(key, "Patch.%s.BCPressure.%s.Value",
                    patch_name,
                    NA_IndexToName(GlobalsIntervalNames[global_cycle],
                                   interval_number));

            data->values[interval_number] = GetDouble(key);
          }
          StoreTypeStruct(public_xtra, data, i);

          break;
        } /* End OverlandDiffusive */
      } /* End switch types */

      switch ((public_xtra)->input_types[(i)])
      {
        case OverlandFlow:
        case OverlandFlowPFB:
        case OverlandKinematic:
        case OverlandDiffusive:
        {
          (public_xtra->using_overland_flow) = TRUE;
          break;
        }
      }
    } /* End for patches */
  } /* if patches */

  NA_FreeNameArray(type_na);
  NA_FreeNameArray(function_na);

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

// This function is a hack to take information from the PublicXtra structure
int BCPressurePackageUsingOverlandFlow(Problem *problem)
{
  PFModule *bc_pressure = ProblemBCPressurePackage(problem);
  PublicXtra *public_xtra = (PublicXtra*)PFModulePublicXtra(bc_pressure);

  return(public_xtra->using_overland_flow);
}

/*-------------------------------------------------------------------------
 * BCPressurePackageFreePublicXtra
 *-------------------------------------------------------------------------*/

void  BCPressurePackageFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  int num_patches, num_cycles;
  int i, interval_number, interval_division;

  if (public_xtra)
  {
    /* Free the well information */
    num_patches = (public_xtra->num_patches);
    NA_FreeNameArray(public_xtra->patches);

    if (num_patches > 0)
    {
      for (i = 0; i < num_patches; i++)
      {
        interval_division = (public_xtra->interval_divisions[(public_xtra->cycle_numbers[i])]);
        switch ((public_xtra)->input_types[(i)])
        {
          case DirEquilRefPatch:
          {
            GetTypeStruct(DirEquilRefPatch, data, public_xtra, i);

            ForEachInterval(interval_division, interval_number)
            {
              tfree((data->value_at_interface[interval_number]));
            }

            tfree((data->value_at_interface));
            tfree((data->values));

            tfree(data);
            break;
          }

          case DirEquilPLinear:
          {
            int interval_number;

            GetTypeStruct(DirEquilPLinear, data, public_xtra, i);

            ForEachInterval(interval_division, interval_number)
            {
              tfree((data->value_at_interface[interval_number]));
              tfree((data->values[interval_number]));
              tfree((data->points[interval_number]));
            }

            tfree((data->value_at_interface));
            tfree((data->points));
            tfree((data->values));

            tfree((data->num_points));

            tfree((data->yupper));
            tfree((data->xupper));
            tfree((data->ylower));
            tfree((data->xlower));

            tfree(data);
            break;
          }

          case FluxConst:
          {
            GetTypeStruct(FluxConst, data, public_xtra, i);
            tfree((data->values));
            tfree(data);
            break;
          }

          case FluxVolumetric:
          {
            GetTypeStruct(FluxVolumetric, data, public_xtra, i);
            tfree((data->values));
            tfree(data);
            break;
          }

          case PressureFile:
          {
            GetTypeStruct(PressureFile, data, public_xtra, i);
            tfree((data->filenames));
            tfree(data);
            break;
          }

          case FluxFile:
          {
            GetTypeStruct(FluxFile, data, public_xtra, i);
            tfree((data->filenames));
            tfree(data);
            break;
          }

          case ExactSolution:
          {
            GetTypeStruct(ExactSolution, data, public_xtra, i);
            tfree(data);
            break;
          }

          //sk
          case OverlandFlow:
          {
            GetTypeStruct(OverlandFlow, data, public_xtra, i);
            tfree((data->values));
            tfree(data);
            break;
          }

          //RMM
          case OverlandFlowPFB:
          {
            GetTypeStruct(OverlandFlowPFB, data, public_xtra, i);
            tfree((data->filenames));
            tfree(data);
            break;
          }

          case SeepageFace:
          {
            GetTypeStruct(SeepageFace, data, public_xtra, i);
            tfree(data->values);
            tfree(data);
            break;
          };

          case OverlandKinematic:
          {
            GetTypeStruct(OverlandKinematic, data, public_xtra, i);
            tfree(data->values);
            tfree(data);
            break;
          }

          case OverlandDiffusive:
          {
            GetTypeStruct(OverlandDiffusive, data, public_xtra, i);
            tfree(data->values);
            tfree(data);
            break;
          }

          default:
          {
            PARFLOW_ERROR("Invalid BC input type");
          }
        } /* End switch type */
      } /* End for patch */

      tfree((public_xtra->data));
      tfree((public_xtra->cycle_numbers));
      tfree((public_xtra->patch_indexes));
      tfree((public_xtra->input_types));

      /* Free the time cycling information */
      num_cycles = (public_xtra->num_cycles);

      tfree((public_xtra->repeat_counts));

      for (i = 0; i < num_cycles; i++)
      {
        tfree((public_xtra->intervals[i]));
      }
      tfree((public_xtra->intervals));

      tfree((public_xtra->interval_divisions));
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
