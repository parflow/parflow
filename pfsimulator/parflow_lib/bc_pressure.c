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

/*****************************************************************************
*
* The functions in this file are for manipulating the BCPressureData structure
*   in ProblemData and work in conjunction with the BCPressurePackage module.
*
* Because of the times things are called, the New function is twinky
* (it was basically put in to be symmetric with the New/Free paradigm
* used through out the code) and is invoked by SetProblemData.  The Alloc
* function actually allocates the data within the sub-structures and is
* invoked by the WellPackage (which has the data needed to compute the array
* sizes and such).  The Free is smart enough to clean up after both the New
* and Alloc functions and is called by SetProblemData.
*
*****************************************************************************/


/*--------------------------------------------------------------------------
 * NewBCPressureData
 *--------------------------------------------------------------------------*/

BCPressureData *NewBCPressureData()
{
  BCPressureData    *bc_pressure_data;

  bc_pressure_data = ctalloc(BCPressureData, 1);

  BCPressureDataNumPhases(bc_pressure_data) = 0;

  BCPressureDataNumPatches(bc_pressure_data) = -1;

  BCPressureDataTypes(bc_pressure_data) = NULL;

  BCPressureDataCycleNumbers(bc_pressure_data) = NULL;

  BCPressureDataPatchIndexes(bc_pressure_data) = NULL;

  BCPressureDataBCTypes(bc_pressure_data) = NULL;

  BCPressureDataValues(bc_pressure_data) = NULL;

  return bc_pressure_data;
}


/*--------------------------------------------------------------------------
 * FreeBCPressureData
 *--------------------------------------------------------------------------*/

void FreeBCPressureData(
                        BCPressureData *bc_pressure_data)
{
  int i, cycle_number, interval_division, interval_number;

  TimeCycleData *time_cycle_data;

  if (bc_pressure_data)
  {
    time_cycle_data = BCPressureDataTimeCycleData(bc_pressure_data);

    if (BCPressureDataNumPatches(bc_pressure_data) > 0)
    {
      if (BCPressureDataValues(bc_pressure_data))
      {
        for (i = 0; i < BCPressureDataNumPatches(bc_pressure_data); i++)
        {
          if (BCPressureDataIntervalValues(bc_pressure_data, i))
          {
            cycle_number = BCPressureDataCycleNumber(bc_pressure_data, i);
            interval_division = TimeCycleDataIntervalDivision(
                                                              time_cycle_data, cycle_number);
            for (interval_number = 0;
                 interval_number < interval_division;
                 interval_number++)
            {
              switch (BCPressureDataType(bc_pressure_data, i))
              {
                case DirEquilRefPatch:
                {
                  GetBCPressureTypeStruct(DirEquilRefPatch, interval_data, bc_pressure_data,
                                          i, interval_number);

                  if (DirEquilRefPatchValueAtInterfaces(
                                                        interval_data))
                  {
                    tfree(DirEquilRefPatchValueAtInterfaces(
                                                            interval_data));
                  }
                  break;
                }

                case DirEquilPLinear:
                {
                  GetBCPressureTypeStruct(DirEquilPLinear, interval_data, bc_pressure_data,
                                          i, interval_number);

                  if (DirEquilPLinearPoints(interval_data))
                  {
                    tfree(DirEquilPLinearPoints(interval_data));
                  }
                  if (DirEquilPLinearValues(interval_data))
                  {
                    tfree(DirEquilPLinearValues(interval_data));
                  }
                  if (DirEquilPLinearValueAtInterfaces(
                                                       interval_data))
                  {
                    tfree(DirEquilPLinearValueAtInterfaces(
                                                           interval_data));
                  }
                  break;
                }

                // @MCB: Doesn't appear to do anything?
                case FluxConst:
                {
                  GetBCPressureTypeStruct(FluxConst, interval_data, bc_pressure_data,
                                          i, interval_number);
                  break;
                }

                // @MCB: Doesn't appear to do anything?
                case FluxVolumetric:
                {
                  GetBCPressureTypeStruct(FluxVolumetric, interval_data, bc_pressure_data,
                                          i, interval_number);
                  break;
                }

                case PressureFile:
                {
                  GetBCPressureTypeStruct(PressureFile, interval_data, bc_pressure_data,
                                          i, interval_number);

                  if (PressureFileName(interval_data))
                  {
                    tfree(PressureFileName(interval_data));
                  }
                  break;
                }

                case FluxFile:
                {
                  GetBCPressureTypeStruct(FluxFile, interval_data, bc_pressure_data,
                                          i, interval_number);

                  if (FluxFileName(interval_data))
                  {
                    tfree(FluxFileName(interval_data));
                  }
                  break;
                }
              }
              if (BCPressureDataIntervalValue(bc_pressure_data, i,
                                              interval_number))
              {
                tfree(BCPressureDataIntervalValue(bc_pressure_data, i,
                                                  interval_number));
              }
            }
            tfree(BCPressureDataIntervalValues(bc_pressure_data, i));
          }
        }
        tfree(BCPressureDataValues(bc_pressure_data));
      }
      if (BCPressureDataBCTypes(bc_pressure_data))
      {
        tfree(BCPressureDataBCTypes(bc_pressure_data));
      }
      if (BCPressureDataPatchIndexes(bc_pressure_data))
      {
        tfree(BCPressureDataPatchIndexes(bc_pressure_data));
      }
      if (BCPressureDataCycleNumbers(bc_pressure_data))
      {
        tfree(BCPressureDataCycleNumbers(bc_pressure_data));
      }
      if (BCPressureDataTypes(bc_pressure_data))
      {
        tfree(BCPressureDataTypes(bc_pressure_data));
      }
    }

    FreeTimeCycleData(time_cycle_data);

    tfree(bc_pressure_data);
  }
}


/*--------------------------------------------------------------------------
 * PrintBCPressureData
 *--------------------------------------------------------------------------*/

void PrintBCPressureData(
                         BCPressureData *bc_pressure_data)
{
  amps_Printf("Pressure BC Information\n");
  if (BCPressureDataNumPatches(bc_pressure_data) == -1)
  {
    amps_Printf("Pressure BCs have not been setup.\n");
  }
  else if (BCPressureDataNumPatches(bc_pressure_data) == 0)
  {
    amps_Printf("No Pressure BCs.\n");
  }
  else
  {
    amps_Printf("Pressure BCs exist.\n");
  }
}
