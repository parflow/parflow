/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

/******************************************************************************
 *
 * The functions in this file are for manipulating the BCPressureData structure
 *   in ProblemData and work in conjuction with the BCPressurePackage module.
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

void FreeBCPressureData(bc_pressure_data)
BCPressureData *bc_pressure_data;
{
   int            i, cycle_number, interval_division, interval_number;

   TimeCycleData *time_cycle_data;

   if ( bc_pressure_data )
   {
      time_cycle_data = BCPressureDataTimeCycleData(bc_pressure_data);

      if (BCPressureDataNumPatches(bc_pressure_data) > 0)
      {
         if (BCPressureDataValues(bc_pressure_data))
         {
            for(i = 0; i < BCPressureDataNumPatches(bc_pressure_data); i++)
            {
               if (BCPressureDataIntervalValues(bc_pressure_data,i))
               {
                  cycle_number = BCPressureDataCycleNumber(bc_pressure_data,i);
                  interval_division = TimeCycleDataIntervalDivision(
					 time_cycle_data, cycle_number);
                  for(interval_number = 0; 
		      interval_number < interval_division; 
		      interval_number++)
                  {
                     switch(BCPressureDataType(bc_pressure_data,i))
                     {
                     case 0:
                     {
                        BCPressureType0 *bc_pressure_type0;

                        bc_pressure_type0 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);
                        if (BCPressureType0ValueAtInterfaces(
                            bc_pressure_type0))
                        {
                           tfree(BCPressureType0ValueAtInterfaces(
                                                           bc_pressure_type0));
                        }
                        break;
                     }
                     case 1:
                     {
                        BCPressureType1 *bc_pressure_type1;

                        bc_pressure_type1 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);
                        if (BCPressureType1Points(bc_pressure_type1))
                        {
                           tfree(BCPressureType1Points(bc_pressure_type1));
                        }
                        if (BCPressureType1Values(bc_pressure_type1))
                        {
                           tfree(BCPressureType1Values(bc_pressure_type1));
                        }
                        if (BCPressureType1ValueAtInterfaces(
                                                            bc_pressure_type1))
                        {
                           tfree(BCPressureType1ValueAtInterfaces(
                                                           bc_pressure_type1));
                        }
                        break;
                     }
                     case 2:
                     {
                        BCPressureType2 *bc_pressure_type2;

                        bc_pressure_type2 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);

                        break;
                     }
                     case 3:
                     {
                        BCPressureType3 *bc_pressure_type3;

                        bc_pressure_type3 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);

                        break;
                     }
                     case 4:
                     {
                        BCPressureType4 *bc_pressure_type4;

                        bc_pressure_type4 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);
                        if (BCPressureType4FileName(bc_pressure_type4))
                        {
                           tfree(BCPressureType4FileName(bc_pressure_type4));
                        }
                        break;
                     }
                     case 5:
                     {
                        BCPressureType5 *bc_pressure_type5;

                        bc_pressure_type5 = BCPressureDataIntervalValue(
                                           bc_pressure_data,i,interval_number);
                        if (BCPressureType5FileName(bc_pressure_type5))
                        {
                           tfree(BCPressureType5FileName(bc_pressure_type5));
                        }
                        break;
                     }
                     }
                     if (BCPressureDataIntervalValue(bc_pressure_data,i,
						     interval_number))
                     {
                        tfree(BCPressureDataIntervalValue(bc_pressure_data,i,
							  interval_number));
                     }
                  }
                  tfree(BCPressureDataIntervalValues(bc_pressure_data,i));
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

void PrintBCPressureData(bc_pressure_data)
BCPressureData *bc_pressure_data;
{
   amps_Printf("Pressure BC Information\n");
   if ( BCPressureDataNumPatches(bc_pressure_data) == -1 )
   {
      amps_Printf("Pressure BCs have not been setup.\n");
   }
   else if ( BCPressureDataNumPatches(bc_pressure_data) == 0 )
   {
      amps_Printf("No Pressure BCs.\n");
   }
   else
   {
      amps_Printf("Pressure BCs exist.\n");
   }
}
