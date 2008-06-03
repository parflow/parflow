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
 * The functions in this file are for manipulating the BCTemperatureData structure
 *   in ProblemData and work in conjuction with the BCTemperaturePackage module.
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
 * NewBCTemperatureData
 *--------------------------------------------------------------------------*/

BCTemperatureData *NewBCTemperatureData()
{
   BCTemperatureData    *bc_temperature_data;

   bc_temperature_data = ctalloc(BCTemperatureData, 1);

   BCTemperatureDataNumPhases(bc_temperature_data) = 0;

   BCTemperatureDataNumPatches(bc_temperature_data) = -1;

   BCTemperatureDataTypes(bc_temperature_data) = NULL;

   BCTemperatureDataCycleNumbers(bc_temperature_data) = NULL;

   BCTemperatureDataPatchIndexes(bc_temperature_data) = NULL;

   BCTemperatureDataBCTypes(bc_temperature_data) = NULL;

   BCTemperatureDataValues(bc_temperature_data) = NULL;

   return bc_temperature_data;
}


/*--------------------------------------------------------------------------
 * FreeBCTemperatureData
 *--------------------------------------------------------------------------*/

void FreeBCTemperatureData(bc_temperature_data)
BCTemperatureData *bc_temperature_data;
{
   int            i, cycle_number, interval_division, interval_number;

   TimeCycleData *time_cycle_data;

   if ( bc_temperature_data )
   {
      time_cycle_data = BCTemperatureDataTimeCycleData(bc_temperature_data);

      if (BCTemperatureDataNumPatches(bc_temperature_data) > 0)
      {
         if (BCTemperatureDataValues(bc_temperature_data))
         {
            for(i = 0; i < BCTemperatureDataNumPatches(bc_temperature_data); i++)
            {
               if (BCTemperatureDataIntervalValues(bc_temperature_data,i))
               {
                  cycle_number = BCTemperatureDataCycleNumber(bc_temperature_data,i);
                  interval_division = TimeCycleDataIntervalDivision(
					 time_cycle_data, cycle_number);
                  for(interval_number = 0; 
		      interval_number < interval_division; 
		      interval_number++)
                  {
                     switch(BCTemperatureDataType(bc_temperature_data,i))
                     {
                     case 0:
                     {
                        BCTemperatureType0 *bc_temperature_type0;
 
                        bc_temperature_type0 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);

                        break;
                     }
                     case 1:
                     {
                        BCTemperatureType1 *bc_temperature_type1;

                        bc_temperature_type1 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);
                        if (BCTemperatureType1Points(bc_temperature_type1))
                        {
                           tfree(BCTemperatureType1Points(bc_temperature_type1));
                        }
                        if (BCTemperatureType1Values(bc_temperature_type1))
                        {
                           tfree(BCTemperatureType1Values(bc_temperature_type1));
                        }
                        if (BCTemperatureType1ValueAtInterfaces(
                                                            bc_temperature_type1))
                        {
                           tfree(BCTemperatureType1ValueAtInterfaces(
                                                           bc_temperature_type1));
                        }
                        break;
                     }
                     case 2:
                     {
                        BCTemperatureType2 *bc_temperature_type2;

                        bc_temperature_type2 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);

                        break;
                     }
                     case 3:
                     {
                        BCTemperatureType3 *bc_temperature_type3;

                        bc_temperature_type3 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);

                        break;
                     }
                     case 4:
                     {
                        BCTemperatureType4 *bc_temperature_type4;

                        bc_temperature_type4 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);
                        if (BCTemperatureType4FileName(bc_temperature_type4))
                        {
                           tfree(BCTemperatureType4FileName(bc_temperature_type4));
                        }
                        break;
                     }
                     case 5:
                     {
                        BCTemperatureType5 *bc_temperature_type5;

                        bc_temperature_type5 = BCTemperatureDataIntervalValue(
                                           bc_temperature_data,i,interval_number);
                        if (BCTemperatureType5FileName(bc_temperature_type5))
                        {
                           tfree(BCTemperatureType5FileName(bc_temperature_type5));
                        }
                        break;
                     }
                     }
                     if (BCTemperatureDataIntervalValue(bc_temperature_data,i,
						     interval_number))
                     {
                        tfree(BCTemperatureDataIntervalValue(bc_temperature_data,i,
							  interval_number));
                     }
                  }
                  tfree(BCTemperatureDataIntervalValues(bc_temperature_data,i));
               }
            }
            tfree(BCTemperatureDataValues(bc_temperature_data));
         }
         if (BCTemperatureDataBCTypes(bc_temperature_data))
         {
            tfree(BCTemperatureDataBCTypes(bc_temperature_data));
         }
         if (BCTemperatureDataPatchIndexes(bc_temperature_data))
         {
            tfree(BCTemperatureDataPatchIndexes(bc_temperature_data));
         }
         if (BCTemperatureDataCycleNumbers(bc_temperature_data))
         {
            tfree(BCTemperatureDataCycleNumbers(bc_temperature_data));
         }
         if (BCTemperatureDataTypes(bc_temperature_data))
         {
            tfree(BCTemperatureDataTypes(bc_temperature_data));
         }
      }

      FreeTimeCycleData(time_cycle_data);

      tfree(bc_temperature_data);
   }
}


/*--------------------------------------------------------------------------
 * PrintBCTemperatureData
 *--------------------------------------------------------------------------*/

void PrintBCTemperatureData(bc_temperature_data)
BCTemperatureData *bc_temperature_data;
{
   amps_Printf("Pressure BC Information\n");
   if ( BCTemperatureDataNumPatches(bc_temperature_data) == -1 )
   {
      amps_Printf("Pressure BCs have not been setup.\n");
   }
   else if ( BCTemperatureDataNumPatches(bc_temperature_data) == 0 )
   {
      amps_Printf("No Pressure BCs.\n");
   }
   else
   {
      amps_Printf("Pressure BCs exist.\n");
   }
}
