/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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

// - Holds a variety of utilities for building and manipulating solid files

#include "parflow_config.h"

#include <stdlib.h>

#include "solidtools.h"
#include "tools_io.h"

#include "string.h"
#include "unistd.h"


/*-----------------------------------------------------------------------
* Returns index of array "in_list[len]" that holds "val" if it exists, -1 otherwise
-----------------------------------------------------------------------*/
int ScanArray(int val, int in_list[], int len) //
  {
   int n=-1;
   int i;

   for (i=0; i< len; ++i)
   {
     if (in_list[i]==val)
     {
       n=i;
       break;
     }
   }

  return n;
  }

  /*-----------------------------------------------------------------------
   * Write a complex solid file with patches requires a mask that is an indicator
   * file where cells with value 1 denote active domain, 0 denotes inactive, but
   * any other integer is an inactive cell with a user defined patch
   *
   * NBE: 2019-07-15
   *
   *-----------------------------------------------------------------------*/
void            PrintPatchySolid(
                             FILE *   fp,
                             FILE *   fp_vtk,
                             Databox *msk,
                             Databox *top,
                             Databox *bot,
                             int bin_out)
{
  int i,j,k;
  int NX = DataboxNx(msk);
  int NY = DataboxNy(msk);
  int NZ = DataboxNz(msk);

  // Default patches
  int DefPatches[] = {1,2,3,4,5,6}; // Bottom, Top, West, East, South, North
  int DefPCounts[]={0,0,0,0,0,0};   // Count for each face patch

  int np_tot=20;
  int UsrPatches[np_tot];           // User defined patches, up to np_tot
  int UsrPCounts[np_tot];           // Counts for number of patches on user patches
  int np_usr=-1;                    // Counter for total user patch
  for (i = 0; i < np_tot; ++i)
  {
    UsrPatches[i]=-1; // Initialize to a negative
    UsrPCounts[i]=0;
  }

  if (NZ!=1)
  {
    printf("\n ERROR (pfpatchysolid): 2-d input required for mask dataset\n");
    return;
  }

  int NXt = DataboxNx(top);
  int NYt = DataboxNy(top);
  int NZt = DataboxNz(top);

  int NXb = DataboxNx(bot);
  int NYb = DataboxNy(bot);
  int NZb = DataboxNz(bot);

  if ((NX!=NXt)||(NX!=NXb)||(NY!=NYt)||(NY!=NYb)||(NZ!=NZt)||(NZ!=NZb))
  {
    printf("\n ERROR (pfpatchysolid): Inconsistent input dataset dimensions\n");
    return;
  }

  int nxyzp = (NX + 1) * (NY + 1) * (NZ + 1);

  // First thing is find the distinct patches in the mask and tally them up
  int mask_val=0;
  int idx=0;
  for (j = 0; j < NY; ++j)
  {
    for (i = 0; i < NX; ++i)
    {
      mask_val = *DataboxCoeff(msk, i, j, 0);
      if (mask_val>1)
      {
      idx=ScanArray(mask_val,UsrPatches,np_tot);
        if (idx<0)
        {
          // Add the integer to the list, augment the count
          np_usr=np_usr+1;
          UsrPatches[np_usr]=mask_val;
        }
      }
    }
  }

  // Should probably sort the patches ASCENDING to make life easier
  /*

            ======== ADD THAT SORTING HERE ========

  */

  // Now move on to building the solids and the patches


  // MIGHT BE WISE to build a PatchObject type so you can store the info better



  // Initialize Xp to a known value to detect unused entries
  // (MIGHT NOT BE NECESSARY SINCE YOU'LL BE KEEPING A RUNNING COUNT ANYWAY)
  double *Xp;
  double def_value=-9.999e-99;
  Xp = (double*)malloc(sizeof(double) * nxyzp * 3);
  for (i=0; i<(nxyzp * 3); ++i)
  {
    Xp[i]=def_value;
  }


  printf(" - - - Verifying patches - - - \n");
  for (i=0; i<np_usr; ++i)
  {
    printf(" Patch %i, Value %i, Count %i \n",i,UsrPatches[i],UsrPCounts[i]);
  }

  /* printf("Element position = %i \n",ScanArray(9,DefPatches,6));

  // printf("Last mask val: %i\n",mask_val);
  // printf("User patch count: %i\n",n);
*/


  // First thing to do is to scan through the mask and get unique values


  int flt=1;            //TEMPORARY, remove for final version...
  char * varname="Temp";  //TEMPORARY, remove for final version...
  double pnts[5];       //TEMPORARY, remove for final version...



  return;

  // DO NOT DELTE THIS PART, you included a VTK option so use this
  // NOTE: This is not the same FP
  //This uses the mixed VTK BINARY legacy format, writes as either double or float

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "ParFlow VTK output\n");
  fprintf(fp, "BINARY\n");
  fprintf(fp, "DATASET STRUCTURED_GRID\n");
  fprintf(fp, "DIMENSIONS %i %i %i\n", NX + 1, NY + 1, NZ + 1);

/* ------------------ Set point mode write ---------------- */
/* To reduce size, write points as float */
  // int i;
  float *pnt;
  pnt = (float*)malloc(tools_SizeofFloat * nxyzp * 3);
  for (i = 0; i < (nxyzp * 3); ++i)
  {
    pnt[i] = (float)pnts[i];
  }
  fprintf(fp, "POINTS %i float\n", nxyzp);
  tools_WriteFloat(fp, pnt, nxyzp * 3);
  free(pnt);

// COMMENT THE PREVIOUS 8 AND UNCOMMENT THE FOLLOWING 3 TO FORCE DOUBLE WRITE
//        /* Write points as double */
//        fprintf(fp,"POINTS %i double\n",nxyzp);
//        tools_WriteDouble(fp, pnts, nxyzp*3);

/* ---------------- End of point mode set ---------------- */

  fprintf(fp, "CELL_DATA %i\n", NX * NY * NZ);

  if (flt == 1)
  {
    /* Write the data as float to reduce file size */
    int j;
    double *DTd;
    float *DTf;
    DTf = (float*)malloc(tools_SizeofFloat * nxyzp);
    DTd = DataboxCoeffs(msk);

    for (j = 0; j < (NX * NY * NZ); ++j)
    {
      DTf[j] = (float)DTd[j];
    }

    fprintf(fp, "SCALARS %s float\n", varname);
    fprintf(fp, "LOOKUP_TABLE default\n");
    tools_WriteFloat(fp, DTf, NX * NY * NZ);
    free(DTf);
  }
  else
  {
    fprintf(fp, "SCALARS %s double\n", varname);
    fprintf(fp, "LOOKUP_TABLE default\n");
    tools_WriteDouble(fp, DataboxCoeffs(msk), NX * NY * NZ);
  }
}



// int CheckList(int * in_list)
// {
//   int n = sizeof(in_list) / sizeof(in_list[0]);
//   return n;
// }
