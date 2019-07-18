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

// #include <stdlib.h>
#include "stdlib.h"

#include "solidtools.h"
#include "tools_io.h"

#include "string.h"
#include "unistd.h"


/*-----------------------------------------------------------------------
* Returns index of array "in_list[len]" that holds "val" if it exists, -1 otherwise
-----------------------------------------------------------------------*/
typedef struct {
  int active;
  int value;
  int patch_cell_count;  // Twice the number of CELL FACES
  int start_idx;    // Limiting indices in the main PATCH list
  int end_idx;
} PatchInfo;

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
int            MakePatchySolid(
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
  double X = DataboxX(msk);
  double Y = DataboxY(msk);
  double Z = DataboxZ(msk);
  double DX = DataboxDx(msk);
  double DY = DataboxDy(msk);
  double DZ = DataboxDz(msk);

  int out_status=-1;

  // Default patches (flagged as negatives here)
  int DefPatches[] = {-1,-2,-3,-4,-5,-6}; // Bottom, Top, West, East, South, North
  int DefPCounts[]={0,0,0,0,0,0};   // Count for each face patch

  int np_tot=20;        // Max number of patches allowed (CAN BE INCREASED IF NEEDED)
  int UsrPatches[np_tot];           // User defined patches, up to np_tot
  int UsrPCounts[np_tot];           // Counts for number of patches on user patches
  int np_usr=-1;                    // Counter for total user patch
  for (i = 0; i < np_tot; ++i)
  {
    UsrPatches[i]=-999; // Initialize to a BIG negative
    UsrPCounts[i]=0;
  }

  if (NZ!=1)
  {
    printf("\n ERROR (pfpatchysolid): 2-d input required for mask dataset\n");
    out_status=-2;
    return out_status;
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
    out_status=-2;
    return out_status;
  }

  int nxyzp = (NX + 1) * (NY + 1); // NOTE: Z is NOT included here

  // First thing is find the distinct patches in the mask and tally them up
  int mask_val=0;
  int test_val=0;
  int idx=0;
  for (j = 0; j < NY; ++j)
  {
    for (i = 0; i < NX; ++i)
    {
      mask_val = *DataboxCoeff(msk, i, j, 0);

      if (mask_val==1)
      {
        DefPCounts[0]=DefPCounts[0]+1;  // Increment BOTTOM and TOP counts
        DefPCounts[1]=DefPCounts[1]+1;
        // Check for edge faces
        if (i==0) {DefPCounts[2]=DefPCounts[2]+1;}
        if (i==(NX-1)) {DefPCounts[3]=DefPCounts[3]+1;}
        if (j==0) {DefPCounts[4]=DefPCounts[4]+1;}
        if (j==(NY-1)) {DefPCounts[5]=DefPCounts[5]+1;}


        // Now check the four neighbors
        if (j>0) // Left neighbor (WEST)
        {
          test_val = *DataboxCoeff(msk, i, j-1, 0);
          if (test_val>1)
          {
            // Find its index and add to the count, or create a new patch
            idx=ScanArray(test_val,UsrPatches,np_tot);
            // printf("\n IDX value (i,j): %i (%i, %i)\n",idx,i,j);
            // return;
              if (idx<0)
              {
                np_usr=np_usr+1; // Add the integer to the list, augment the count
                if (np_usr>np_tot) {printf("ERROR: Too many patches\n"); out_status=-2; return out_status;}
                UsrPatches[np_usr]=test_val;
                UsrPCounts[np_usr]=UsrPCounts[np_usr]+1;
              }
              else
              {
                UsrPCounts[idx]=UsrPCounts[idx]+1;
              }
          }
        } // End of WEST
        if (j<(NY-1)) // Right neighbor (EAST)
        {
          test_val = *DataboxCoeff(msk, i, j+1, 0);
          if (test_val>1)
          {
            // Find its index and add to the count, or create a new patch
            idx=ScanArray(test_val,UsrPatches,np_tot);
              if (idx<0)
              {
                np_usr=np_usr+1; // Add the integer to the list, augment the count
                if (np_usr>np_tot) {printf("ERROR: Too many patches\n"); out_status=-2; return out_status;}
                UsrPatches[np_usr]=test_val;
                UsrPCounts[np_usr]=UsrPCounts[np_usr]+1;
              } else
              {
                UsrPCounts[idx]=UsrPCounts[idx]+1;
              }
          }
        } // End of EAST
        if (i>0) // Lower neighbor (SOUTH)
        {
          test_val = *DataboxCoeff(msk, i-1, j, 0);
          if (test_val>1)
          {
            // Find its index and add to the count, or create a new patch
            idx=ScanArray(test_val,UsrPatches,np_tot);
              if (idx<0)
              {
                np_usr=np_usr+1; // Add the integer to the list, augment the count
                if (np_usr>np_tot) {printf("ERROR: Too many patches\n"); out_status=-2; return out_status;}
                UsrPatches[np_usr]=test_val;
                UsrPCounts[np_usr]=UsrPCounts[np_usr]+1;
              } else
              {
                UsrPCounts[idx]=UsrPCounts[idx]+1;
              }
          }
        } // End of SOUTH
        if (i<(NX-1)) // Upper neighbor (NORTH)
        {
          test_val = *DataboxCoeff(msk, i+1, j, 0);
          if (test_val>1)
          {
            // Find its index and add to the count, or create a new patch
            idx=ScanArray(test_val,UsrPatches,np_tot);
              if (idx<0)
              {
                np_usr=np_usr+1; // Add the integer to the list, augment the count
                if (np_usr>np_tot) {printf("ERROR: Too many patches\n"); out_status=-2; return out_status;}
                UsrPatches[np_usr]=test_val;
                UsrPCounts[np_usr]=UsrPCounts[np_usr]+1;
              } else
              {
                UsrPCounts[idx]=UsrPCounts[idx]+1;
              }
          }
        } // End of NORTH
      } // end of mask_val test
    } // end of j loop
  } // end of i loop

  int ix_off[4],iy_off[4],jx_off[4],jy_off[4];
  int off_ref[]={0,-1,1,0};

  int p_cnt=0;
  int p_idx=0;

  double z_bot,z_top;
  double dx_b,dy_b,dx_t,dy_t;

  double *Xp_BT; // x, y, Z-bottom, Z-top for ENTIRE grid (nxyzp entries)
  Xp_BT = (double*)malloc(sizeof(double) * nxyzp * 4);

  // Build the point database, not efficient to loop again but oh well
  for (j=0; j<NY+1; ++j)
  {
  for (i=0; i<NX+1; ++i)
  {
    // Set some default values to catch errors...
    z_bot=-9999.99;
    z_bot=-3333.33;
    dx_b=0.0; dy_b=0.0;
    dx_t=0.0; dy_t=0.0;

    Xp_BT[p_idx]= X + i*DX;
    Xp_BT[p_idx+1]= Y + j*DY;

    /* INDEPENDENT LINEAR INTERPOLATION: Super simple but most of the time the top
        and bottom surfaces aren't affected by the small errors this may have  */
    // Linearly interpolate the top and bottom
    for (k=0;k<4; ++k)
    {
      ix_off[k]=off_ref[k];
      jy_off[k]=off_ref[k];

      jx_off[k]=0;
      iy_off[k]=0;
    }

    if (i==0)
    {
      ix_off[0]=1; ix_off[1]=0;
    } else if ((i+1)==(NX))
    {
      ix_off[2]=0; ix_off[3]=-1;
      for (k=0; k<4; ++k) {iy_off[k]=-1;}
    }  else if (i==(NX))
    {
      ix_off[0]=-1; ix_off[1]=-2; ix_off[2]=-1; ix_off[3]=-2;
      for (k=0; k<4; ++k) {iy_off[k]=-1;}
    }

    if (j==0)
    {
      jy_off[0]=1; jy_off[1]=0;
    } else if ((j+1)==(NY))
    {
      jy_off[2]=0; jy_off[3]=-1;
      for (k=0; k<4; ++k) {jx_off[k]=-2;}
    } else if (j==(NY))
    {
      jy_off[0]=-1; jy_off[1]=-2; jy_off[2]=-1; jy_off[3]=-2;
      for (k=0; k<4; ++k) {jx_off[k]=-2;}
    }

    dx_b=((*DataboxCoeff(bot, i+ix_off[0], j+jx_off[0], 0) - *DataboxCoeff(bot, i+ix_off[1], j+jx_off[1], 0))/DX +
        (*DataboxCoeff(bot, i+ix_off[2], j+jx_off[2], 0) - *DataboxCoeff(bot, i+ix_off[3], j+jx_off[3], 0))/DX)/2.0;
    dx_t=((*DataboxCoeff(top, i+ix_off[0], j+jx_off[0], 0) - *DataboxCoeff(top, i+ix_off[1], j+jx_off[1], 0))/DX +
            (*DataboxCoeff(top, i+ix_off[2], j+jx_off[2], 0) - *DataboxCoeff(top, i+ix_off[3], j+jx_off[3], 0))/DX)/2.0;

    dy_b=((*DataboxCoeff(bot, i+iy_off[0], j+jy_off[0], 0) - *DataboxCoeff(bot, i+iy_off[1], j+jy_off[1], 0))/DX +
        (*DataboxCoeff(bot, i+iy_off[2], j+jy_off[2], 0) - *DataboxCoeff(bot, i+iy_off[3], j+jy_off[3], 0))/DX)/2.0;
    dy_t=((*DataboxCoeff(top, i+iy_off[0], j+jy_off[0], 0) - *DataboxCoeff(top, i+iy_off[1], j+jy_off[1], 0))/DX +
            (*DataboxCoeff(top, i+iy_off[2], j+jy_off[2], 0) - *DataboxCoeff(top, i+iy_off[3], j+jy_off[3], 0))/DX)/2.0;

    if ((i<(NX-1))&(j<(NY-1)))
    {
      z_bot=*DataboxCoeff(bot, i, j, 0) - dx_b*DX/2.0 - dy_b*DY/2.0;
      z_top=*DataboxCoeff(top, i, j, 0) - dx_t*DX/2.0 - dy_t*DY/2.0;
    } else if ((i>=(NX-1))&(j<(NY-1))) // The edges go here...
    {
      z_bot=*DataboxCoeff(bot, i-1, j, 0) + dx_b*DX/2.0 - dy_b*DY/2.0;
      z_top=*DataboxCoeff(top, i-1, j, 0) + dx_t*DX/2.0 - dy_t*DY/2.0;;
    } else if ((i<(NX-1))&(j>=(NY-1))) // The edges go here...
    {
      z_bot=*DataboxCoeff(bot, i, j-1, 0) - dx_b*DX/2.0 + dy_b*DY/2.0;
      z_top=*DataboxCoeff(top, i, j-1, 0) - dx_t*DX/2.0 + dy_t*DY/2.0;
    } else if ((i>=(NX-1))&(j>=(NY-1))) // The edges go here...
    {
      z_bot=*DataboxCoeff(bot, i-1, j-1, 0) + dx_b*DX/2.0 + dy_b*DY/2.0;
      z_top=*DataboxCoeff(top, i-1, j-1, 0) + dx_t*DX/2.0 + dy_t*DY/2.0;
    }

    Xp_BT[p_idx+2]=z_bot;   // BOTTOM elevation
    Xp_BT[p_idx+3]=z_top;   // TOP elevation
    p_idx=p_idx+4;        // Then augment our counter
  }

  }

  // for (k=0; k<(nxyzp*4); k=k+4)
  // {
  // printf("%f, %f, %f, %f  \n",(k/4.0)+1.0,Xp_BT[k],Xp_BT[k+1],Xp_BT[k+2]);
  // }

  out_status=1;
  return out_status;

  // Should probably sort the patches ASCENDING to make life easier
  /*
            ======== ADD THAT SORTING HERE? ========
      0. Duplicate UsrPCounts and UsrPatches
      1. Find minimum value in UsrPatches and its index
      2. Copy count and ID to n-th position in new array
      3. Move on to the n+1-th position, greater than n-th but less than rest
      4. Clear both and re-create with correct size
  */

  // Now move on to building the solids and the patches
  int np=0;
  int cell_faces=0;
  // Count up all the cell faces
  for (i=0; i<6; ++i) cell_faces=cell_faces+DefPCounts[i];

  // printf("BEFORE: %i\n",cell_faces);

  // for (i=0; i<np_tot; ++i) cell_faces=cell_faces+UsrPCounts[i];
  for (i=0; i<np_tot; ++i)
  {
    cell_faces=cell_faces+UsrPCounts[i];
  }

  // Create 2-d array to hold the vertex ID numbers, [cell_faces][6] since 6 per face
  int *Patches[cell_faces];
  for (i=0; i<cell_faces; i++) Patches[i] = (int *)malloc(6 * sizeof(int));
  // And initialize to -1
  for (i=0;i<cell_faces; i++) for (j=0;j<6; j++) Patches[i][j]=-1;

  // Create the array structure to hold the info for those patches
  PatchInfo AllPatches[6+np_usr];

  np=0;
  for (i=0; i<6; ++i)
  {
    if (DefPCounts[i]>0)
    {
      AllPatches[i].active=1;
      AllPatches[i].value=DefPatches[i];
      AllPatches[i].patch_cell_count=DefPCounts[i];
      AllPatches[i].start_idx=np;
      AllPatches[i].end_idx=np+AllPatches[i].patch_cell_count-1;
      np=AllPatches[i].end_idx+1;
    } else {
      AllPatches[i].active=0;
      AllPatches[i].value=DefPatches[i];
      AllPatches[i].patch_cell_count=0;
      AllPatches[i].start_idx=-1;
      AllPatches[i].end_idx=-1;
    }
  }
  for (i=6; i<(6+np_usr+1); ++i)
  {
    if (UsrPCounts[i-6]>0)
    {
      AllPatches[i].active=1;
      AllPatches[i].value=UsrPatches[i-6];
      AllPatches[i].patch_cell_count=UsrPCounts[i-6];
      AllPatches[i].start_idx=np;
      AllPatches[i].end_idx=np+AllPatches[i].patch_cell_count-1;
      np=AllPatches[i].end_idx+1;
    } else {
      // AllPatches[i].active=0;
      // AllPatches[i].patch_cell_count=0;
      // AllPatches[i].value=-1;
      // AllPatches[i].start_idx=-1;
      // AllPatches[i].end_idx=-1;
      printf("\n This shouldn't be able to happen...STOPPING\n");
      printf("   Value if: %i \n",i);
      return out_status;
    }
  }

  /*
  printf("Last IDX: %i, Total cells: %i \n",AllPatches[6+np_usr].end_idx,cell_faces);
  for (j=0; j<(6+np_usr+1); ++j)
  printf("First: %i, Last: %i, Count %i, Value %i \n",AllPatches[j].start_idx,AllPatches[j].end_idx,AllPatches[j].patch_cell_count,AllPatches[j].value);
  */

  // Initialize Xp to a known value to detect unused entries
  // (MIGHT NOT BE NECESSARY SINCE YOU'LL BE KEEPING A RUNNING COUNT ANYWAY)
  double *Xp;
  Xp = (double*)malloc(sizeof(double) * nxyzp * 3);
  double def_value=Xp[0];

  // Initialize point database
  for (i=0; i<(nxyzp * 3); i=i+3)
  {
    Xp[i]=def_value;
    Xp[i+1]=def_value;
    Xp[i+2]=def_value;
  }

  // Each cell has eight points
  int p_pnts[8];
  p_cnt=-1;
  for (j = 0; j < NY; ++j)
  {
    for (i = 0; i < NX; ++i)
    {
      // MAIN LOOP over all the cells
      mask_val = *DataboxCoeff(msk, i, j, 0);
      if (mask_val==1)
      {
        for (i=0;i<8;++i) {p_pnts[i]=-1;} // Always start fresh
        if ((j==1)&(i==1))
        {
          // It's the corner so add them all
          k=1;

          p_cnt=p_cnt+8;
        } else if (i==1) {
          k=1;
        }









      }
    }
  }



  // out_status=0;  // If all good
  out_status=1;  // My debugging mode...
  return out_status;


  printf(" - - - Verifying default patches - - - \n");
  for (i=0; i<6; ++i)
  {
    // printf(" Patch %i, Value %i, Count %i \n",i,DefPatches[i],DefPCounts[i]);
    printf(" Patch %i, Value %i, Count %i, Active %i \n",i,AllPatches[i].value,AllPatches[i].patch_cell_count,AllPatches[i].active);
  }

  printf("\n - - - Verifying user patches - - - \n");
  // for (i=0; i<np_usr; ++i)
  // {
  //   printf(" Patch %i, Value %i, Count %i \n",i,UsrPatches[i],UsrPCounts[i]);
  // }
  for (i=6; i<(6+np_usr); ++i)
  {
  printf(" Patch %i, Value %i, Count %i, Active %i \n",i,AllPatches[i].value,AllPatches[i].patch_cell_count,AllPatches[i].active);
  }


  /* printf("Element position = %i \n",ScanArray(9,DefPatches,6));



  // printf("Last mask val: %i\n",mask_val);
  // printf("User patch count: %i\n",n);
*/


  // First thing to do is to scan through the mask and get unique values


  int flt=1;            //TEMPORARY, remove for final version...
  char * varname="Temp";  //TEMPORARY, remove for final version...
  double pnts[5];       //TEMPORARY, remove for final version...



  return out_status;

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
