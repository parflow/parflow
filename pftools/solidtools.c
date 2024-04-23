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

// - Holds a variety of utilities for building and manipulating solid files

#include "parflow_config.h"

// #include <stdlib.h>
#include "stdlib.h"

#include "solidtools.h"
#include "tools_io.h"

#include "string.h"
#include "unistd.h"


// Structure to hold information about the patches
typedef struct {
  int active;
  int p_write;
  int value;
  int patch_cell_count;
  int start_idx;
  int end_idx;
  int total_patches;
  int act_faces[6];
} PatchInfo;

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
 * NBE: 2019-07-20
 *
 *-----------------------------------------------------------------------*/
int            MakePatchySolid(
                             FILE *   fp,
                             FILE *   fp_vtk,
                             Databox *msk,
                             Databox *top,
                             Databox *bot,
                             int sub_patches,
                             int bin_out)
{
  int out_status=-1;
  int i,j,k,m;
  int NX = DataboxNx(msk);
  int NY = DataboxNy(msk);
  int NZ = DataboxNz(msk);
  double X = DataboxX(msk);
  double Y = DataboxY(msk);
  double DX = DataboxDx(msk);
  double DY = DataboxDy(msk);

  /* ---------- Max number of patches ---------- */
  int np_tot=30;        // Max number of patches allowed (CAN BE INCREASED IF NEEDED)

  int debugger=0;

  // #include "time.h"
  // clock_t start_time, end_time;
  // double run_time;

  // Default patches (flagged as negatives here)
  int DefPatches[] = {-1,-2,-3,-4,-5,-6}; // Bottom, Top, West, East, South, North
  int DefPCounts[]={0,0,0,0,0,0};   // Count for each face patch

  int UsrPatches[np_tot];           // User defined patches, up to np_tot
  int UsrPCounts[np_tot];           // Counts for number of patches on user patches
  int np_usr=-1;                    // Counter for total user patch
  for (i = 0; i < np_tot; ++i)
  {
    UsrPatches[i]=-99999; // Initialize to a BIG negative
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

  int any_zeros=0;

  // Find the distinct patches in the mask and tally them up
  int mask_val=0;
  int test_val=0;
  int idx=0;
  for (j = 0; j < NY; ++j)
  {
    for (i = 0; i < NX; ++i)
    {
      mask_val = *DataboxCoeff(msk, i, j, 0);

      if ( (any_zeros==0) && (mask_val==0))
      {
        any_zeros=1;
      }

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
        if (i>0) // Left neighbor (WEST)
        {
          test_val = *DataboxCoeff(msk, i-1, j, 0);
          if (test_val!=1)
          {
            // Find its index and add to the count, or create a new patch
            idx=ScanArray(test_val,UsrPatches,np_tot);
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
        if (i<(NX-1)) // Right neighbor (EAST)
        {
          test_val = *DataboxCoeff(msk, i+1, j, 0);
          if (test_val!=1)
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
        if (j>0) // Lower neighbor (SOUTH)
        {
          test_val = *DataboxCoeff(msk, i, j-1, 0);
          if (test_val!=1)
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
        if (j<(NY-1)) // Upper neighbor (NORTH)
        {
          test_val = *DataboxCoeff(msk, i, j+1, 0);
          if (test_val!=1)
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

  double x_test,y_test;
  double z_bot;
  double dx_b,dy_b;

  // =========================================================================
  // Sort the patches in ASCENDING order to make life easier

  int TempUPCnt[np_usr+1],TempUPLst[np_usr+1];
    for (i=0; i<(np_usr+1); ++i)
  {
    // printf(" %i %i \n",UsrPatches[i],UsrPCounts[i]);
    TempUPCnt[i]=UsrPCounts[i];
    TempUPLst[i]=UsrPatches[i];
    UsrPCounts[i]=-1;       // Reset so they can be sorted
    UsrPatches[i]=-99999;
  }
  // Find the minimum

  int min_val,min_idx;
  for (j=0; j<(np_usr+1); ++j)
  {
      min_val=TempUPLst[j];
      min_idx=j;
    for (k=(j+1); k<(np_usr+1); ++k)
    {
      if (TempUPLst[k]<min_val)
      {
        min_val=TempUPLst[k];
        min_idx=k;
      }
    } // k loop end
  UsrPCounts[j]=TempUPCnt[min_idx];
  UsrPatches[j]=TempUPLst[min_idx];
  // Now swap the values
  TempUPLst[min_idx]=TempUPLst[j]; TempUPLst[j]=UsrPatches[j];
  TempUPCnt[min_idx]=TempUPCnt[j]; TempUPCnt[j]=UsrPCounts[j];
  }

  // Now move on to building the solids and the patches
  int np=0;
  int cell_faces=0;
  // Count up all the cell faces

  for (i=0; i<6; ++i) {cell_faces=cell_faces+DefPCounts[i];}
  for (i=0; i<np_tot; ++i)
  {
    cell_faces=cell_faces+UsrPCounts[i];
  }

  // Create the array structure to hold the info for those patches
  PatchInfo AllPatches[6+np_usr];
  int non_blanks=0;

  np=0;
  for (i=0; i<6; ++i)
  {
    if (DefPCounts[i]>0)
    {
      AllPatches[i].p_write=-1;
      AllPatches[i].active=1;
      AllPatches[i].value=DefPatches[i];
      AllPatches[i].patch_cell_count=DefPCounts[i];
      AllPatches[i].start_idx=np;
      AllPatches[i].end_idx=np+AllPatches[i].patch_cell_count-1;
      np=AllPatches[i].end_idx+1;
      non_blanks=non_blanks+1;
      for (j=0; j<6; ++j) {AllPatches[i].act_faces[j]=0;} // Bottom, Top, West, East, South, North
      AllPatches[i].total_patches=0;
    } else {
      AllPatches[i].p_write=-1;
      AllPatches[i].active=0;
      AllPatches[i].value=DefPatches[i];
      AllPatches[i].patch_cell_count=0;
      AllPatches[i].start_idx=-1;
      AllPatches[i].end_idx=-1;
      for (j=0; j<6; ++j) {AllPatches[i].act_faces[j]=0;} // Bottom, Top, West, East, South, North
    }
  }
  for (i=6; i<(6+np_usr+1); ++i)
  {
    if (UsrPCounts[i-6]>0)
    {
      AllPatches[i].p_write=-1;
      AllPatches[i].active=1;
      AllPatches[i].value=UsrPatches[i-6];
      AllPatches[i].patch_cell_count=UsrPCounts[i-6];
      AllPatches[i].start_idx=np;
      AllPatches[i].end_idx=np+AllPatches[i].patch_cell_count-1;
      np=AllPatches[i].end_idx+1;
      non_blanks=non_blanks+1;
      for (j=0; j<6; ++j) {AllPatches[i].act_faces[j]=0;} // Bottom, Top, West, East, South, North
      AllPatches[i].total_patches=0;
    } else {
      // AllPatches[i].active=0;
      // AllPatches[i].patch_cell_count=0;
      // AllPatches[i].value=-1;
      // AllPatches[i].start_idx=-1;
      // AllPatches[i].end_idx=-1;
      printf("\n ERROR: Patch indexing failure...MUST STOP\n");
      printf("    Value if: %i (%i) \n",i-6,i);
      printf("Pcount, Pval: %i %i\n",UsrPCounts[i-6],UsrPatches[i-6]);
      return out_status;
    }
  }

  // if (debugger==1) {
  // for (i=0;i<6+np_usr+1; ++i)
  // {
  // printf("%i %i %i %i %i \n",i,AllPatches[i].value,AllPatches[i].patch_cell_count,AllPatches[i].start_idx,AllPatches[i].end_idx);
  // }
  // }

  // Used to easily find the ID of user patches and zeros later
  int patch_values[6+np_usr];
  for (i=0; i<(6+np_usr+1); ++i)
  {
    patch_values[i]=AllPatches[i].value;
  }

  // int Patch[cell_faces*7];
  int *Patch;
  // Patch=(int*)calloc(cell_faces*7,sizeof(int));
  Patch=(int*)calloc(cell_faces*8,sizeof(int));

  // for (i=0;i<(cell_faces*7); i++) {Patch[i]=-1111;}
  for (i=0;i<(cell_faces*8); i++) {Patch[i]=-1111;}

  if (debugger==1)
  {
    printf(" *** Cell faces: %i *** \n",cell_faces);
    printf("Should be: %i\n",NX*2 + NY*2 + (NX*NY)*2);
    printf("NX,NY: %i %i\n",NX, NY);

    printf("any_zeros value: %i\n",any_zeros);
  }

  // ==========================================================================
  // Build a 2-d array to hold the active points

  if (debugger==1) printf("NXY points: %i\n",nxyzp);

  // double Xp_Act[nxyzp*6];
  double *Xp_Act;
  Xp_Act=(double*)calloc(nxyzp*6,sizeof(double));

  for (i=0;i<nxyzp; i++) {for (j=0;j<3; j++) {Xp_Act[3*i + j]=-11111.11;}}

  // Which corners of the cell to add to the Xp_Act dBase, who belongs to this cell
  int add_pnt[8],our_pnts[8];
  int i_off=0, j_off=0;
  int km,n_t;
  int np_act=-1;

  // if (debugger==1) start_time=clock(); // Enable <time.h> at top of function

  // Build the point database, not efficient to loop again but oh well
  for (j=0; j<NY; ++j) //NY
  {
  for (i=0; i<NX; ++i) //NX
  {
    // if (debugger==1) printf("Current i,j,np_act: %i, %i, %i\n",i,j,np_act);

    mask_val = *DataboxCoeff(msk, i, j, 0);
    if (mask_val==1)
    {
    for (k=0;k<8;++k){add_pnt[k]=1;} // Initially assume everybody gets added
    for (k=0;k<8;++k){our_pnts[k]=-1;}

    // Determine who doesn't need to be added
    if (i>0) {
        if (*DataboxCoeff(msk, i-1, j, 0)==1) {
          add_pnt[0]=0; add_pnt[2]=0; add_pnt[4]=0; add_pnt[6]=0;
        }
    }
    if ((i>0)&(j>0)) {
        if (*DataboxCoeff(msk, i-1, j-1, 0)==1) {
          add_pnt[0]=0; add_pnt[4]=0;
        }
    }
    if (j>0) {
        if (*DataboxCoeff(msk, i, j-1, 0)==1) {
          add_pnt[0]=0; add_pnt[1]=0; add_pnt[4]=0; add_pnt[5]=0;
        }
    }
    if ((i<(NX-1))&(j>0)) {
        if (*DataboxCoeff(msk, i+1, j-1, 0)==1) {
          add_pnt[1]=0; add_pnt[5]=0;
        }
    }

    // Now add the new points to Xp_Act
    for (km=0; km<8; ++km)
    {
      if (add_pnt[km]==1)
      {
              i_off=0; j_off=0;
              np_act=np_act+1;
              our_pnts[km]=np_act;
              if ((km==1)|(km==3)|(km==5)|(km==7)) {i_off=1;}
              if ((km==2)|(km==3)|(km==6)|(km==7)) {j_off=1;}

              Xp_Act[3*np_act]= X + (i+i_off)*DX;
              Xp_Act[3*np_act+1]= Y + (j+j_off)*DY;

              // The simple linear interpolation
              z_bot=-3333.33;
              dx_b=0.0; dy_b=0.0;

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
              }

              if (j==0)
              {
                jy_off[0]=1; jy_off[1]=0;
              } else if ((j+1)==(NY))
              {
                jy_off[2]=0; jy_off[3]=-1;
                for (k=0; k<4; ++k) {jx_off[k]=-2;}
              }

              if ((km==0)|(km==1)|(km==2)|(km==3))
              {
                dx_b=((*DataboxCoeff(bot, i+ix_off[0], j+jx_off[0], 0) - *DataboxCoeff(bot, i+ix_off[1], j+jx_off[1], 0))/DX +
                    (*DataboxCoeff(bot, i+ix_off[2], j+jx_off[2], 0) - *DataboxCoeff(bot, i+ix_off[3], j+jx_off[3], 0))/DX)/2.0;
                dy_b=((*DataboxCoeff(bot, i+iy_off[0], j+jy_off[0], 0) - *DataboxCoeff(bot, i+iy_off[1], j+jy_off[1], 0))/DX +
                    (*DataboxCoeff(bot, i+iy_off[2], j+jy_off[2], 0) - *DataboxCoeff(bot, i+iy_off[3], j+jy_off[3], 0))/DX)/2.0;

                    // HERE: These need a km check instead of just an i j check
                    if ((km==1)|(km==3)) {dx_b=-dx_b;}
                    if ((km==2)|(km==3)) {dy_b=-dy_b;}
                    z_bot=*DataboxCoeff(bot, i, j, 0) - dx_b*DX/2.0 - dy_b*DY/2.0;

              } else {
                dx_b=((*DataboxCoeff(top, i+ix_off[0], j+jx_off[0], 0) - *DataboxCoeff(top, i+ix_off[1], j+jx_off[1], 0))/DX +
                        (*DataboxCoeff(top, i+ix_off[2], j+jx_off[2], 0) - *DataboxCoeff(top, i+ix_off[3], j+jx_off[3], 0))/DX)/2.0;
                dy_b=((*DataboxCoeff(top, i+iy_off[0], j+jy_off[0], 0) - *DataboxCoeff(top, i+iy_off[1], j+jy_off[1], 0))/DX +
                        (*DataboxCoeff(top, i+iy_off[2], j+jy_off[2], 0) - *DataboxCoeff(top, i+iy_off[3], j+jy_off[3], 0))/DX)/2.0;

                    if ((km==5)|(km==7)) {dx_b=-dx_b;}
                    if ((km==6)|(km==7)) {dy_b=-dy_b;}
                    z_bot=*DataboxCoeff(top, i, j, 0) - dx_b*DX/2.0 - dy_b*DY/2.0;
              }
              Xp_Act[3*np_act+2]=z_bot;   // BOTTOM elevation

      } else {
        // Point already exists so go find its index
      i_off=0; j_off=0;
      if ((km==1)|(km==3)|(km==5)|(km==7)) {i_off=1;}
      if ((km==2)|(km==3)|(km==6)|(km==7)) {j_off=1;}
      x_test = X + (i+i_off)*DX;
      y_test = Y + (j+j_off)*DY;

      /* Loop last to first since points will be closer */
      for (m=np_act; m>0; m=m-1)
      {
        if ((Xp_Act[3*m]==x_test)&(Xp_Act[3*m+1]==y_test))
        {
          break;
        }
      }
      // If it's a bottom point keep going to find the next instance of the x-y coodrinates
      if (km<=3) {
        for (m=m-1; m>0; m=m-1)
        {
          if ((Xp_Act[3*m]==x_test)&(Xp_Act[3*m+1]==y_test))
          {
            break;
          }
        }
      }

      our_pnts[km]=m;

      }
    }

    // Always add a top and bottom patch
    /* ----- BOTTOM patch ----- */
    AllPatches[0].p_write=AllPatches[0].p_write+1;
    n_t=AllPatches[0].start_idx+AllPatches[0].p_write;
    Patch[8*n_t]=our_pnts[1]; // First triangle of the cell face
    Patch[8*n_t + 1]=our_pnts[0];
    Patch[8*n_t + 2]=our_pnts[2];

    Patch[8*n_t + 3]=our_pnts[2]; // Second triangle of the cell face
    Patch[8*n_t + 4]=our_pnts[3];
    Patch[8*n_t + 5]=our_pnts[1];
    Patch[8*n_t + 6]=-1;
    Patch[8*n_t + 7]=0;
    AllPatches[0].act_faces[0]+=1;

    /* ----- TOP patch ----- */
    AllPatches[1].p_write=AllPatches[1].p_write+1;
    n_t=AllPatches[1].start_idx+AllPatches[1].p_write;
    Patch[8*n_t + 0]=our_pnts[6];
    Patch[8*n_t + 1]=our_pnts[4];
    Patch[8*n_t + 2]=our_pnts[5];

    Patch[8*n_t + 3]=our_pnts[5];
    Patch[8*n_t + 4]=our_pnts[7];
    Patch[8*n_t + 5]=our_pnts[6];
    Patch[8*n_t + 6]=-2;
    Patch[8*n_t + 7]=1;
    AllPatches[1].act_faces[1]+=1;

    if (i==0) // Write a WEST EDGE patch
    {
      AllPatches[2].p_write=AllPatches[2].p_write+1;
      n_t=AllPatches[2].start_idx+AllPatches[2].p_write;
      Patch[8*n_t]=our_pnts[6];
      Patch[8*n_t + 1]=our_pnts[2];
      Patch[8*n_t + 2]=our_pnts[0];
      Patch[8*n_t + 3]=our_pnts[0];
      Patch[8*n_t + 4]=our_pnts[4];
      Patch[8*n_t + 5]=our_pnts[6];
      Patch[8*n_t + 6]=-3;
      Patch[8*n_t + 7]=2;
      AllPatches[2].act_faces[2]+=1;
    }

    if (i>0) // Check for an WEST facing user patch
    {
      test_val=*DataboxCoeff(msk, i-1, j, 0);
      if ((test_val == 0)|(test_val > 1))
      {
        if (test_val<0) {return -2;} // Error so bail
        idx=ScanArray(test_val,patch_values,6+np_usr+1);
        if (idx<0) {return -1;} // Error so bail
        AllPatches[idx].p_write=AllPatches[idx].p_write+1;
        n_t=AllPatches[idx].start_idx+AllPatches[idx].p_write;
        Patch[8*n_t]=our_pnts[6];
        Patch[8*n_t + 1]=our_pnts[2];
        Patch[8*n_t + 2]=our_pnts[0];
        Patch[8*n_t + 3]=our_pnts[0];
        Patch[8*n_t + 4]=our_pnts[4];
        Patch[8*n_t + 5]=our_pnts[6];
        Patch[8*n_t + 6]=test_val;
        AllPatches[idx].act_faces[2]+=1;
        Patch[8*n_t + 7]=2;
      }
    }

    if (i==(NX-1)) // Write an EAST EDGE patch
    {
      AllPatches[3].p_write=AllPatches[3].p_write+1;
      n_t=AllPatches[3].start_idx+AllPatches[3].p_write;
      Patch[8*n_t]=our_pnts[1];
      Patch[8*n_t + 1]=our_pnts[3];
      Patch[8*n_t + 2]=our_pnts[7];
      Patch[8*n_t + 3]=our_pnts[7];
      Patch[8*n_t + 4]=our_pnts[5];
      Patch[8*n_t + 5]=our_pnts[1];
      Patch[8*n_t + 6]=-4;
      Patch[8*n_t + 7]=3;
      AllPatches[3].act_faces[3]+=1;
    }

    if (i<(NX-1)) // Check for an EAST facing user patch
    {
      test_val=*DataboxCoeff(msk, i+1, j, 0);
      if ((test_val == 0)|(test_val > 1))
      {
        if (test_val<0) {return -2;} // Error so bail
        idx=ScanArray(test_val,patch_values,6+np_usr+1);
        if (idx<0) {return -1;} // Error so bail
        AllPatches[idx].p_write=AllPatches[idx].p_write+1;
        n_t=AllPatches[idx].start_idx+AllPatches[idx].p_write;
        Patch[8*n_t]=our_pnts[1];
        Patch[8*n_t + 1]=our_pnts[3];
        Patch[8*n_t + 2]=our_pnts[7];
        Patch[8*n_t + 3]=our_pnts[7];
        Patch[8*n_t + 4]=our_pnts[5];
        Patch[8*n_t + 5]=our_pnts[1];
        Patch[8*n_t + 6]=test_val;
        Patch[8*n_t + 7]=3;
        AllPatches[idx].act_faces[3]+=1;
      }
    }

    if (j==0) // Write a SOUTH EDGE patch
    {
      AllPatches[4].p_write=AllPatches[4].p_write+1;
      n_t=AllPatches[4].start_idx+AllPatches[4].p_write;
      Patch[8*n_t]=our_pnts[4];
      Patch[8*n_t + 1]=our_pnts[0];
      Patch[8*n_t + 2]=our_pnts[1];
      Patch[8*n_t + 3]=our_pnts[1];
      Patch[8*n_t + 4]=our_pnts[5];
      Patch[8*n_t + 5]=our_pnts[4];
      Patch[8*n_t + 6]=-5;
      Patch[8*n_t + 7]=4;
      AllPatches[4].act_faces[4]+=1;
    }

    if (j>0) // Check for an SOUTH facing user patch
    {
      test_val=*DataboxCoeff(msk, i, j-1, 0);
      if ((test_val == 0)|(test_val > 1))
      {
        if (test_val<0) {return -2;} // Error so bail
        idx=ScanArray(test_val,patch_values,6+np_usr+1);
        if (idx<0) {return -1;} // Error so bail
        AllPatches[idx].p_write=AllPatches[idx].p_write+1;
        n_t=AllPatches[idx].start_idx+AllPatches[idx].p_write;
        Patch[8*n_t]=our_pnts[4];
        Patch[8*n_t + 1]=our_pnts[0];
        Patch[8*n_t + 2]=our_pnts[1];
        Patch[8*n_t + 3]=our_pnts[1];
        Patch[8*n_t + 4]=our_pnts[5];
        Patch[8*n_t + 5]=our_pnts[4];
        Patch[8*n_t + 6]=test_val;
        Patch[8*n_t + 7]=4;
        AllPatches[idx].act_faces[4]+=1;
      }
    }

    if (j==(NY-1)) // Write a NORTH EDGE patch
    {
      AllPatches[5].p_write=AllPatches[5].p_write+1;
      n_t=AllPatches[5].start_idx+AllPatches[5].p_write;
      Patch[8*n_t]=our_pnts[3];
      Patch[8*n_t + 1]=our_pnts[2];
      Patch[8*n_t + 2]=our_pnts[6];

      Patch[8*n_t + 3]=our_pnts[6];
      Patch[8*n_t + 4]=our_pnts[7];
      Patch[8*n_t + 5]=our_pnts[3];
      Patch[8*n_t + 6]=-6;
      Patch[8*n_t + 7]=5;
      AllPatches[5].act_faces[5]+=1;
    }

    if (j<(NY-1)) // Check for an NORTH facing user patch
    {
      test_val=*DataboxCoeff(msk, i, j+1, 0);
      if ((test_val == 0)|(test_val > 1))
      {
        if (test_val<0) {return -2;} // Error so bail
        idx=ScanArray(test_val,patch_values,6+np_usr+1);
        if (idx<0) {return -1;} // Error so bail
        AllPatches[idx].p_write=AllPatches[idx].p_write+1;
        n_t=AllPatches[idx].start_idx+AllPatches[idx].p_write;
        Patch[8*n_t]=our_pnts[3];
        Patch[8*n_t + 1]=our_pnts[2];
        Patch[8*n_t + 2]=our_pnts[6];
        Patch[8*n_t + 3]=our_pnts[6];
        Patch[8*n_t + 4]=our_pnts[7];
        Patch[8*n_t + 5]=our_pnts[3];
        Patch[8*n_t + 6]=test_val;
        AllPatches[idx].act_faces[5]+=1;
        Patch[8*n_t + 7]=5;
      }
    }

    test_val=0;

    } // End of mask_val==1 block
  } // End of i loop
  } // End of j loop

  // if (debugger==1) {
  //   // To use this, uncomment this AND 1) the line with start_time, and 2) the declarations up top
  //   end_time=clock();
  //   run_time=(double)(end_time-start_time) / CLOCKS_PER_SEC;
  //   printf("Elapsed time: %f\n", run_time);
  // }

  if (debugger==1) printf("DBG: Done building point-patch database\n");

  char *dir_labels[]={"Bottom","Top","Left","Right","Front","Back"};

  // --------------------------------------------------------------------------
  //              ===== Write out the solid file =====
  // --------------------------------------------------------------------------
  // Write the patch order to the terminal so the user can copy them
  printf("PFPATCHYSOLID - Patch write order is:\n\n");
  if (AllPatches[0].patch_cell_count>0) {printf(" Bottom ");}
  if (AllPatches[1].patch_cell_count>0) {printf(" Top ");}

  if (AllPatches[2].patch_cell_count>0) {printf(" Left ");}  // West
  if (AllPatches[3].patch_cell_count>0) {printf(" Right ");} // East

  if (AllPatches[4].patch_cell_count>0) {printf(" Front ");} // South
  if (AllPatches[5].patch_cell_count>0) {printf(" Back ");}  // North
  for (i=6; i<(6+np_usr+1); ++i)
  {
    if (sub_patches==1) {
      if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
      {
        for (j=0; j<6; ++j) {
        if (AllPatches[i].act_faces[j]>0)
        printf(" Usr_%i_%s ",AllPatches[i].value,dir_labels[j]);
        }
      }
    } else {
    if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
    {printf(" Usr_%i ",AllPatches[i].value);}
    }
  }
  printf("\n \n");

  // ====== Now write out the solid =====
  if (bin_out==0)
  {
  fprintf(fp,"1 \n"); // VERSION #
  fprintf(fp,"%i \n",np_act+1);   // # of POINTS/VERTICIES
  for (i=0; i<=np_act; ++i)
  {
    fprintf(fp,"%17.4f %17.4f %17.4f\n",Xp_Act[3*i],Xp_Act[3*i+1],Xp_Act[3*i+2]);
  }
  fprintf(fp,"1 \n"); // Number of SOLIDS (only one allowed here)
  fprintf(fp,"%i \n",(cell_faces)*2); // Total number of triangles

  for (i=0; i<cell_faces; ++i)
  {
  fprintf(fp," %i %i %i\n",Patch[8*i],Patch[8*i + 1],Patch[8*i + 2]);
  fprintf(fp," %i %i %i\n",Patch[8*i + 3],Patch[8*i + 4],Patch[8*i + 5]);
  }

  // NEED TO MODIFY FOR SUB PATCH OPTION
  if (sub_patches==1) {
    int all_faces=0;
    // Count up all the ACTUAL patches
    for (i=0; i<(6+np_usr+1); ++i) {
      // printf("F%i, ",i);
      for (j=0; j<6; ++j) {
        // printf(" %i ",AllPatches[i].act_faces[j]);
        if ( (AllPatches[i].value!=0) && (AllPatches[i].act_faces[j]>0)) {all_faces+=1;}
      }
      // printf("    Val=%i, Cnt=%i \n",AllPatches[i].value,AllPatches[i].patch_cell_count);
    }

    // printf("non_blanks=%i, all_faces=%i \n",non_blanks,all_faces);

    fprintf(fp,"%i \n",all_faces); // -1 since zero is omitted

    for (i=0; i<(6+np_usr+1); ++i)
    {
      if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
      {
        for (k=0; k<6; ++k) {
          if (AllPatches[i].act_faces[k]>0) {
            fprintf(fp,"%i \n",AllPatches[i].act_faces[k]*2);
            for (j=AllPatches[i].start_idx; j<=AllPatches[i].end_idx; ++j)
            {
              if (Patch[8*j + 7]==k) {
              fprintf(fp,"%i \n",2*j);
              fprintf(fp,"%i \n",2*j+1);
              }
            }
          }
        } // End of k loop
      }
    }
  // ----- End of sub patch section for ASCII write -----
  } else {
  if (any_zeros==1)
  {
    fprintf(fp,"%i \n",non_blanks-1); // -1 since zero is omitted
  } else {
    fprintf(fp,"%i \n",non_blanks); // -1 since zero is omitted
  }

  for (i=0; i<(6+np_usr+1); ++i)
  {
    if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
    {
      fprintf(fp,"%i \n",AllPatches[i].patch_cell_count*2);
      for (j=2*AllPatches[i].start_idx; j<=2*AllPatches[i].end_idx+1; ++j)
      {
        fprintf(fp,"%i \n",j);
      }
    }
  }
  }
  if (debugger==1) printf("DBG: Done writing ASCII solid\n");
  } else {
    /* ---------- PLACE HOLDER for future feature ---------- */
    // This writes a BINARY solid file. The file can be written and read with
    // ParFlow BUT a bug limits this to a processor topology of (1,1,1)
    //  * * A fix is coming so just use ASCII solid files for now * *

    int write_int=1;

    tools_WriteInt(fp,&write_int, 1); // VERSION #
    write_int=np_act+1;
    tools_WriteInt(fp,&write_int, 1); // # of POINTS/VERTICIES
    for (i=0; i<=np_act; ++i)
    {
      tools_WriteDouble(fp,&Xp_Act[3*i], 1);
      tools_WriteDouble(fp,&Xp_Act[3*i+1], 1);
      tools_WriteDouble(fp,&Xp_Act[3*i+2], 1);
    }
    write_int=1;
    tools_WriteInt(fp,&write_int, 1); // Number of SOLIDS (only one allowed here)
    write_int=cell_faces*2;
    tools_WriteInt(fp,&write_int, 1); // Total number of triangles
    for (i=0; i<(cell_faces); ++i)
    {
      tools_WriteInt(fp,&Patch[8*i], 1);
      tools_WriteInt(fp,&Patch[8*i + 1], 1);
      tools_WriteInt(fp,&Patch[8*i + 2], 1);
      tools_WriteInt(fp,&Patch[8*i + 3], 1);
      tools_WriteInt(fp,&Patch[8*i + 4], 1);
      tools_WriteInt(fp,&Patch[8*i + 5], 1);
    }

    if (sub_patches==1) {
        int all_faces=0;
        // Count up all the ACTUAL patches
        for (i=0; i<(6+np_usr+1); ++i) {
          for (j=0; j<6; ++j) {
            if ((AllPatches[i].value!=0) && (AllPatches[i].act_faces[j]>0)) {all_faces+=1;}
          }
        }

        write_int=all_faces;
        tools_WriteInt(fp,&write_int, 1); // -1 since zero is omitted

        for (i=0; i<(6+np_usr+1); ++i)
        {
          if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
          {
            for (k=0; k<6; ++k) {
              if (AllPatches[i].act_faces[k]>0) {
                write_int=AllPatches[i].act_faces[k]*2;
                tools_WriteInt(fp,&write_int, 1);
                for (j=AllPatches[i].start_idx; j<=AllPatches[i].end_idx; ++j)
                {
                  if (Patch[8*j + 7]==k) {
                  write_int=2*j;
                  tools_WriteInt(fp,&write_int, 1);
                  write_int=2*j+1;
                  tools_WriteInt(fp,&write_int, 1);
                  }
                }
              }
            } // End of k loop
          }
        }
      // ----- End of sub patch section for BINARY write -----

    } else {
    if (any_zeros==1)
    {
      write_int=non_blanks-1;
    } else {
      write_int=non_blanks;
    }
    tools_WriteInt(fp,&write_int, 1); // -1 since zero is omitted

    for (i=0; i<(6+np_usr+1); ++i)
    {
      if ((AllPatches[i].patch_cell_count>0)&(AllPatches[i].value!=0))
      {
        write_int=AllPatches[i].patch_cell_count*2;
        tools_WriteInt(fp,&write_int, 1);
        for (j=2*AllPatches[i].start_idx; j<=2*AllPatches[i].end_idx+1; ++j)
        {
          write_int=j;
          tools_WriteInt(fp,&write_int, 1);
        }
      }
    }
    if (debugger==1) printf("DBG: Done writing BINARY solid\n");
  }  // End of ascii/binary solid file test
  }
  // --------------------------------------------------------------------------
  //        ===== Write the VTK if the file ID is not NULL =====
  // --------------------------------------------------------------------------
  if (fp_vtk!=NULL) {

  // If an ASCII is preferred you can easily change that here
  int write_ascii=0;      // Default is 0 to write a binary
  int write_float=1;      // Only compatible with BINARY, but writes data as float to save space

  if (debugger==1) {
    if (write_ascii==1) {
       printf("DBG: Writing ASCII vtk\n");
     } else {
       printf("DBG: Writing BINARY vtk \n");
       if (write_float==1) {
         printf("DBG: -> Storing vtk points as FLOAT \n");
       } else {
         printf("DBG: -> Storing vtk points as DOUBLE \n");
       }
     }
   }

  double *PatchEl;   // patch mean elevations
  int *PatchVal;    // patch ID numbers
  PatchEl=(double*)calloc(cell_faces*2,sizeof(double));
  PatchVal=(int*)calloc(cell_faces*2,sizeof(int));

  k=0;
  for (i=0; i<cell_faces*2; i=i+2)
  {
    PatchEl[i]=(Xp_Act[3*Patch[8*k]+2]+Xp_Act[3*Patch[8*k+1]+2]+Xp_Act[3*Patch[8*k+2]+2])/3.0;
    PatchEl[i+1]=(Xp_Act[3*Patch[8*k+3]+2]+Xp_Act[3*Patch[8*k+4]+2]+Xp_Act[3*Patch[8*k+5]+2])/3.0;

    PatchVal[i]=Patch[8*k + 6];
    PatchVal[i+1]=Patch[8*k + 6];
    k=k+1;
  }

  float *fPatchEl;
  int *fPatchVal;
  float *fXp_Act;

  if (write_float==1)
  {
    fPatchEl=(float*)calloc(cell_faces*2,sizeof(float));
    fXp_Act=(float*)calloc((np_act+1)*3,sizeof(float));
    fPatchVal=(int*)calloc(cell_faces*2,sizeof(int));

    for (i=0; i<=(np_act); i++)
    {
      fXp_Act[3*i]=(float)Xp_Act[3*i];
      fXp_Act[3*i+1]=(float)Xp_Act[3*i+1];
      fXp_Act[3*i+2]=(float)Xp_Act[3*i+2];
    }

    for (i=0; i<cell_faces*2; ++i)
    {
      fPatchEl[i]=(float)PatchEl[i];
      fPatchVal[i]=(int)PatchVal[i];
    }
  }

  if (debugger==1) printf("DBG: Done with VTK prep, writing file...\n");

  //This uses the mixed VTK BINARY legacy format, writes as either double or float
  fprintf(fp_vtk, "# vtk DataFile Version 2.0\n");
  fprintf(fp_vtk, "ParFlow Solid file VTK output\n");
  if (write_ascii==1) {fprintf(fp_vtk, "ASCII\n");} else {fprintf(fp_vtk, "BINARY\n");}
  fprintf(fp_vtk, "DATASET POLYDATA\n");
  if (write_ascii==1) { // ASCII VTK
  fprintf(fp_vtk,"%s %i %s\n","POINTS",np_act+1,"float");
    for (i=0; i<=np_act; ++i)
    {
      fprintf(fp_vtk,"%17.4f %17.4f %17.4f\n",Xp_Act[3*i],Xp_Act[3*i+1],Xp_Act[3*i+2]);
    }
  } else {
  // BINARY VTK
  if (write_float) {
    fprintf(fp_vtk,"%s %i %s\n","POINTS",np_act+1,"float");
    for (i=0; i<=np_act; ++i)
    {
      for (j=0; j<3; ++j)
      {
        tools_WriteFloat(fp_vtk,&fXp_Act[3*i+j], 1);
      }
    }
  } else {
    fprintf(fp_vtk,"%s %i %s\n","POINTS",np_act+1,"double");
    for (i=0; i<=np_act; ++i)
    {
      for (j=0; j<3; ++j)
      {
        tools_WriteDouble(fp_vtk,&Xp_Act[3*i+j], 1);
      }
    }
  }
  }

  int vrtx=3;
  int* nvrtx=&vrtx;
  fprintf(fp_vtk,"%s %i %i\n","POLYGONS",(cell_faces)*2,(cell_faces)*2*4);
  if (write_ascii==1) { // ASCII VTK
    for (i=0; i<cell_faces; ++i)
    {
    fprintf(fp_vtk," %i %i %i %i\n",*nvrtx,Patch[8*i],Patch[8*i+1],Patch[8*i+2]);
    fprintf(fp_vtk," %i %i %i %i\n",*nvrtx,Patch[8*i+3],Patch[8*i+4],Patch[8*i+5]);
    }
  } else {
  // BINARY VTK
  for (i=0; i<cell_faces; ++i)
  {
    tools_WriteInt(fp_vtk,nvrtx, 1);
    for (j=0; j<3; ++j)
    {
      tools_WriteInt(fp_vtk,&Patch[8*i+j], 1);
    }
    tools_WriteInt(fp_vtk,nvrtx, 1);
    for (j=3; j<6; ++j)
    {
      tools_WriteInt(fp_vtk,&Patch[8*i+j], 1);
    }
  }
  }

  // Now write the cell data
  fprintf(fp_vtk,"%s %i\n","CELL_DATA ",(cell_faces)*2);
  if (write_ascii==1) { // ASCII VTK
    fprintf(fp_vtk,"%s %s %s\n","SCALARS","Elev","float");
    fprintf(fp_vtk,"LOOKUP_TABLE default\n");
    for (i=0; i<cell_faces*2; ++i)
    {
      fprintf(fp_vtk,"%17.4f \n",PatchEl[i]);
    }

    fprintf(fp_vtk,"%s %s %s\n","SCALARS","Patch","integer");
    fprintf(fp_vtk,"LOOKUP_TABLE default\n");
    for (i=0; i<cell_faces*2; ++i)
    {
      fprintf(fp_vtk,"%i \n",PatchVal[i]);
    }
  } else {
    if (write_float)
    {
      fprintf(fp_vtk,"\n%s %s %s\n","SCALARS","Elev","float");
      fprintf(fp_vtk,"LOOKUP_TABLE default\n");
      for (i=0; i<cell_faces*2; ++i)
      {
        tools_WriteFloat(fp_vtk,&fPatchEl[i], 1);
      }

      fprintf(fp_vtk,"%s %s %s\n","SCALARS","Patch","integer");
      fprintf(fp_vtk,"LOOKUP_TABLE default\n");
      for (i=0; i<cell_faces*2; ++i)
      {
        tools_WriteInt(fp_vtk,&fPatchVal[i], 1);
      }
    } else
    {
      fprintf(fp_vtk,"\n%s %s %s\n","SCALARS","Elev","double");
      fprintf(fp_vtk,"LOOKUP_TABLE default\n");
      for (i=0; i<cell_faces*2; ++i)
      {
        tools_WriteDouble(fp_vtk,&PatchEl[i], 1);
      }

      fprintf(fp_vtk,"%s %s %s\n","SCALARS","Patch","integer");
      fprintf(fp_vtk,"LOOKUP_TABLE default\n");
      for (i=0; i<cell_faces*2; ++i)
      {
        tools_WriteInt(fp_vtk,&PatchVal[i], 1);
      }
    }
  }

  } // End of VTK write
  out_status=0;
  return out_status;

}  // End of MakePatchySolid

// ****************************************************************************
//  ConvertPfsolAscii2Bin (part of pfsolidfmtconvert command)
//  Conversion function for ASCII solid file to Binary solid file
// ****************************************************************************
int         ConvertPfsolAscii2Bin(FILE *fp_asc,
                                  FILE *fp_bin)
{
  int out_status=-1;

  int debug=0;

  if (debug) printf("\n * * Debugging for Ascii2Bin converter * * \n");

  int read_int,read_ivec[3],i,j,k;
  float read_fltV[3];
  double read_dble;

  int n_vertex,n_tins,n_patches,n_obj;

  fscanf(fp_asc,"%i",&read_int); // Read VERSION #
  tools_WriteInt(fp_bin,&read_int, 1); // Write VERSION #
  if (debug) printf("\n--> Version %i \n",read_int);

  fscanf(fp_asc,"%i",&n_vertex); // Read # of POINTS/VERTICIES
  tools_WriteInt(fp_bin,&n_vertex, 1); //
  if (debug) printf("--> N-Vertices %i \n",n_vertex);

  for (i=0; i<n_vertex; ++i)
  {
    fscanf(fp_asc,"%f %f %f",&read_fltV[0],&read_fltV[1],&read_fltV[2]);

    for (j=0; j<3; ++j)
    {
      read_dble=(double)read_fltV[j];
      tools_WriteDouble(fp_bin,&read_dble, 1);
    }
  }

  int nS;
  fscanf(fp_asc,"%i",&nS); // Read Number of SOLIDS (only one allowed here)
  // if (nS>1)
  // {
  //   printf("ERROR (pfsolascii2asc): More than one solid object detected. Only one is allowed");
  // } else if (nS<1) {
  //   printf("ERROR (pfsolascii2asc): No one solid objects detected in pfsol");
  // }
  tools_WriteInt(fp_bin,&nS, 1);
  if (debug) printf("--> N-Solids %i \n",nS);

  for (k=0; k<nS; ++k) {
  fscanf(fp_asc,"%i",&n_tins); // Total number of triangles
  tools_WriteInt(fp_bin,&n_tins, 1);
  if (debug) printf("--> N-tins %i \n",n_tins);

  for (i=0; i<n_tins; ++i)
  {
    fscanf(fp_asc,"%i %i %i",&read_ivec[0],&read_ivec[1],&read_ivec[2]);
    // printf("%i %i %i\n",read_ivec[0],read_ivec[1],read_ivec[2]);
    for (j=0; j<3; ++j)
    {
      read_int=read_ivec[j];
      tools_WriteInt(fp_bin,&read_int, 1);
      // printf(" %i ",read_int);
    }
    // printf("\n");
  }

  fscanf(fp_asc,"%i",&n_patches); // Total number of triangles
  tools_WriteInt(fp_bin,&n_patches, 1);
  if (debug) printf("--> N-patch %i \n",n_patches);

  for (i=0; i<n_patches; ++i)
  {
    fscanf(fp_asc,"%i",&n_obj); // Elements in this patch
    tools_WriteInt(fp_bin,&n_obj, 1);
    if (debug) printf("--> Patch members: %i\n",n_obj);
    for (j=0; j<n_obj; ++j)
    {
      fscanf(fp_asc,"%i",&read_int);
      tools_WriteInt(fp_bin,&read_int, 1);
    }
  }
  } // End of loop over solids

  out_status=0;
  return out_status;
} // End of ascii 2 binary solid file conversion

// ****************************************************************************
//  ConvertPfsolBin2Ascii (part of pfsolidfmtconvert command)
//  Conversion function for Binary solid file to ASCII solid file
// ****************************************************************************
int         ConvertPfsolBin2Ascii(FILE *fp_bin,
                                  FILE *fp_asc)
{
  // printf("\n ERROR (pfsolbin2ascii): Routine not enabled\n");

  int out_status=-1;

  int debug=0;

  if (debug) printf("\n * * Debugging for Bin2Ascii converter * * \n");

  int read_int,read_ivec[3],i,j,k;
  float read_fltV[3];
  double read_dble;

  int n_vertex,n_tins,n_patches,n_obj;

  tools_ReadInt(fp_bin,&read_int, 1); // Read VERSION #
  fprintf(fp_asc,"%i\n",read_int); // Write VERSION #
  if (debug) printf("\n--> Version %i \n",read_int);

  tools_ReadInt(fp_bin,&n_vertex, 1); // Read # of POINTS/VERTICIES
  fprintf(fp_asc,"%i\n",n_vertex); //
  if (debug) printf("--> N-Vertices %i \n",n_vertex);
  for (i=0; i<n_vertex; ++i)
  {
    for (j=0; j<3; ++j)
    {
      tools_ReadDouble(fp_bin,&read_dble, 1);
      read_fltV[j]=read_dble;
    }
    fprintf(fp_asc,"%17.4f %17.4f %17.4f\n",read_fltV[0],read_fltV[1],read_fltV[2]);
  }

  int nS;
  tools_ReadInt(fp_bin,&nS, 1); // Read Number of SOLIDS (only one allowed here)
  // if (nS>1)
  // {
  //   printf("ERROR (pfsolascii2asc): More than one solid object detected. Only one is allowed");
  // } else if (nS<1) {
  //   printf("ERROR (pfsolascii2asc): No one solid objects detected in pfsol");
  // }
  fprintf(fp_asc,"%i\n",nS);
  if (debug) printf("--> N-Solids %i \n",nS);

  for (k=0; k<nS; ++k) {
  tools_ReadInt(fp_bin,&n_tins, 1); // Total number of triangles
  fprintf(fp_asc,"%i\n",n_tins);
  if (debug) printf("--> N-tins %i \n",n_tins);

  for (i=0; i<n_tins; ++i)
  {
    for (j=0; j<3; ++j)
    {
      tools_ReadInt(fp_bin,&read_int, 1);
      read_ivec[j]=read_int;
    }
    fprintf(fp_asc,"%i %i %i\n",read_ivec[0],read_ivec[1],read_ivec[2]);
  }

  tools_ReadInt(fp_bin,&n_patches, 1);  // Total number of triangles
  fprintf(fp_asc,"%i\n",n_patches);
  if (debug) printf("--> N-patch %i \n",n_patches);

  for (i=0; i<n_patches; ++i)
  {
    tools_ReadInt(fp_bin,&n_obj, 1);  // Elements in this patch
    fprintf(fp_asc,"%i\n",n_obj);
    if (debug) printf("--> Patch members: %i\n",n_obj);

    for (j=0; j<n_obj; ++j)
    {
      tools_ReadInt(fp_bin,&read_int, 1);
      fprintf(fp_asc,"%i\n",read_int);
    }
  }
  } // End of loop over solids

  out_status=0;
  return out_status;
} // End of binary 2 ascii solid file conversion
