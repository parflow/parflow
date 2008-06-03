/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <string.h>
#include <avs/avs.h>
#include <avs/port.h>
#include <avs/field.h>
 
/* ----> START OF USER-SUPPLIED CODE SECTION #1 (INCLUDE FILES, GLOBAL VARIABLES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #1                            */
 
/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int Zero_Outer_desc()
{

	int in_port, out_port, param, iresult;
	extern int Zero_Outer_compute();

	AVSset_module_name("Zero_Outer", MODULE_FILTER);

	/* Input Port Specifications               */

	in_port = AVScreate_input_port("PF_field_in", 
		"field 3D 3-space scalar rectilinear float", REQUIRED);

	in_port = AVScreate_input_port("X_inter_by_yz", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);
	in_port = AVScreate_input_port("Y_inter_by_zx", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);
	in_port = AVScreate_input_port("Z_inter_by_xy", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);


	/* Output Port Specifications              */

	out_port = AVScreate_output_port("PF_field_out", 
		"field 3D scalar 3-coord  rectilinear float");
	/* Parameter Specifications                */
	param = AVSadd_parameter("Solid_num", "integer", 0, 0, 10);
	AVSconnect_widget(param, "idial");

	param = AVSadd_float_parameter("Outer_val", 0.00000, FLOAT_UNBOUND, 
		FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_parameter("Ghost_Points", "choice", "No_Ghost_pt", 
		"Ghost_pt:No_Ghost_pt", ":");

	AVSset_compute_proc(Zero_Outer_compute);

	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int Zero_Outer_compute( PF_field_in,  X_inter_by_yz, Y_inter_by_zx,
 Z_inter_by_xy, PF_field_out, Solid_num, Outer_val, Ghost_Points)
     AVSfield_float *PF_field_in;
     AVSfield_float *X_inter_by_yz;
     AVSfield_float *Y_inter_by_zx;
     AVSfield_float *Z_inter_by_xy;
     AVSfield_float **PF_field_out;
     int Solid_num;
     float *Outer_val;
     char *Ghost_Points;
{
#define  ABS(A)  ((A) > (0) ? (A) : (-A))
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  INSIDE 1
#define  OUTSIDE 0
        int i, j, k;
        int l, m, n, lp;
	float *data_plane_temp;
        int   dim_X;
        int   dim_Y;
        int   dim_Z;
	int *temp_intersect_C1;
	float *temp_intersect_V1;
	int hold_I;
	int hold_intersect_C1;
	float hold_intersect_V1;
	float Max_intersect_V1;
	float *grid_1;
	float *grid_2;
	float *grid_3;
        int num_inter_at_l;
        int num_inter_at_lp1;
        int num_inter_at_m;
        int num_inter_at_mp1;
        int got_start; 
        int got_fin;
        int I;
	int Ip;
	int Iv;
        int S;
        int S_index;
        int S_num;
        int Picked_Solid_Num;
        int temp_solid_num;
        float temp_inter_sec;
	float temp_start, temp_fin;
        int max_intersects;
	int *in_or_out;
	int in_out_flag;
	int in_out_ct;
	int num_intersect;
	int dims[3];
	float  *temp_Data;
	int N_solids; 
	float ave_pf_field;
	int num_not_ave;
	int num_pf_field;

/* Scan intersection data for max solid number    */

	N_solids = 0;
	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(I=1; I < max_intersects; I++)
	  {
	    temp_solid_num =ABS((int) I3DV(Z_inter_by_xy,i,j,I)[0]);
	    N_solids = MAX(N_solids,temp_solid_num);
	  }

	AVSmodify_parameter("Solid_num",AVS_MAXVAL,0,0,N_solids-1);

	Picked_Solid_Num =Solid_num + 1;
 
	dim_X=(PF_field_in)->dimensions[0];
	dim_Y=(PF_field_in)->dimensions[1];
	dim_Z=(PF_field_in)->dimensions[2];

	dims[0] = dim_X;
	dims[1] = dim_Y;
	dims[2] = dim_Z;

	*PF_field_out = (AVSfield_float *) 
	  AVSdata_alloc("field 3D scalar 3-coord rectilinear float", dims);

	AVSfield_copy_points(PF_field_in,*PF_field_out); 

	    data_plane_temp = (float *)malloc(dim_X * dim_Y * dim_Z * sizeof(float));
	    grid_1 = (float *)malloc(dim_X * sizeof(float));
	    grid_2 = (float *)malloc(dim_Y * sizeof(float));
	    grid_3 = (float *)malloc(dim_Z * sizeof(float));

	for(i=0; i < dim_X; i++)
	    grid_1[i] = (PF_field_in)->points[i];

	for(j=0; j < dim_Y; j++)
	    grid_2[j] = (PF_field_in)->points[dim_X + j];

	for(k=0; k < dim_Z; k++)
	    grid_3[k] = (PF_field_in)->points[dim_X + dim_Y + k];


	for(i=0; i < dim_X; i++)
	  for(j=0; j < dim_Y; j++)
	    for(k=0; k < dim_Z; k++)
	      {
		data_plane_temp[i + dim_X * j + dim_X * dim_Y * k] = 
		  I3D(PF_field_in, i, j, k); 
	      }


	    max_intersects  = (Z_inter_by_xy)->dimensions[2];


	temp_intersect_C1 = 
	  (int *)malloc(dim_X * dim_Y * max_intersects *  sizeof(int));
	temp_intersect_V1 = 
	  (float *)malloc(dim_X * dim_Y * max_intersects *  sizeof(float));

	in_or_out = (int *)malloc(dim_X * dim_Y *  dim_Z * sizeof(int));

	      for(i=0; i < dim_X; i++)
	      for(j=0; j < dim_Y; j++)
	      for(I=0; I < max_intersects; I++)
		{
		  temp_intersect_C1[i + dim_X*j + dim_X*dim_Y*I] = 0;
		  temp_intersect_V1[i + dim_X*j + dim_X*dim_Y*I] = 0.0;
		}

	      for(i=0; i < dim_X; i++)
	      for(j=0; j < dim_Y; j++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Z_inter_by_xy,i,j,I)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip +1;
			  temp_intersect_C1[i + dim_X * j + dim_X * dim_Y * Ip] = 
			    (int) I3DV(Z_inter_by_xy,i,j,I)[0]; 
			  temp_intersect_V1[i + dim_X * j + dim_X * dim_Y * Ip] = 
			    I3DV(Z_inter_by_xy,i,j,I)[1];
			}
		    }
		  temp_intersect_C1[i + dim_X * j + dim_X * dim_Y * 0] = Ip; 
	        }



/* reorder intersections */
	for(l=0; l < dim_X; l++)
        for(m=0; m < dim_Y; m++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*0];
	    for(Ip=num_inter_at_l; Ip > 1; Ip--)
	      {
		Max_intersect_V1 = -1.0e9;
		for(I=1; I <= Ip; I++)
		  {
		    if(temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I] >Max_intersect_V1)
		      {
			Max_intersect_V1 = temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I];
			hold_I = I;
			hold_intersect_C1 = temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*I];
			hold_intersect_V1 = temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I];
		      }
	          }
		temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*hold_I] = 
		  temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*Ip];
		temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*hold_I] = 
		  temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*Ip];
		temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*Ip] = hold_intersect_C1; 
		temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*Ip] = hold_intersect_V1;
	      }
	  }


/*initilize in_or_out flag */
	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(k=0; k < dim_Z; k++)
	    {
	      in_or_out[i + dim_X * j + dim_X * dim_Y * k] = OUTSIDE;
	    }

	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
	  {
	    num_inter_at_l = temp_intersect_C1[i + dim_X*j + dim_X*dim_Y*0];
/* loop over intersection pairs */
	    for( I=1; I <= num_inter_at_l; I=I+2)
	    {
		for(k=0; k < dim_Z; k++)
		  {
		    if(grid_3[k] >= temp_intersect_V1[i + dim_X*j + dim_X*dim_Y*I] &&
		       grid_3[k] < temp_intersect_V1[i + dim_X*j + dim_X*dim_Y*(I+1)])
		      {
			in_or_out[i + dim_X * j + dim_X * dim_Y * k] = INSIDE;
		      }
		  }
	      }
	  }


        temp_Data = (*PF_field_out)->data;

 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i, j, k); 
	      }
	    else
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * k) = *Outer_val;
	      }
	  }          

/*	ave_pf_field=0.0;
	num_pf_field=0;
	
 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
	for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		ave_pf_field = ave_pf_field + I3D(PF_field_in, i, j, k);
		num_pf_field = num_pf_field + 1;
	      }
	  }
	ave_pf_field = ave_pf_field/num_pf_field;

	printf("ave_pf_field %f\n",ave_pf_field);

	num_not_ave = 0;
	
 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
	for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE &&
	       I3D(PF_field_in, i, j, k) != ave_pf_field)
	      {
		num_not_ave = num_not_ave + 1;
	      }
	  }

	 printf("num_not_ave %d\n",num_not_ave); */

/*   Fill in ghost points   */
   if(strncmp(Ghost_Points,"Ghost_pt",8) == 0)
     {
 
 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(k=0; k < dim_Z-1; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == OUTSIDE &&
	       in_or_out[i + dim_X * j + dim_X * dim_Y * (k+1)] == INSIDE)
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i, j, k+1); 
/*	    printf("k,PF_field_in %d %f\n",k,I3D(PF_field_in, i, j, k)); */
	      }
	  }          

 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(k=dim_Z-2; k > 0; k--)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * (k+1)] == OUTSIDE &&
	       in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * (k+1)) =
		  I3D(PF_field_in, i, j, k); 
	      }
	  }          

 	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y-1; j++)
        for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == OUTSIDE &&
	       in_or_out[i + dim_X * (j+1) + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i, j+1, k); 
	      }
	  }          

 	for(i=0; i < dim_X; i++)
        for(j=dim_Y-2; j > 0; j--)
        for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * (j+1) + dim_X * dim_Y * k] == OUTSIDE &&
	       in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i + dim_X * (j+1) + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i, j, k); 
	      }
	  }          

 	for(i=0; i < dim_X-1; i++)
        for(j=0; j < dim_Y; j++)
        for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i + dim_X * j + dim_X * dim_Y * k] == OUTSIDE &&
	       in_or_out[i+1 + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i + dim_X * j + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i+1, j, k); 
	      }
	  }          

 	for(i=dim_X-2; i > 0; i--)
        for(j=0; j < dim_Y; j++)
        for(k=0; k < dim_Z; k++)
	  {
	    if(in_or_out[i+1 + dim_X * j + dim_X * dim_Y * k] == OUTSIDE &&
	       in_or_out[i + dim_X * j + dim_X * dim_Y * k] == INSIDE)
	      {
		*(temp_Data + i+1 + dim_X * j + dim_X * dim_Y * k) =
		  I3D(PF_field_in, i, j, k); 
	      }
	  }          
      }

/* free any locally allocated data */
	free(data_plane_temp);
	free(grid_1);
	free(grid_2);
	free(grid_3);
	free(temp_intersect_C1);
	free(temp_intersect_V1);
	free(in_or_out);

/* THIS IS THE END OF THE 'HINTS' AREA. ADD YOUR OWN CODE BETWEEN THE    */
/* FOLLOWING COMMENTS. THE CODE YOU SUPPLY WILL NOT BE OVERWRITTEN.      */
 
/* ----> START OF USER-SUPPLIED CODE SECTION #3 (COMPUTE ROUTINE BODY)   */
/* <---- END OF USER-SUPPLIED CODE SECTION #3                            */
	return(1);
}

/* <---- END OF USER-SUPPLIED CODE SECTION #4                            */
