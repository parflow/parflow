/* mod_gen Version 1                                                     */
/* Module Name: "PF_Slicer" (Mapper) (Subroutine)                        */
/* Author: Dan &,2-4019,B451 R2024                                      */
/* Date Created: Fri Jun  7 14:26:30 1996                                */
/*                                                                       */

#include <stdio.h>
#include <string.h>
#include <avs/avs.h>
#include <avs/port.h>
#include <avs/field.h>
#include <avs/geom.h>
#include <avs/colormap.h>
 
/* ----> START OF USER-SUPPLIED CODE SECTION #1 (INCLUDE FILES, GLOBAL VARIABLES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #1                            */
 
/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int PF_Slicer_desc()
{

	int in_port, out_port, param, iresult;
	extern int PF_Slicer_compute();

	AVSset_module_name("PF_Slicer", MODULE_MAPPER);

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
	out_port = AVScreate_output_port("Slice_out", "geom");

	/* Parameter Specifications                */
	param = AVSadd_parameter("Slice_Axis", "choice", "X_Axis", 
		"X_Axis:Y_Axis:Z_Axis", ":");
	  AVSconnect_widget(param, "radio_buttons");
	param = AVSadd_parameter("Slice_plane", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_1_start", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_1_fin", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_2_start", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_2_fin", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");

	param = AVSadd_parameter("Solid_num", "integer", 0, 0, 10);
	AVSconnect_widget(param, "typein_integer");

	AVSset_compute_proc(PF_Slicer_compute);
	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int PF_Slicer_compute( PF_field_in,  X_inter_by_yz, Y_inter_by_zx,
 Z_inter_by_xy, Slice_out,
 Slice_Axis, Slice_plane,
 Grid_1_start, Grid_1_fin,
 Grid_2_start, Grid_2_fin, Solid_num)
	AVSfield_float *PF_field_in;
	AVSfield_float *X_inter_by_yz;
	AVSfield_float *Y_inter_by_zx;
	AVSfield_float *Z_inter_by_xy;
	GEOMedit_list *Slice_out;
	char *Slice_Axis;
	int Slice_plane;
	int Grid_1_start, Grid_1_fin;
	int Grid_2_start, Grid_2_fin;
	int Solid_num;
{
#define  ABS(A)  ((A) > (0) ? (A) : (-A))
#define  ABSF(A)  ((A) > (0.0) ? (A) : (-A))
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  MIN(A, B)  ((A) < (B) ? (A) : (B))
#define  MAX_SECTIONS 6
#define  X_AXIS 1
#define  Y_AXIS 2
#define  Z_AXIS 3
#define  NO 0
#define  YES 1
#define  INSIDE 1
#define  OUTSIDE 0
	GEOMobj *Tmp_obj;
        int axis;
        int i, j, k;
        int l, m, lp;
        int l_start, l_fin;
        int m_start, m_fin;
        int m_start_1, m_fin_1;
        int m_start_2, m_fin_2;
	float *x_grid;
	float *y_grid;
	float *z_grid;
	float *data_plane_temp;
        int   dim_X;
        int   dim_Y;
        int   dim_Z;
        int   dim_1;
        int   dim_2;
        int   dim_3;
        int   dim_I1;
        int   dim_I2;
        int   dim_I3;
	int *temp_intersect_C1;
	float *temp_intersect_V1;
	int *temp_intersect_C2;
	float *temp_intersect_V2;
	int hold_I;
	int hold_intersect_C1;
	float hold_intersect_V1;
	float Max_intersect_V1;
	int hold_intersect_C2;
	float hold_intersect_V2;
	float Max_intersect_V2;
	float *grid_1;
	float *grid_2;
	float del_grid_1;
	float del_grid_2;
	float grid_3;
	float verts[6][3];
	float norm_verts[6][3];
        int cindex;
        float hue, sat, value;
        float test_val_0, test_val_1, test_val_2, test_val_3, test_val_4;
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
        int *num_sections;
	float temp_start, temp_fin;
        int max_intersects;
	int lc, lf, nc, nf, nmax;
	float sum_temp, max_dist;
	float diff_1, diff_2;
	float test_range_l, test_range_lp1, test_distance;
	float dist_cut_off;
	int *in_or_out;
	int in_out_flag;
	int intersect_ct;
	int in_out_ct;
	int num_intersect;
	float test_intersection;
	float get_intersection();
	float get_intersection_diag();
	int l_of_node[4],m_of_node[4];
	int l_0, l_1, l_2, l_3;
	int m_0, m_1, m_2, m_3;
	float u_0, v_0 ,u_1, v_1, u_2, v_2, u_3, v_3, u_4 ,v_4;
	float weight_1, weight_2;
	float normal_dir;
        float line_verts[2][3];
	int N_solids; 

	normal_dir = 1.0;

	dim_X = (PF_field_in)->dimensions[0];
	dim_Y = (PF_field_in)->dimensions[1];
	dim_Z = (PF_field_in)->dimensions[2];
        x_grid = (float *)malloc(dim_X * sizeof(float));
        y_grid = (float *)malloc(dim_Y * sizeof(float));
        z_grid = (float *)malloc(dim_Z * sizeof(float));

	for(i=0; i < dim_X; i++)
	  x_grid[i] = (PF_field_in)->points[i];
	for(j=0; j < dim_Y; j++)
	  y_grid[j] = (PF_field_in)->points[dim_X+j];
	for(k=0; k < dim_Z; k++)
	  z_grid[k] = (PF_field_in)->points[dim_X+dim_Y+k];


      if(strncmp(Slice_Axis,"X_Axis",6) == 0) axis = X_AXIS; 
      if(strncmp(Slice_Axis,"Y_Axis",6) == 0) axis = Y_AXIS; 
      if(strncmp(Slice_Axis,"Z_Axis",6) == 0) axis = Z_AXIS; 

/*printf(" axis %d \n",axis); */
/*printf("Slice_plane  %d \n",Slice_plane); */

	if(axis == X_AXIS)
	  {
	    dim_3 = dim_X;
	    dim_1 = dim_Y;
	    dim_2 = dim_Z;
	    dim_I1 = (Z_inter_by_xy)->dimensions[0];
	    dim_I2 = (Z_inter_by_xy)->dimensions[1];
	    dim_I3 = (Z_inter_by_xy)->dimensions[2];
	    del_grid_1 = RECT_Y(PF_field_in)[2] - RECT_Y(PF_field_in)[1];
	    del_grid_2 = RECT_Z(PF_field_in)[2] - RECT_Z(PF_field_in)[1];
	    if(Slice_plane > (dim_3-1))Slice_plane = dim_3-1;
	    grid_3 = (PF_field_in)->points[Slice_plane];
	    AVSmodify_parameter("Slice_plane",AVS_MAXVAL,0,0,dim_X-1);
	  }

	    
	if(axis == Y_AXIS)
	  {
	    dim_3 = dim_Y;
	    dim_1 = dim_Z;
	    dim_2 = dim_X;
	    dim_I1 = (X_inter_by_yz)->dimensions[0];
	    dim_I2 = (X_inter_by_yz)->dimensions[1];
	    dim_I3 = (X_inter_by_yz)->dimensions[2];
	    del_grid_1 = RECT_Z(PF_field_in)[2] - RECT_Z(PF_field_in)[1];
	    del_grid_2 = RECT_X(PF_field_in)[2] - RECT_X(PF_field_in)[1];
	    if(Slice_plane > (dim_3-1))Slice_plane = dim_3-1;
	    grid_3 = (PF_field_in)->points[dim_X + Slice_plane];
	    AVSmodify_parameter("Slice_plane",AVS_MAXVAL,0,0,dim_Y-1);
	  }
	    
	if(axis == Z_AXIS)
	  {
	    dim_3 = dim_Z;
	    dim_1 = dim_X;
	    dim_2 = dim_Y;
	    dim_I1 = (Y_inter_by_zx)->dimensions[0];
	    dim_I2 = (Y_inter_by_zx)->dimensions[1];
	    dim_I3 = (Y_inter_by_zx)->dimensions[2];
	    del_grid_1 = RECT_X(PF_field_in)[2] - RECT_X(PF_field_in)[1];
	    del_grid_2 = RECT_Y(PF_field_in)[2] - RECT_Y(PF_field_in)[1];
	    if(Slice_plane > (dim_3-1))Slice_plane = dim_3-1;
	    grid_3 = (PF_field_in)->points[dim_X + dim_Y + Slice_plane];
	    AVSmodify_parameter("Slice_plane",AVS_MAXVAL,0,0,dim_Z-1);
	  }

/**** MAX are turned off ***************/
	Grid_1_fin =dim_1-1;     
	Grid_2_fin =dim_2-1;     
	AVSparameter_visible("Grid_1_fin",0);     
	AVSparameter_visible("Grid_2_fin",0);     

	AVSmodify_parameter("Grid_1_start",AVS_MAXVAL,0,0,dim_1-1);
	AVSmodify_parameter("Grid_1_fin",AVS_MAXVAL,0,0,dim_1-1);
	AVSmodify_parameter("Grid_2_start",AVS_MAXVAL,0,0,dim_2-1);
	AVSmodify_parameter("Grid_2_fin",AVS_MAXVAL,0,0,dim_2-1);

	if(Grid_1_start < 0)Grid_1_start =0;
	if(Grid_1_fin > dim_1-1)Grid_1_fin =dim_1-1;
	if(Grid_2_start < 0)Grid_2_start =0;
	if(Grid_2_fin > dim_2-1)Grid_2_fin =dim_2-1;

/* printf(" dim_1 %d \n",dim_1); */
/* printf(" dim_2 %d \n",dim_2); */
/* printf(" grid_3 %f \n",grid_3); */


	    data_plane_temp = (float *)malloc(dim_1 * dim_2 * sizeof(float));
	    grid_1 = (float *)malloc(dim_1 * sizeof(float));
	    grid_2 = (float *)malloc(dim_2 * sizeof(float));
	    num_sections = (int *)malloc(dim_1  * sizeof(int));

	for(l=0; l < dim_1; l++)
	  {
	      if(axis == X_AXIS)grid_1[l] = (PF_field_in)->points[dim_X + l];
	      if(axis == Y_AXIS)grid_1[l] = (PF_field_in)->points[dim_X + dim_Y + l];
	      if(axis == Z_AXIS)grid_1[l] = (PF_field_in)->points[l];
	  }

	for(m=0; m < dim_2; m++)
	  {
	      if(axis == X_AXIS)grid_2[m] = (PF_field_in)->points[dim_X + dim_Y + m];
	      if(axis == Y_AXIS)grid_2[m] = (PF_field_in)->points[m];
	      if(axis == Z_AXIS)grid_2[m] = (PF_field_in)->points[dim_X + m];
	  }


	for(l=0; l < dim_1; l++)
	  for(m=0; m < dim_2; m++)
	    {
	      if(axis == X_AXIS)data_plane_temp[l +dim_1 * m] = 
                I3D(PF_field_in, Slice_plane, l, m); 

 	      if(axis == Y_AXIS)data_plane_temp[l +dim_1 * m] = 
                I3D(PF_field_in, m, Slice_plane, l); 

 	      if(axis == Z_AXIS)data_plane_temp[l +dim_1 * m] = 
                I3D(PF_field_in, l, m, Slice_plane); 
	    }


/*  THIS IS A 'HINTS' AREA - YOU MAY CUT AND PASTE AT WILL FROM IT       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/* Create the Slice_out object                                           */
/*	Tmp_obj = GEOMcreate_obj(GEOM_POLYTRI, NULL);   */
  Tmp_obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);

/*                                                                       */

	if(axis == X_AXIS) max_intersects = dim_I3;
	if(axis == Y_AXIS) max_intersects = dim_I1;
	if(axis == Z_AXIS) max_intersects = dim_I2;

 
/* Scan intersection data for max solid number    */

	N_solids = 0;
	for(i=0; i < dim_X; i++)
        for(j=0; j < dim_Y; j++)
        for(I=1; I < max_intersects; I++)
	  {
	    temp_solid_num =ABS((int) I3DV(Z_inter_by_xy,i,j,I)[0]);
	    N_solids = MAX(N_solids,temp_solid_num);
	  }

/* make sure Solid_num is valid, in range     */
	if(Solid_num < 0)Solid_num = 0;
	if(Solid_num > (N_solids-1))Solid_num = N_solids-1;

	AVSmodify_parameter("Solid_num",AVS_MAXVAL,0,0,N_solids-1);

	Picked_Solid_Num = Solid_num + 1;
 

/* 	printf("max_intersects %d\n",max_intersects);  */

	temp_intersect_C1 = 
	  (int *)malloc(dim_1 * max_intersects *  sizeof(int));
	temp_intersect_V1 = 
	  (float *)malloc(dim_1 * max_intersects *  sizeof(float));

	temp_intersect_C2 = 
	  (int *)malloc(dim_2 * max_intersects *  sizeof(int));
	temp_intersect_V2 = 
	  (float *)malloc(dim_2 * max_intersects *  sizeof(float));

	in_or_out = (int *)malloc(dim_1 * dim_2 *  sizeof(int));

	      for(l=0; l < dim_1; l++)
	      for(I=0; I < max_intersects; I++)
		{
		  temp_intersect_C1[l + dim_1 * I] = 0;
		  temp_intersect_V1[l + dim_1 * I] = 0.0;
		}

	      for(m=0; m < dim_2; m++)
	      for(I=0; I < max_intersects; I++)
		{
		  temp_intersect_C2[m + dim_2 * I] = 0;
		  temp_intersect_V2[m + dim_2 * I] = 0.0;
		}

          if(axis == X_AXIS)
	    {
	      for(l=0; l < dim_1; l++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Z_inter_by_xy,Slice_plane,l,I)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip +1;
			  temp_intersect_C1[l + dim_1 * Ip] = 
			    (int) I3DV(Z_inter_by_xy,Slice_plane,l,I)[0]; 
			  temp_intersect_V1[l + dim_1 * Ip] = 
			    I3DV(Z_inter_by_xy,Slice_plane,l,I)[1];
			}
		    }
		  temp_intersect_C1[l + dim_1 * 0] = Ip; 
	        }
	      for(m=0; m < dim_2; m++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Y_inter_by_zx,Slice_plane,I,m)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip +1;
			  temp_intersect_C2[m + dim_2 * Ip] =
			    (int)  I3DV(Y_inter_by_zx,Slice_plane,I,m)[0]; 
			  temp_intersect_V2[m + dim_2 * Ip] =
			    I3DV(Y_inter_by_zx,Slice_plane,I,m)[1]; 
			}
		    }
		  temp_intersect_C2[m + dim_2 * 0] = Ip; 
		}
	    }
	
          if(axis == Y_AXIS)
	    {
	      for(l=0; l < dim_1; l++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(X_inter_by_yz,I,Slice_plane,l)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip + 1;
			  temp_intersect_C1[l + dim_1 * Ip] =
			    (int) I3DV(X_inter_by_yz,I,Slice_plane,l)[0]; 
			  temp_intersect_V1[l + dim_1 * Ip] =
			    I3DV(X_inter_by_yz,I,Slice_plane,l)[1]; 
			}
		    }
		  temp_intersect_C1[l + dim_1 * 0] = Ip; 
	        }
	      for(m=0; m < dim_2; m++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Z_inter_by_xy,m,Slice_plane,I)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip + 1;
			  temp_intersect_C2[m + dim_2 * Ip] = 
			    (int) I3DV(Z_inter_by_xy,m,Slice_plane,I)[0]; 
			  temp_intersect_V2[m + dim_2 * Ip] = 
			    I3DV(Z_inter_by_xy,m,Slice_plane,I)[1]; 
			}
		    }
		  temp_intersect_C2[m + dim_2 * 0] = Ip; 
		}
	    }

          if(axis == Z_AXIS)
	    {
	      for(l=0; l < dim_1; l++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Y_inter_by_zx,l,I,Slice_plane)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip + 1;
			  temp_intersect_C1[l + dim_1 * Ip] =
			    (int)  I3DV(Y_inter_by_zx,l,I,Slice_plane)[0]; 
			  temp_intersect_V1[l + dim_1 * Ip] =
			    I3DV(Y_inter_by_zx,l,I,Slice_plane)[1]; 
			}
		    }
		  temp_intersect_C1[l + dim_1 * 0] = Ip; 
		}

	      for(m=0; m < dim_2; m++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(X_inter_by_yz,I,m,Slice_plane)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip + 1;
			  temp_intersect_C2[m + dim_2 * Ip] =
			    (int) I3DV(X_inter_by_yz,I,m,Slice_plane)[0]; 
			  temp_intersect_V2[m + dim_2 * Ip] =
			    I3DV(X_inter_by_yz,I,m,Slice_plane)[1]; 
			}
		    }
		  temp_intersect_C2[m + dim_2 * 0] = Ip; 
		}
	    }

/* 	for(l=0; l < dim_1; l++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_1 * 0];
	    for(I=1; I <= num_inter_at_l; I++)
	      {
	    printf(" l,I,temp_intersects %d %d %d %f\n",l,I,
		   temp_intersect_C1[l + dim_1 * I],
		   temp_intersect_V1[l + dim_1 * I]);
	      }
	  }                 

 	for(m=0; m < dim_2; m++)
	  {
	    num_inter_at_m = temp_intersect_C2[m + dim_2 * 0];
	    for(I=1; I <= num_inter_at_m; I++)
	      {
	    printf(" m,I,temp_intersects %d %d %d %f\n",m,I,
		   temp_intersect_C2[m + dim_2 * I],
		   temp_intersect_V2[m + dim_2 * I]);
	      }
	  }        */             


/* reorder intersections */

	for(l=0; l < dim_1; l++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_1 * 0];
	    for(Ip=num_inter_at_l; Ip > 1; Ip--)
	      {
		Max_intersect_V1 = -1.0e9;
		for(I=1; I <= Ip; I++)
		  {
		    if(temp_intersect_V1[l + dim_1 * I] >Max_intersect_V1)
		      {
			Max_intersect_V1 = temp_intersect_V1[l + dim_1 * I];
			hold_I = I;
			hold_intersect_C1 = temp_intersect_C1[l + dim_1 * I];
			hold_intersect_V1 = temp_intersect_V1[l + dim_1 * I];
		      }
	          }
		temp_intersect_C1[l + dim_1 * hold_I] = 
		  temp_intersect_C1[l + dim_1 * Ip];
		temp_intersect_V1[l + dim_1 * hold_I] = 
		  temp_intersect_V1[l + dim_1 * Ip];
		temp_intersect_C1[l + dim_1 * Ip] = hold_intersect_C1; 
		temp_intersect_V1[l + dim_1 * Ip] = hold_intersect_V1;
	      }
	  }

	for(m=0; m < dim_2; m++)
	  {
	    num_inter_at_m = temp_intersect_C2[m + dim_2 * 0];
	    for(Ip=num_inter_at_m; Ip > 1; Ip--)
	      {
		Max_intersect_V2 = -1.0e9;
		for(I=1; I <= Ip; I++)
		  {
		    if(temp_intersect_V2[m + dim_2 * I] >Max_intersect_V2)
		      {
			Max_intersect_V2 = temp_intersect_V2[m + dim_2 * I];
			hold_I = I;
			hold_intersect_C2 = temp_intersect_C2[m + dim_2 * I];
			hold_intersect_V2 = temp_intersect_V2[m + dim_2 * I];
		      }
	          }
		temp_intersect_C2[m + dim_2 * hold_I] = 
		  temp_intersect_C2[m + dim_2 * Ip];
		temp_intersect_V2[m + dim_2 * hold_I] = 
		  temp_intersect_V2[m + dim_2 * Ip];
		temp_intersect_C2[m + dim_2 * Ip] = hold_intersect_C2; 
		temp_intersect_V2[m + dim_2 * Ip] = hold_intersect_V2;
	      }
	  }


/*   	for(l=0; l < dim_1; l++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_1 * 0];
	    for(I=1; I <= num_inter_at_l; I++)
	      {
	    printf("post sort, l,I,temp_intersects %d %d %d %f\n",l,I,
		   temp_intersect_C1[l + dim_1 * I],
		   temp_intersect_V1[l + dim_1 * I]);
	      }
	  }        */

/*  	for(m=0; m < dim_2; m++)
	  {
	    num_inter_at_m = temp_intersect_C2[m + dim_2 * 0];
	    for(I=1; I <= num_inter_at_m; I++)
	      {
	    printf("post sort, m,I,temp_intersects %d %d %d %f\n",m,I,
		   temp_intersect_C2[m + dim_2 * I],
		   temp_intersect_V2[m + dim_2 * I]);
	      }
	  }         */


/*initilize in_or_out flag */
	for(l=0; l < dim_1; l++)
	  for(m=0; m < dim_2; m++)
	    {
	      in_or_out[l + dim_1 * (m)] = OUTSIDE;
	    }

	for(l=0; l < dim_1; l++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_1 * 0];
/* loop over intersection pairs */
	    for( I=1; I <= num_inter_at_l; I=I+2)
	    {
	      for(m=0; m < dim_2; m++)
		{
		  if(grid_2[m] > temp_intersect_V1[l + dim_1 * I] &&
		     grid_2[m] < temp_intersect_V1[l + dim_1 * (I+1)])
		    {
		      in_or_out[l + dim_1 * (m)] = INSIDE;
		    }
		}
	    }
	  }

/*	for(m=0; m < dim_2; m++)
	  {
	    num_inter_at_m = temp_intersect_C2[m + dim_2 * 0];
	    for( I=1; I <= num_inter_at_m; I=I+2)
	    {
	      for(l=0; l < dim_1; l++)
		{
		  if(grid_1[l] >= temp_intersect_V2[m + dim_2 * I] &&
		     grid_1[l] < temp_intersect_V2[m + dim_2 * (I+1)])
		    {
		      in_or_out[l + dim_1 * (m)] = INSIDE;
		    }
		}
	    }
	  } */


 /*	for(l=0; l < dim_1; l++) 
	  for(m=0; m < dim_2; m++)
		{
		  printf("l,m, in_or_out,grid_2[m]  %d  %d %d %f\n",
			 l,m,in_or_out[l + dim_1 * (m)],grid_2[m]);    
		} */         

	  for(l=Grid_1_start; l < Grid_1_fin; l++) 
	  for(m=Grid_2_start; m < Grid_2_fin; m++) 
	    {
	     in_out_ct=0;
	     if(in_or_out[l + dim_1 * (m)] == INSIDE)
		{
		  l_of_node[in_out_ct] = l;
		  m_of_node[in_out_ct] = m;
		  in_out_ct = in_out_ct + 1;
		}
	     if(in_or_out[l + dim_1 * (m+1)] == INSIDE)
		{
		  l_of_node[in_out_ct] = l;
		  m_of_node[in_out_ct] = m+1;
		  in_out_ct = in_out_ct + 1;
		}
	     if(in_or_out[l+1 + dim_1 * (m+1)] == INSIDE) 
		{
		  l_of_node[in_out_ct] = l+1;
		  m_of_node[in_out_ct] = m+1;
		  in_out_ct = in_out_ct + 1;
		}
	     if(in_or_out[l+1 + dim_1 * (m)] == INSIDE)
		{
		  l_of_node[in_out_ct] = l+1;
		  m_of_node[in_out_ct] = m;
		  in_out_ct = in_out_ct + 1;
		}

/* -------------------------------------------------------------------------------- */

   if(in_out_ct == 4)
     {
     
      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l];
	  verts[0][2]=grid_2[m];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l];
	   verts[1][2]=grid_2[m+1];
	    verts[2][0]=grid_3;
	    verts[2][1]=grid_1[l+1];
	    verts[2][2]=grid_2[m+1];
	     verts[3][0]=grid_3;
	     verts[3][1]=grid_1[l+1];
	     verts[3][2]=grid_2[m];
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l];
	   verts[1][0]=grid_2[m+1];
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l];
	    verts[2][0]=grid_2[m+1];
	    verts[2][1]=grid_3;
	    verts[2][2]=grid_1[l+1];
	     verts[3][0]=grid_2[m];
	     verts[3][1]=grid_3;
	     verts[3][2]=grid_1[l+1];
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l];
	  verts[0][1]=grid_2[m];
	  verts[0][2]=grid_3;
	   verts[1][0]=grid_1[l];
	   verts[1][1]=grid_2[m+1];
	   verts[1][2]=grid_3;
	    verts[2][0]=grid_1[l+1];
	    verts[2][1]=grid_2[m+1];
	    verts[2][2]=grid_3;
	     verts[3][0]=grid_1[l+1];
	     verts[3][1]=grid_2[m];
	     verts[3][2]=grid_3;
	}


	  for (i = 0; i <= 3; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_COPY_DATA); */
     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_NOT_SHARED,0);
    }          /* end of if all points(4) are IN */

/* -------------------------------------------------------------------------------- */

	     if(in_out_ct == 0)
	       {
	       }
/* -------------------------------------------------------------------------------- */
	     if(in_out_ct == 1)
	       {
/* 		 printf("in_out_ct == 1  at l,m, %d %d\n",l,m); */
		 m_1 =MIN(m,m_of_node[0]);
		 m_2 =m_1 + 1;
		 v_0 = get_intersection(
			&grid_2[m_1],&grid_2[m_2],l_of_node[0],dim_1,
			&temp_intersect_V1[0],&temp_intersect_C1[l_of_node[0]]);
		 l_1 =MIN(l,l_of_node[0]);
		 l_2 =l_1 + 1;
		 u_1 = get_intersection(
			&grid_1[l_1],&grid_1[l_2],m_of_node[0],dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_of_node[0]]);

      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l_of_node[0]];
	  verts[0][2]=grid_2[m_of_node[0]];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l_of_node[0]];
	   verts[1][2]=v_0;
	    verts[2][0]=grid_3;
	    verts[2][1]=u_1;
	    verts[2][2]=grid_2[m_of_node[0]];
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m_of_node[0]];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l_of_node[0]];
	   verts[1][0]=v_0;
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l_of_node[0]];
	    verts[2][0]=grid_2[m_of_node[0]];
	    verts[2][1]=grid_3;
	    verts[2][2]=u_1;
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l_of_node[0]];
	  verts[0][1]=grid_2[m_of_node[0]];
	  verts[0][2]=grid_3;
	  verts[1][0]=grid_1[l_of_node[0]];
	  verts[1][1]=v_0;
	  verts[1][2]=grid_3;
	  verts[2][0]=u_1;
	  verts[2][1]=grid_2[m_of_node[0]];
	  verts[2][2]=grid_3;
	}


	  for (i = 0; i <= 2; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,3,GEOM_COPY_DATA); */

     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,3,GEOM_SHARED,0);

	       }          /* end of if 1 */
/* -------------------------------------------------------------------------------- */

   if(in_out_ct == 2 && m_of_node[0] == m_of_node[1])
     {
/*        printf("in_out_ct == 2, m=m at l,m, %d %d\n",l,m);    */
       m_1 =MIN(m,m_of_node[0]);
       m_2 =m_1 + 1;
       v_2 = get_intersection(
		     &grid_2[m_1],&grid_2[m_2],(l+1),dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l+1]);
       v_3 = get_intersection(
		     &grid_2[m_1],&grid_2[m_2],l,dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l]);
      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l];
	  verts[0][2]=grid_2[m_of_node[0]];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l+1];
	   verts[1][2]=grid_2[m_of_node[0]];
	    verts[2][0]=grid_3;
	    verts[2][1]=grid_1[l+1];
	    verts[2][2]=v_2;
	     verts[3][0]=grid_3;
	     verts[3][1]=grid_1[l];
	     verts[3][2]=v_3;
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m_of_node[0]];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l];
	   verts[1][0]=grid_2[m_of_node[0]];
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l+1];
	    verts[2][0]=v_2;
	    verts[2][1]=grid_3;
	    verts[2][2]=grid_1[l+1];
	     verts[3][0]=v_3;
	     verts[3][1]=grid_3;
	     verts[3][2]=grid_1[l];
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l];
	  verts[0][1]=grid_2[m_of_node[0]];
	  verts[0][2]=grid_3;
	   verts[1][0]=grid_1[l+1];
	   verts[1][1]=grid_2[m_of_node[0]];
	   verts[1][2]=grid_3;
	    verts[2][0]=grid_1[l+1];
	    verts[2][1]=v_2;
	    verts[2][2]=grid_3;
	     verts[3][0]=grid_1[l];
	     verts[3][1]=v_3;
	     verts[3][2]=grid_3;
	}


	  for (i = 0; i <= 3; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_COPY_DATA); */
     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_NOT_SHARED,0);
	   }

/* -------------------------------------------------------------------------------- */

      if(in_out_ct == 2 &&  l_of_node[0] == l_of_node[1]) 
	     {

/*   		 printf("in_out_ct == 2, l=l at l,m, %d %d\n",l,m);    */

		 l_1 =MIN(l,l_of_node[0]);
		 l_2 =l_1 + 1;
		 u_2 = get_intersection(
			&grid_1[l_1],&grid_1[l_2],(m+1),dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m+1]);
		 u_3 = get_intersection(
			&grid_1[l_1],&grid_1[l_2],m,dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m]);

      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l_of_node[0]];
	  verts[0][2]=grid_2[m];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l_of_node[0]];
	   verts[1][2]=grid_2[m+1];
	    verts[2][0]=grid_3;
	    verts[2][1]=u_2;
	    verts[2][2]=grid_2[m+1];
	     verts[3][0]=grid_3;
	     verts[3][1]=u_3;
	     verts[3][2]=grid_2[m];
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l_of_node[0]];
	   verts[1][0]=grid_2[m+1];
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l_of_node[0]];
	    verts[2][0]=grid_2[m+1];
	    verts[2][1]=grid_3;
	    verts[2][2]=u_2;
	     verts[3][0]=grid_2[m];
	     verts[3][1]=grid_3;
	     verts[3][2]=u_3;
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l_of_node[0]];
	  verts[0][1]=grid_2[m];
	  verts[0][2]=grid_3;
	   verts[1][0]=grid_1[l_of_node[0]];
	   verts[1][1]=grid_2[m+1];
	   verts[1][2]=grid_3;
	    verts[2][0]=u_2;
	    verts[2][1]=grid_2[m+1];
	    verts[2][2]=grid_3;
	     verts[3][0]=u_3;
	     verts[3][1]=grid_2[m];
	     verts[3][2]=grid_3;
	}


	  for (i = 0; i <= 3; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_COPY_DATA); */
     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_NOT_SHARED,0);
    }       

/* -------------------------------------------------------------------------------- */

  if((in_out_ct == 2) && 
     (l_of_node[0] != l_of_node[1]) && 
     (m_of_node[0] != m_of_node[1]))
    {
/*        printf("in_out_ct == 2, l!=l && m!=m at l,m, %d %d\n",l,m);   */
		 m_1 =MIN(m,m_of_node[0]);
		 m_2 =m_1 + 1;
		 v_0 = get_intersection(
			&grid_2[m_1],&grid_2[m_2],l_of_node[0],dim_1,
			&temp_intersect_V1[0],&temp_intersect_C1[l_of_node[0]]);
		 l_1 =MIN(l,l_of_node[0]);
		 l_2 =l_1 + 1;
		 u_1 = get_intersection(
			&grid_1[l_1],&grid_1[l_2],m_of_node[0],dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_of_node[0]]);

      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l_of_node[0]];
	  verts[0][2]=grid_2[m_of_node[0]];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l_of_node[0]];
	   verts[1][2]=v_0;
	    verts[5][0]=grid_3;
	    verts[5][1]=u_1;
	    verts[5][2]=grid_2[m_of_node[0]];
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m_of_node[0]];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l_of_node[0]];
	   verts[1][0]=v_0;
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l_of_node[0]];
	    verts[5][0]=grid_2[m_of_node[0]];
	    verts[5][1]=grid_3;
	    verts[5][2]=u_1;
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l_of_node[0]];
	  verts[0][1]=grid_2[m_of_node[0]];
	  verts[0][2]=grid_3;
	  verts[1][0]=grid_1[l_of_node[0]];
	  verts[1][1]=v_0;
	  verts[1][2]=grid_3;
	  verts[5][0]=u_1;
	  verts[5][1]=grid_2[m_of_node[0]];
	  verts[5][2]=grid_3;
	}

		 m_1 =MIN(m,m_of_node[1]);
		 m_2 =m_1 + 1;
		 v_0 = get_intersection(
			&grid_2[m_1],&grid_2[m_2],l_of_node[1],dim_1,
			&temp_intersect_V1[0],&temp_intersect_C1[l_of_node[1]]);
		 l_1 =MIN(l,l_of_node[1]);
		 l_2 =l_1 + 1;
		 u_1 = get_intersection(
			&grid_1[l_1],&grid_1[l_2],m_of_node[1],dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_of_node[1]]);

      if(axis == X_AXIS)
	{
	  verts[3][0]=grid_3;
	  verts[3][1]=grid_1[l_of_node[1]];
	  verts[3][2]=grid_2[m_of_node[1]];
	   verts[4][0]=grid_3;
	   verts[4][1]=grid_1[l_of_node[1]];
	   verts[4][2]=v_0;
	    verts[2][0]=grid_3;
	    verts[2][1]=u_1;
	    verts[2][2]=grid_2[m_of_node[1]];
	}
      if(axis == Y_AXIS)
	{
	  verts[3][0]=grid_2[m_of_node[1]];
	  verts[3][1]=grid_3;
	  verts[3][2]=grid_1[l_of_node[1]];
	   verts[4][0]=v_0;
	   verts[4][1]=grid_3;
	   verts[4][2]=grid_1[l_of_node[1]];
	    verts[2][0]=grid_2[m_of_node[1]];
	    verts[2][1]=grid_3;
	    verts[2][2]=u_1;
	}
      if(axis == Z_AXIS)
	{
	  verts[3][0]=grid_1[l_of_node[1]];
	  verts[3][1]=grid_2[m_of_node[1]];
	  verts[3][2]=grid_3;
	  verts[4][0]=grid_1[l_of_node[1]];
	  verts[4][1]=v_0;
	  verts[4][2]=grid_3;
	  verts[2][0]=u_1;
	  verts[2][1]=grid_2[m_of_node[1]];
	  verts[2][2]=grid_3;
	}

		 for (i = 0; i <= 5; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,6,GEOM_COPY_DATA); */
     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,6,GEOM_NOT_SHARED,0);
 
    }

/* -------------------------------------------------------------------------------- */

     if(in_out_ct == 3)
       {

/*      printf("in_out_ct == 3 at l,m, %d %d\n",l,m);  */

	 if(l_of_node[0] == l_of_node[1])
	   {
	     if(m_of_node[2] == m)
	       {
		 l_0 = l+1;
		 m_0 = m;
		  l_1 = l;
		  m_1 = m;
		   l_2 = l;
		   m_2 = m+1;
		    l_3 = l+1;
		    m_3 = m+1;
		 u_3 = get_intersection(
			&grid_1[l_2],&grid_1[l_3],m_3,dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_3]);
		 v_3 = grid_2[m_3];
		 u_4 = grid_1[l_3];
                 v_4 = get_intersection(
		     &grid_2[m_0],&grid_2[m_3],l_3,dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l_3]);
	       }
	     else
	       {
		 l_3 = l+1;
		 m_3 = m;
		 l_0 = l;
		 m_0 = m;
		 l_1 = l;
		 m_1 = m+1;
		 l_2 = l+1;
		 m_2 = m+1;
		 u_3 = grid_1[l_3];
                 v_3 = get_intersection(
		     &grid_2[m_3],&grid_2[m_2],l_3,dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l_3]);
		 u_4 = get_intersection(
			&grid_1[l_0],&grid_1[l_3],m_3,dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_3]);
		 v_4 = grid_2[m_3];
	       }
	   }
	 else
	   {
	     if(m_of_node[0] == m+1)
	       {
		 l_2 = l+1;
		 m_2 = m;
		 l_3 = l;
		 m_3 = m;
		 l_0 = l;
		 m_0 = m+1;
		 l_1 = l+1;
		 m_1 = m+1;
		 u_3 = get_intersection(
			&grid_1[l_3],&grid_1[l_2],m_3,dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_3]);
		 v_3 = grid_2[m_3];
		 u_4 = grid_1[l_3];
                 v_4 = get_intersection(
		     &grid_2[m_3],&grid_2[m_0],l_3,dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l_3]);
	       }
	     else
	       {
		 l_1 = l+1;
		 m_1 = m;
		 l_2 = l;
		 m_2 = m;
		 l_3 = l;
		 m_3 = m+1;
		 l_0 = l+1;
		 m_0 = m+1;
		 u_3 = grid_1[l_3];
                 v_3 = get_intersection(
		     &grid_2[m_2],&grid_2[m_3],l_3,dim_1,
		     &temp_intersect_V1[0],&temp_intersect_C1[l_3]);
		 u_4 = get_intersection(
			&grid_1[l_3],&grid_1[l_0],m_3,dim_2,
			&temp_intersect_V2[0],&temp_intersect_C2[m_3]);
		 v_4 = grid_2[m_3];
	       }

	   }
      if(axis == X_AXIS)
	{
	  verts[0][0]=grid_3;
	  verts[0][1]=grid_1[l_0];
	  verts[0][2]=grid_2[m_0];
	   verts[1][0]=grid_3;
	   verts[1][1]=grid_1[l_1];
	   verts[1][2]=grid_2[m_1];
	    verts[2][0]=grid_3;
	    verts[2][1]=grid_1[l_2];
	    verts[2][2]=grid_2[m_2];
	     verts[3][0]=grid_3;
	     verts[3][1]=u_3;
	     verts[3][2]=v_3;
	      verts[4][0]=grid_3;
	      verts[4][1]=u_4;
	      verts[4][2]=v_4;
	}
      if(axis == Y_AXIS)
	{
	  verts[0][0]=grid_2[m_0];
	  verts[0][1]=grid_3;
	  verts[0][2]=grid_1[l_0];
	   verts[1][0]=grid_2[m_1];
	   verts[1][1]=grid_3;
	   verts[1][2]=grid_1[l_1];
	    verts[2][0]=grid_2[m_2];
	    verts[2][1]=grid_3;
	    verts[2][2]=grid_1[l_2];
	     verts[3][0]=v_3;
	     verts[3][1]=grid_3;
	     verts[3][2]=u_3;
	      verts[4][0]=v_4;
	      verts[4][1]=grid_3;
	      verts[4][2]=u_4;
	}
      if(axis == Z_AXIS)
	{
	  verts[0][0]=grid_1[l_0];
	  verts[0][1]=grid_2[m_0];
	  verts[0][2]=grid_3;
	   verts[1][0]=grid_1[l_1];
	   verts[1][1]=grid_2[m_1];
	   verts[1][2]=grid_3;
	    verts[2][0]=grid_1[l_2];
	    verts[2][1]=grid_2[m_2];
	    verts[2][2]=grid_3;
	     verts[3][0]=u_3;
	     verts[3][1]=v_3;
	     verts[3][2]=grid_3;
	      verts[4][0]=u_4;
	      verts[4][1]=v_4;
	      verts[4][2]=grid_3;
	}

	  for (i = 0; i <= 4; i++)
	    {
	      if(axis == X_AXIS)
		{
		  norm_verts[i][0]=normal_dir;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Y_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=normal_dir;
		  norm_verts[i][2]=0.0;
	        }
	      if(axis == Z_AXIS)
		{
		  norm_verts[i][0]=0.0;
		  norm_verts[i][1]=0.0;
		  norm_verts[i][2]=normal_dir;
	        }
	    }
 
/*        GEOMadd_polytriangle
         (Tmp_obj,verts,norm_verts,GEOM_NULL,4,GEOM_COPY_DATA); */
     GEOMadd_disjoint_polygon
       (Tmp_obj,verts,norm_verts,GEOM_NULL,5,GEOM_NOT_SHARED,0);

	       }          /* end of if 3 are IN */

/* -------------------------------------------------------------------------------- */

          }   /* end of m and l loop */

       GEOMgen_normals(Tmp_obj,0);

       GEOMset_extent(Tmp_obj);
  
	*Slice_out = GEOMinit_edit_list(*Slice_out);
	GEOMedit_geometry(*Slice_out, "Slice_out", Tmp_obj);
	GEOMdestroy_obj(Tmp_obj);


/* free any locally allocated data */
	free(x_grid);
	free(y_grid);
	free(z_grid);
	free(data_plane_temp);
	free(grid_1);
	free(grid_2);
	free(num_sections);
	free(temp_intersect_C1);
	free(temp_intersect_V1);
	free(temp_intersect_C2);
	free(temp_intersect_V2);
	free(in_or_out);

/* THIS IS THE END OF THE 'HINTS' AREA. ADD YOUR OWN CODE BETWEEN THE    */
/* FOLLOWING COMMENTS. THE CODE YOU SUPPLY WILL NOT BE OVERWRITTEN.      */
 
/* ----> START OF USER-SUPPLIED CODE SECTION #3 (COMPUTE ROUTINE BODY)   */
/* <---- END OF USER-SUPPLIED CODE SECTION #3                            */
	return(1);
}
 
 
/* ----> START OF USER-SUPPLIED CODE SECTION #4 (SUBROUTINES, FUNCTIONS, UTILITY ROUTINES)*/
float get_intersection(g_1,g_2,n,dim,intersect_data,max_inter)
float *intersect_data;
float *g_1, *g_2;
int *max_inter;
int n, dim;
{
#define  ABSF(A)  ((A) > (0.0) ? (A) : (-A))
  float temp_intersect;
  int I;
  float test;
  float intersect_limit;

  intersect_limit = 1.0e-3 *(*g_2 - *g_1);
  for(I=1; I<=*max_inter; I++)
    {
      temp_intersect =*(intersect_data + n + dim*(I));
      if(*g_1 <= temp_intersect && *g_2 >= temp_intersect)
	{
	  return(temp_intersect);
	}
      if(ABSF(*g_1 - temp_intersect) < intersect_limit) return(*g_1);
      if(ABSF(*g_2 - temp_intersect) < intersect_limit) return(*g_2);
    }
  temp_intersect = 0.5*(*g_1+*g_2); 
  printf(" No intersection found in get_intersection\n");
  printf("   g_1,g_2 n, dim, *max_inter; %f %f %d %d %d\n",
	 *g_1, *g_2, n, dim, *max_inter); 
  for(I=1; I<=*max_inter; I++)
    {
      temp_intersect =*(intersect_data + n + dim*(I));
      printf("    get_intersection, I, temp_intersect %d %f\n",I, temp_intersect); 
    }
  return(temp_intersect); 
/*  return(0.0); */
}
float get_intersection_diag(g_1,g_2,n,dim,intersect_data,max_inter)
float *intersect_data;
float *g_1, *g_2;
int *max_inter;
int n, dim;
{
  float temp_intersect;
  int I;
 	  printf("g_1,g_2, n, dim, *max_inter; %f %f %d %d %d\n", *g_1, *g_2, n, dim, *max_inter);
  for(I=1; I<=*max_inter; I++)
    {
      temp_intersect =*(intersect_data + n + dim*(I));
 	  printf("I,temp_intersect %d %f\n",I,temp_intersect);
      if(*g_1 <= temp_intersect && *g_2 > temp_intersect)
	{
 	  printf("get_intersection, I, temp_intersect %d %f\n",I, temp_intersect); 
	  return(temp_intersect);
	}
    }
	  printf(" No intersection found in get_intersection\n");
  return(0.0);
}

/* <---- END OF USER-SUPPLIED CODE SECTION #4                            */
