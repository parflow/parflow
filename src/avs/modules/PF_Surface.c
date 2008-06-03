/* mod_gen Version 1                                                     */
/* Module Name: "PF_Surface" (Mapper) (Subroutine)                        */
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
int PF_Surface_desc()
{

	int in_port, out_port, param, iresult;
	extern int PF_Surface_compute();

	AVSset_module_name("PF_Surface", MODULE_MAPPER);

	/* Input Port Specifications               */

	in_port = AVScreate_input_port("X_inter_by_yz", 
		"field 3D 3-space 2-vector rectilinear float", OPTIONAL);
	in_port = AVScreate_input_port("Y_inter_by_zx", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);
       	in_port = AVScreate_input_port("Z_inter_by_xy", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);


	/* Output Port Specifications              */
	out_port = AVScreate_output_port("Surface_out", "geom");

	/* Parameter Specifications                */
	param = AVSadd_parameter("Grid_X_start", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_X_fin", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");

	param = AVSadd_parameter("Grid_Y_start", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Grid_Y_fin", "integer", 0, 0, 1);
          AVSconnect_widget(param, "idial");

	param = AVSadd_parameter("Solid_num", "integer", 0, 0, 10);
          AVSconnect_widget(param, "idial");

	param = AVSadd_parameter("Show_Top", "choice", "Show_Top", 
		"Show_Top:Dont_Show_Top", ":");
	  AVSconnect_widget(param, "radio_buttons");
	param = AVSadd_parameter("Show_Bottom", "choice", "Show_Bottom", 
		"Show_Bottom:Dont_Show_Bottom", ":");
	  AVSconnect_widget(param, "radio_buttons");
	param = AVSadd_parameter("Show_Sides", "choice", "Show_Sides", 
		"Show_Sides:Dont_Show_Sides", ":");
	  AVSconnect_widget(param, "radio_buttons");
	
	AVSset_compute_proc(PF_Surface_compute);
/* ----> START OF USER-SUPPLIED CODE SECTION #2 (ADDITIONAL SPECIFICATION INFO)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #2                            */
	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int PF_Surface_compute(X_inter_by_yz, Y_inter_by_zx, Z_inter_by_xy, Surface_out,
 Grid_X_start, Grid_X_fin,
 Grid_Y_start, Grid_Y_fin, 
 Solid_num,Show_Top,Show_Bottom,Show_Sides)
     AVSfield_float *X_inter_by_yz;
     AVSfield_float *Y_inter_by_zx;
     AVSfield_float *Z_inter_by_xy;
     GEOMedit_list *Surface_out;
     int Grid_X_start, Grid_X_fin;
     int Grid_Y_start, Grid_Y_fin;
     int Solid_num;
     char *Show_Top; 
     char *Show_Bottom;
     char *Show_Sides;
{
#define  ABS(A)  ((A) > (0) ? (A) : (-A))
#define  ABSF(A)  ((A) > (0.0) ? (A) : (-A))
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  MIN(A, B)  ((A) < (B) ? (A) : (B))
#define  MAX_SECTIONS 6
	GEOMobj *Bottom_obj;
	GEOMobj *Top_obj;
	GEOMobj *Side_obj;
        int i, j, k;
        int l, m, n, lp;
        int l_start, l_fin;
        int m_start, m_fin;
        int   dim_X;
        int   dim_Y;
        int   dim_Z;
	int *temp_intersect_C1;
	float *temp_intersect_V1;
	int hold_I;
	int hold_intersect_C1;
	float hold_intersect_V1;
	float Max_intersect_V1;
	float *Grid_X;
	float *Grid_Y;
	float del_Grid_X;
	float del_Grid_Y;
	float *Grid_Z;
	float verts[6][3];
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
	float normal_dir;
        float line_verts[2][3];
        int N_solids;
        float Z_ave;

	normal_dir = -1.0;
 
	dim_X = (Z_inter_by_xy)->dimensions[0];
	dim_Y = (Z_inter_by_xy)->dimensions[1];
	dim_Z = (Y_inter_by_zx)->dimensions[2];
	del_Grid_X = RECT_X(Z_inter_by_xy)[2] - RECT_X(Z_inter_by_xy)[1];
	del_Grid_Y = RECT_Y(Z_inter_by_xy)[2] - RECT_Y(Z_inter_by_xy)[1];
	
	if(Grid_X_start < 0)Grid_X_start =0;
	if(Grid_X_fin > dim_X-1)Grid_X_fin =dim_X-1;
	if(Grid_Y_start < 0)Grid_Y_start =0;
	if(Grid_Y_fin > dim_Y-1)Grid_Y_fin =dim_Y-1;

/****** MAX are turned off     ************/
	Grid_X_fin =dim_X-1;
	Grid_Y_fin =dim_Y-1;

	AVSmodify_parameter("Grid_X_start",AVS_MAXVAL,0,0,dim_X-1);
	AVSmodify_parameter("Grid_X_fin",AVS_MAXVAL,0,0,dim_X-1);
	AVSmodify_parameter("Grid_Y_start",AVS_MAXVAL,0,0,dim_Y-1);
	AVSmodify_parameter("Grid_Y_fin",AVS_MAXVAL,0,0,dim_Y-1);

/* printf(" dim_X %d \n",dim_X); */
/* printf(" dim_Y %d \n",dim_Y); */


	Grid_X = (float *)malloc(dim_X * sizeof(float));
	Grid_Y = (float *)malloc(dim_Y * sizeof(float));
	Grid_Z = (float *)malloc(dim_Z * sizeof(float));


	max_intersects = (Z_inter_by_xy)->dimensions[2];

/* 	printf("max_intersects %d\n",max_intersects);  */

	for(l=0; l < dim_X; l++)
	  {
	      Grid_X[l] = (Z_inter_by_xy)->points[l];
	  }

	for(m=0; m < dim_Y; m++)
	  {
	      Grid_Y[m] = (Z_inter_by_xy)->points[dim_X + m];
	  }

	for(n=0; n < dim_Z; n++)
	  {
	      Grid_Z[n] = (Y_inter_by_zx)->points[dim_X + max_intersects + n];
	  }



/*  THIS IS A 'HINTS' AREA - YOU MAY CUT AND PASTE AT WILL FROM IT       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/* Create the Surface_out object                                           */
/*	Bottom_obj = GEOMcreate_obj(GEOM_POLYTRI, NULL);   */
  Bottom_obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);
  Top_obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);
  Side_obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);

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

/* make sure Solid_num is valid, in range     */
	if(Solid_num < 0)Solid_num = 0;
	if(Solid_num > (N_solids-1))Solid_num = N_solids-1;

	Picked_Solid_Num =Solid_num + 1;
 

	temp_intersect_C1 = 
	  (int *)malloc(dim_X*dim_Y*max_intersects *  sizeof(int));
	temp_intersect_V1 = 
	  (float *)malloc(dim_X*dim_Y*max_intersects *  sizeof(float));


	      for(l=0; l < dim_X; l++)
	      for(m=0; m < dim_Y; m++)
	      for(I=0; I < max_intersects; I++)
		{
		  temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*I] = 0;
		  temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I] = 0.0;
		}

	
	      for(l=0; l < dim_X; l++)
	      for(m=0; m < dim_Y; m++)
		{
		  Ip=0;
		  for(I=1; I < max_intersects; I++)
		    {
		      temp_solid_num =(int) I3DV(Z_inter_by_xy,l,m,I)[0];
		      if(ABS(temp_solid_num) == Picked_Solid_Num)
			{
			  Ip = Ip +1;
			  temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*Ip] = 
			    (int) I3DV(Z_inter_by_xy,l,m,I)[0]; 
			  temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*Ip] = 
			    I3DV(Z_inter_by_xy,l,m,I)[1];
			}
		    }
		  temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*0] = Ip; 
	        }


/* 	for(l=0; l < dim_X; l++)
        for(m=0; m < dim_Y; m++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*0];
	    for(I=1; I <= num_inter_at_l; I++)
	      {
	    printf(" l,I,temp_intersects %d %d %d %f\n",l,I,
		   temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*I],
		   temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I]);
	      }
	  }             */     


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


/*   	for(l=0; l < dim_X; l++)
	for(m=0; m < dim_Y; m++)
	  {
	    num_inter_at_l = temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*0];
	    for(I=1; I <= num_inter_at_l; I++)
	      {
	    printf("post sort, l,I,temp_intersects %d %d %d %f\n",l,I,
		   temp_intersect_C1[l + dim_X*m + dim_X*dim_Y*I],
		   temp_intersect_V1[l + dim_X*m + dim_X*dim_Y*I]);
	      }
	  }          */



       for(l=Grid_X_start; l < Grid_X_fin; l++) 
       for(m=Grid_Y_start; m < Grid_Y_fin; m++) 
	 {
	   I = 1;
	   if(Picked_Solid_Num == ABS(temp_intersect_C1[l   + dim_X*(m  ) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l+1 + dim_X*(m  ) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l   + dim_X*(m+1) + dim_X*dim_Y*I]))
	     {
   if(strncmp(Show_Bottom,"Show_Bottom",11) == 0)
     {
       verts[0][2]=temp_intersect_V1[l   + dim_X*(m  ) + dim_X*dim_Y*I];
       verts[1][2]=temp_intersect_V1[l   + dim_X*(m+1) + dim_X*dim_Y*I];
       verts[2][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I];
       verts[3][2]=temp_intersect_V1[l+1 + dim_X*(m  ) + dim_X*dim_Y*I];
/* only show surface if average of z values is inside grid   */
       Z_ave = (verts[0][2] + verts[1][2] + verts[2][2] + verts[3][2])/4.0;
       if(Z_ave < Grid_Z[dim_Z-1] && Z_ave > Grid_Z[0])
	 {
	   verts[0][0]=Grid_X[l];
	   verts[0][1]=Grid_Y[m];
	   verts[1][0]=Grid_X[l];
	   verts[1][1]=Grid_Y[m+1];
	   verts[2][0]=Grid_X[l+1];
	   verts[2][1]=Grid_Y[m+1];
	   verts[3][0]=Grid_X[l+1];
	   verts[3][1]=Grid_Y[m];

	   GEOMadd_disjoint_polygon
	     (Bottom_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
	 }
     }
	       /* check for edge */
	       if(strncmp(Show_Sides,"Show_Sides",10) == 0)
		 {
 	       if(l > Grid_X_start && l <  Grid_X_fin)
		 {

		   /* check cell in minus l direction for edge */
		   if(Picked_Solid_Num != ABS(temp_intersect_C1[l-1 + dim_X*(m  ) + dim_X*dim_Y*I]) ||
		      Picked_Solid_Num != ABS(temp_intersect_C1[l-1 + dim_X*(m+1) + dim_X*dim_Y*I]))
		      {
			verts[0][0]=Grid_X[l];
			verts[0][1]=Grid_Y[m];
			verts[0][2]=temp_intersect_V1[l + dim_X*(m  ) + dim_X*dim_Y*I];
			verts[1][0]=Grid_X[l];
			verts[1][1]=Grid_Y[m+1];
			verts[1][2]=temp_intersect_V1[l + dim_X*(m+1) + dim_X*dim_Y*I];
			verts[2][0]=Grid_X[l];
			verts[2][1]=Grid_Y[m+1];
			verts[2][2]=temp_intersect_V1[l + dim_X*(m+1) + dim_X*dim_Y*(I+1)];
			verts[3][0]=Grid_X[l];
			verts[3][1]=Grid_Y[m];
			verts[3][2]=temp_intersect_V1[l + dim_X*(m  ) + dim_X*dim_Y*(I+1)];

			GEOMadd_disjoint_polygon
			  (Side_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
		      }

		   /* check cell in plus l direction for edge */
		   if(Picked_Solid_Num != ABS(temp_intersect_C1[l+2 + dim_X*(m  ) + dim_X*dim_Y*I]) ||
		      Picked_Solid_Num != ABS(temp_intersect_C1[l+2 + dim_X*(m+1) + dim_X*dim_Y*I]))
		      {
			verts[0][0]=Grid_X[l+1];
			verts[0][1]=Grid_Y[m];
			verts[0][2]=temp_intersect_V1[l+1 + dim_X*(m  ) + dim_X*dim_Y*I];
			verts[1][0]=Grid_X[l+1];
			verts[1][1]=Grid_Y[m+1];
			verts[1][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I];
			verts[2][0]=Grid_X[l+1];
			verts[2][1]=Grid_Y[m+1];
			verts[2][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*(I+1)];
			verts[3][0]=Grid_X[l+1];
			verts[3][1]=Grid_Y[m];
			verts[3][2]=temp_intersect_V1[l+1 + dim_X*(m  ) + dim_X*dim_Y*(I+1)];

			GEOMadd_disjoint_polygon
			  (Side_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
		      }
		 }
	       if(m > Grid_Y_start && m <  Grid_Y_fin)
		 {

		   /* check cell in minus m direction for edge */
		   if(Picked_Solid_Num != ABS(temp_intersect_C1[l   + dim_X*(m-1) + dim_X*dim_Y*I]) ||
		      Picked_Solid_Num != ABS(temp_intersect_C1[l+1 + dim_X*(m-1) + dim_X*dim_Y*I]))
		      {
			verts[0][0]=Grid_X[l];
			verts[0][1]=Grid_Y[m];
			verts[0][2]=temp_intersect_V1[l + dim_X*(m) + dim_X*dim_Y*I];
			verts[1][0]=Grid_X[l+1];
			verts[1][1]=Grid_Y[m];
			verts[1][2]=temp_intersect_V1[l+1 + dim_X*(m) + dim_X*dim_Y*I];
			verts[2][0]=Grid_X[l+1];
			verts[2][1]=Grid_Y[m];
			verts[2][2]=temp_intersect_V1[l+1 + dim_X*(m) + dim_X*dim_Y*(I+1)];
			verts[3][0]=Grid_X[l];
			verts[3][1]=Grid_Y[m];
			verts[3][2]=temp_intersect_V1[l + dim_X*(m) + dim_X*dim_Y*(I+1)];

			GEOMadd_disjoint_polygon
			  (Side_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
		      }

		   /* check cell in plus m direction for edge */
		   if(Picked_Solid_Num != ABS(temp_intersect_C1[l   + dim_X*(m+2) + dim_X*dim_Y*I]) ||
		      Picked_Solid_Num != ABS(temp_intersect_C1[l+1 + dim_X*(m+2) + dim_X*dim_Y*I]))
		      {
			verts[0][0]=Grid_X[l];
			verts[0][1]=Grid_Y[m+1];
			verts[0][2]=temp_intersect_V1[l + dim_X*(m+1) + dim_X*dim_Y*I];
			verts[1][0]=Grid_X[l+1];
			verts[1][1]=Grid_Y[m+1];
			verts[1][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I];
			verts[2][0]=Grid_X[l+1];
			verts[2][1]=Grid_Y[m+1];
			verts[2][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*(I+1)];
			verts[3][0]=Grid_X[l];
			verts[3][1]=Grid_Y[m+1];
			verts[3][2]=temp_intersect_V1[l + dim_X*(m+1) + dim_X*dim_Y*(I+1)];

			GEOMadd_disjoint_polygon
			  (Side_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
		      }


		 }
	     }
	 }   /* end of m and l loop */
     }

   if(strncmp(Show_Top,"Show_Top",8) == 0)
     {
       for(l=Grid_X_start; l < Grid_X_fin; l++) 
       for(m=Grid_Y_start; m < Grid_Y_fin; m++) 
	 {
	   I = 2;
	   if(Picked_Solid_Num == ABS(temp_intersect_C1[l   + dim_X*(m  ) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l+1 + dim_X*(m  ) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I]) &&
	      Picked_Solid_Num == ABS(temp_intersect_C1[l   + dim_X*(m+1) + dim_X*dim_Y*I]))
	     {
	       verts[0][2]=temp_intersect_V1[l   + dim_X*(m  ) + dim_X*dim_Y*I];
	       verts[1][2]=temp_intersect_V1[l   + dim_X*(m+1) + dim_X*dim_Y*I];
	       verts[2][2]=temp_intersect_V1[l+1 + dim_X*(m+1) + dim_X*dim_Y*I];
	       verts[3][2]=temp_intersect_V1[l+1 + dim_X*(m  ) + dim_X*dim_Y*I];
/* only show surface if average of z values is inside grid   */
	       Z_ave = (verts[0][2] + verts[1][2] + verts[2][2] + verts[3][2])/4.0;
	       if(Z_ave < Grid_Z[dim_Z-1] && Z_ave > Grid_Z[0])
		 {
		   verts[0][0]=Grid_X[l];
		   verts[0][1]=Grid_Y[m];
		   verts[1][0]=Grid_X[l];
		   verts[1][1]=Grid_Y[m+1];
		   verts[2][0]=Grid_X[l+1];
		   verts[2][1]=Grid_Y[m+1];
		   verts[3][0]=Grid_X[l+1];
		   verts[3][1]=Grid_Y[m];

		   GEOMadd_disjoint_polygon
		      (Top_obj,verts,GEOM_NULL,GEOM_NULL,4,GEOM_NOT_SHARED,0);
		 }
	     }
	 }   /* end of m and l loop */
     }
	
       GEOMgen_normals(Bottom_obj,0);

       GEOMset_extent(Bottom_obj);
  
	*Surface_out = GEOMinit_edit_list(*Surface_out);

	GEOMedit_geometry(*Surface_out, "Top_Surface", Top_obj);
	GEOMdestroy_obj(Top_obj);

	GEOMedit_geometry(*Surface_out, "Bottom_Surface", Bottom_obj);
	GEOMdestroy_obj(Bottom_obj);

	GEOMedit_geometry(*Surface_out, "Side_Surface", Side_obj);
	GEOMdestroy_obj(Side_obj);


/* free any locally allocated data */
	free(Grid_X);
	free(Grid_Y);
	free(temp_intersect_C1);
	free(temp_intersect_V1);

/* THIS IS THE END OF THE 'HINTS' AREA. ADD YOUR OWN CODE BETWEEN THE    */
/* FOLLOWING COMMENTS. THE CODE YOU SUPPLY WILL NOT BE OVERWRITTEN.      */
 
/* ----> START OF USER-SUPPLIED CODE SECTION #3 (COMPUTE ROUTINE BODY)   */
/* <---- END OF USER-SUPPLIED CODE SECTION #3                            */
	return(1);
}
 
 
/* ----> START OF USER-SUPPLIED CODE SECTION #4 (SUBROUTINES, FUNCTIONS, UTILITY ROUTINES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #4                            */
