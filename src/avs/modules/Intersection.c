/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <avs/avs.h>
#include <avs/port.h>
#include <avs/field.h>
 
/* ----> START OF USER-SUPPLIED CODE SECTION #1 (INCLUDE FILES, GLOBAL VARIABLES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #1                            */
 
/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int Intersection_desc()
{

	int in_port, out_port, param, iresult;
	extern int Intersection_compute();

	AVSset_module_name("Intersection", MODULE_DATA);

	/* Input Port Specifications               */
	in_port = AVScreate_input_port("ParFlow_in", 
		"field 3D 3-space scalar rectilinear float", REQUIRED);

	/* Output Port Specifications              */
	out_port = AVScreate_output_port("X_inter_by_yz", 
		"field 3D 3-space 2-vector rectilinear float");
	out_port = AVScreate_output_port("Y_inter_by_zx", 
		"field 3D 3-space 2-vector rectilinear float");
	out_port = AVScreate_output_port("Z_inter_by_xy", 
		"field 3D 3-space 2-vector rectilinear float");
	out_port = AVScreate_output_port("N_solids_out", "integer");

	/* Parameter Specifications                */
	param = AVSadd_parameter("Solid_file_in", "string", "", "", ":.pfsol");
	AVSconnect_widget(param, "browser");

	AVSset_compute_proc(Intersection_compute);

	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int Intersection_compute( ParFlow_in, X_inter_by_yz, 
   Y_inter_by_zx, Z_inter_by_xy, N_solids_out, Solid_file_in)
	AVSfield_float *ParFlow_in;
	AVSfield_float **X_inter_by_yz;
	AVSfield_float **Y_inter_by_zx;
	AVSfield_float **Z_inter_by_xy;
        int *N_solids_out;
        char *Solid_file_in;
{

/*  THIS IS A 'HINTS' AREA - YOU MAY CUT AND PASTE AT WILL FROM IT       */
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  MIN(A, B)  ((A) < (B) ? (A) : (B))
#define  MAX_SOLIDS 10
#define  MAX_INTERSECTION 40
	int dims0[3];
	int dims1[3];
	int dims2[3];
	int dim_X,dim_Y,dim_Z;
        float *X_grid, *Y_grid, *Z_grid;
        float del_X_grid, del_Y_grid, del_Z_grid;
        float line_verts[2][3];
        float verts[3][3];
        float normals[3][3];
        float color_verts[3][3]; 
        int Max_z_interset_xy;
        int Max_x_interset_yz;
        int Max_y_interset_zx;
        FILE *fp;
        int File_Version_Num;
        int N_vertex;
        float *Verts_in;
        float Tmp_Verts[3];
	int N_solids; 
        int Num_solid_tri[MAX_SOLIDS];
        int *Vert_cont_list;
        int Max_vertex;
        int Tmp_cont[3];
        int i,j,k,m,n,read_pass,xyz;
        int Num_paches, Num_paches_tri, Paches_tri;
        int Tri_Num_Tmp[3];
        float *Temp_grid_pt_Zxys;
        float *Temp_grid_pt_Xyzs;
	float *Temp_grid_pt_Yzxs;
	float Max_pt_X, Min_pt_X;
	float Max_pt_Y, Min_pt_Y;
	float Max_pt_Z, Min_pt_Z;

	Max_pt_X = -1.0e8;
	Min_pt_X = 1.0e8;
	Max_pt_Y = -1.0e8;
	Min_pt_Y = 1.0e8;
	Max_pt_Z = -1.0e8;
	Min_pt_Z = 1.0e8;
     for (read_pass = 0; read_pass <2 ; read_pass++)
       {
        fp = fopen(Solid_file_in, "r");

        fscanf(fp,"%d\n",&File_Version_Num);
        fscanf(fp,"%d\n",&N_vertex);

         if(read_pass == 0)
           Verts_in = (float *) malloc(3 * N_vertex * sizeof(float));
        for (m = 0; m <N_vertex ; m++)
           {
            fscanf(fp,"%f %f %f\n",&Tmp_Verts[0],&Tmp_Verts[1],&Tmp_Verts[2]);
           if(read_pass == 0)
             {
              Verts_in[3*m] = Tmp_Verts[0];
              Verts_in[3*m+1] = Tmp_Verts[1];
              Verts_in[3*m+2] =  Tmp_Verts[2];
	      Max_pt_X = MAX(Max_pt_X,Tmp_Verts[0]); 
	      Min_pt_X = MIN(Min_pt_X,Tmp_Verts[0]); 
	      Max_pt_Y = MAX(Max_pt_Y,Tmp_Verts[1]); 
	      Min_pt_Y = MIN(Min_pt_Y,Tmp_Verts[1]); 
	      Max_pt_Z = MAX(Max_pt_Z,Tmp_Verts[2]); 
	      Min_pt_Z = MIN(Min_pt_Z,Tmp_Verts[2]); 
             }
           }
	fscanf(fp,"%d\n",&N_solids);
	*N_solids_out = N_solids;
        Max_vertex = 0;
        for (m = 0; m <N_solids ; m++)
          {
             if(read_pass == 0)printf("m N_solid %d %d\n",m,N_solids);
             fscanf(fp,"%d\n",&Num_solid_tri[m]);
             Max_vertex = MAX(Max_vertex,Num_solid_tri[m]);      
             if(read_pass == 0)printf("  m,Num_solid_tri[m] %d %d\n",
				    m,Num_solid_tri[m]);
             for (n = 0; n <Num_solid_tri[m] ; n++)
               {
                 fscanf(fp,"%d %d %d\n",&Tmp_cont[0],&Tmp_cont[1],&Tmp_cont[2]);
                if(read_pass == 1)
                  {
                    Vert_cont_list[3*(m + N_solids * n) + 0] = Tmp_cont[0];
                    Vert_cont_list[3*(m + N_solids * n) + 1] = Tmp_cont[1];
                    Vert_cont_list[3*(m + N_solids * n) + 2] = Tmp_cont[2];
	          }

                }
             fscanf(fp,"%d\n",&Num_paches);
             for (n = 0; n <Num_paches ; n++)
               {
                 fscanf(fp,"%d\n",&Num_paches_tri);
                 for (k = 0; k <Num_paches_tri ; k++)
                   {
                     fscanf(fp,"%d\n",&Paches_tri);
		   }
               }      
	   }
          if(read_pass == 0)printf("Max_vertex %d\n", Max_vertex);
          if(read_pass == 0)
            Vert_cont_list = (int *) malloc(3 * Max_vertex * N_solids *sizeof(int));  
       fclose(fp);
      }


	printf("Min and Max X vertex %f %f \n", Min_pt_X, Max_pt_X);
	printf("Min and Max Y vertex %f %f \n", Min_pt_Y, Max_pt_Y);
	printf("Min and Max Z vertex %f %f \n", Min_pt_Z, Max_pt_Z);

/* the following must all have the same value   */

        Max_z_interset_xy = MAX_INTERSECTION;
        Max_x_interset_yz = MAX_INTERSECTION;
        Max_y_interset_zx = MAX_INTERSECTION;

       dim_X = (ParFlow_in)->dimensions[0];
       dim_Y = (ParFlow_in)->dimensions[1];
       dim_Z = (ParFlow_in)->dimensions[2];                                             

       X_grid = (float *) malloc(dim_X * sizeof(float));                          
       Y_grid = (float *) malloc(dim_Y * sizeof(float));                          
       Z_grid = (float *) malloc(dim_Z * sizeof(float));
     
       for (i = 0; i < dim_X; i++)
         {
           X_grid[i] = RECT_X(ParFlow_in)[i];
         }                         
	del_X_grid = X_grid[1] - X_grid[0];

       for (j = 0; j < dim_Y; j++)
         {
           Y_grid[j] = RECT_Y(ParFlow_in)[j];
         }                         
	del_Y_grid = Y_grid[1] - Y_grid[0];

       for (k = 0; k < dim_Z; k++)
         {
           Z_grid[k] = RECT_Z(ParFlow_in)[k];
         }                         
	del_Z_grid = Z_grid[1] - Z_grid[0];

/*                                                                       */
/*  FILL IN THE OUTPUT FIELD  Here!                                      */
/*                                                                       */
/* Free old field data                                                   */
	if (*X_inter_by_yz) AVSfield_free(*X_inter_by_yz);
/* Allocate space for new field output                                   */
	dims1[0] = Max_x_interset_yz;
	dims1[1] = (ParFlow_in)->dimensions[1];
	dims1[2] = (ParFlow_in)->dimensions[2];
	*X_inter_by_yz = (AVSfield_float *) 
           AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims1);
	if (*X_inter_by_yz == NULL) {
	    AVSerror("Allocation of output field failed.");
	    return(0);
	}

       Temp_grid_pt_Xyzs = (*X_inter_by_yz)->points; 
       for (i = 0; i < Max_x_interset_yz; i++)
	 {
           *(Temp_grid_pt_Xyzs + i) = 0.0;
	 }

       for (j = 0; j < dim_Y; j++)
	 {
           *(Temp_grid_pt_Xyzs + j + Max_x_interset_yz) = RECT_Y(ParFlow_in)[j]; 
	 }

       for (k = 0; k < dim_Z; k++)
	 {
	   *(Temp_grid_pt_Xyzs + k + Max_x_interset_yz + dim_Y) = RECT_Z(ParFlow_in)[k];
	 }

        get_Intersect_data(N_solids, 0, Y_grid, Z_grid, 
			    dim_Y, dim_Z, X_grid, dim_X, Max_x_interset_yz,
         Verts_in, Vert_cont_list, Num_solid_tri, X_inter_by_yz);

/*  FILL IN THE OUTPUT FIELD  Here!                                      */
/*                                                                       */
/* Free old field data                                                   */
	if (*Y_inter_by_zx) AVSfield_free(*Y_inter_by_zx);
/* Allocate space for new field output                                   */
	dims2[0] = (ParFlow_in)->dimensions[0];
	dims2[1] = Max_y_interset_zx;
	dims2[2] = (ParFlow_in)->dimensions[2];
	*Y_inter_by_zx = (AVSfield_float *) 
           AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims2);
	if (*Y_inter_by_zx == NULL) {
	    AVSerror("Allocation of output field failed.");
	    return(0);
	}

       Temp_grid_pt_Yzxs = (*Y_inter_by_zx)->points; 
       for (i = 0; i < dim_X; i++)
	 {
         *(Temp_grid_pt_Yzxs + i) = RECT_X(ParFlow_in)[i]; 
	 }

       for (j = 0; j < Max_y_interset_zx; j++)
	 {
	   *(Temp_grid_pt_Yzxs + j + dim_X) = 0.0;
	 }

       for (k = 0; k < dim_Z; k++)
	 {
          *(Temp_grid_pt_Yzxs + k + dim_X + Max_y_interset_zx) = RECT_Z(ParFlow_in)[k];
	 }

        get_Intersect_data(N_solids, 1, Z_grid, X_grid, dim_Z, dim_X, 
			    Y_grid, dim_Y, Max_y_interset_zx,
         Verts_in, Vert_cont_list, Num_solid_tri, Y_inter_by_zx);

/*  FILL IN THE OUTPUT FIELD  Here!                                      */
/* Free old field data                                                   */
	if (*Z_inter_by_xy) AVSfield_free(*Z_inter_by_xy);
/* Allocate space for new field output                                   */
	dims0[0] = (ParFlow_in)->dimensions[0];
	dims0[1] = (ParFlow_in)->dimensions[1];
	dims0[2] = Max_z_interset_xy;
	*Z_inter_by_xy = (AVSfield_float *) 
           AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims0);
	if (*Z_inter_by_xy == NULL) {
	    AVSerror("Allocation of output field failed.");
	    return(0);
	}

       Temp_grid_pt_Zxys = (*Z_inter_by_xy)->points; 
       for (i = 0; i < dim_X; i++)
	 {
         *(Temp_grid_pt_Zxys + i) = RECT_X(ParFlow_in)[i]; 
	 }

       for (j = 0; j < dim_Y; j++)
	 {
	   *(Temp_grid_pt_Zxys + j + dim_X) = RECT_Y(ParFlow_in)[j];
	 }

       for (k = 0; k < Max_z_interset_xy; k++)
	 {
          *(Temp_grid_pt_Zxys + k + dim_X + dim_Y) = 0.0;
	 }

        get_Intersect_data(N_solids, 2, X_grid, Y_grid, dim_X, dim_Y,
			     Z_grid, dim_Z, Max_z_interset_xy,
         Verts_in, Vert_cont_list, Num_solid_tri, Z_inter_by_xy);

/*                                                                       */
/* free any locally allocated data                                       */
/* THIS IS THE END OF THE 'HINTS' AREA. ADD YOUR OWN CODE BETWEEN THE    */
/* FOLLOWING COMMENTS. THE CODE YOU SUPPLY WILL NOT BE OVERWRITTEN.      */
 
/* ----> START OF USER-SUPPLIED CODE SECTION #3 (COMPUTE ROUTINE BODY)   */
/* <---- END OF USER-SUPPLIED CODE SECTION #3                            */
	return(1);
}
 
/* ----> START OF USER-SUPPLIED CODE SECTION #4 (SUBROUTINES, FUNCTIONS, UTILITY ROUTINES)*/
/* *****************************************/
/* Module get_Intersect_data                 */
/* *****************************************/
int get_Intersect_data(N_solids, Direction, grid_1, grid_2,  
     Dim_1, Dim_2, grid_I, Dim_I, Dim_3,
     Verts, Vert_cont_list, Num_triangles, Intersect_data)
     int Direction; 
     int N_solids;
     float *grid_1; 
     float *grid_2; 
     float *grid_I;
     int Dim_1; 
     int Dim_2;
     int Dim_I;
     int Dim_3;
     float *Verts; 
     int *Vert_cont_list; 
     int *Num_triangles; 
     AVSfield_float **Intersect_data;
    {
#define NO_INTERSECTION -1
#define INTERSECTION 1 
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  MIN(A, B)  ((A) < (B) ? (A) : (B))

      int Solid_Index;
      float  *temp_Data;
      int i,j,k;
      float grid_1_test; 
      float grid_2_test; 
      float U_tri[3];
      float V_tri[3];
      int CrossCount;
      int ip, jp, kp;
      int t0;
      int t1;
      int t2;
      int intersec_flag;
      int total_intsect;
      int *Intersection_counter;
      float u_0, v_0, w_0, v_1, u_1, w_1;
      float A_tmp, B_tmp, C_tmp, D_tmp;
      int coor_1, coor_2, coor_3;
      float U_max, U_min;
      float V_max,V_min;
      int i_start, i_fin;
      int j_start, j_fin;
      float del_grid_1;
      float del_grid_2;
      float del_grid_I;
      float temp_intersect;
      float f_test_intersect;
      int i_test_intersect;

      del_grid_1=grid_1[2] - grid_1[1];
      del_grid_2=grid_2[2] - grid_2[1];
      del_grid_I=grid_I[2] - grid_I[1];

/*       printf(" Solid_Index %d\n",Solid_Index); */
/*       printf(" N_solids %d\n",N_solids);       */
/*       printf("Direction  %d\n",Direction);     */
/*       printf("Dim_1  %d\n",Dim_1);             */
/*       printf("Dim_2  %d\n",Dim_2);             */
 
      if(Direction == 0)    /* intersection in X direction  */
	{
	  coor_1 =1;
	  coor_2 =2;
	  coor_3 =0;
	}

      if(Direction == 1)    /* intersection in Y direction  */
	{
	  coor_1 =2;
	  coor_2 =0;
	  coor_3 =1;
	}

      if(Direction == 2)    /* intersection in Z direction  */
	{
	  coor_1 =0;
	  coor_2 =1;
	  coor_3 =2;
	}

	  
     Intersection_counter = (int *) malloc(Dim_1 * Dim_2 * sizeof(int));

/*        for (i = 0; i < Dim_1; i++)
         {
          printf("i, grid_1 %d %f\n",i,grid_1[i]);
         }                          */

/*        for (j = 0; j < Dim_2; j++)
         {
          printf("j, grid_2 %d %f\n",j,grid_2[j]);
         }                          */

/*      printf(" Num_triangles %d\n",Num_triangles); */

        temp_Data = (*Intersect_data)->data;

/*  First element in data array in the intersection counter */

        for (j = 0; j < Dim_2; j++)
        for (i = 0; i < Dim_1; i++)
	  {
            Intersection_counter[i + Dim_1 * j] =0;
	  }
 
  total_intsect = 0;
      for(Solid_Index = 0; Solid_Index < N_solids; Solid_Index++) 
      for (k = 0; k < Num_triangles[Solid_Index]; k++) 
	{
	  t0 = Vert_cont_list[3*(Solid_Index + N_solids * k) + 0];
	  t1 = Vert_cont_list[3*(Solid_Index + N_solids * k) + 1];
	  t2 = Vert_cont_list[3*(Solid_Index + N_solids * k) + 2];
	  U_max=MAX(MAX(Verts[3*t0+coor_1],Verts[3*t1+coor_1]),Verts[3*t2+coor_1]);
	  V_max=MAX(MAX(Verts[3*t0+coor_2],Verts[3*t1+coor_2]),Verts[3*t2+coor_2]);
/*        printf("U_max,V_max %f %f\n",U_max,V_max); */
	  U_min=MIN(MIN(Verts[3*t0+coor_1],Verts[3*t1+coor_1]),Verts[3*t2+coor_1]);
	  V_min=MIN(MIN(Verts[3*t0+coor_2],Verts[3*t1+coor_2]),Verts[3*t2+coor_2]);
/*        printf("U_min,V_min %f %f\n",U_min,V_min); */

/*	  i_start = (int) ((U_min - grid_1[0])/del_grid_1);
	  i_fin = (int) ((U_max - grid_1[0])/del_grid_1)+1;
	  j_start = (int) ((V_min - grid_2[0])/del_grid_2);
	  j_fin = (int) ((V_max - grid_2[0])/del_grid_2)+1; */

	  i_start = (int) ((U_min - grid_1[0])/del_grid_1)-1;
	  i_fin = (int) ((U_max - grid_1[0])/del_grid_1)+2;
	  j_start = (int) ((V_min - grid_2[0])/del_grid_2)-1;
	  j_fin = (int) ((V_max - grid_2[0])/del_grid_2)+2;

	  if(i_start < 0)i_start = 0;
	  if(j_start < 0)j_start = 0;
	  if(i_fin > Dim_1) i_fin = Dim_1;
	  if(j_fin > Dim_2) j_fin = Dim_2;

/*        printf("i_start, i_fin  %d %d\n",i_start, i_fin);   */
/*        printf("j_start, j_fin  %d %d\n",j_start, j_fin);   */

	  for (i = i_start; i < i_fin; i++)
	    {
	    for (j = j_start; j < j_fin; j++)  
	      {
		grid_1_test=grid_1[i];
		grid_2_test=grid_2[j];  
		U_tri[0] = Verts[3*t0+coor_1] - grid_1_test;
		V_tri[0] = Verts[3*t0+coor_2] - grid_2_test;
		U_tri[1] = Verts[3*t1+coor_1] - grid_1_test;
		V_tri[1] = Verts[3*t1+coor_2] - grid_2_test;
		U_tri[2] = Verts[3*t2+coor_1] - grid_1_test;
		V_tri[2] = Verts[3*t2+coor_2] - grid_2_test; 

		CrossCount = 0;
		for (ip=0,jp=1; ip<3; ip++, jp++)
		  {
		    if( jp == 3) jp = 0;
		    if((V_tri[ip] < 0  && V_tri[jp] >= 0) ||
		       (V_tri[ip] >= 0 && V_tri[jp] < 0))
		      {
			if(U_tri[ip] >= 0 && U_tri[jp] >= 0)
			  CrossCount++;
			else if(U_tri[ip] >= 0 || U_tri[jp] >=0)
			  if(U_tri[ip]-V_tri[ip] * (U_tri[jp] - U_tri[ip])/
			     (V_tri[jp]-V_tri[ip]) > 0)
			    CrossCount++;
		      }
		  }
		if(CrossCount%2 == 0)
		  intersec_flag=NO_INTERSECTION;
		else{
		  total_intsect = total_intsect + 1;
		  intersec_flag=INTERSECTION;

/* set intersection counter */
		  kp=0;
		  Intersection_counter[i + Dim_1 * j] =
		    Intersection_counter[i + Dim_1 * j] +1;
		  if(Direction == 0)*(temp_Data + 2*(kp + Dim_3 * i + Dim_3 * Dim_1 * j))
		    = Intersection_counter[i + Dim_1 * j];
		  if(Direction == 1)*(temp_Data + 2*(j + Dim_2 * kp + Dim_2 * Dim_3 * i))
		    = Intersection_counter[i + Dim_1 * j];
		  if(Direction == 2)*(temp_Data + 2*(i + Dim_1 * j + Dim_1 * Dim_2 * kp))
		    = Intersection_counter[i + Dim_1 * j];
/* find intersection value */
		  u_0 =  Verts[3*t1+coor_1] - Verts[3*t0+coor_1];
		  v_0 =  Verts[3*t1+coor_2] - Verts[3*t0+coor_2]; 
		  w_0 =  Verts[3*t1+coor_3] - Verts[3*t0+coor_3];
		  u_1 =  Verts[3*t2+coor_1] - Verts[3*t0+coor_1];
		  v_1 =  Verts[3*t2+coor_2] - Verts[3*t0+coor_2];
		  w_1 =  Verts[3*t2+coor_3] - Verts[3*t0+coor_3];

		  A_tmp = v_0 * w_1 - v_1 * w_0;
		  B_tmp = w_0 * u_1 - w_1 * u_0;
		  C_tmp = u_0 * v_1 - u_1 * v_0;
		  D_tmp = A_tmp * Verts[3*t0+coor_1]  + 
		    B_tmp * Verts[3*t0+coor_2] + C_tmp * Verts[3*t0+coor_3];
		  if (C_tmp != 0.0)
		    {
/* set intersection value */
		      kp=Intersection_counter[i + Dim_1 * j];
/******/
		      if(kp >Dim_3)printf("WARNING at i,j kp > Dim_3, Direction %d %d %d %d %d\n",
                    i,j,kp,Dim_3,Direction);
/******/
		      temp_intersect = (D_tmp - A_tmp * grid_1_test - B_tmp * grid_2_test) / C_tmp;
/* check to see if intersection is ON a grid point  */
	f_test_intersect = (temp_intersect - grid_I[0]) / del_grid_I;	      
	i_test_intersect = (int)((temp_intersect - grid_I[0]) / del_grid_I);
/* if intersection is on a grid point, move it up to avoid more if test in following modules */
		      if((float)i_test_intersect == f_test_intersect)
			{
/*			  printf("fandi_test_intersect %f %d\n",f_test_intersect,i_test_intersect); */
/*			  temp_intersect = temp_intersect +1.0e-3 * del_grid_I; */
			}
		      if(Direction == 0)
			*(temp_Data + 2*(kp + Dim_3 * i + Dim_3 * Dim_1 * j)+1) = temp_intersect;
		      if(Direction == 1)
			*(temp_Data + 2*(j + Dim_2 * kp + Dim_2 * Dim_3 * i)+1) = temp_intersect;
		      if(Direction == 2)
			*(temp_Data + 2*(i + Dim_1 * j + Dim_1 * Dim_2 * kp)+1) = temp_intersect;

		      if (A_tmp > 0.0)
			{
/*               *normal_component = 1; */
/*              set intersection solid type */
			  kp=Intersection_counter[i + Dim_1 * j];
			  if(Direction == 0)*(temp_Data + 2*(kp + Dim_3 * i + Dim_3 * Dim_1 * j)) 
			    = Solid_Index+1;
			  if(Direction == 1)*(temp_Data + 2*(j + Dim_2 * kp + Dim_2 * Dim_3 * i)) 
			    = Solid_Index+1;
			  if(Direction == 2)*(temp_Data + 2*(i + Dim_1 * j + Dim_1 * Dim_2 * kp)) 
			    = Solid_Index+1;
			}
		      else
			{
/*               *normal_component = -1; */
/*               set intersection solid type */
			  kp=Intersection_counter[i + Dim_1 * j];
			  if(Direction == 0)*(temp_Data + 2*(kp + Dim_3 * i + Dim_3 * Dim_1 * j))
			    = -(Solid_Index+1);
			  if(Direction == 1)*(temp_Data + 2*(j + Dim_2 * kp + Dim_2 * Dim_3 * i)) 
			    = -(Solid_Index+1);
			  if(Direction == 2)*(temp_Data + 2*(i + Dim_1 * j + Dim_1 * Dim_2 * kp)) 
			    = -(Solid_Index+1);
			}
		    } 
		  } 
	      }       /* end of grid_1 grid_2 loop, index i,j */

	   }            /* end of triangle loop, index k */                    
        }            /* end of solid loop Solid_Index*/                    

      printf(" total_intsect Dim_1*Dim_2  %d %d\n",total_intsect, Dim_1*Dim_2); 

      free(Intersection_counter);
	return(1);
    }
/* <---- END OF USER-SUPPLIED CODE SECTION #4                            */
