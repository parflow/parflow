/* mod_gen Version 1                                                     */
/* Module Name: "PFsol_to_Geom" (Mapper) (Subroutine)                      */
/* Author: Dan &,2-4019,B451 R2024                                      */
/* Date Created: Tue May 28 15:39:56 1996                                */
/*                                                                       */

#include <stdio.h>
#include <math.h>
#include <avs/avs.h>
#include <avs/port.h>
#include <avs/field.h>
#include <avs/geom.h>
 
/* ----> START OF USER-SUPPLIED CODE SECTION #1 (INCLUDE FILES, GLOBAL VARIABLES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #1                            */
 
/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int PFsol_to_Geom_desc()
{

	int in_port, out_port, param, iresult;
	extern int PFsol_to_Geom_compute();

	AVSset_module_name("PFsol_to_Geom", MODULE_MAPPER);

	/* Input Port Specifications               */
	in_port = AVScreate_input_port("Vertical_Expan", "real", REQUIRED);

	/* Output Port Specifications              */
	out_port = AVScreate_output_port("Geo_Unit", "geom");

	/* Parameter Specifications                */
	param = AVSadd_parameter("Solid_file_in", "string", "", "", ":.pfsol");
	AVSconnect_widget(param, "browser");

	param = AVSadd_parameter("Solid_or_Wire", "choice", "Solid", 
		"Solid:Wire:Band:Triangles:None", ":");
	AVSconnect_widget(param, "radio_buttons");

	param = AVSadd_parameter("Band_Size", "choice", "Relative", 
		"Fixed:Relative", ":");
	AVSconnect_widget(param, "radio_buttons");

	param = AVSadd_parameter("Solid_num", "integer", 0, 0, 1);
	AVSconnect_widget(param, "idial");

	param = AVSadd_float_parameter("Vert_Displace", 0.00000, 
		FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Band_Width", 0.1, 0.00000, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Band_Width_Min", 0.1, 0.00000, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("X_min_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("X_max_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Y_min_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Y_max_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Z_min_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_float_parameter("Z_max_cutoff", 0.5, FLOAT_UNBOUND, FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	param = AVSadd_parameter("Limits_or_not", "choice", "No Limits", 
		"Use Limits:No Limits", ":");
	AVSconnect_widget(param, "radio_buttons");



	AVSset_compute_proc(PFsol_to_Geom_compute);
/* ----> START OF USER-SUPPLIED CODE SECTION #2 (ADDITIONAL SPECIFICATION INFO)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #2                            */
	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int PFsol_to_Geom_compute( Vertical_Expan,
  Geo_Unit, Solid_file_in, Solid_or_Wire, Band_Size, Solid_num,
  Vert_Displace, Band_Width, Band_Width_Min,
  X_min_cutoff, X_max_cutoff,
  Y_min_cutoff, Y_max_cutoff,
  Z_min_cutoff, Z_max_cutoff,
  Limits_or_not)
	float *Vertical_Expan;
	GEOMedit_list *Geo_Unit;
	char *Solid_file_in;
	char *Solid_or_Wire;
	char *Band_Size;
	int Solid_num;
	float *Vert_Displace;
	float *Band_Width;
	float *Band_Width_Min;
	float *X_min_cutoff;
	float *X_max_cutoff;
	float *Y_min_cutoff;
	float *Y_max_cutoff;
	float *Z_min_cutoff;
	float *Z_max_cutoff;
	char *Limits_or_not;
{

/*  THIS IS A 'HINTS' AREA - YOU MAY CUT AND PASTE AT WILL FROM IT       */
#define  MAX(A, B)  ((A) > (B) ? (A) : (B))
#define  MIN(A, B)  ((A) < (B) ? (A) : (B))
#define  MAX_SOLIDS 20
#define  MAX_PFsol_to_Geom 40
  FILE *fp;
  GEOMobj *Tmp_Obj;
  float *Temp_grid_pt_Xyzs;
  float *Temp_grid_pt_Yzxs;
  float *Temp_grid_pt_Zxys;
  float *Verts_in;
  float *X_grid, *Y_grid, *Z_grid;
  float MIN_of_extent, MAX_of_extent;
  float Tmp_Verts[3];
  float line_verts[2][3];
  float normals[3][3];
  float verts[3][3];
  int *Vert_cont_list;
  int File_Version_Num;
  int Max_vertex;
  int N_solids;
  int N_vertex;
  int Num_paches, Num_paches_tri, Paches_tri;
  int Num_solid_tri[MAX_SOLIDS];
  int Tmp_cont[3];
  int Tri_Num_Tmp[3];
  int dim_X,dim_Y,dim_Z;
  int dims0[3];
  int dims1[3];
  int dims2[3];
  int i,j,k,m,n,read_pass,xyz;
  float band_width;
  float inner_verts[3][3];
  float triangle_cent[3];
  float band_verts[4][3];
  float X_max,X_min;
  float Y_max,Y_min;
  float Z_max,Z_min;
  float X_view_center, X_view_width;
  float Y_view_center, Y_view_width;
  float Z_view_center, Z_view_width;
  float X_center, X_half_width;
  float Y_center, Y_half_width;
  float Z_center, Z_half_width;
  float edge_dist[3];
  float edge_tmp;
  float center_dist[3];
  float tri_h, tri_half_p;
  float actual_band_width;

/* Read pfsol file in two pases, first to get number of solids    */
/* and number of triangles per solid.                              */

     for (read_pass = 0; read_pass <2 ; read_pass++)
       {
        fp = fopen(Solid_file_in, "r");

        fscanf(fp,"%d\n",&File_Version_Num);
        fscanf(fp,"%d\n",&N_vertex);

         if(read_pass == 0)
           Verts_in = (float *) malloc(3 * N_vertex * sizeof(float));
        for (m = 0; m < N_vertex ; m++)
           {
            fscanf(fp,"%f %f %f\n",&Tmp_Verts[0],&Tmp_Verts[1],&Tmp_Verts[2]);
           if(read_pass == 0)
             {
              Verts_in[3*m] = Tmp_Verts[0];
              Verts_in[3*m+1] = Tmp_Verts[1];
              Verts_in[3*m+2] = *Vertical_Expan *  Tmp_Verts[2] + *Vert_Displace;
             }
           }
       fscanf(fp,"%d\n",&N_solids);
             if(read_pass == 0)printf("This file contains %d solids\n",N_solids); 
        Max_vertex = 0;
        for (m = 0; m <N_solids ; m++)
          {
             fscanf(fp,"%d\n",&Num_solid_tri[m]);
             Max_vertex = MAX(Max_vertex,Num_solid_tri[m]);      
             if(read_pass == 0)printf(" Solid %d contains %d triangles \n", m,
                   Num_solid_tri[m]);
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
/*           if(read_pass == 0)printf("Max_vertex %d\n", Max_vertex);    */
          if(read_pass == 0)
            Vert_cont_list = (int *) malloc(3 * Max_vertex * N_solids *sizeof(int));  
       fclose(fp);
      }

	    if(N_solids > 1)AVSmodify_parameter("Solid_num",AVS_MAXVAL,0,0,N_solids-1);


/*                                                                       */
/* Create the Geo_Unit object                                            */
  if(strncmp(Solid_or_Wire,"None",4) == 0)
    Tmp_Obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL); 
  if(strncmp(Solid_or_Wire,"Solid",5) == 0)
    Tmp_Obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL); 
  if(strncmp(Solid_or_Wire,"Band",4) == 0)
    Tmp_Obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL); 
  if(strncmp(Solid_or_Wire,"Wire",4) == 0)
    Tmp_Obj = GEOMcreate_obj(GEOM_POLYTRI,GEOM_NULL );
  if(strncmp(Solid_or_Wire,"Triangles",9) == 0)
    Tmp_Obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL); 



       if(strncmp(Solid_or_Wire,"None",4) != 0)
	 {

/* Get max in x y and z     */
   X_max=-1.0e9;
   X_min=1.0e9;
   Y_max=-1.0e9;
   Y_min=1.0e9;
   Z_max=-1.0e9;
   Z_min=1.0e9;

   for (n = 0; n <Num_solid_tri[Solid_num] ; n++)    
	     {
	       Tri_Num_Tmp[0]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 0];
	       Tri_Num_Tmp[1]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 1];
	       Tri_Num_Tmp[2]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 2];
               for (xyz = 0; xyz < 3 ; xyz++)
                 {
                   verts[0][xyz]=Verts_in[3*Tri_Num_Tmp[0]+xyz];
                   verts[1][xyz]=Verts_in[3*Tri_Num_Tmp[1]+xyz];
                   verts[2][xyz]=Verts_in[3*Tri_Num_Tmp[2]+xyz];
                 }
               for (m = 0; m < 3 ; m++)
		 {
		   X_max = MAX(X_max, verts[m][0]);
		   Y_max = MAX(Y_max, verts[m][1]);
		   Z_max = MAX(Z_max, verts[m][2]);
		   X_min = MIN(X_min, verts[m][0]);
		   Y_min = MIN(Y_min, verts[m][1]);
		   Z_min = MIN(Z_min, verts[m][2]);
		 }
	     }
   X_view_center = 0.5;
   X_view_width = 0.5;
   Y_view_center = 0.5;
   Y_view_width = 0.5;
   Z_view_center = 0.5;
   Z_view_width = 0.5;

   X_center = X_view_center*X_min + (1.0-X_view_center)*X_max;
   X_half_width = 0.5*X_view_width*(X_max-X_min);

   Y_center = Y_view_center*Y_min + (1.0-Y_view_center)*Y_max;
   Y_half_width =  0.5*Y_view_width*(Y_max-Y_min);

   Z_center = Z_view_center*Z_min + (1.0-Z_view_center)*Z_max;
   Z_half_width =  0.5*Z_view_width*(Z_max-Z_min);

   for (n = 0; n <Num_solid_tri[Solid_num] ; n++) 
     {
       if(n % 1000 ==0)
	 {
	   printf("PFsol_to_Geom is %f percent done\n",(100.0*n)/Num_solid_tri[Solid_num]);
	 }
       Tri_Num_Tmp[0]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 0];
       Tri_Num_Tmp[1]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 1];
       Tri_Num_Tmp[2]=Vert_cont_list[3*(Solid_num  + N_solids * n) + 2];
       for (xyz = 0; xyz < 3 ; xyz++)
	 {
	   verts[0][xyz]=Verts_in[3*Tri_Num_Tmp[0]+xyz];
	   verts[1][xyz]=Verts_in[3*Tri_Num_Tmp[1]+xyz];
	   verts[2][xyz]=Verts_in[3*Tri_Num_Tmp[2]+xyz];
	 }
       for (xyz = 0; xyz < 3 ; xyz++)
	 {
	   triangle_cent[xyz] = 0.0;
	   for (m = 0; m < 3 ; m++)
	     {
	       triangle_cent[xyz] = triangle_cent[xyz] + verts[m][xyz];
	     }
	   triangle_cent[xyz] = triangle_cent[xyz]/3.0;
	 }

       for (m = 0; m < 3 ; m++)
	 {
	   edge_dist[m]=0.0;
	   center_dist[m]=0.0;
	 }

       for (xyz = 0; xyz < 3 ; xyz++)
	 {
	   edge_dist[0]=edge_dist[0] +
	     (verts[0][xyz]-verts[1][xyz])*(verts[0][xyz]-verts[1][xyz]);
	   edge_dist[1]=edge_dist[1] +
	     (verts[1][xyz]-verts[2][xyz])*(verts[1][xyz]-verts[2][xyz]);
	   edge_dist[2]=edge_dist[2] +
	     (verts[0][xyz]-verts[2][xyz])*(verts[0][xyz]-verts[2][xyz]);
	 }
/*  printf("edge_dist %f %f %f \n",edge_dist[0],edge_dist[1],edge_dist[2]); */

       for (m = 0; m < 3 ; m++)
	 {
	   center_dist[m]=0.0;
	   for (xyz = 0; xyz < 3 ; xyz++)
	     {
	       center_dist[m]=center_dist[m] +
		 (triangle_cent[xyz]-verts[m][xyz])*(triangle_cent[xyz]-verts[m][xyz]);
	     }
	 }

       for (m = 0; m < 3 ; m++)
	 {
	   center_dist[m]=(float)sqrt((double)center_dist[m]);
	   edge_dist[m]=(float)sqrt((double)edge_dist[m]);
	 }

       tri_half_p = 0.5*(edge_dist[0] + edge_dist[1] + edge_dist[2]);

       tri_h =((tri_half_p-edge_dist[0])*
	       (tri_half_p-edge_dist[1])*
	       (tri_half_p-edge_dist[2]))/tri_half_p;
       tri_h = (float)sqrt((double)tri_h);

/*       printf("n tri_h %d %f\n",n, tri_h);       */
/*  printf("edge_dist %f %f %f \n",edge_dist[0],edge_dist[1],edge_dist[2]);*/
/*  printf("center_dist %f %f %f \n",center_dist[0],center_dist[1],center_dist[2]);*/
/*_______________________________________________________________________*/

   if(strncmp(Solid_or_Wire,"Solid",5) == 0)
     {
       if(((strncmp(Limits_or_not,"Use Limits",10) == 0 &&
	    triangle_cent[0] < *X_max_cutoff &&
	    triangle_cent[0] > *X_min_cutoff &&
	    triangle_cent[1] < *Y_max_cutoff &&
	    triangle_cent[1] > *Y_min_cutoff &&
	    triangle_cent[2] < *Z_max_cutoff &&
	    triangle_cent[2] > *Z_min_cutoff))
	  ||
	  (strncmp(Limits_or_not,"No Limits",9) == 0))
 
	 {
	GEOMadd_disjoint_polygon
	     (Tmp_Obj,verts,GEOM_NULL,GEOM_NULL,3,GEOM_SHARED,0); 
	 }
     }
/*_______________________________________________________________________*/


  if(strncmp(Solid_or_Wire,"Band",4) == 0)
    {
      band_width = *Band_Width;
      if(((strncmp(Limits_or_not,"Use Limits",10) == 0 &&
	   triangle_cent[0] < *X_max_cutoff &&
	   triangle_cent[0] > *X_min_cutoff &&
	   triangle_cent[1] < *Y_max_cutoff &&
	   triangle_cent[1] > *Y_min_cutoff &&
	   triangle_cent[2] < *Z_max_cutoff &&
	   triangle_cent[2] > *Z_min_cutoff)) 
	 ||
	 (strncmp(Limits_or_not,"No Limits",9) == 0))
	{

	  if(strncmp(Band_Size,"Relative",8) == 0)
	    {
	      band_width = *Band_Width;
	    }
	  if(strncmp(Band_Size,"Fixed",5) == 0)
	    {
	      band_width = *Band_Width/tri_h;
	    }

	  actual_band_width = tri_h*band_width;
	    if(actual_band_width < *Band_Width_Min)
	      {
	      band_width = *Band_Width_Min/tri_h;
	      }

	  if(band_width < 1.0 )
	    {
	      for (xyz = 0; xyz < 3 ; xyz++)
		{
		  for (m = 0; m < 3 ; m++)
		    {
		      inner_verts[m][xyz] = 
			band_width*triangle_cent[xyz] + (1.0-band_width)* verts[m][xyz];
		    }
		}
	   
	      for (xyz = 0; xyz < 3 ; xyz++)
		{
		  band_verts[0][xyz] =  verts[0][xyz];
		  band_verts[1][xyz] =  verts[1][xyz];
		  band_verts[2][xyz] =  inner_verts[1][xyz];
		  band_verts[3][xyz] =  inner_verts[0][xyz];
		}

	      GEOMadd_disjoint_polygon
		(Tmp_Obj,band_verts,GEOM_NULL,GEOM_NULL,4,GEOM_SHARED,0); 

	      for (xyz = 0; xyz < 3 ; xyz++)
		{
		  band_verts[0][xyz] =  verts[1][xyz];
		  band_verts[1][xyz] =  verts[2][xyz];
		  band_verts[2][xyz] =  inner_verts[2][xyz];
		  band_verts[3][xyz] =  inner_verts[1][xyz];
		}
	      
	      GEOMadd_disjoint_polygon
		(Tmp_Obj,band_verts,GEOM_NULL,GEOM_NULL,4,GEOM_SHARED,0); 

	      for (xyz = 0; xyz < 3 ; xyz++)
		{
		  band_verts[0][xyz] =  verts[2][xyz];
		  band_verts[1][xyz] =  verts[0][xyz];
		  band_verts[2][xyz] =  inner_verts[0][xyz];
		  band_verts[3][xyz] =  inner_verts[2][xyz];
		}

	      GEOMadd_disjoint_polygon
		(Tmp_Obj,band_verts,GEOM_NULL,GEOM_NULL,4,GEOM_SHARED,0); 

	    }
	  else
	    {
	      GEOMadd_disjoint_polygon
		(Tmp_Obj,verts,GEOM_NULL,GEOM_NULL,3,GEOM_SHARED,0); 
	    }
	}  
    }  
/*_______________________________________________________________________*/


   if(strncmp(Solid_or_Wire,"Triangles",9) == 0)
     {
       if(((strncmp(Limits_or_not,"Use Limits",10) == 0 &&
	    triangle_cent[0] < *X_max_cutoff &&
	    triangle_cent[0] > *X_min_cutoff &&
	    triangle_cent[1] < *Y_max_cutoff &&
	    triangle_cent[1] > *Y_min_cutoff &&
	    triangle_cent[2] < *Z_max_cutoff &&
	    triangle_cent[2] > *Z_min_cutoff)) 
	  ||
	  (strncmp(Limits_or_not,"No Limits",9) == 0))
	 {

	   if(strncmp(Band_Size,"Relative",8) == 0)
	     {
	       band_width = *Band_Width;
	     }
	   if(strncmp(Band_Size,"Fixed",5) == 0)
	     {
	       band_width = *Band_Width/tri_h;
	     }  

	  actual_band_width = tri_h*band_width;
	    if(actual_band_width < *Band_Width_Min)
	      {
	      band_width = *Band_Width_Min/tri_h;
	      }

	   if(band_width < 1.0 )
	     {
	       for (xyz = 0; xyz < 3 ; xyz++)
		 {
		   for (m = 0; m < 3 ; m++)
		     {
		       inner_verts[m][xyz] = 
			 band_width*triangle_cent[xyz] + 
			   (1.0-band_width)* verts[m][xyz];
		     }
		 }

	       GEOMadd_disjoint_polygon
		 (Tmp_Obj,inner_verts,GEOM_NULL,GEOM_NULL,3,GEOM_SHARED,0); 

	     }  
	 }  
     }  


/*_______________________________________________________________________*/

       if(strncmp(Solid_or_Wire,"Wire",4) == 0)
	 {
       if(((strncmp(Limits_or_not,"Use Limits",10) == 0 &&
	  triangle_cent[0] < *X_max_cutoff &&
	  triangle_cent[0] > *X_min_cutoff &&
	  triangle_cent[1] < *Y_max_cutoff &&
	  triangle_cent[1] > *Y_min_cutoff &&
	  triangle_cent[2] < *Z_max_cutoff &&
	  triangle_cent[2] > *Z_min_cutoff)) 
	 ||
	   (strncmp(Limits_or_not,"No Limits",9) == 0))
	 {
	   for (xyz = 0; xyz < 3 ; xyz++)
	     {
	       line_verts[0][xyz] = verts[0][xyz];
	       line_verts[1][xyz] = verts[1][xyz];
	     }
	   GEOMadd_disjoint_line
	     (Tmp_Obj,line_verts,GEOM_NULL,2,GEOM_COPY_DATA);
	   for (xyz = 0; xyz < 3 ; xyz++)
	     {
	       line_verts[0][xyz] = verts[1][xyz];
	       line_verts[1][xyz] = verts[2][xyz];
	     }
	   GEOMadd_disjoint_line
	     (Tmp_Obj,line_verts,GEOM_NULL,2,GEOM_COPY_DATA);
	   for (xyz = 0; xyz < 3 ; xyz++)
	     {
	       line_verts[0][xyz] = verts[2][xyz];
	       line_verts[1][xyz] = verts[0][xyz];
	     }
	   GEOMadd_disjoint_line
	     (Tmp_Obj,line_verts,GEOM_NULL,2,GEOM_COPY_DATA);
	 }
     }
     

/*_______________________________________________________________________*/
     }

	 }     /* if(strncmp(Solid_or_Wire,"None",4) != 0) */
 

      GEOMgen_normals(Tmp_Obj,0);

        GEOMset_extent(Tmp_Obj);
  
	*Geo_Unit = GEOMinit_edit_list(*Geo_Unit);
	GEOMedit_geometry(*Geo_Unit, "Geo_Unit", Tmp_Obj);
	GEOMdestroy_obj(Tmp_Obj);


	return(1);
}
