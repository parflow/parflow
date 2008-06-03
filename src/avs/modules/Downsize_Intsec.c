/* mod_gen Version 1                                                     */
/* Module Name: "Downsize_Intsec" (Mapper) (Subroutine)                        */
/* Author: Dan &,2-4019,B451 R2024                                      */
/* Date Created: Fri Jun  7 14:26:30 1996                                */
/*                                                                       */

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
int Downsize_Intsec_desc()
{

	int in_port, out_port, param, iresult;
	extern int Downsize_Intsec_compute();

	AVSset_module_name("Downsize_Intsec", MODULE_FILTER);

	/* Input Port Specifications               */

	in_port = AVScreate_input_port("PF_field_1", 
		"field 3D 3-space scalar rectilinear float", OPTIONAL);

	in_port = AVScreate_input_port("PF_field_2", 
		"field 3D 3-space scalar rectilinear float", OPTIONAL);

	in_port = AVScreate_input_port("X_inter_by_yz", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);
	in_port = AVScreate_input_port("Y_inter_by_zx", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);
	in_port = AVScreate_input_port("Z_inter_by_xy", 
		"field 3D 3-space 2-vector rectilinear float", REQUIRED);

	/* Output Port Specifications              */

	out_port = AVScreate_output_port("dn_PF_field_1", 
		"field 3D 3-space scalar rectilinear float");

	out_port = AVScreate_output_port("dn_PF_field_2", 
		"field 3D 3-space scalar rectilinear float");

	out_port = AVScreate_output_port("dn_X_inter_by_yz", 
		"field 3D 3-space 2-vector rectilinear float");
	out_port = AVScreate_output_port("dn_Y_inter_by_zx", 
		"field 3D 3-space 2-vector rectilinear float");
	out_port = AVScreate_output_port("dn_Z_inter_by_xy", 
		"field 3D 3-space 2-vector rectilinear float");

	/* Parameter Specifications                */
	param = AVSadd_parameter("Downsize", "integer", 1, 1, 16);
	AVSconnect_widget(param, "idial");
	param = AVSadd_float_parameter("Z_expand", 1.00000, FLOAT_UNBOUND, 
		FLOAT_UNBOUND);
	AVSconnect_widget(param, "typein_real");

	AVSset_compute_proc(Downsize_Intsec_compute);

	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int Downsize_Intsec_compute( PF_field_1, PF_field_2,
   X_inter_by_yz, Y_inter_by_zx, Z_inter_by_xy,
   dn_PF_field_1, dn_PF_field_2,
   dn_X_inter_by_yz, dn_Y_inter_by_zx, dn_Z_inter_by_xy,
   Downsize, Z_expand)
     AVSfield_float *PF_field_1;
     AVSfield_float *PF_field_2;
     AVSfield_float *X_inter_by_yz;
     AVSfield_float *Y_inter_by_zx;
     AVSfield_float *Z_inter_by_xy;
     AVSfield_float **dn_PF_field_1;
     AVSfield_float **dn_PF_field_2;
     AVSfield_float **dn_X_inter_by_yz;
     AVSfield_float **dn_Y_inter_by_zx;
     AVSfield_float **dn_Z_inter_by_xy;
     int Downsize;
     float *Z_expand;
{
  int i, j, k;

  float *x_grid;
  float *y_grid;
  float *z_grid;

  float *x_grid_out;
  float *y_grid_out;
  float *z_grid_out;

  float *data_1;
  float *data_2;

  int dim_X;
  int dim_Y;
  int dim_Z;

  int dim_X_out;
  int dim_Y_out;
  int dim_Z_out;

  int dim_I0;
  int dim_I1;
  int dim_I2;
  int dims[3];

  float *Temp_data_pt_Xyzs;
  float *Temp_grid_pt_Xyzs;

  float *Temp_data_pt_Yzxs;
  float *Temp_grid_pt_Yzxs;
 
  float *Temp_data_pt_Zxys;
  float *Temp_grid_pt_Zxys;

  float *Temp_data_pt_PF_1;
  float *Temp_grid_pt_PF_1;

  float *Temp_data_pt_PF_2;
  float *Temp_grid_pt_PF_2;

  dim_X = (Z_inter_by_xy)->dimensions[0];
  dim_Y = (Z_inter_by_xy)->dimensions[1];
  dim_Z = (X_inter_by_yz)->dimensions[2];

  dim_X_out = (dim_X-1)/Downsize + 1;
  dim_Y_out = (dim_Y-1)/Downsize + 1;
  dim_Z_out = (dim_Z-1)/Downsize + 1;
 
  dim_I0 = (X_inter_by_yz)->dimensions[0];
  dim_I1 = (Y_inter_by_zx)->dimensions[1];
  dim_I2 = (Z_inter_by_xy)->dimensions[2];

  x_grid = (float *)malloc(dim_X * sizeof(float));
  y_grid = (float *)malloc(dim_Y * sizeof(float));
  z_grid = (float *)malloc(dim_Z * sizeof(float));

  for (i = 0; i < dim_X; i++)
    {
      x_grid[i] = RECT_X(Z_inter_by_xy)[i];
    }                         

  for (j = 0; j < dim_Y; j++)
    {
      y_grid[j] = RECT_Y(Z_inter_by_xy)[j];
    }                         

  for (k = 0; k < dim_Z; k++)
    {
      z_grid[k] = *Z_expand * RECT_Z(X_inter_by_yz)[k];
    }                         

  dims[0] = dim_X_out;
  dims[1] = dim_Y_out;
  dims[2] = dim_Z_out;

  *dn_PF_field_1 = (AVSfield_float *) 
    AVSdata_alloc("field 3D 3-space scalar rectilinear float", dims);
  if (*dn_PF_field_1 == NULL) {
    AVSerror("Allocation of output field failed.");
    return(0);
  }
  
  *dn_PF_field_2 = (AVSfield_float *) 
    AVSdata_alloc("field 3D 3-space scalar rectilinear float", dims);
  if (*dn_PF_field_2 == NULL) {
    AVSerror("Allocation of output field failed.");
    return(0);
  }

  dims[0] = dim_I0;
  dims[1] = dim_Y_out;
  dims[2] = dim_Z_out;

  *dn_X_inter_by_yz = (AVSfield_float *) 
    AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims);
  if (*dn_X_inter_by_yz == NULL) {
    AVSerror("Allocation of output field failed.");
    return(0);
  }

  dims[0] = dim_X_out;
  dims[1] = dim_I1;
  dims[2] = dim_Z_out;

  *dn_Y_inter_by_zx = (AVSfield_float *) 
    AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims);
  if (*dn_Y_inter_by_zx == NULL) {
    AVSerror("Allocation of output field failed.");
    return(0);
  }

  dims[0] = dim_X_out;
  dims[1] = dim_Y_out;
  dims[2] = dim_I2;

  *dn_Z_inter_by_xy = (AVSfield_float *) 
    AVSdata_alloc("field 3D 3-space 2-vector rectilinear float",dims);
  if (*dn_Z_inter_by_xy == NULL) {
    AVSerror("Allocation of output field failed.");
    return(0);
  }

  Temp_grid_pt_Xyzs = (*dn_X_inter_by_yz)->points; 

  for(i=0; i < dim_I0; i++)
    {
      *(Temp_grid_pt_Xyzs + i) = 0.0;
    }
  for(j=0; j < dim_Y_out; j++)
    {
      *(Temp_grid_pt_Xyzs + dim_I0 + j) = y_grid[j*Downsize];
    }
  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_grid_pt_Xyzs  + dim_Y_out + dim_I0 + k) = z_grid[k*Downsize];
    }

  Temp_data_pt_Xyzs = (*dn_X_inter_by_yz)->data; 
  for(i=0; i < dim_I0; i++)
  for(j=0; j < dim_Y_out; j++)
  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_data_pt_Xyzs + 2*(i + j * dim_I0 + dim_Y_out * dim_I0 * k) + 0) =
	I3DV(X_inter_by_yz, i, j*Downsize,  k*Downsize)[0];
      *(Temp_data_pt_Xyzs + 2*(i + j * dim_I0 + dim_Y_out * dim_I0 * k) + 1) =
	I3DV(X_inter_by_yz, i, j*Downsize,  k*Downsize)[1];
    }

  Temp_grid_pt_Yzxs = (*dn_Y_inter_by_zx)->points; 

  for(i=0; i < dim_X_out; i++)
    {
      *(Temp_grid_pt_Yzxs + i) = x_grid[i*Downsize];
    }
  for(j=0; j < dim_I1; j++)
    {
      *(Temp_grid_pt_Yzxs + dim_X_out + j) = 0.0;
    }
  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_grid_pt_Yzxs  + dim_X_out + dim_I1 + k) = z_grid[k*Downsize];
    }

  Temp_data_pt_Yzxs = (*dn_Y_inter_by_zx)->data; 
  for(i=0; i < dim_X_out; i++)
  for(j=0; j < dim_I1; j++)
  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_data_pt_Yzxs + 2*(i + j * dim_X_out + dim_X_out * dim_I1 * k) + 0) =
	I3DV(Y_inter_by_zx, i*Downsize, j,  k*Downsize)[0];
      *(Temp_data_pt_Yzxs + 2*(i + j * dim_X_out + dim_X_out * dim_I1 * k) + 1) =
	I3DV(Y_inter_by_zx, i*Downsize, j,  k*Downsize)[1];  
    }

  Temp_grid_pt_Zxys = (*dn_Z_inter_by_xy)->points; 

  for(i=0; i < dim_X_out; i++)
    {
      *(Temp_grid_pt_Zxys + i) = x_grid[i*Downsize];
    }
  for(j=0; j < dim_Y_out; j++)
    {
      *(Temp_grid_pt_Zxys + dim_X_out + j) = y_grid[j*Downsize];
    }
  for(k=0; k < dim_I2; k++)
    {
      *(Temp_grid_pt_Zxys + dim_X_out + dim_Y_out + k ) = 0.0;
    }

  Temp_data_pt_Zxys = (*dn_Z_inter_by_xy)->data; 
  for(i=0; i < dim_X_out; i++)
  for(j=0; j < dim_Y_out; j++)
  for(k=0; k < dim_I2; k++)
    {
      *(Temp_data_pt_Zxys + 2*(i + j * dim_X_out + dim_X_out * dim_Y_out * k) + 0) =
	I3DV(Z_inter_by_xy, i*Downsize, j*Downsize,  k)[0];
      *(Temp_data_pt_Zxys + 2*(i + j * dim_X_out + dim_X_out * dim_Y_out * k) + 1) =
	*Z_expand * I3DV(Z_inter_by_xy, i*Downsize, j*Downsize,  k)[1];  
    } 

  Temp_grid_pt_PF_1 = (*dn_PF_field_1)->points; 
  Temp_grid_pt_PF_2 = (*dn_PF_field_2)->points; 

  for(i=0; i < dim_X_out; i++)
    {
      *(Temp_grid_pt_PF_1 + i) = x_grid[i*Downsize];
      *(Temp_grid_pt_PF_2 + i) = x_grid[i*Downsize];
    }

  for(j=0; j < dim_Y_out; j++)
    {
      *(Temp_grid_pt_PF_1 + j + dim_X_out) = y_grid[j*Downsize];
      *(Temp_grid_pt_PF_2 + j + dim_X_out) = y_grid[j*Downsize];
    }

  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_grid_pt_PF_1 + k + dim_X_out + dim_Y_out) = z_grid[k*Downsize];
      *(Temp_grid_pt_PF_2 + k + dim_X_out + dim_Y_out) = z_grid[k*Downsize];
    } 

  Temp_data_pt_PF_1 = (*dn_PF_field_1)->data; 
  Temp_data_pt_PF_2 = (*dn_PF_field_2)->data; 

  for(i=0; i < dim_X_out; i++)
  for(j=0; j < dim_Y_out; j++)
  for(k=0; k < dim_Z_out; k++)
    {
      *(Temp_data_pt_PF_1 + i + dim_X_out * j + dim_X_out * dim_Y_out * k) = 
	I3D(PF_field_1, i*Downsize, j*Downsize, k*Downsize);
      *(Temp_data_pt_PF_2 + i + dim_X_out * j + dim_X_out * dim_Y_out * k) = 2.02;
      I3D(PF_field_2, i*Downsize, j*Downsize, k*Downsize); 
    }  

  free(x_grid);
  free(y_grid);
  free(z_grid);

	return(1);
}
