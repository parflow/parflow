/* mod_gen Version 1                                                     */
/* Module Name: "Brick_cont" (Mapper) (Subroutine)                       */
/* Author: Dan &,2-4019,B451 R2024                                       */
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
int Brick_cont_desc()
{

	int in_port, out_port, param, iresult;
	extern int Brick_cont_compute();

	AVSset_module_name("Brick_cont", MODULE_MAPPER);

	/* Input Port Specifications               */

	in_port = AVScreate_input_port("PF_field_in", 
		"field 3D 3-space scalar rectilinear float", REQUIRED);
	in_port = AVScreate_input_port("N_solids", "integer", REQUIRED);

	/* Output Port Specifications              */
	out_port = AVScreate_output_port("X_plane_out", "integer");
	out_port = AVScreate_output_port("Y_plane_out", "integer");
	out_port = AVScreate_output_port("Z_plane_out", "integer");
	out_port = AVScreate_output_port("Picked_solid_out", "integer");

       /* Parameter Specifications                */
        param = AVSadd_parameter("X_plane", "integer", 0, 0, 1);
	AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Y_plane", "integer", 0, 0, 1);
	AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Z_plane", "integer", 0, 0, 1);
	AVSconnect_widget(param, "idial");
	param = AVSadd_parameter("Picked_solid", "integer", 0, 0, 1);
	AVSconnect_widget(param, "idial");

	AVSset_compute_proc(Brick_cont_compute);
/* ----> START OF USER-SUPPLIED CODE SECTION #2 (ADDITIONAL SPECIFICATION INFO)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #2                            */
	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int Brick_cont_compute( PF_field_in, N_solids,
X_plane_out, Y_plane_out, Z_plane_out, Picked_solid_out,
X_plane, Y_plane, Z_plane, Picked_solid)
	AVSfield_float *PF_field_in;
	int N_solids;
	int *X_plane_out, *Y_plane_out, *Z_plane_out;
	int *Picked_solid_out;
	int X_plane, Y_plane, Z_plane;
	int Picked_solid;
{
        int   dim_X;
        int   dim_Y;
        int   dim_Z;

	dim_X=(PF_field_in)->dimensions[0];
	dim_Y=(PF_field_in)->dimensions[1];
	dim_Z=(PF_field_in)->dimensions[2];

	AVSmodify_parameter("X_plane",AVS_MAXVAL,0,0,dim_X-1);
	AVSmodify_parameter("Y_plane",AVS_MAXVAL,0,0,dim_Y-1);
	AVSmodify_parameter("Z_plane",AVS_MAXVAL,0,0,dim_Z-1);
	AVSmodify_parameter("Picked_solid",AVS_MAXVAL,0,0,N_solids-1);

	*X_plane_out = X_plane;
	*Y_plane_out = Y_plane;
	*Z_plane_out = Z_plane;
	*Picked_solid_out = Picked_solid;	 

/* THIS IS THE END OF THE 'HINTS' AREA. ADD YOUR OWN CODE BETWEEN THE    */
/* FOLLOWING COMMENTS. THE CODE YOU SUPPLY WILL NOT BE OVERWRITTEN.      */
 
/* ----> START OF USER-SUPPLIED CODE SECTION #3 (COMPUTE ROUTINE BODY)   */
/* <---- END OF USER-SUPPLIED CODE SECTION #3                            */
	return(1);
}
 
 
/* ----> START OF USER-SUPPLIED CODE SECTION #4 (SUBROUTINES, FUNCTIONS, UTILITY ROUTINES)*/
/* <---- END OF USER-SUPPLIED CODE SECTION #4                            */
