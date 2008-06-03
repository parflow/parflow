/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * Perspective_Zoom
 *
 * AVS coroutine to rotate images in the Geometry Viewer.
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include <avs.h>


/*--------------------------------------------------------------------------
 * Perspective_Zoom
 *--------------------------------------------------------------------------*/

Perspective_Zoom()
{
   int     Perspective_Zoom_compute();
   int     p;


   AVSset_module_name("perspective_zoom", MODULE_DATA);

   p = AVSadd_float_parameter("FOV init", 30.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("FOV fin", 10.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("At X init", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

    p = AVSadd_float_parameter("At Y init", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

    p = AVSadd_float_parameter("At Z init", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("At X fin", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

    p = AVSadd_float_parameter("At Y fin", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

    p = AVSadd_float_parameter("At Z fin", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_parameter("steps", "integer", 4, 1, INT_UNBOUND);
   AVSconnect_widget(p, "typein_integer");

   p = AVSadd_parameter("go", "oneshot", 0, 0, 0);
   AVSconnect_widget(p, "oneshot");

   p = AVSadd_parameter("switch fin-init", "oneshot", 0, 0, 0);
   AVSconnect_widget(p, "oneshot");

   p = AVScreate_output_port("counter", "integer");
}

	
/*--------------------------------------------------------------------------
 * Perspective_Zoom_compute
 *--------------------------------------------------------------------------*/

#define EPSILON 1.0e-8

main(argc, argv)
int    argc;
char  *argv[];
{
   int      Perspective_Zoom();

   float   *angle_start, *angle_fin;
   float   *at_x_init, *at_y_init, *at_z_init;
   float   *at_x_fin, *at_y_fin, *at_z_fin;
   double   angle_s, angle_f, x, y, z;
   double   at_x_i, at_y_i, at_z_i;
   double   at_x_f, at_y_f, at_z_f;
   int      n;
   int      go;
   int      change;

   char     com_buff[200], com_ptr;
   char    *out_buff, *err_buff;

   double   s;
   double   angle, del_angle;
   double   at_x, del_at_x;
   double   at_y, del_at_y;
   double   at_z, del_at_z;
   double    a;

   int      i;


   AVScorout_init(argc, argv, Perspective_Zoom);

   AVScorout_set_sync(0);

   while (1)
   {
      AVScorout_wait();

      AVScorout_input(&angle_start, &angle_fin, 
		      &at_x_init, &at_y_init, &at_z_init,
		      &at_x_fin, &at_y_fin, &at_z_fin,
		      &n, &go, &change);

      if(change)
	{
	  angle_f = *angle_start;
	  angle_s = *angle_fin;
 printf ("changing angle_f angle_s %f %f \n",angle_f, angle_s);
	  AVSmodify_float_parameter("FOV fin", AVS_VALUE, angle_f,0,0);
	  AVSmodify_float_parameter("FOV init", AVS_VALUE, angle_s,0,0);
	  at_x_i = *at_x_fin;
	  at_y_i = *at_y_fin;
	  at_z_i = *at_z_fin;
	  at_x_f = *at_x_init;
	  at_y_f = *at_y_init;
	  at_z_f = *at_z_init;
	  AVSmodify_float_parameter("At X init", AVS_VALUE, at_x_i, 0, 0);
	  AVSmodify_float_parameter("At Y init", AVS_VALUE, at_y_i, 0, 0);
	  AVSmodify_float_parameter("At Z init", AVS_VALUE, at_z_i, 0, 0);
	  AVSmodify_float_parameter("At X fin", AVS_VALUE, at_x_f, 0, 0);
	  AVSmodify_float_parameter("At Y fin", AVS_VALUE, at_y_f, 0, 0);
	  AVSmodify_float_parameter("At Z fin", AVS_VALUE, at_z_f, 0, 0);
	}
	 angle_s = *angle_start;
	 angle_f = *angle_fin;
 printf ("new angles angle_f angle_s %f %f \n",angle_f, angle_s);
	  at_x_f = *at_x_fin;
	  at_y_f = *at_y_fin;
	  at_z_f = *at_z_fin;
	  at_x_i = *at_x_init;
	  at_y_i = *at_y_init;
	  at_z_i = *at_z_init;

      if (go && n)
      {
	 del_angle = (angle_f - angle_s)/n;
	 angle = angle_s;

	 del_at_x = (at_x_f - at_x_i)/n;
	 del_at_y = (at_y_f - at_y_i)/n;
	 del_at_z = (at_z_f - at_z_i)/n;
	 at_x = at_x_i;
	 at_y = at_y_i;
	 at_z = at_z_i;

	 for (i=1; i<=n; i++)
	 {
	   a=angle + i*del_angle;
	   x=at_x + i*del_at_x;
	   y=at_y + i*del_at_y;
	   z=at_z + i*del_at_z;

	    sprintf(com_buff,
		    "geom_set_camera_params -fov %lf -at %lf %lf %lf", a, x, y, z); 
	 
	    AVScommand("kernel", com_buff, &out_buff, &err_buff);
	    AVScommand("kernel", "geom_refresh", &out_buff, &err_buff);
      AVScorout_output(i);

	    AVScorout_exec();
	 }
      }
   }
}
