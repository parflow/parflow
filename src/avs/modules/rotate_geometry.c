/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * RotateGeom
 *
 * AVS coroutine to rotate images in the Geometry Viewer.
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include <avs.h>


/*--------------------------------------------------------------------------
 * RotateGeom
 *--------------------------------------------------------------------------*/

RotateGeom()
{
   int     RotateGeom_compute();
   int     p;


   AVSset_module_name("rotate geometry", MODULE_DATA);

   p = AVSadd_float_parameter("angle", 360.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("x", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("y", 0.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_float_parameter("z", 1.0, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   p = AVSadd_parameter("segs", "integer", 36, 1, INT_UNBOUND);
   AVSconnect_widget(p, "typein_integer");

   p = AVSadd_parameter("go", "oneshot", 0, 0, 0);
   AVSconnect_widget(p, "oneshot");
}

	
/*--------------------------------------------------------------------------
 * RotateGeom_compute
 *--------------------------------------------------------------------------*/

#define EPSILON 1.0e-8

main(argc, argv)
int    argc;
char  *argv[];
{
   int      RotateGeom();

   float   *angle_p, *x_p, *y_p, *z_p;
   double   angle, x, y, z;
   int      n;
   int      go;

   char     com_buff[200], com_ptr;
   char    *out_buff, *err_buff;

   double   s;
   double   d1, d2;
   double   ax, ay, az;

   int      i;


   AVScorout_init(argc, argv, RotateGeom);

   AVScorout_set_sync(0);

   while (1)
   {
      AVScorout_wait();

      AVScorout_input(&angle_p, &x_p, &y_p, &z_p, &n, &go);

      if (go && n)
      {
	 angle = *angle_p;
	 x     = *x_p;
	 y     = *y_p;
	 z     = *z_p;

	 s  = 180.0 / acos(-1.0);
	 d1 = sqrt(y*y + z*z);
	 d2 = sqrt(x*x + y*y + z*z);

	 if (d1 < EPSILON)
	    ax = 90.0;
	 else
	    ax = asin(y/d1)*s;

	 if (d2 < EPSILON)
	    ay = 0.0;
	 else
	    ay = asin(x/d2)*s;

	 az = angle/n;

	 for (i=1; i<=n; i++)
	 {
	    sprintf(com_buff,
		    "geom_concat_matrix -rx %lf -ry %lf -rz %lf -ry %lf -rx %lf", ax, ay, az, -ay, -ax);
	 
	    AVScommand("kernel", com_buff, &out_buff, &err_buff);
	    AVScommand("kernel", "geom_refresh", &out_buff, &err_buff);

	    AVScorout_exec();
	 }
      }
   }
}
