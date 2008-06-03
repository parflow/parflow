/****************************************************************************
                  INTERNATIONAL AVS CENTER
	(This disclaimer must remain at the top of all files)

WARRANTY DISCLAIMER

This module and the files associated with it are distributed free of charge.
It is placed in the public domain and permission is granted for anyone to use,
duplicate, modify, and redistribute it unless otherwise noted.  Some modules
may be copyrighted.  You agree to abide by the conditions also included in
the AVS Licensing Agreement, version 1.0, located in the main module
directory located at the International AVS Center ftp site and to include
the AVS Licensing Agreement when you distribute any files downloaded from 
that site.

The International AVS Center, MCNC, the AVS Consortium and the individual
submitting the module and files associated with said module provide absolutely
NO WARRANTY OF ANY KIND with respect to this software.  The entire risk as to
the quality and performance of this software is with the user.  IN NO EVENT
WILL The International AVS Center, MCNC, the AVS Consortium and the individual
submitting the module and files associated with said module BE LIABLE TO
ANYONE FOR ANY DAMAGES ARISING FROM THE USE OF THIS SOFTWARE, INCLUDING,
WITHOUT LIMITATION, DAMAGES RESULTING FROM LOST DATA OR LOST PROFITS, OR ANY
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

This AVS module and associated files are public domain software unless
otherwise noted.  Permission is hereby granted to do whatever you like with
it, subject to the conditions that may exist in copyrighted materials. Should
you wish to make a contribution toward the improvement, modification, or
general performance of this module, please send us your comments:  why you
liked or disliked it, how you use it, and most important, how it helps your
work. We will receive your comments at avs@ncsc.org.

Please send AVS module bug reports to avs@ncsc.org.

******************************************************************************/
/* Analog Clock Module  --------   Lars M Bishop                    */
/*                                 National Center for              */
/*     Version 2.0                 Computational Electronics        */
/*                                 Beckman Institute                */
/*                                 University of Illinois at UC     */
/*                                 bishop@viz1.ceg.uiuc.edu         */
/*                                 lmb@cs.brown.edu                 */


#include <stdio.h>
#include <avs/flow.h>
#include <avs/avs_data.h>
#include <avs/field.h>
#include <avs/geom.h>
#include <avs/colormap.h>
/* IAC CODE CHANGE : #include <math.h> */
/* IAC CODE CHANGE : #include <avs/avs_math.h> */
/* IAC CODE CHANGE : #include <avs/avs_math.h> */
/* IAC CODE CHANGE : #include <avs/avs_math.h> */
/* IAC CODE CHANGE : #include <avs/avs_math.h> */
#include <avs/avs_math.h>


void AC_HSV_to_RGB(float *, float *, float *, float, float, float);


int AC_Analog_Clock()
{
  int AC_Analog_Clock_compute();

  
  /* Set the module name and type */
  AVSset_module_name("Analog Clock", MODULE_MAPPER);
  
  AVScreate_input_port("time","integer",REQUIRED);
  AVScreate_input_port("colors","colormap",OPTIONAL);
  
  AVSadd_parameter("Cycle Length","integer",100,2,INT_UNBOUND);
  AVSadd_parameter("# of Polygons","integer",50,4,1000);
  AVSadd_parameter("Max Value","integer",1000,0,INT_UNBOUND);
  AVSadd_parameter("Pie Clock","boolean",0,0,1);
  AVSadd_parameter("Chaining On","boolean",0,0,1);
  
  /* Create an output port for the result */
    AVScreate_output_port("Geometry", "geom");
    AVScreate_output_port("Chaining", "integer");
  
  /* Tell avs what subroutine to call to do the compute */
  AVSset_compute_proc(AC_Analog_Clock_compute);
}

int AC_Analog_Clock_compute(int time, AVScolormap *colormap, GEOMedit_list *output, 
			    int *chain, int steps, int resolution, int max, int pie,
			    int chaining)
{  
  GEOMobj *obj;
  float theta, max_theta, delta, old_theta;
  float *verts, *colors;
  int *plist;
  int index, flags, cindex;
  float color, r, g, b;

  delta = 2.0*M_PI/((float)steps);
  max_theta = (time % steps) * delta;

  *chain = time/steps;

  if(pie)
    {
      AVSparameter_visible("# of Polygons",TRUE);

      if(chaining)
	{
	  AVSparameter_visible("Max Value",FALSE);
	  
	  r = 0.75; g = 0.75; b = 0.75;
	}
      else
	{
	  AVSparameter_visible("Max Value",TRUE);

	  /* calculate the color value */
	  color = ((float)time / (float)max) - floor((float)time / (float)max);
	  color = (color > 1.0) ? 1.0 :  color;
	  
	  if(colormap)
	    {
	      cindex = ((float) (colormap->size)) * color;
	      cindex = (cindex>=colormap->size) ? colormap->size-1 : cindex; 
	      AC_HSV_to_RGB(&r,&g,&b,colormap->hue[cindex],
			    colormap->saturation[cindex],
			    colormap->value[cindex]);
	    }
	  else
	    {
	      r = color; g = 0.25; b = 1.0-color;
	    }
	}

      /* the size of the wedges used to make the "smooth" pie */
      delta = 2.0*M_PI/((float)resolution);
      
      /* space for vertices */
      verts = malloc(sizeof(float)*3*4);
      colors = malloc(sizeof(float)*3*4);
      
      *output = GEOMinit_edit_list(*output);

      /* dummy objects to clear away any parts of the other mode of clock
	 that may be left over */
      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      GEOMedit_geometry(*output,"slow_hand",obj);
      GEOMdestroy_obj(obj);

      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      GEOMedit_geometry(*output,"fast_hand",obj);
      GEOMdestroy_obj(obj);

      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);

      /* the color of the current pie */
      colors[0]   = r; colors[1]   = g; colors[2]   = b;
      colors[3+0] = r; colors[3+1] = g; colors[3+2] = b;
      colors[6+0] = r; colors[6+1] = g; colors[6+2] = b;
      colors[9+0] = r; colors[9+1] = g; colors[9+2] = b;
      
      /* if a pie needs to be drawn at all, draw it */
      if((time%steps)!=0 && (delta<=max_theta))
	{
	  /* some of the vertex positions are constant - factor them out */
	  verts[3+2] = 0.0; verts[6+2] = 0.0;
	      
	  /* loops to create all of the little segments of the pie */
	  for(theta=delta, old_theta=0; 
	      theta<=max_theta; 
	      old_theta = theta, theta+=delta)
	    {	      
	      /* front face polygon */
	      verts[0] = 0.0; verts[1] = 0.0; verts[2] = 0.0;
	      verts[3+0] = sin(old_theta); verts[3+1] = cos(old_theta);
	      verts[6+0] = sin(theta); verts[6+1] = cos(theta);
	      if(!chaining)
		GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,3,
					 GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	      else
		GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,3,
					 GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	     
	      /* side face polygon */
	      verts[0] = verts[3+0]; verts[1] = verts[3+1]; verts[2] = -1.0;
	      verts[9+0] = verts[6+0]; verts[9+1] = verts[6+1]; verts[9+2] = -1.0;
	      if(!chaining)
		GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,4,
					 GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	      else
		GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,4,
					 GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);		
	    }
	  
	  /* end cap (0 degrees) */
	  verts[0] = 0.0; verts[1] = 0.0; verts[2] = -1.0;
	  verts[3+0] = 0.0; verts[3+1] = 0.0; verts[3+2] = 0.0;
	  if(!chaining)
	    GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,4,
				     GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  else
	    GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,4,
				     GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  
	  /* end cap (max degrees) */
	  verts[6+0] = 0.0; verts[6+1] = 1.0; verts[6+2] = 0.0;
	  verts[9+0] = 0.0; verts[9+1] = 1.0; verts[9+2] = -(1.0);
	  if(!chaining)
	    GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,4,
				     GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  else
	    GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,4,
				     GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  
	  /* add the pie to the edit list */
	  GEOMgen_normals(obj,GEOM_FACET_NORMALS);
	}
      else
	obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
	  
      GEOMedit_geometry(*output,"Analog_Clock_pie",obj);
      GEOMdestroy_obj(obj);
      
      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      
      /* vertices for the back plane */
      verts[0] = -1.25; verts[1] = -1.25; verts[2] = -1.0;
      verts[3+0] = 1.25; verts[3+1] = -1.25; verts[3+2] = -1.0;
      verts[6+0] = 1.25; verts[6+1] = 1.25; verts[6+2] = -1.0;
      verts[9+0] = -1.25; verts[9+1] = 1.25; verts[9+2] = -1.0;
      GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,4,
			       GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
      
      /* add back plane to edit list */
      GEOMgen_normals(obj,GEOM_FACET_NORMALS);
      GEOMedit_geometry(*output,"Analog_Clock_back",obj);
      GEOMdestroy_obj(obj);
      
      /* create a parent object */
      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      GEOMedit_geometry(*output,"Analog_Clock",obj);
      GEOMdestroy_obj(obj);
      
      /* add the parts to the parent object */
      GEOMedit_parent(*output,"Analog_Clock_pie","Analog_Clock");
      GEOMedit_parent(*output,"Analog_Clock_back","Analog_Clock");	  
      
      
      /* free the allocated space */

/* IAC CODE CHANGE :       free(colors); free(verts); */

/* IAC CODE CHANGE :        free(colors); free(verts); */

/* IAC CODE CHANGE :         free(colors); free(verts); */

/* IAC CODE CHANGE :          free(colors); free(verts); */

/* IAC CODE CHANGE :           free(colors); free(verts); */
           free(colors); free(verts);
    }
  else
    {
      if(chaining)
	AVSparameter_visible("Max Value",FALSE);
      else
	AVSparameter_visible("Max Value",TRUE);

      AVSparameter_visible("# of Polygons",FALSE);

      *output = GEOMinit_edit_list(*output);

      
      /* dummy objects to clear away any parts of the other mode of clock
	 that may be left over */
      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      GEOMedit_geometry(*output,"Analog_Clock_pie",obj);
      GEOMdestroy_obj(obj);
      
      /* space for vertices */
      colors = malloc(sizeof(float)*3*3);
      verts = malloc(sizeof(float)*3*3);

      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);

      /* the color of the hands is constant */
      colors[0]=colors[1]=0.5; colors[2]=1.0;
      colors[3+0]=colors[3+1]=0.5; colors[3+2]=1.0;
      colors[6+0]=colors[6+1]=0.5; colors[6+2]=1.0;
      
      /* the vertices for the fast hand */
      verts[0] = sin(max_theta); verts[1] = cos(max_theta); verts[2] = 0.0;
      verts[3+0] = sin(max_theta-M_PI/2.0)/12.0;
      verts[3+1] = cos(max_theta-M_PI/2.0)/12.0;
      verts[3+2] = 0.0;
      verts[6+0] = verts[6+1] = 0.0; verts[6+2] = 1.0/7.0;
      GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,3,
			       GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
      
      verts[0] = sin(max_theta); verts[1] = cos(max_theta); verts[2] = 0.0;
      verts[3 + 0] = sin(max_theta+M_PI/2.0)/12.0;
      verts[3 + 1] = cos(max_theta+M_PI/2.0)/12.0;
      verts[3 + 2] = 0.0;
      verts[6+0] = verts[6+1] = 0.0; verts[6+2] = 1.0/7.0;
      GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,3,
			       GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
      
      /* add the fast hand to the edit list */
      GEOMgen_normals(obj,GEOM_FACET_NORMALS);
      GEOMedit_geometry(*output,"fast_hand",obj);
      GEOMdestroy_obj(obj);     




      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);

      if(!chaining)
	{
	  /* calculate the angle of the slow hand */
	  max_theta = ((float)time / (float)max) * 2.0*M_PI;
	  
	  /* the vertex positions for the slow hand */
	  verts[0] = sin(max_theta)/2.0; 
	  verts[1] = cos(max_theta)/2.0; 
	  verts[2] = 0.0;
	  verts[3+0] = sin(max_theta-M_PI/2.0)/7.0;
	  verts[3+1] = cos(max_theta-M_PI/2.0)/7.0;
	  verts[3+2] = 0.0;
	  verts[6+0] = verts[6+1] = 0.0; verts[6+2] = 1.0/7.0;
	  GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,3,
				   GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  
	  verts[0] = sin(max_theta)/2.0; 
	  verts[1] = cos(max_theta)/2.0;
	  verts[2] = 0.0;
	  verts[3 + 0] = sin(max_theta+M_PI/2.0)/7.0;
	  verts[3 + 1] = cos(max_theta+M_PI/2.0)/7.0;
	  verts[3 + 2] = 0.0;
	  verts[6+0] = verts[6+1] = 0.0; verts[6+2] = 1.0/7.0;
	  GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,colors,3,
				   GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
	  
	  /* add the slow hand to the edit list */
	  GEOMgen_normals(obj,GEOM_FACET_NORMALS);
	}

      GEOMedit_geometry(*output,"slow_hand",obj);
      GEOMdestroy_obj(obj);     


      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      
      /* vertices for the back plane */
      verts[0] = -1.25; verts[1] = -1.25; verts[2] = 0.0;
      verts[3+0] = 1.25; verts[3+1] = -1.25; verts[3+2] = 0.0;
      verts[6+0] = 1.25; verts[6+1] = 1.25; verts[6+2] = 0.0;
      verts[9+0] = -1.25; verts[9+1] = 1.25; verts[9+2] = 0.0;
      GEOMadd_disjoint_polygon(obj,verts,GEOM_NULL,GEOM_NULL,4,
			       GEOM_CONVEX | GEOM_SHARED,GEOM_COPY_DATA);
      
      /* add the back plane */
      GEOMgen_normals(obj,GEOM_FACET_NORMALS);
      GEOMedit_geometry(*output,"Analog_Clock_back",obj);
      GEOMdestroy_obj(obj);     

      /* create a parent object */
      obj = GEOMcreate_obj(GEOM_POLYHEDRON, GEOM_NULL);
      GEOMedit_geometry(*output,"Analog_Clock",obj);
      GEOMdestroy_obj(obj);
      
      /* add the parts to the parent object */
      GEOMedit_parent(*output,"fast_hand","Analog_Clock");
      GEOMedit_parent(*output,"slow_hand","Analog_Clock");
      GEOMedit_parent(*output,"Analog_Clock_back","Analog_Clock");
	  
      /* free allocated space */

/* IAC CODE CHANGE :       free(colors); free(verts); */

/* IAC CODE CHANGE :        free(colors); free(verts); */

/* IAC CODE CHANGE :         free(colors); free(verts); */

/* IAC CODE CHANGE :          free(colors); free(verts); */

/* IAC CODE CHANGE :           free(colors); free(verts); */
           free(colors); free(verts);
    }


  return(1);
}




/**********************************************************
   Following hsv_to_rgb routine obtained from field2mesh
   module written by:   
	Wes Bethel   
	Lawrence Berkeley Laboratory
	1 Cyclotron Rd.   Mail Stop 50-F
	Berkeley CA 94720
	415-486-6626
	ewbethel@lbl.gov

     This software is copyright (C) 1991,  Regents  of  the
University  of  California.   Anyone may reproduce field2mesh2.c,
the software in this distribution, in whole or in part, pro-
vided that:

(1)  Any copy  or  redistribution  of  field2mesh2.c  must  show  the
     Regents  of  the  University of California, through its
     Lawrence Berkeley Laboratory, as the source,  and  must
     include this notice;

(2)  Any use of this software must reference this  distribu-
     tion,  state that the software copyright is held by the
     Regents of the University of California, and  that  the
     software is used by their permission.

     It is acknowledged that the U.S. Government has  rights
in  field2mesh2.c  under  Contract DE-AC03-765F00098 between the U.S.
Department of Energy and the University of California.

     field2mesh2.c is provided as a professional  academic  contribu-
tion  for  joint exchange.  Thus it is experimental, is pro-
vided ``as is'', with no warranties of any kind  whatsoever,
no  support,  promise  of updates, or printed documentation.
The Regents of the University of California  shall  have  no
liability  with respect to the infringement of copyrights by
field2mesh2.c, or any part thereof.     
*************************************************************/



void AC_HSV_to_RGB(float *r,float *g,float *b,float h,float s,float v)
{
    float f, p, q, t;
    float ht;
    int i;
 
/*  Make sure not to trash the input colormap */
    ht = h;

    if (v == 0.0)
    {
	*r=0.0;
	*g=0.0;
	*b=0.0;
    }
    else
    {
	if (s == 0.0)
	{
	    *r = v;
	    *g = v;
	    *b = v;
	}
	else
	{
	    ht = ht * 6.0;
	    if (ht >= 6.0)
		ht = 0.0;
      
	    i = ht;
	    f = ht - i;
	    p = v*(1.0-s);
	    q = v*(1.0-s*f);
	    t = v*(1.0-s*(1.0-f));
      
 	    if (i == 0) 
	    {
		*r = v;
		*g = t;
		*b = p;
	    }
	    else if (i == 1)
	    {
		*r = q;
		*g = v;
		*b = p;
	    }
	    else if (i == 2)
	    {
		*r = p;
		*g = v;
		*b = t;
	    }
	    else if (i == 3)
	    {
		*r = p;
		*g = q;
		*b = v;
	    }
	    else if (i == 4)
	    {
		*r = t;
		*g = p;
		*b = v;
	    }
	    else if (i == 5)
	    {
		*r = v;
		*g = p;
		*b = q;
	    }
	}
    }

}



