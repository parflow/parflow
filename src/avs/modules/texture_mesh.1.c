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
/*   
     This software is copyright (C) 1992,  Regents  of  the
University  of  California.   Anyone may reproduce texture_mesh.c,
the software in this distribution, in whole or in part, pro-
vided that:

(1)  Any copy  or  redistribution  of  texture_mesh.c  must  show  the
     Regents  of  the  University of California, through its
     Lawrence Berkeley Laboratory, as the source,  and  must
     include this notice;

(2)  Any use of this software must reference this  distribu-
     tion,  state that the software copyright is held by the
     Regents of the University of California, and  that  the
     software is used by their permission.

     It is acknowledged that the U.S. Government has  rights
in  texture_mesh.c  under  Contract DE-AC03-765F00098 between the U.S.
Department of Energy and the University of California.

     texture_mesh.c is provided as a professional  academic  contribu-
tion  for  joint exchange.  Thus it is experimental, is pro-
vided ``as is'', with no warranties of any kind  whatsoever,
no  support,  promise  of updates, or printed documentation.
The Regents of the University of California  shall  have  no
liability  with respect to the infringement of copyrights by
texture_mesh.c, or any part thereof.     

Author:
	Wes Bethel   
	Lawrence Berkeley Laboratory
	1 Cyclotron Rd.   Mail Stop 50-F
	Berkeley CA 94720
	510-486-6626
	ewbethel@lbl.gov
*/

#include <stdio.h>
#include <avs/avs.h>
#include <avs/field.h>
#include <avs/geom.h>
#include <avs/colormap.h>

/**
  * this module is conceptually similar to the AVS-supplied field 2 mesh
  * in that it takes an input field and produces an output quad mesh.
  *
  * there are a number of important distinctions:
  *
  * 1. the input field is of a specific type: a 2d rectilinear field. the
  *    data in the field is assumed to be an elevation (z-value) present
  *    at each grid location.  (modified to take 2d uniform and 2d, 3space
  *    irregular fields, as well)
  *
  * this module provides a way to do texture mapping.  an image is mapped
  * onto the 2d mesh.  the mapping may be done explicitly, where vertex
  * colors are a function of the input image and map parameters.  or
  * the mapping may be done implicitly, where uv coordinates in the output
  * geometry may be set.  this latter way is AVS-specific.
  *
  * w.bethel summer 1992
**/

#define OK 0
#define ERROR 1
    
#define UV_SURF_MIN "Surface UV Min"
#define UV_SURF_MAX "Surface UV Max"
    
mesh_block()
{
    int p,mesh_block_p();

    AVSset_module_name("texture mesh",MODULE_MAPPER);

    p=AVScreate_input_port("input field","field 2D float",REQUIRED);
    p=AVScreate_input_port("texture","field 2D 4-vector byte",OPTIONAL);

    p=AVScreate_output_port("mesh block","geom");

    /* create the texture mapping options buttons. */
    
    p = AVSadd_parameter("dummy1","string","Texture Mapping Options:", "","");
    AVSconnect_widget(p,"text");
    AVSadd_parameter_prop(p,"width","integer",4);

    p = AVSadd_parameter("MapChoice","choice","Dynamic","Dynamic!Explicit","!");
    AVSconnect_widget(p,"radio_buttons");

    /* create the surface uv coordinate parameters. */
    
    p = AVSadd_parameter("dummy2","string","Surface UV Coordinates", "","");
    AVSconnect_widget(p,"text");
    AVSadd_parameter_prop(p,"width","integer",4);

    p = AVSadd_parameter(UV_SURF_MIN,"string","0. 0.",NULL,NULL);
    p = AVSadd_parameter(UV_SURF_MAX,"string","1. 1.",NULL,NULL);
    
    p = AVSadd_parameter("dummy3","string","Explicit Mapping Interpolation", "","");
    AVSconnect_widget(p,"text");
    AVSadd_parameter_prop(p,"width","integer",4);

    p = AVSadd_parameter("InterpChoice","choice","Point","Point!Bilinear","!");
    AVSconnect_widget(p,"radio_buttons");

    AVSset_compute_proc(mesh_block_p);
}

mesh_block_p(inf,texture,outgeom,dummy1,mapchoice,dummy2,uv_surf_min,
	     uv_surf_max,dummy3,interpchoice)
AVSfield *inf;
AVSfield_char *texture;
GEOMedit_list *outgeom;
char *dummy1,*dummy2,*dummy3;
char *mapchoice,*uv_surf_min,*uv_surf_max,*interpchoice;
{
    int uverts,vverts;
    int i,j,index;
    float *top_verts,*t;
    float *uv,u,du,v,dv;
    float *x,*y,*z;
    float dx,dy;
    GEOMobj *top_mesh,*top2;
    float s_umin,s_umax,s_vmin,s_vmax;
    int texture_method;
    unsigned long *colors;
    int interp_method;

    /* first, create the geometry for the mesh object. */
    
    uverts = inf->dimensions[0];
    vverts = inf->dimensions[1];
    
    *outgeom = GEOMinit_edit_list(*outgeom);

    /* start by creating a vertex array */
    
    if (inf->uniform == UNIFORM)
    {
	/**
	  * for uniform fields, we'll just create arrays containing
	  * coordinates.  this makes the code the same as for
	  * rectilinear fields.
	**/
	x = (float *)malloc(sizeof(float)*inf->dimensions[0]);
	y = (float *)malloc(sizeof(float)*inf->dimensions[1]);
	if (inf->points) /* there is extents info in this field. use it. */
	{
	    dx = (*(inf->points+1) - *(inf->points))/(inf->dimensions[0]-1);
	    dy = (*(inf->points+3) - *(inf->points+2))/(inf->dimensions[1]-1);
	}
	else
	    dx = dy = 1.0;
	
	x[0] = *(inf->points);
	for (i=1;i<inf->dimensions[0];i++)
	    x[i] = x[i-1] + dx;
	y[0] = *(inf->points+2);
	for (i=1;i<inf->dimensions[1];i++)
	    y[i] = y[i-1] + dy;
	z = inf->field_union.field_data_float_u;
    }
    else if (inf->uniform == RECTILINEAR)
    {
	x = inf->points;
	y = x+inf->dimensions[0];
	z = inf->field_union.field_data_float_u;
    }
    else /* irregular */
    {
	x = inf->points;
	y = x + inf->dimensions[0]*inf->dimensions[1];
	
	if (inf->nspace == 3)
	    z = y + inf->dimensions[0]*inf->dimensions[1];
	else if (inf->nspace == 2)
	    z = inf->field_union.field_data_float_u;
	else
	{
	    AVSwarning(" Only 2- or 3-space 2D irregular fields are permitted.");
	    return(0);
	}
	    
    }
    
    top_verts = (float *)malloc(sizeof(float)*uverts*vverts*3);
    if (top_verts == NULL)
    {
	AVSwarning(" Unable to malloc space for the mesh vertices.");
	return(0);
    }
    t = top_verts;

    index = 0;

    if (inf->uniform != IRREGULAR)
    {
	for (j=0;j<vverts;j++)
	{
	    for (i=0;i<uverts;i++)
	    {
		*t++ = x[i];
		*t++ = y[j];
		*t++ = z[index++];
	    }
	}
    }
    else
    {
	for (j=0;j<vverts*uverts;j++)
	{
	    *t++ = x[j];
	    *t++ = y[j];
	    *t++ = z[j];
	}
    }

    top_mesh = (GEOMobj *)GEOMcreate_mesh(GEOM_NULL,top_verts,uverts,vverts,GEOM_COPY_DATA);
    GEOMgen_normals(top_mesh,0);
    GEOMedit_geometry(*outgeom,"textured_mesh",top_mesh);
    
    /* next, deal with doing the texture. */

    /* 0 means dynamic, 1 means explicit. */
    texture_method = AVSchoice_number("MapChoice",mapchoice) - 1;
    
    if (texture_method == 0)  /* dynamic */
    {
	uv = (float *)malloc(sizeof(float)*uverts*vverts*2);
	
	if (uv == NULL)
	{
	    AVSwarning(" Unable to malloc space for UV coords. ");
	    return(0);
	}
	t = uv;

	/* validate the surface min/max uv coordinates. */
	if (get_2_floats(uv_surf_min,&s_umin,&s_vmin) == ERROR)
	{
	    AVSerror(" Invalid floating point value for umin or vmin.");
	    return(0);
	}
	
	if (get_2_floats(uv_surf_max,&s_umax,&s_vmax) == ERROR)
	{
	    AVSerror(" Invalid floating point value for umin or vmin.");
	    return(0);
	}
	u = s_umin;
	v = s_vmin;

	du = (s_umax - s_umin)/(uverts-1);
	dv = (s_vmax - s_vmin)/(vverts-1);

	for (j=0;j<vverts;j++,v+=dv)
	{
	    for (u=s_umin,i=0;i<uverts;i++,u+=du)
	    {
		*t++ = u;
		*t++ = v;
	    }
	}
    
	GEOMadd_uvs(top_mesh,uv,uverts*vverts,GEOM_COPY_DATA);

/* IAC CODE CHANGE : 	free((char *)uv); */
	 free(uv);
    }
    else
    {
	if (texture == NULL)
	{
	    AVSerror(" There must be an AVS image on the texture input port for explicit mapping.");
	    return(0);
	}
	
	if (get_2_floats(uv_surf_min,&s_umin,&s_vmin) == ERROR)
	{
	    AVSerror(" Invalid floating point value for umin or vmin.");
	    return(0);
	}
	
	if (get_2_floats(uv_surf_max,&s_umax,&s_vmax) == ERROR)
	{
	    AVSerror(" Invalid floating point value for umin or vmin.");
	    return(0);
	}
	
	interp_method = AVSchoice_number("InterpChoice",interpchoice) - 1;

	if ((interp_method == 1) && (inf->uniform == IRREGULAR))
	{
	    AVSwarning(" Bilinear sampling NOT supported for irregular fields. Using point sampling instead. ");
	    AVSmodify_parameter("InterpChoice",AVS_VALUE,"Point",NULL,NULL);
	    interp_method = 0;
	}

	colors = (unsigned long *)malloc(sizeof(unsigned long)*uverts*vverts);

	if (colors == NULL)
	{
	    AVSwarning(" Unable to malloc space for the explicit texture colors.");
	    return(0);
	}

	compute_colors(colors,texture->data,uverts,vverts,&s_umin,
		       &s_vmin,&s_umax,&s_vmax,interp_method,x,y,
		       texture->dimensions[0],texture->dimensions[1],inf);

	GEOMadd_int_colors(top_mesh,colors,uverts*vverts,GEOM_COPY_DATA);

/* IAC CODE CHANGE : 	free((char *)colors); */
	 free(colors);
    }
	    

    GEOMdestroy_obj(top_mesh);


/* IAC CODE CHANGE :     free((char *)top_verts); */
     free(top_verts);

    if (inf->uniform != IRREGULAR)
    {

/* IAC CODE CHANGE : 	free((char *)x); */
	 free(x);

/* IAC CODE CHANGE : 	free((char *)y); */
	 free(y);
    }
    
    return(1);
    
}

static int
get_2_floats(s,f1,f2)
char *s;
float *f1,*f2;
{
    char *t;
    extern double strtod();
    double foo;

    t = s;
    foo = strtod(s,&t);
    if (t==s)
	return(ERROR);
    *f1 = foo;

    s = t;
    foo = strtod(s,&t);
    if (t==s)
	return(ERROR);
    *f2 = foo;

    return(OK);
}

compute_colors(colors,tex,iu,iv,umin,vmin,umax,vmax,interp_method,xc,yc,
	       texture_iu,texture_iv,inf)
unsigned long *colors;
unsigned char *tex;
int iu,iv;
float *umin,*vmin,*umax,*vmax;
int interp_method;
float *xc,*yc;
int texture_iu,texture_iv;
AVSfield *inf;
{
    /**
      * the texture map consists of argb tuples of unsigned char's
      * (an AVS image)
      *
      * vertex coloring (ie texture mapping) for uniform and rectilinear
      * fields is done using a spatial method.  the texture is mapped
      * uniformly through space on these types of fields.
      *
      * irregular fields, in contrast, do not use a spatial mapping, but
      * rather, a (hack) technique based upon indeces.  in the default
      * case (of mapping the entire image to the entire 2D irregular field)
      * some "rubber sheeting" may occur, where the degree of rubber
      * sheeting depends upon the severity of deformation present in
      * the coordinates of the input field.
      * 
    **/
    
    float *dU,*dV;
    double u_axis_length,v_axis_length;
    int i,j;
    double t;
    float uacc,vacc,u,v;
    float urange,vrange;
    int tu_offset,tv_offset;
    int u_out_of_range,v_out_of_range;
    unsigned long white = 0xffffffff;
    float usamples,vsamples;
    float min_extent[3],max_extent[3];
    extern double fabs();
    
    dU = (float *)malloc(sizeof(float)*(iu-1));
    dV = (float *)malloc(sizeof(float)*(iv-1));

    /* this way of computing u and v axes lengths is valid only for
       rectilinear fields (will work with uniform fields for which we
       provide explicit coordinate values in the x,y arrays).  it will
       not work for irregular fields because 1. we need Z information, and
       2. need use of Pythagoreus formula for computing then summing
       segment lengths. */

    if (inf->uniform != IRREGULAR)
    {
	for (i=0;i<iv-1;i++)
	    dV[i] = yc[i] - yc[i+1];
	
	for (i=0;i<iu-1;i++)
	    dU[i] = xc[i] - xc[i+1];

	u_axis_length = 0.;
	for (i=0;i<iu-1;i++)
	{
	    t = dU[i];
	    t = fabs(t);
	    dU[i] = t;
	    u_axis_length += t;
	}
    
	v_axis_length = 0.;
	for (i=0;i<iv-1;i++)
	{
	    t = dV[i];
	    t = fabs(t);
	    dV[i] = t;
	    v_axis_length += t;
	}
    }

    urange = *umax - *umin;
    vrange = *vmax - *vmin;

    for (vacc=0.,j=0;j<iv;j++)
    {
	if (inf->uniform == IRREGULAR)
	    v = (float)j/(float)(iv-1);
	else
	{
	    v = vacc / v_axis_length;
	    vacc += dV[j];
	}
	
	v = v*vrange + *vmin;
	
	if ((v < 0.) || (v > 1.0))
	    v_out_of_range = 1;
	else
	    v_out_of_range = 0;
	
	
	tv_offset = v*(texture_iv-1);
	tv_offset *= (texture_iu<<2);
	
	for (uacc=0.,i=0;i<iu;i++)
	{
	    if (inf->uniform == IRREGULAR)
		u = (float)i/(float)(iu-1);
	    else
	    {
		u = uacc/u_axis_length;
		uacc += dU[i];
	    }

	    u = u*urange + *umin;

	    tu_offset = u*(texture_iu-1);

	    if ((u < 0.) || (u > 1.0))
		u_out_of_range = 1;
	    else
		u_out_of_range = 0;
	    
	    if (interp_method == 0)
	    {
		/* point sampling. */
		
		if (u_out_of_range || v_out_of_range)
		    memcpy((char *)(colors+j*iu+i),
			   (char *)&white,
			   sizeof(unsigned long));
		else
		{
		    memcpy((char *)(colors+j*iu+i),
			   (char *)(tex+(tu_offset<<2)+tv_offset),
			   sizeof(unsigned long));
		}
	    }
	    else
	    {
		/* compute number of texture samples to use in
		   "antialiasing" the texture at this output point */

		usamples = texture_iu * dU[i] / u_axis_length;
		vsamples = texture_iv * dV[j] / v_axis_length;

		if ((usamples <= 1.) || (vsamples <= 1.))
		{
		    if (u_out_of_range || v_out_of_range)
			memcpy((char *)(colors+j*iu+i),
			       (char *)&white,
			       sizeof(unsigned long));
		    else
			memcpy((char *)(colors+j*iu+i),
			       (char *)(tex+(tu_offset<<2)+tv_offset),
			       sizeof(unsigned long));
		}
		else
		    blend_texture(colors+j*iv+i,tex,iu,iv,usamples,vsamples,tu_offset,tv_offset,texture_iu);
	    }
	}
    }


/* IAC CODE CHANGE :     free((char *)dU);   */
     free(dU);  

/* IAC CODE CHANGE :     free((char *)dV); */
     free(dV);
    return(1);
}

blend_texture(d,tex,iu,iv,useg,vseg,u_offset,v_offset,texture_width)
unsigned long *d;
unsigned char *tex;
int iu,iv;
float useg,vseg;
int u_offset,v_offset;
int texture_width;
{
    int npoints;
    double weight;
    int i,j;
    double r,g,b,t; /* don't worry about opacity */
    int base_offset;
    int ir,ig,ib;

    /**
      * we are assuming that useg and vset are both greater than 1.
      * this corresponds to the case of many texture samples mapping
      * into a destination sample, which is typically the case when
      * we have an output grid upon which we are mapping a dense image.
      *
      * we are performing a quick and dirty integration of colors from
      * the input texture.  fractional input samples are ignored.
      *
      * since we are dealing with rectilinear grids, texture mapping
      * techniques such as Smith's two-pass method are not valid.  we
      * must do an explicit backwards mapping.
    **/
    
    npoints = useg*vseg;
    weight = 1./(double)npoints;

    r = 0.;
    g = 0.;
    b = 0.;

    base_offset = (u_offset<<2) + v_offset;
    
    /* byte ordering in texture is ARGB */

    for (j=0;j<(int)vseg;j++)
    {
	for (i=0;i<(int)useg;i++)
	{
	    r += weight * (double)(tex[base_offset+1+(i<<2)]);
	    g += weight * (double)(tex[base_offset+2+(i<<2)]);
	    b += weight * (double)(tex[base_offset+3+(i<<2)]);
	}
	base_offset += (texture_width << 2);
    }

    ir = (int)r;
    ig = (int)g;
    ib = (int)b;

    *d = (ir << 16) | (ig << 8) | ib;
}

