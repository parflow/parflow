
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

 AUTHOR : Lawrence Berkeley Laboratory
******************************************************************************/
/*********************************************************************

Copyright (C) 1991,1992,1993,1994, Lawrence Berkeley Laboratory.  All Rights
Reserved.  Permission to copy and modify this software and its
documentation (if any) is hereby granted, provided that this notice
is retained thereon and on all copies.  

This software is provided as a professional academic contribution
for joint exchange.   Thus it is experimental and scientific
in nature, undergoing development, and is provided "as is" with
no warranties of any kind whatsoever, no support, promise of
updates or printed documentation.

This work is supported by the U. S. Department of Energy under 
contract number DE-AC03-76SF00098 between the U. S. Department 
of Energy and the University of California.


	Author: Wes Bethel
		Lawrence Berkeley Laboratory

  "this software is 100% hand-crafted by a human being in the USA"

*********************************************************************/



/**
  *  this module is intended to replace the collage module sent out
  *  with the hips distribution.  noteworthy changes:
  *
  *  1. image x,y offsets are user-controllable for each of the two
  *     input images.  the offset specifies delta in pixels from the
  *     upper left-hand corner of the output image.
  *
  *  2. the standard collage module takes an arbitrary number of input
  *     images.  this module takes just two.
  *  3. a dial is available for setting the "blank" areas of the output
  *     image to a shade of gray.
  *  4. regular r,b,g images are allowed, as are single byte gray images,
  *     or any other 2d byte field (of arbitrary vector length).
  *
  *  17 jan 1991   wes bethel, lawrence berkeley laboratory
  *  30 sep 1994   w.bethel, avoided bug in AVS whereby a % character in a
  *      parm name string causes AVS to dump core by changing a parm's name.
**/

#include <stdio.h>
#include <memory.h>
#include <avs/avs.h>
#include <avs/field.h>

collage()
{
    int collage_p();
    int p;

    AVSset_module_name("collage",MODULE_FILTER);
    AVScreate_input_port("img 1","field uniform 2D byte",REQUIRED);
    AVScreate_input_port("img 2","field uniform 2D byte",REQUIRED);
    AVScreate_output_port("new image","field uniform 2D byte");
    AVSadd_float_parameter("Image 1 X-offset",0.,0.,FLOAT_UNBOUND);
    AVSadd_float_parameter("Image 1 Y-offset",0.,0.,FLOAT_UNBOUND);
    AVSadd_float_parameter("Image 2 X-offset",0.,0.,FLOAT_UNBOUND);
    AVSadd_float_parameter("Image 2 Y-offset",0.,0.,FLOAT_UNBOUND);
#if 0    
    AVSadd_float_parameter("Gray Fill %",0.,0.,1.);
#endif
    AVSadd_float_parameter("Gray Fill Pct.",0.,0.,1.);
    AVSset_compute_proc(collage_p);
}

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

collage_p(img1,img2,outf,img1_x,img1_y,img2_x,img2_y,pct_gray)
AVSfield_char *img1,*img2,**outf;
float *img2_x,*img2_y,*img1_x,*img1_y,*pct_gray;
{
    AVSfield_char template;
    int dims[2];
    int xsize,ysize;

    /**
      * error checking:
      * make sure that both input images have same veclen.
    **/
    if (img1->veclen != img2->veclen)
    {
	AVSwarning(" Input images must have the same vector length.");
	return(0);
    }
    /**
      *  size of resulting image is calculated as:
      *  xsize = max((img1->xsize + img1_x_offset),(img2->xsize+img2_x_offset))
      *  ysize = max((img1->ysize + img2_y_offset),(img2->ysize+img2_y_offset))
    **/

    xsize = img1->dimensions[0] + (int)*img1_x;
    xsize = MAX(xsize,img2->dimensions[0]+ (int)*img2_x);

    ysize = img1->dimensions[1] + (int)*img1_y;
    ysize = MAX(ysize,img2->dimensions[1] + (int)*img2_y);

    memset((char *)&template,0,sizeof(AVSfield_char));

    template.ndim=2;
    template.nspace = 0;
    template.uniform = UNIFORM;
    template.veclen = img1->veclen;
    template.size = sizeof(char);
    template.type = AVS_TYPE_BYTE;
    dims[0] = xsize;
    dims[1] = ysize;
    if (*outf)
	AVSfield_free(*outf);
    *outf = (AVSfield_char *)AVSfield_alloc(&template,dims);
    if (*outf == NULL)
    {
	AVSwarning(" Field malloc error.");
	return(0);
    }

    fill_background(*outf,xsize,ysize,pct_gray);
    
    copy_to_out(img1,*outf,xsize,ysize,img1_x,img1_y);
    copy_to_out(img2,*outf,xsize,ysize,img2_x,img2_y);

    return(1);
}

fill_background(outf,xsize,ysize,pct_gray)
AVSfield_char *outf;
int xsize,ysize;
float *pct_gray;
{
    int i,count;
    unsigned char *dest;
    unsigned char color;

    count = xsize*ysize*outf->veclen;

    color = *pct_gray * 255;
    dest = outf->data;

    memset((char *)dest,(int)color,count);
#if 0
    /* use this if you don't have memset */
    for (i=0;i<count;i++)
	*dest++ = color;
#endif
}

copy_to_out(img1,outf,xsize,ysize,dx,dy)
AVSfield_char *img1,*outf;
int xsize,ysize;
float *dx,*dy;
{
    int i,j,k,count;
    unsigned char *dest,*src;
    int x_offset,y_offset;
    int veclen;

    veclen = img1->veclen;
    x_offset = (int)(*dx);
    y_offset = (int)(*dy);

    src = img1->data;
    for (j=0;j<img1->dimensions[1];j++)
    {
	dest = outf->data + veclen*xsize*j + veclen*xsize*y_offset + veclen*x_offset;
	count = img1->dimensions[0]*img1->veclen; 
	memcpy((char *)dest,(char *)src,count);
	src += img1->dimensions[0]*img1->veclen;
#if 0
	/* use this if you don't have memcpy */
	for (i=0;i<img1->dimensions[0];i++)
	    for (k=0;k<img1->veclen;k++)
		*dest++ = *src++;
#endif
    }
}

