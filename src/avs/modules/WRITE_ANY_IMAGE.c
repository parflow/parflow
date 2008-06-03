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
#include <stdio.h>
#include <avs/avs.h>
#include <avs/port.h>
#include <avs/field.h>
#include "im.h"
 
/* This module is based upon the SDSC image tool library */
/* Written May 18,1992 by 
	 Terry Myerson
	 International AVS Center 
	 North Carolina Supercomputing Center 

   Modifications:
   -------------
   11/23/92  SRT  added ImVfbFree call to free up memory 
*/

#define CHOICE_LIST "eps:gif:hdf:icon:iff:mpnt:pbm:pcx:pgm:pic:pict:pix:pnm:ps:ras:rgb:rla:rle:rpbm:rpgm:rpnm:rppm:synu:tiff:x:xbm:xwd"

/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int write_ANY_image_desc()
{

	int in_port, out_port, param;
	extern int write_ANY_image_compute();

	AVSset_module_name("WRITE ANY IMAGE", MODULE_RENDER);

	/* Output Port Specifications              */
	in_port = AVScreate_input_port("image", 
		"field 2D 4-vector uniform byte",REQUIRED);

	/* Parameter Specifications                */
        param = AVSadd_parameter("WRITE ANY IMAGE browser","string",NULL,NULL,NULL);
	AVSconnect_widget(param, "browser");

	param = AVSadd_parameter("image type","choice","x",CHOICE_LIST,":");
	AVSconnect_widget(param, "radio_buttons");
	AVSadd_parameter_prop(param,"columns","integer",2);


	AVSset_compute_proc(write_ANY_image_compute);
	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int write_ANY_image_compute( image, filename,image_type)
	AVSfield_char *image;
	char *filename,*image_type;
{
	FILE *image_file;
	ImVfb *image_buffer;
	TagTable *dataTable;
	TagEntry *imageEntry;
	ImVfbPtr ptr;
	int i,j;

        if (!filename) return 0;

	if ( (image_file = fopen ( filename,"w" )) == NULL) 
	{
	   fprintf(stderr,"WRITE ANY IMAGE: Error opening filename %s\n",filename);
	   return 0;
        }

        image_buffer=ImVfbAlloc(MAXX(image),MAXY(image),IMVFBRGB | IMVFBALPHA);
	for (j=0;j<MAXY(image);j++)
	{

#ifdef DEBUG
   fprintf(stderr,"Converting Scanline %d\n",j);
#endif

           for (i=0;i<MAXX(image);i++)
	   {
	      ptr=ImVfbQPtr(image_buffer,i,j);
	      ImVfbSRed(image_buffer,ptr,I2DV(image,i,j)[AVS_RED_BYTE]);
	      ImVfbSBlue(image_buffer,ptr,I2DV(image,i,j)[AVS_BLUE_BYTE]);
	      ImVfbSGreen(image_buffer,ptr,I2DV(image,i,j)[AVS_GREEN_BYTE]);
	      ImVfbSAlpha(image_buffer,ptr,I2DV(image,i,j)[AVS_ALPHA_BYTE]);
	   }
        }

        dataTable=TagTableAlloc();
	TagTableAppend (dataTable,
	   TagEntryAlloc("image vfb",POINTER,&image_buffer));

	if ((ImFileFWrite (image_file,image_type,NULL,dataTable )) == -1)
	{
	   fprintf(stderr,"ERROR in ImFileFRead\n");
	   return 0;
	}

        fclose(image_file);
	ImVfbFree(image_buffer);  /* added 11/23/92 SRT */

	return(1);
}
 

