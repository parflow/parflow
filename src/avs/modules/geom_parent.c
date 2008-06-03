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
/* Module Name: "geom_parent" (Output) (Subroutine)                      */
/* Author: Ed Bender,Greenbelt,                                          */
/* Date Created: Wed Dec 16 16:12:15 1992                                */
/*                                                                       */

#include <stdio.h>
#include <string.h>
#include <avs/avs.h>
#include <avs/port.h>
 
/* *****************************************/
/*  Module Description                     */
/* *****************************************/
int geom_parent_desc()
{

	int in_port, out_port, param;
	extern int geom_parent_compute();

	AVSset_module_name("geom_parent", MODULE_RENDER);

	/* Parameter Specifications                */
	param = AVSadd_parameter("Group Name","string","Group", NULL,NULL);

	param = AVSadd_parameter("Include Current Obj","boolean",0, 0, 1);

	AVSset_compute_proc(geom_parent_compute);

	return(1);
}
 
/* *****************************************/
/* Module Compute Routine                  */
/* *****************************************/
int geom_parent_compute( group_name, include
)
	char *group_name;
	int include;
{

	char obj_buf[2000];
	char *objs[21];
	char *out_buf;
	char *err_buf;
	char obj_number[100]; 
	int i;
	char command[100];
	int obj_toggle[21];


	if (include) {
	  AVSmodify_parameter("Include Current Obj",AVS_VALUE,0,0,1);
	  AVScommand("kernel", "geom_get_cur_obj_name",&out_buf,&err_buf);
	  strcpy(obj_buf,out_buf);
	  if ( strncmp(obj_buf,group_name,(int)strlen(group_name))== 0 
				    || strncmp(obj_buf,"top",3)==0){
	    AVSwarning("Illegal grouping, command ignored");
	  }
	  else {
	    sprintf(command, "geom_set_parent %s",group_name);
	    AVScommand("kernel", command, &out_buf,&err_buf);
	  }
	}
	return(1);
}
 
