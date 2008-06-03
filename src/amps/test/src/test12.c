/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/* 
   This is a simple "ring" test.  It send a message from the host
   to all the nodes.

*/

#define V1_len 5
#define V2_len 11
#define V1_stride 7
#define V2_stride 3


#include <stdio.h>
#include "amps.h"

int main (argc, argv)
int argc;
char *argv[];
{
   amps_Invoice invoice;
   amps_Invoice send_invoice;

   int v1_len=V1_len;
   int v2_len=V2_len;
   int v1_stride=V1_stride;
   int v2_stride=V2_stride;
   
   int dim=2;
   int len_array[] = { V1_len, V2_len };
   int stride_array[] = { V1_stride, V2_stride };

   int total_length;

   double *vector;

   int   num;
   int   me;

   int i, j;
   int c;

   char *recvd_string = NULL;
   int length;

   int loop;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      exit(1);
   }
   
   loop=atoi(argv[1]);

   total_length = ((v1_len-1)*v1_stride+1)*v2_len + (v2_stride-1)*(v2_len-1);
   /* Init Vector */
   if( (vector = (double *)calloc(total_length, sizeof(double)))==NULL)
	amps_Printf("Error mallocing vector\n");

   
   for(;loop;loop--)
   {
      /* SGS order of args */
      invoice = amps_NewInvoice("%&.&D(*)", &len_array, &stride_array, 
				dim, vector);
      
      num = amps_Size(amps_CommWorld);

      me = amps_Rank(amps_CommWorld);
      
      if(me)
      {
	 amps_Recv(amps_CommWorld, me-1, invoice);
	 amps_Send(amps_CommWorld, (me+1) % num, invoice);
      }
      else
      {
	 /* Set up the Vector */
	 for(c = 0; c < total_length; c++)
	    vector[c] = c;
	 
	 amps_Send(amps_CommWorld, 1, invoice);
	 
	 /* clear the array */
	 /* Set up the Vector */
	 for(c = 0; c < total_length; c++)
	    vector[c] = 0;
	 
	 amps_Recv(amps_CommWorld, num-1, invoice);
      }
      
      /* check the result */
      for( j = 0; j < v2_len;  j ++)
	 for(i = 0; i < v1_len; i++)
	    if (vector[j*((v1_len-1)*v1_stride+v2_stride) + i*(v1_stride)] 
		!= j*((v1_len-1)*v1_stride+v2_stride) + i*(v1_stride))
	       result = 1;
	    else
	       vector[j*((v1_len-1)*v1_stride+v2_stride) + i*(v1_stride)] = 0.0;
      
      for(c = 0; c < total_length; c++)
	 if(vector[c])
	    result =1;
      
      if(result)
      {
	 amps_Printf("ERROR!!!!! vectors do not match\n");
	 result = 1;
      }
      else
      {
	 amps_Printf("Success\n");
	 result = 0;
      }

      amps_FreeInvoice(invoice);
      
   }

   free(vector);



   amps_Finalize();

   return result;
}

