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

#include <stdio.h>
#include "amps.h"

#define v1_len 5
#define v2_len 4
#define v3_len 3
#define v1_stride 5
#define v2_stride 4
#define v3_stride 2

int main (argc, argv)
int argc;
char *argv[];
{
   amps_Invoice invoice;
   amps_Invoice send_invoice;

   int dim=3;
   int len_array[] = { v1_len, v2_len, v3_len };
   int stride_array[] = { v1_stride, v2_stride, v3_stride };

   int total_length;

   int loop;

   double *vector;

   int   num;
   int   me;

   int i, j, k;
   int index;
   int c;

   char *recvd_string = NULL;
   int length;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      exit(1);
   }

   loop=atoi(argv[1]);


   total_length = ((v1_len-1)*v1_stride+1)*v2_len + (v2_stride-1)*(v2_len-1);
   total_length = total_length * v3_len + (v2_stride-1) * (v3_len-1);
   /* Init Vector */
   vector = calloc(total_length, sizeof(double));

   
   /* SGS order of args */
   invoice = amps_NewInvoice("%&.&D(*)", &len_array, &stride_array, 
			     dim, vector);
   
   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   for(;loop;loop--)
   {

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
      for( k = 0; k < v3_len;  k ++)
      {
	 for( j = 0; j < v2_len;  j ++)
	 {
	    for(i = 0; i < v1_len; i++)
	    {
	       index = k * 
		  (((v1_len-1)*v1_stride)*v2_len + v2_stride*(v2_len-1) 
		   + v3_stride)
		     + j*((v1_len-1)*v1_stride+v2_stride) + i*(v1_stride); 
	       
	       if (vector[index] != index )
		  result = 1;
	       else
		  vector[index] = 0.0;
	    }
	    
	 }
      }
	 

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
      
   }
   free(vector);

   amps_FreeInvoice(invoice);

   amps_Finalize();

   return result;
}

