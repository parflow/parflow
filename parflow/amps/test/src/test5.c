/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/* 

*/

#include <stdio.h>
#include <stdlib.h>
#include "amps.h"

int sum(x)
int x;
{
    int i, result=0;

    for(i=1; i <=x ; i++)
	result += i;

    return result;
}

int main (argc, argv)
int argc;
char *argv[];
{

   amps_Invoice invoice;

   int test;
   double d_result;
   int i_result;

   int   num;
   int   me;

   int loop, i;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   loop = atoi(argv[1]);


   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   invoice = amps_NewInvoice("%d", &d_result);


   for(i = loop;i; i--)
     {
       /* Test the Max function */
       
       d_result = me + 1;
       
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Max);

       
       if( (d_result != (double)num))
	 {
	   amps_Printf("ERROR!!!!! MAX result is incorrect: %f  %d\n",
		       d_result, i_result);
	   result = 1;
	 }
       else
	 if(me == 0)
	   amps_Printf("Success\n");

       /* Test the Min function */
       
       d_result = me + 1;
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Min);

       
       if( (d_result != (double)1))
	 {
	   amps_Printf("ERROR!!!!! MIN result is incorrect: %f  %d\n",
		       d_result, i_result);
	   result = 1;
	 }
       else
	 if(me == 0)
	   amps_Printf("Success\n");
       
       /* Test the Add function */
       
       d_result = me + 1;
       
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Add);
       
       
       test = sum(num);
       if( (d_result != (double)test) )
	 {
	   amps_Printf("ERROR!!!!! Add result is incorrect: %f  %d want %d\n",
		       d_result, i_result, test);
	   result = 1;
	 }
       else
	 if(me == 0)
	 amps_Printf("Success\n");
     }
       

   amps_FreeInvoice(invoice);

   invoice = amps_NewInvoice("%i%d", &i_result, &d_result);

   for(i = loop;i; i--)
     {
       /* Test the Max function */
       
       d_result = i_result = me + 1;
       
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Max);

       
       if( (d_result != (double)num) || (i_result != num))
	 {
	   amps_Printf("ERROR!!!!! MAX result is incorrect: %f  %d\n",
		       d_result, i_result);
	   result = 1;
	 }
       else
	 if(me == 0)
	   amps_Printf("Success\n");

       /* Test the Min function */
       
       d_result = i_result = me + 1;
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Min);

       
       if( (d_result != (double)1) || (i_result != 1))
	 {
	   amps_Printf("ERROR!!!!! MIN result is incorrect: %f  %d\n",
		       d_result, i_result);
	   result = 1;
	 }
       else
	 if(me == 0)
	   amps_Printf("Success\n");
       
       /* Test the Add function */
       
       d_result = i_result = me + 1;
       
       
       amps_AllReduce(amps_CommWorld, invoice, amps_Add);
       
       
       test = sum(num);
       if( (d_result != (double)test) || (i_result != test))
	 {
	   amps_Printf("ERROR!!!!! Add result is incorrect: %f  %d want %d\n",
		       d_result, i_result, test);
	   result = 1;
	 }
       else
	 if(me == 0)
	 amps_Printf("Success\n");
     }
       

   amps_FreeInvoice(invoice);

   amps_Finalize();

   return result;
}

