/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

#define AMPS_USE_PVM_BCAST 1

/*
   This should probably be written using Log type alg. since PVM is stupid.
*/

#if AMPS_USE_PVM_BCAST

int amps_BCast(comm, source, invoice) 
amps_Comm comm;
int source;
amps_Invoice invoice;
{

   if(source == amps_Rank(comm))
   {
      amps_pack(comm, invoice);
      pvm_bcast(comm, amps_MsgTag);
   }
   else
   {
      /* in case there is some memory still allocated */
      amps_ClearInvoice(invoice);
      pvm_recv(-1, amps_MsgTag);
      amps_unpack(comm, invoice);
   }
   
   return 0;
}

#else

#define vtor(node) \
     (node > source ? node : ( node ) ? node - 1 : source)

#define rtov(node) \
     (node > source ? node : ( node < source ) ? node + 1 : 0)

int amps_BCast(comm, source, invoice) 
amps_Comm comm;
int source;
amps_Invoice invoice;
{
   int n;
   int N;
   int d;
   int poft, log, npoft;
   int start;
   int startd;
   int node;

   char *in_buffer;
   char *out_buffer;

   int size = 0;
   
   int recvd_flag;

   N = nnodes();
   n = rtov(mynode());

   amps_FindPowers(N, &log, &npoft, &poft);

   start = 0;       /* start of sub "power of two" block we are working on */
   recvd_flag = (n==start);

   if( n == 0 )
   {
       amps_pack(comm, invoice, &in_buffer);
       size = xlength(in_buffer - sizeof(double));
   }
   else
       AMPS_CLEAR_INVOICE(invoice);

   if(n >= poft)
   {
	 node = vtor(n - poft);
	 in_buffer = amps_recvb(node);
   }
   else
   {
      for( d = poft >> 1; d >= 1; d >>=1)
      {

	 startd = start ^ d;
	 
	 if(!recvd_flag && (startd != n))
	 {
	    if( n >= startd )
	       start = startd;
	    continue;
	 }
	 
	 if(recvd_flag)
	 {
	    node = vtor( (n ^ d) );

	    if( (out_buffer = amps_new(comm, size)) == NULL)
		amps_Error("amps_bcast", OUT_OF_MEMORY, "malloc of letter", HALT);
	    memcpy(out_buffer, in_buffer, size-sizeof(double));
	    amps_xsend(out_buffer, node);
	 }
	 else
	 {
	    node = vtor(start);
	    recvd_flag = 1;
	    in_buffer = amps_recvb(node);
	    size = xlength(in_buffer - sizeof(double));
	 }
      }

      if( n < N - poft)
      {
	 node = vtor(poft + n);
	 if( (out_buffer = amps_new(comm, size)) == NULL)
	     amps_Error("amps_bcast", OUT_OF_MEMORY, "malloc of letter", HALT);
	 memcpy(out_buffer, in_buffer, size-sizeof(double));
	 amps_xsend(out_buffer, node);
      }
   }

   if( n == 0 )
   {
#if 0
       AMPS_CLEAR_INVOICE(invoice);
#endif
   }
   else
   {
      amps_unpack(comm, invoice, in_buffer);
   }

   AMPS_PACK_FREE_LETTER(comm, invoice, in_buffer);

   return 0;
}

#endif
