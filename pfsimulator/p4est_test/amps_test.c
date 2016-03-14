#include <stdio.h>
#include <parflow.h>

int
main(int argc, char **argv)
{
  int   dim,i, me, comm_size;
  double *array_s,*array_r;
  int   len[3] = {3,4,1};
  int   str[3];
  amps_Invoice send_invoices[2];
  amps_Invoice recv_invoices[2];
  amps_Package package;
  amps_Handle  handle;
  int   src[2];
  int   dest[2];

  if (amps_Init(&argc, &argv))
    {
      amps_Printf("Error amps_Init\n");
      exit(1);
    }

  me = amps_Rank(amps_CommWorld);
  comm_size = amps_Size(amps_CommWorld);

  array_s  = amps_CTAlloc(double,58);
  array_r  = amps_CTAlloc(double,58);
  dim = 2;

  if (me == 0){
      str[0] = 2, str[1] = 3, str[2]=1;
      for(i=0;i<58;i++){
          array_s[i] = i;
        }
      send_invoices[0] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s);
      recv_invoices[0] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r);
      send_invoices[1] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s + 29);
      recv_invoices[1] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r + 29);

    }else{
      for(i=0;i<58;i++){
          array_s[i] = me;
        }
      str[0] = str[1] = str[2] = 1;
      send_invoices[0] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s);
      recv_invoices[0] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r);
      send_invoices[1] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s+29);
      recv_invoices[1] = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r+29);
    }

  src[0] = src[1]  = (me == 0) ? (comm_size - 1) : (me - 1);
  dest[0] = dest[1] = (me + 1) % comm_size;


  /* communication using amps packages*/
  package = amps_NewPackage(amps_CommWorld,
                            2, &dest[0], &send_invoices[0],
                            2, &src[0],  &recv_invoices[0]);
  handle = amps_IExchangePackage(package);
  amps_Wait(handle);
  amps_FreePackage(package);

  printf("Rank %i received as message 1: \n", me);
  for(i=0;i<29;i++){
      printf( "%0.0f ", array_r[i]);
    }
  printf("\n");

  printf("Rank %i received as message 2: \n", me);
  for(i=0;i<29;i++){
      printf( "%0.0f ", array_r[i + 29]);
    }
  printf("\n");

  amps_TFree(array_r);
  amps_TFree(array_s);
  amps_FreeInvoice(send_invoices[0]);
  amps_FreeInvoice(send_invoices[1]);
  amps_FreeInvoice(recv_invoices[0]);
  amps_FreeInvoice(recv_invoices[1]);
  amps_Finalize();

  return 0;
}
