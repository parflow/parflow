#include <stdio.h>
#include <parflow.h>

int
main(int argc, char **argv)
{
  int simple;
  int   dim,i, me;
  double *array_s,*array_r;
  int   len[3] = {3,4,1};
  int   str[3];
  amps_Invoice send_invoice;
  amps_Invoice recv_invoice;
  amps_Package package;
  amps_Handle  handle;
  int   src;
  int   dest;

  if (amps_Init(&argc, &argv))
    {
      amps_Printf("Error amps_Init\n");
      exit(1);
    }

  simple  = atoi(argv[1]);
  me = amps_Rank(amps_CommWorld);

  array_s  = amps_CTAlloc(double,29);
  array_r  = amps_CTAlloc(double,29);
  dim = 2;

  if (me == 0){
      str[0] = 2, str[1] = 3, str[2]=1;
      for(i=0;i<29;i++){
          array_s[i] = i;
        }
      send_invoice = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s);
      recv_invoice = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r);
    }
  if (me == 1){
      for(i=0;i<29;i++){
          array_s[i] = 1.;
        }
      str[0] = str[1] = str[2] = 1;
      send_invoice = amps_NewInvoice("%&.&D(*)",len, str, dim, array_s);
      recv_invoice = amps_NewInvoice("%&.&D(*)",len, str, dim, array_r);
    }

  src  = (me == 0) ? 1 : 0;
  dest = (me == 0) ? 1 : 0;

  if (simple) {

      /*Simple communication*/
      amps_Send(amps_CommWorld, dest, send_invoice);
      amps_Recv(amps_CommWorld, src, recv_invoice);

    }else{

      /* communication using amps packages*/
      package = amps_NewPackage(amps_CommWorld,
                                1, &dest, &send_invoice,
                                1, &src,  &recv_invoice);
      handle = amps_IExchangePackage(package);
      amps_Wait(handle);
      amps_FreePackage(package);
    }

  printf("Rank %i received: \n", me);
  for(i=0;i<29;i++){
      printf( "%0.0f ", array_r[i]);
    }
  printf("\n");

  amps_TFree(array_r);
  amps_TFree(array_s);
  amps_FreeInvoice(send_invoice);
  amps_FreeInvoice(recv_invoice);
  amps_Finalize();

  return 0;
}
