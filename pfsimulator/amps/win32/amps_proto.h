#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif


/* _rand48.c */
void _dorand48 P((unsigned short xseed [3 ]));

/* amps_allreduce.c */
int amps_ReduceOperation P((amps_Comm comm, amps_Invoice invoice, char *buf_dest, char *buf_src, int operation));
int amps_AllReduce P((amps_Comm comm, amps_Invoice invoice, int operation));

/* amps_bcast.c */
int amps_BCast P((amps_Comm comm, int source, amps_Invoice invoice));

/* amps_clear.c */
void amps_ClearInvoice P((amps_Invoice inv));

/* amps_createinvoice.c */
int amps_CreateInvoice P((amps_Comm comm, amps_Invoice inv));

/* amps_exchange.c */
void _amps_wait_exchange P((amps_Handle handle));
amps_Handle amps_IExchangePackage P((amps_Package package));

/* amps_ffopen.c */
amps_File amps_FFopen P((amps_Comm comm, char *filename, char *type, long size));

/* amps_finalize.c */
int amps_Finalize P((void));

/* amps_find_powers.c */
void amps_FindPowers P((int N, int *log, int *Nnext, int *Nprev));

/* amps_fopen.c */
amps_File amps_Fopen P((char *filename, char *type));

/* amps_init.c */
int amps_Init P((int *argc, char **argv[]));
unsigned amps_main P((void *arg));
int main P((int argc, char *argv []));
void *_amps_CTAlloc P((int count, char *filename, int line));
void *_amps_TAlloc P((int count, char *filename, int line));

/* amps_invoice.c */
void amps_AppendInvoice P((amps_Invoice *invoice, amps_Invoice append_invoice));
amps_Invoice amps_new_empty_invoice P((void));
int amps_FreeInvoice P((amps_Invoice inv));
int amps_add_invoice P((amps_Invoice *inv, int ignore, int type, int len_type, int len, int *ptr_len, int stride_type, int stride, int *ptr_stride, int dim_type, int dim, int *ptr_dim, int data_type, void *data));
amps_Invoice amps_NewInvoice P((const char *fmt0, ...));
int amps_num_package_items P((amps_Invoice inv));

/* amps_io.c */
void amps_ScanChar P((amps_File file, char *data, int len, int stride));
void amps_ScanShort P((amps_File file, short *data, int len, int stride));
void amps_ScanInt P((amps_File file, int *data, int len, int stride));
void amps_ScanLong P((amps_File file, long *data, int len, int stride));
void amps_ScanFloat P((amps_File file, float *data, int len, int stride));
void amps_ScanDouble P((amps_File file, double *data, int len, int stride));
void amps_WriteDouble P((amps_File file, double *ptr, int len));
void amps_WriteInt P((amps_File file, int *ptr, int len));
void amps_ReadDouble P((amps_File file, double *ptr, int len));
void amps_ReadInt P((amps_File file, int *ptr, int len));

/* amps_irecv.c */
char *amps_recv P((int src));
amps_Handle amps_IRecv P((amps_Comm comm, int source, amps_Invoice invoice));

/* amps_newhandle.c */
amps_Handle amps_NewHandle P((amps_Comm comm, int id, amps_Invoice invoice, amps_Package package));

/* amps_newpackage.c */
amps_Package amps_NewPackage P((amps_Comm comm, int num_send, int *dest, amps_Invoice *send_invoices, int num_recv, int *src, amps_Invoice *recv_invoices));
void amps_FreePackage P((amps_Package package));

/* amps_pack.c */
int amps_pack P((amps_Comm comm, amps_Invoice inv, char **buffer));

/* amps_recv.c */
char *amps_recvb P((int src));
int amps_Recv P((amps_Comm comm, int source, amps_Invoice invoice));

/* amps_send.c */
int amps_xsend P((char *buffer, int dest));
int amps_Send P((amps_Comm comm, int dest, amps_Invoice invoice));

/* amps_sfbcast.c */
int amps_SFBCast P((amps_Comm comm, amps_File file, amps_Invoice invoice));

/* amps_sfclose.c */
int amps_SFclose P((amps_File file));

/* amps_sfopen.c */
amps_File amps_SFopen P((char *filename, char *type));

/* amps_sizeofinvoice.c */
int amps_sizeof_invoice P((amps_Comm comm, amps_Invoice inv));

/* amps_sync.c */
int amps_Sync P((amps_Comm comm));

/* amps_test.c */
int amps_Test P((amps_Handle handle));

/* amps_unpack.c */
int amps_unpack P((amps_Comm comm, amps_Invoice inv, char *buffer));

/* amps_vector.c */
void amps_vector_out P((amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride));
void amps_vector_in P((amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride));
int amps_vector_align P((amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride));
int amps_vector_sizeof_buffer P((amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride));
int amps_vector_sizeof_local P((amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride));

/* amps_wait.c */
int amps_Wait P((amps_Handle handle));

/* drand48.c */
double drand48 P((void));

/* erand48.c */
double erand48 P((unsigned short xseed [3 ]));

/* jrand48.c */
long jrand48 P((unsigned short xseed [3 ]));

/* lrand48.c */
long lrand48 P((void));

/* mrand48.c */
long mrand48 P((void));

/* nrand48.c */
long nrand48 P((unsigned short xseed [3 ]));

/* srand48.c */
void srand48 P((long seed));

/* unix_port.c */
double d_sign P((doublereal *a, doublereal *b));

#undef P
