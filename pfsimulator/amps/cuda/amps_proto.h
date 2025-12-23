/* amps_allreduce.c */
int amps_AllReduce(amps_Comm comm, amps_Invoice invoice, MPI_Op operation);

/* amps_bcast.c */
int amps_BCast(amps_Comm comm, int source, amps_Invoice invoice);

/* amps_clear.c */
void amps_ClearInvoice(amps_Invoice inv);

/* amps_clock.c */
void amps_clock_init(void);
amps_Clock_t amps_Clock(void);
amps_CPUClock_t amps_CPUClock(void);

/* amps_createinvoice.c */
int amps_CreateInvoice(amps_Comm comm, amps_Invoice inv);

/* amps_exchange.c */
void _amps_wait_exchange(amps_Handle handle);
amps_Handle amps_IExchangePackage(amps_Package package);
void _amps_wait_exchange(amps_Handle handle);
amps_Handle amps_IExchangePackage(amps_Package package);

/* amps_ffopen.c */
amps_File amps_FFopen(amps_Comm comm, char *filename, char *type, long size);

/* amps_finalize.c */
int amps_Finalize(void);

/* amps_find_powers.c */
void amps_FindPowers(int N, int *log, int *Nnext, int *Nprev);

/* amps_fopen.c */
amps_File amps_Fopen(char *filename, char *type);

/* amps_init.c */
int amps_Init(int *argc, char **argv []);
int amps_EmbeddedInit(void);
int amps_EmbeddedInit(MPI_COMM comm);
int amps_EmbeddedInitFComm(MPI_Fint *f_handle);
int amps_EmbeddedInitComm(MPI_Comm com);

/* amps_invoice.c */
void amps_AppendInvoice(amps_Invoice *invoice, amps_Invoice append_invoice);
amps_Invoice amps_new_empty_invoice(void);
int amps_FreeInvoice(amps_Invoice inv);
int amps_add_invoice(amps_Invoice *inv, int ignore, int type, int len_type, int len, int *ptr_len, int stride_type, int stride, int *ptr_stride, int dim_type, int dim, int *ptr_dim, int data_type, void *data);
amps_Invoice amps_NewInvoice(const char *fmt0, ...);
int amps_num_package_items(amps_Invoice inv);

/* amps_io.c */
void amps_ScanByte(amps_File file, char *data, int len, int stride);
void amps_ScanChar(amps_File file, char *data, int len, int stride);
void amps_ScanShort(amps_File file, short *data, int len, int stride);
void amps_ScanInt(amps_File file, int *data, int len, int stride);
void amps_ScanLong(amps_File file, long *data, int len, int stride);
void amps_ScanFloat(amps_File file, float *data, int len, int stride);
void amps_ScanDouble(amps_File file, double *data, int len, int stride);

#ifndef CASC_HAVE_BIGENDIAN
void amps_WriteDouble(amps_File file, double *ptr, int len);
void amps_ReadDouble(amps_File file, double *ptr, int len);
void amps_WriteInt(amps_File file, int *ptr, int len);
void amps_ReadInt(amps_File file, int *ptr, int len);
#endif


/* amps_irecv.c */
amps_Handle amps_IRecv(amps_Comm comm, int source, amps_Invoice invoice);

/* amps_newhandle.c */
amps_Handle amps_NewHandle(amps_Comm comm, int id, amps_Invoice invoice, amps_Package package);
void amps_FreeHandle(amps_Handle handle);

/* amps_newpackage.c */
amps_Package amps_NewPackage(amps_Comm comm, int num_send, int *dest, amps_Invoice *send_invoices, int num_recv, int *src, amps_Invoice *recv_invoices);
void amps_FreePackage(amps_Package package);
amps_Package amps_NewPackage(amps_Comm comm, int num_send, int *dest, amps_Invoice *send_invoices, int num_recv, int *src, amps_Invoice *recv_invoices);
void amps_FreePackage(amps_Package package);

/* amps_pack.c */
int amps_create_mpi_cont_send_type(amps_Comm comm, amps_Invoice inv);
void amps_create_mpi_type(amps_Comm comm, amps_Invoice inv);
int amps_pack(amps_Comm comm, amps_Invoice inv, char *buffer, int *streams_hired);

/* amps_print.c */
FILE* amps_SetConsole(FILE* stream);
void amps_Printf(const char *fmt, ...);

/* amps_recv.c */
char *amps_recvb(int src, int *size);
int amps_Recv(amps_Comm comm, int source, amps_Invoice invoice);

/* amps_send.c */
int amps_xsend(amps_Comm comm, int dest, amps_Invoice invoice, char *buffer);
int amps_Send(amps_Comm comm, int dest, amps_Invoice invoice);

/* amps_sfbcast.c */
int amps_SFBCast(amps_Comm comm, amps_File file, amps_Invoice invoice);

/* amps_sfclose.c */
int amps_SFclose(amps_File file);

/* amps_sfopen.c */
amps_File amps_SFopen(const char *filename, const char *type);

/* amps_sizeofinvoice.c */
long amps_sizeof_invoice(amps_Comm comm, amps_Invoice inv);

/* amps_test.c */
int amps_Test(amps_Handle handle);

/* amps_unpack.c */
int amps_unpack(amps_Comm comm, amps_Invoice inv, char *buffer, int *streams_hired);
int amps_unpack_mpi1(amps_Comm comm, amps_Invoice inv, char *buffer, int buf_size);

/* amps_vector.c */
void amps_vector_in(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride);
void amps_vector_out(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride);
int amps_vector_align(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride);
int amps_vector_sizeof_buffer(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride);
int amps_vector_sizeof_local(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride);

/* signal.c */
void handler_ill(void);
void handler_bus(void);
void handler_seg(void);
void handler_sys(void);
void handler_fpe(void);
void handler_division(void);
void handler_overflow(void);
void handler_invalid(void);
void Fsignal(void);

int amps_Wait(amps_Handle handle);

void _amps_Abort(
                 char *message,
                 char *filename,
                 int   line);
