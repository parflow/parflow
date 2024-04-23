/* amps_clear.c */
void amps_ClearInvoice(amps_Invoice inv);

/* amps_clock.c */
void amps_clock_init(void);
amps_Clock_t amps_Clock(void);
amps_CPUClock_t amps_CPUClock(void);

/* amps_ffopen.c */
amps_File amps_FFopen(amps_Comm comm, char *filename, char *type, long size);

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
void amps_WriteDouble(amps_File file, double *ptr, int len);
void amps_WriteInt(amps_File file, int *ptr, int len);
void amps_ReadDouble(amps_File file, double *ptr, int len);
void amps_ReadInt(amps_File file, int *ptr, int len);
void amps_WriteInt(amps_File file, int *ptr, int len);
void amps_ReadInt(amps_File file, int *ptr, int len);

/* amps_newpackage.c */
amps_Package amps_NewPackage(amps_Comm comm, int num_send, int *dest, amps_Invoice *send_invoices, int num_recv, int *src, amps_Invoice *recv_invoices);
void amps_FreePackage(amps_Package package);

/* amps_sfbcast.c */
int amps_SFBCast(amps_Comm comm, amps_File file, amps_Invoice invoice);

void _amps_Abort(
                 char *message,
                 char *filename,
                 int   line);

/* amps_print.c */
FILE* amps_SetConsole(FILE* stream);
void amps_Printf(const char *fmt, ...);
