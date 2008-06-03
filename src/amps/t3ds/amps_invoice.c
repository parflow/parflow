#include <string.h>
#include <ctype.h>
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
/* Warning HACK HACK HACK                                                 */
/* This should be fixed.                                                  */
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
#if defined(NCUBE2) && defined(_HOST_PROG) && defined(__GNUC__)
#include </Net/local/gnu/lib/gcc-lib/sun4-sunos4.1/2.3.3/include/stdarg.h>
#else
#include <stdarg.h>
#endif

#include "amps.h"

#define to_digit(c)     ((c) - '0')

void amps_AppendInvoice(invoice, append_invoice)
amps_Invoice *invoice;
amps_Invoice append_invoice;
{
   if(*invoice)
   {
      (*invoice) -> num += append_invoice -> num;
      ((*invoice) -> end_list) -> next  = append_invoice -> list;
      (*invoice) -> end_list = append_invoice -> end_list;
      free(append_invoice);
   }
   else
      *invoice = append_invoice;
}

amps_Invoice amps_new_empty_invoice()
{
   amps_Invoice temp;

   if( (temp = (amps_Invoice)calloc(1, sizeof(amps_InvoiceStruct))) == NULL)
      amps_Error("zip_new_empty_invoice", OUT_OF_MEMORY, "", HALT);

   return temp;
}


int amps_FreeInvoice(inv)
amps_Invoice inv;
{
    amps_InvoiceEntry *ptr, *next;

    if(inv == NULL) return 0;

    /* Delete any storage associated with this invoice */
    amps_ClearInvoice(inv);

    if ((ptr = inv -> list) == NULL) return 0;

    next = ptr -> next;

    while(next)
    {
	free(ptr);
	ptr = next;
	next = ptr -> next;
    }

    free(ptr);

    free(inv);

    return 0;
}

int amps_add_invoice(inv, ignore, type,  
		     len_type, len, ptr_len, 
		     stride_type, stride, ptr_stride, 
		     dim_type, dim,
		     data_type, data)
amps_Invoice *inv;
int ignore;
int type, len_type, stride_type;
int len, *ptr_len;
int stride, *ptr_stride;
int data_type;
void *data;
int dim_type;
int *dim;
{
   amps_InvoiceEntry *ptr, *new_entry;
   
   if( (new_entry = (amps_InvoiceEntry*)
	                         calloc(1, sizeof(amps_InvoiceEntry))) == NULL)
      amps_Error("amps_new_empty_invoice", OUT_OF_MEMORY, "", HALT);
   
   new_entry -> next = NULL;
   
   new_entry -> type = type;
   
   new_entry -> len_type = len_type;
   new_entry -> len  = len;
   new_entry -> ptr_len = ptr_len;
   
   new_entry -> stride_type = stride_type;
   new_entry -> stride = stride;
   new_entry -> ptr_stride = ptr_stride;

   
   new_entry -> dim_type = dim_type;
   new_entry -> dim  = dim;

   
   new_entry -> ignore = ignore;
   
   new_entry -> data_type = data_type;
   new_entry -> data  = data;
   
   /* check if the invoice is null                                          */
   if (*inv == NULL)
      *inv = amps_new_empty_invoice();
   
   /* check if list is empty                                                */
   if( (ptr = (*inv) -> end_list) == NULL)
      (*inv) -> list = (*inv) -> end_list = new_entry;
   else
   {
      (*inv) -> end_list -> next = new_entry;
      (*inv) -> end_list = new_entry;
   }
   
   return TRUE;
}

amps_Invoice amps_NewInvoice(const char *fmt0, ...)
{
    va_list ap;

    short ignore;
    char *fmt;	
    int ch;	
    int n;	
    char *cp;	
    int len;	
    int *ptr_len;	
    int len_type;
    int stride;	
    int *ptr_stride;	
    int stride_type;	
    int dim_type;
    int *dim;
    void *ptr_data;
    int ptr_data_type;
    int ret;
    int type;
    int num = 0;
    amps_Invoice inv;


    va_start(ap, fmt0);

    inv = NULL;

    fmt = (char *)fmt0;
    ret = 0;
    
    for (;;) 
    {
	
	for (cp = fmt; (ch = *fmt) != '\0' && ch != '%'; fmt++)
	    return 0;                         /*error */
	if ((n = fmt - cp) != 0)
	    /* error condition */
	    return 0;
	
	if (ch == '\0')
	    goto done;
	fmt++;		/* skip over '%' */
	
	
	ptr_data = NULL;
	stride_type = AMPS_INVOICE_CONSTANT;
	len_type = AMPS_INVOICE_CONSTANT;
	len = 1;
	stride  = 1;
	ignore = FALSE;
	ptr_data_type = AMPS_INVOICE_CONSTANT;
	
    rflag:	
	ch = *fmt++;
    reswitch:
	switch (ch)
	{
	case ' ':
	    /* 
	     * ignore spaces between % and start of format 
	     */
	    goto rflag;
	case '-':
	    /* user wants to skip the space */
	    ignore = TRUE;
	    goto rflag;
	case '*':
	    len = va_arg(ap, int);
	    goto rflag;
	case '&':
	    ptr_len = va_arg(ap, int *);
	    len_type = AMPS_INVOICE_POINTER;
	    goto rflag;
	case '@':
	    /* we are getting a pointer to pointer to data */
	    ptr_data_type = AMPS_INVOICE_POINTER;
	    goto rflag;
	case '.':
	    if ((ch = *fmt++) == '&') 
	    {
		ptr_stride = va_arg(ap, int *);
		stride_type = AMPS_INVOICE_POINTER;
		goto rflag;
	    } 
	    else if (ch =='*')
	    {
		stride = va_arg(ap, int);
		goto rflag;
	    }
	    stride = 0;
	    while (isdigit(ch)) 
	    {
		stride = 10 * stride + to_digit(ch);
		ch = *fmt++;
	    }
	    goto reswitch;
	case '0':
	case '1': case '2': case '3': case '4':
	case '5': case '6': case '7': case '8': case '9':
	    n = 0;
	    do
	    {
		n = 10 * n + to_digit(ch);
		ch = *fmt++;
	    } while (isdigit(ch));
	    len = n;
	    goto reswitch;
	case 'c':
	    type = AMPS_INVOICE_CHAR_CTYPE;
	    break;
	case 's':
	    type= AMPS_INVOICE_SHORT_CTYPE;
	    break;
	case 'i':
	    type = AMPS_INVOICE_INT_CTYPE;
	    break;
	case 'l':
	    type = AMPS_INVOICE_LONG_CTYPE;
	    break;
	case 'd':
	    type = AMPS_INVOICE_DOUBLE_CTYPE;
	    break;
	case 'f':
	    type = AMPS_INVOICE_FLOAT_CTYPE;
	    break;
	case 'D':
	    type = AMPS_INVOICE_DOUBLE_CTYPE + AMPS_INVOICE_LAST_CTYPE;
	    /* skip over "(" */
	    fmt++;
	    if ( (ch = *fmt++) == '&' )
	    {

		dim_type = AMPS_INVOICE_POINTER;
		dim = va_arg(ap, int *);
		fmt++;
	    }
	    else
	    {
		dim_type = AMPS_INVOICE_CONSTANT;
		if (  ch == '*' )
		{
		   dim = (int *)va_arg(ap, int );
		   fmt++;
		}
		else
		{
		    dim = 0;
		    while( isdigit( ch ) )
		    {
		       dim = (int *)(10 * (int)(dim) + to_digit(ch));
		       ch = *fmt++;
		    }
		}
	    }
	    break;
	default:
	    printf("AMPS Error: invalid invoice specification\n");
	    printf("character %c", ch);
	    exit(1);
	    break;
        }
	
	/* if user had an extra we already have grabbed the data pointer */
	if(!ptr_data && !ignore)
	    if(ptr_data_type == AMPS_INVOICE_POINTER)
		ptr_data = va_arg(ap, void  **);
	    else
		ptr_data = va_arg(ap, void *);
	
        amps_add_invoice(&inv, ignore, type, 
			 len_type, len, ptr_len,
			 stride_type, stride, ptr_stride,
			 dim_type, dim,
			 ptr_data_type, ptr_data);
	num++;
    }
    
 done:

    inv -> num = num;
    
    return inv;
}


