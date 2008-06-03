/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
 *********************************************************************EHEADER*/
/* Basic I/O routines */

#ifndef MAXPATHLEN
#ifdef MAX_PATH
#define MAXPATHLEN MAX_PATH
#else
#define MAXPATHLEN 1024
#endif
#endif

/*---------------------------------------------------------------------------*/
/* The following routines are used to actually write data to a file.         */
/* We use XDR like representation for all values written.                    */
/*---------------------------------------------------------------------------*/

#define tools_SizeofChar sizeof(char)
#define tools_SizeofShort sizeof(short)
#define tools_SizeofInt sizeof(int)
#define tools_SizeofLong sizeof(long)
#define tools_SizeofFloat sizeof(float)
#define tools_SizeofDouble sizeof(double)


#ifdef TOOLS_BYTE_SWAP
 
void tools_WriteInt();
 
void tools_WriteDouble();
 
void tools_ReadInt();
 
void tools_ReadDouble();
 
#else
#ifdef TOOLS_CRAY 

void tools_WriteInt();
 
void tools_WriteDouble();
 
void tools_ReadInt();
 
void tools_ReadDouble();

#else
#ifdef TOOLS_INTS_ARE_64

#define tools_WriteDouble(file, ptr, len) \
    fwrite( (ptr), sizeof(double), (len), (FILE *)(file) )

#define tools_ReadDouble(file, ptr, len) \
    fread( (ptr), sizeof(double), (len), (FILE *)(file) )

#else

/*****************************************************************************/
/* Normal I/O for machines that use IEEE                                     */
/*****************************************************************************/
 
#define tools_WriteInt(file, ptr, len) \
    fwrite( (ptr), sizeof(int), (len), (FILE *)(file) )
 
#define tools_WriteDouble(file, ptr, len) \
    fwrite( (ptr), sizeof(double), (len), (FILE *)(file) )
 
#define tools_ReadInt(file, ptr, len) \
    fread( (ptr), sizeof(int), (len), (FILE *)(file) )
 
#define tools_ReadDouble(file, ptr, len) \
    fread( (ptr), sizeof(double), (len), (FILE *)(file) )
 
#endif
#endif
#endif
