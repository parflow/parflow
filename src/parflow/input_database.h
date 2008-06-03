/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _INPUT_DATABASE_HEADER
#define _INPUT_DATABASE

#include "hbt.h"

/**
  Maximum length for the key (including the trailing NUL).
  User can thus enter 256 characters.
  */
#define IDB_MAX_KEY_LEN 257

/**
  Maximum length for the value (including the trailing NUL).
  User can thus enter 4096 characters.
  */
#define IDB_MAX_VALUE_LEN 4097

/**
  Entry value for the HBT.  Contains the key and the value pair.
  */
typedef struct _IDB_Entry
{
   char *key;
   char *value;

   /* Flag indicating if the key was used */
   char used;
} IDB_Entry;

/**
  The input database type.  Currently uses a HBT (height balanced tree) for
  storage.
  */
typedef HBT IDB;

typedef struct NameArray__ 
{
   int num;
   char **names;

   char *tok_string;
   char *string;

} NameArrayStruct;

typedef NameArrayStruct *NameArray;

#define WHITE " \t\n"

#define NA_MAX_KEY_LENGTH 2048

#endif 
