/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#ifndef _INPUT_DATABASE_HEADER
#define _INPUT_DATABASE_HEADER

#include "hbt.h"

/**
 * Maximum length for the key (including the trailing NUL).
 * User can thus enter 2047 characters.
 */
#define IDB_MAX_KEY_LEN 2048

/**
 * Maximum length for the value (including the trailing NUL).
 * User can thus enter 65335 characters.
 */
#define IDB_MAX_VALUE_LEN 65536

/**
 * Entry value for the HBT.  Contains the key and the value pair.
 */
typedef struct _IDB_Entry {
  char *key;
  char *value;

  /* Flag indicating if the key was used */
  char used;
} IDB_Entry;

/**
 * The input database type.  Currently uses a HBT (height balanced tree) for
 * storage.
 */
typedef HBT IDB;

/**
 * NameArray is a specialized string array used in ParFlow input parsing.
 */
typedef struct NameArray__ {
  int num;  /*!< Number of names (strings) stored */
  char **names; /*!< The names (strings) stored */

  char *tok_string; /*!< Not sure why this is here */
  char *string; /*!< Whitespace separated string of all entries; seems odd to store this, why? */
} NameArrayStruct;

typedef NameArrayStruct *NameArray;

/**
 * Prints out a database entry.
 *
 * If it was not used during a run it is flagged as being unused.
 *
 * @param file File pointer
 * @param entry The entry to print
 * @return void
 */
void IDB_Print(FILE *file, void *entry);

/**
 * Compare two IDB_Entry's for determining order.  This orders they key's
 * in alphabetic (as defined by strcmp) order.
 *
 * if a -> key > b -> key return > 0
 * else if a -> key < b -> key return < 0
 * else return 0 (a -> key == b -> key)
 *
 * @param a First entry to compare
 * @param b Second entry to compare
 * @return comparsion value of the keys
 */
int IDB_Compare(void *a, void *b);

/**
 * Free an IDB_Entry.  Free's up the character space for the key and value.
 *
 * @param a Entry to free
 * @return N/A
 */
void IDB_Free(void *a);
/**
 * Create an IDB_Entry
 * Allocates a new IDB_Entry structure with the key and value that is passed
 * in.  The strings are copied so the user can free them.
 *
 * @param key The search key for this entry
 * @param value The value associated with the key
 * @return The new IDB_Entry structure
 */
IDB_Entry *IDB_NewEntry(char *key, char *value);

/**
 * Read in an input database from a flat file.  The returned database
 * can be then used for querying of user input options.
 *
 * A return of NULL indicates and error occured while reading the database.
 *
 * @param filename The name of the input file containing the database [IN]
 * @return The database
 */
IDB *IDB_NewDB(char *filename);

/**
 * Frees up the input database.
 *
 * @param database The database to free
 * @return N/A
 */
void IDB_FreeDB(IDB *database);
/**
 * Prints all of the keys and flag those that have been used during
 * the run.
 *
 * @param file  The File pointer to print the information to
 * @param database The database to print usage information for
 * @return N/A
 */
void IDB_PrintUsage(FILE *file, IDB *database);

/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @param database The database to search
 * @param key The key to search for
 * @return The string which matches the search key
 */
char *IDB_GetString(IDB *database, const char *key);

/**
 * Get an input string from the input database.  If the key is not
 * found use the default value.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The string which matches the search key
 */
char *IDB_GetStringDefault(IDB *database, const char *key, char *default_value);

/**
 * Get an double value from the input database.  If the key is not
 * found use the default value.
 *
 * The program halts if the value is not a valid double.
 *
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The double which matches the search key
 */
double IDB_GetDoubleDefault(IDB *database, const char *key, double default_value);

/**
 * Get a double value from the input database.  If the key is not
 * found print an error and exit.
 *
 * @param database The database to search
 * @param key The key to search for
 * @return The double which matches the search key
 */
double IDB_GetDouble(IDB *database, const char *key);

/**
 * Get a integer value from the input database.  If the key is not
 * found use the default value.
 *
 * The program halts if the value is not a valid integer.
 *
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The integer which matches the search key
 */
int IDB_GetIntDefault(IDB *database, const char *key, int default_value);

/**
 * Get a integer value from the input database.  If the key is not
 * found print an error and exit.
 *
 * @param database The database to search
 * @param key The key to search for
 * @return The integer which matches the search key
 */
int IDB_GetInt(IDB *database, const char *key);

/**
 * Construct a name array from an input string.
 *
 * Provided string is split by whitespace and entries are created for
 * each space separated words.
 *
 * @param string the input string with white space seperated names
 * @return The new name array
 */
NameArray NA_NewNameArray(char *string);

/**
 * Append entries to a name array.
 *
 * The provided string is split by whitespace and entries are created
 * for each space separated words.
 *
 * @param name_array The name array
 * @param name  The name to append
 */
int NA_AppendToArray(NameArray name_array, char *string);

/**
 * Free a name array.
 *
 * Free the name array structure and free all the stored names.
 *
 * @param name_array Name array to free
 */
void NA_FreeNameArray(NameArray name_array);

/**
 * Returns Index in the name array for the specified name.
 *
 * Returns -1 if name is not found.
 *
 * @param name_array  The name array for the lookup
 * @param name The name to find
 */
int NA_NameToIndex(NameArray name_array, char *name);

/**
 * Returns Index in the name array for the specified name, exit if not found
 *
 * Prints error message and exits if name is not found.
 *
 * @param name_array  The name array for the lookup
 * @param name The name to find
 * @param key The key that was being accessed.
 */
int NA_NameToIndexExitOnError(NameArray name_array, const char *name, const char* key);

/**
 * Returns Name name corresponding to the provided index.
 *
 * @param name_array  The name array for the lookup
 * @param index The index into the name array
 */
char *NA_IndexToName(NameArray name_array, int index);

int NA_Sizeof(NameArray name_array);

/**
 * I/O wrapper for InputErrors.
 *
 * Intent of this isto avoid having the rank and repeated all over.
 */
void InputError(const char *format, const char *s1, const char *s2);

/**
 * Output error for invalid value provided to a NameArray key and exits.
 *
 * @param name_array the NameArray
 * @param switch_name the invalid value provided
 * @param key the database key being parsed
 */
void NA_InputError(NameArray name_array, const char *switch_name, const char *key);

#endif
