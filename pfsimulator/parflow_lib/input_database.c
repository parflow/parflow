/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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

#include "parflow.h"

#include <string.h>

/**
 * Prints out a database entry.  If it was not used during the
 * it is flagged as being unused.
 *
 * @memo Print out a database entry
 * @param file File pointer
 * @param entry The entry to pring
 * @return void
 */
void IDB_Print(FILE *file, void *obj)
{
  IDB_Entry *entry = (IDB_Entry*)obj;

  if (!entry->used)
    fprintf(file, "# Not used\n");

  fprintf(file, "pfset %s \"%s\"\n", entry->key, entry->value);
}

/**
 * Compare two IDB_Entry's for determining order.  This orders they key's
 * in alphabetic (as defined by strcmp) order.
 *
 * if a -> key > b -> key return > 0
 * else if a -> key < b -> key return < 0
 * else return 0 (a -> key == b -> key)
 *
 * @memo Compare two IDB_Entry's (Internal)
 * @param a First entry to compare
 * @param b Second entry to compare
 * @return comparsion value of the keys
 */
int IDB_Compare(void *obj_a, void* obj_b)
{
  IDB_Entry *a = (IDB_Entry*)obj_a;
  IDB_Entry *b = (IDB_Entry*)obj_b;

  return strcmp(a->key, b->key);
}

/**
 * Free an IDB_Entry.  Free's up the character space for the key and value.
 *
 * @memo Free an IDB_Entry (Internal)
 * @param a Entry to free
 * @return N/A
 */
void IDB_Free(void *obj)
{
  IDB_Entry *a = (IDB_Entry*)obj;

  free(a->key);
  free(a->value);
  free(a);
}

/**
 * Create an IDB_Entry
 * Allocates a new IDB_Entry structure with the key and value that is passed
 * in.  The strings are copied so the user can free them.
 *
 * @memo Create an IDB_Entry (Internal)
 * @param key The search key for this entry
 * @param value The value associated with the key
 * @return The new IDB_Entry structure
 */
IDB_Entry *IDB_NewEntry(char *key, char *value)
{
  IDB_Entry *a;

  a = ctalloc(IDB_Entry, 1);

  a->key = strdup(key);
  a->value = strdup(value);

  return a;
}

/**
 * Read in an input database from a flat file.  The returned database
 * can be then used for querying of user input options.
 *
 * A return of NULL indicates and error occured while reading the database.
 *
 * @memo Read in users input into a datbase
 * @param filename The name of the input file containing the database [IN]
 * @return The database
 */
IDB *IDB_NewDB(char *filename)
{
  int num_entries;
  amps_Invoice invoice;

  IDB *db;
  IDB_Entry *entry;

  char key[IDB_MAX_KEY_LEN];
  char value[IDB_MAX_VALUE_LEN];

  int key_len;
  int value_len;

  int i;

  amps_File file;

  /* Initalize the db structure */
  db = (IDB*)HBT_new(IDB_Compare,
                     IDB_Free,
                     IDB_Print,
                     NULL,
                     0);

  if ((file = amps_SFopen(filename, "r")) == NULL)
  {
    InputError("Error: can't open file %s%s\n", filename, "");
  }

  /* Read in the number of items in the database */
  invoice = amps_NewInvoice("%i", &num_entries);
  amps_SFBCast(amps_CommWorld, file, invoice);
  amps_FreeInvoice(invoice);

  /* Read in each of the items in the file and put them in the HBT */

  invoice = amps_NewInvoice("%i%&c%i%&c", &key_len, &key_len, key,
                            &value_len, &value_len, value);
  for (i = 0; i < num_entries; i++)
  {
    /* Read the key and value from the input file */
    amps_SFBCast(amps_CommWorld, file, invoice);
    key[key_len] = '\0';
    value[value_len] = '\0';

    /* Create an new entry */
    entry = IDB_NewEntry(key, value);

    /* Insert into the height balanced tree */
    HBT_insert(db, entry, 0);
  }

  amps_FreeInvoice(invoice);

  amps_SFclose(file);

  return db;
}

/**
 * Frees up the input database.
 *
 * @memo Frees up the input database.
 * @param database The database to free
 * @return N/A
 */
void IDB_FreeDB(IDB *database)
{
  HBT_free(database);
}

/**
 * Prints all of the keys and flag those that have been used during
 * the run.
 *
 * @memo Prints all of the keys and flag those that have been used during
 * the run.
 * @param file  The File pointer to print the information to
 * @param database The database to print usage information for
 * @return N/A
 */
void IDB_PrintUsage(FILE *file, IDB *database)
{
  HBT_printf(file, database);
}

/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param database The database to search
 * @param key The key to search for
 * @return The string which matches the search key
 */
char *IDB_GetString(IDB *database, const char *key)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    result->used = 1;
    return result->value;
  }
  else
  {
    InputError("Error: Can't find required key <%s>%s\n", key, "");
    return 0;
  }
}

/**
 * Get an input string from the input database.  If the key is not
 * found use the default value.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The string which matches the search key
 */
char *IDB_GetStringDefault(IDB *       database,
                           const char *key,
                           char *      default_value)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;
  IDB_Entry *entry;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    result->used = 1;
    return result->value;
  }
  else
  {
    /* Create an new entry */
    entry = IDB_NewEntry((char*)key, default_value);
    entry->used = 1;

    /* Insert into the height balanced tree */
    HBT_insert(database, entry, 0);

    return default_value;
  }
}

/**
 * Get an double value from the input database.  If the key is not
 * found use the default value.
 *
 * The program halts if the value is not a valid double.
 *
 * @memo Get a double value from the input database
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The double which matches the search key
 */
double IDB_GetDoubleDefault(IDB *       database,
                            const char *key,
                            double      default_value)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;
  double value;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    if (sscanf(result->value, "%lf", &value) != 1)
    {
      InputError("Error: The key <%s> is not a valid double: value is <%s>\n", key, result->value);
    }

    result->used = 1;
    return value;
  }
  else
  {
    char default_string[IDB_MAX_KEY_LEN];
    IDB_Entry *entry;

    /* Create a string to insert into the database */
    /* This is used so only a single default value can be found
     * for a given key */
    sprintf(default_string, "%f", default_value);

    /* Create an new entry */
    entry = IDB_NewEntry((char*)key, default_string);
    entry->used = 1;

    /* Insert into the height balanced tree */
    HBT_insert(database, entry, 0);

    return default_value;
  }
}

/**
 * Get a double value from the input database.  If the key is not
 * found print an error and exit.
 *
 * @memo Get a double from the input database
 * @param database The database to search
 * @param key The key to search for
 * @return The double which matches the search key
 */
double IDB_GetDouble(IDB *database, const char *key)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;
  double value;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    if (sscanf(result->value, "%lf", &value) != 1)
    {
      InputError("Error: The key <%s> is not a valid double: value is <%s>\n",
                 key, result->value);
    }

    result->used = 1;
    return value;
  }
  else
  {
    InputError("Input Error: Can't find required key <%s>%s\n", key, "");
    return 0;
  }
}


/**
 * Get a integer value from the input database.  If the key is not
 * found use the default value.
 *
 * The program halts if the value is not a valid integer.
 *
 * @memo Get a integer value from the input database
 * @param database The database to search
 * @param key The key to search for
 * @param default_value The default to use if not found
 * @return The integer which matches the search key
 */
int IDB_GetIntDefault(IDB *       database,
                      const char *key,
                      int         default_value)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;
  int value;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    if (sscanf(result->value, "%d", &value) != 1)
    {
      InputError("Error: The key <%s> is not a valid integer: value is <%s>\n",
                 key, result->value);
    }

    result->used = 1;
    return value;
  }
  else
  {
    char default_string[IDB_MAX_KEY_LEN];
    IDB_Entry *entry;

    /* Create a string to insert into the database */
    /* This is used so only a single default value can be found
     * for a given key */
    sprintf(default_string, "%d", default_value);

    /* Create an new entry */
    entry = IDB_NewEntry((char*)key, default_string);
    entry->used = 1;

    /* Insert into the height balanced tree */
    HBT_insert(database, entry, 0);

    return default_value;
  }
}

/**
 * Get a integer value from the input database.  If the key is not
 * found print an error and exit.
 *
 * @memo Get a integer from the input database
 * @param database The database to search
 * @param key The key to search for
 * @return The integer which matches the search key
 */
int IDB_GetInt(IDB *database, const char *key)
{
  IDB_Entry lookup_entry;
  IDB_Entry *result;
  int value;

  lookup_entry.key = (char*)key;

  result = (IDB_Entry*)HBT_lookup(database, &lookup_entry);

  if (result)
  {
    if (sscanf(result->value, "%d", &value) != 1)
    {
      InputError("Error: The key <%s> is not a valid int: value is <%s>\n",
                 key, result->value);
    }

    result->used = 1;
    return value;
  }
  else
  {
    InputError("Input Error: Can't find required key <%s>%s\n", key, "");
    return 0;
  }
}

/**
 * Construct a name array from an input string.
 *
 * @memo Construct a name array from an input string.
 * @param string the input string with white space seperated names
 * @return The new name array
 */
NameArray NA_NewNameArray(char *string)
{
  NameArray name_array;


  char *ptr;
  char *new_string;
  int size;

#ifdef THREADS
  char *lasts = NULL;
#endif

  name_array = ctalloc(NameArrayStruct, 1);

  /* parse the string and put into the string */

#ifdef THREADS
#else
  new_string = strdup(string);

  /* SGS Warning this is NOT thread safe */
  if (strtok(new_string, WHITE) == NULL)
  {
    free(new_string);
    return name_array;
  }

  size = 1;
  while (strtok(NULL, WHITE) != NULL)
    size++;

  free(new_string);

  new_string = strdup(string);

  name_array->names = talloc(char *, size);
  name_array->num = size;
  name_array->tok_string = new_string;
  name_array->string = strdup(string);

  ptr = strtok(new_string, WHITE);
  size = 0;
  name_array->names[size++] = strdup(ptr);

  while ((ptr = strtok(NULL, WHITE)) != NULL)
    name_array->names[size++] = strdup(ptr);
#endif

  return name_array;
}

int NA_AppendToArray(NameArray name_array, char *string)
{
  int size;
  int i;

  char *ptr;

  char *both_string;
  char *temp_string;

  both_string = talloc(char, strlen(string) + strlen(name_array->string) + 2);

  sprintf(both_string, "%s %s", name_array->string, string);

  /* Determine the number of entries */

  temp_string = strdup(both_string);

  /* SGS Warning this is NOT thread safe of strtok is not
   * thread safe (which is probably is not */
  if (strtok(temp_string, WHITE) == NULL)
  {
    free(temp_string);
    return 0;
  }

  size = 1;
  while (strtok(NULL, WHITE) != NULL)
    size++;

  free(temp_string);

  if (name_array->string)
    free(name_array->string);
  if (name_array->tok_string)
    free(name_array->tok_string);

  for (i = 0; i < name_array->num; i++)
  {
    free(name_array->names[i]);
  }
  if (name_array->names)
    free(name_array->names);

  name_array->names = talloc(char *, size);
  name_array->num = size;
  name_array->string = strdup(both_string);
  name_array->tok_string = both_string;

  ptr = strtok(both_string, WHITE);
  size = 0;
  name_array->names[size++] = strdup(ptr);

  while ((ptr = strtok(NULL, WHITE)) != NULL)
    name_array->names[size++] = strdup(ptr);

  return 0;
}


void NA_FreeNameArray(NameArray name_array)
{
  int i;

  if (name_array)
  {
    if (name_array->string)
      free(name_array->string);
    if (name_array->tok_string)
      free(name_array->tok_string);
    for (i = 0; i < name_array->num; i++)
    {
      free(name_array->names[i]);
    }
    if (name_array->names)
      free(name_array->names);
    free(name_array);
  }
}

int NA_NameToIndex(NameArray name_array, char *name)
{
  int i;

  for (i = 0; i < name_array->num; i++)
  {
    if (!strcmp(name_array->names[i], name))
      return i;
  }

  return -1;
}

char *NA_IndexToName(NameArray name_array, int index)
{
  return name_array->names[index];
}

int NA_Sizeof(NameArray name_array)
{
  return name_array->num;
}

void InputError(const char *format, const char *s1, const char *s2)
{
  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf(format, s1, s2);
  }

  exit(1);
}


