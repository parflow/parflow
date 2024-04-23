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

#include "parflow.h"

#include <string.h>

/** Whitespace characters */
#define NA_WHITE_SPACE " \t\n"


void IDB_Print(FILE *file, void *obj)
{
  IDB_Entry *entry = (IDB_Entry*)obj;

  if (!entry->used)
    fprintf(file, "# Not used\n");

  fprintf(file, "pfset %s \"%s\"\n", entry->key, entry->value);
}

int IDB_Compare(void *obj_a, void* obj_b)
{
  IDB_Entry *a = (IDB_Entry*)obj_a;
  IDB_Entry *b = (IDB_Entry*)obj_b;

  return strcmp(a->key, b->key);
}

void IDB_Free(void *obj)
{
  IDB_Entry *a = (IDB_Entry*)obj;

  free(a->key);
  free(a->value);
  free(a);
}

IDB_Entry *IDB_NewEntry(char *key, char *value)
{
  IDB_Entry *a;

  a = ctalloc(IDB_Entry, 1);

  a->key = strdup(key);
  a->value = strdup(value);

  return a;
}

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
    if ((value_len + 1) > IDB_MAX_VALUE_LEN) {
      key[key_len] = '\0';
      char s[128];
      sprintf(s, "%d", IDB_MAX_VALUE_LEN-1);
      InputError("Error: The value associated with input database "
                 "key <%s> is too long. The maximum length is %s. ",
                 key, s); 
    }
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

void IDB_FreeDB(IDB *database)
{
  HBT_free(database);
}

void IDB_PrintUsage(FILE *file, IDB *database)
{
  HBT_printf(file, database);
}

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
  if (strtok(new_string, NA_WHITE_SPACE) == NULL)
  {
    free(new_string);
    return name_array;
  }

  size = 1;
  while (strtok(NULL, NA_WHITE_SPACE) != NULL)
    size++;

  free(new_string);

  new_string = strdup(string);

  name_array->names = talloc(char *, size);
  name_array->num = size;
  name_array->tok_string = new_string;
  name_array->string = strdup(string);

  ptr = strtok(new_string, NA_WHITE_SPACE);
  size = 0;
  name_array->names[size++] = strdup(ptr);

  while ((ptr = strtok(NULL, NA_WHITE_SPACE)) != NULL)
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
  if (strtok(temp_string, NA_WHITE_SPACE) == NULL)
  {
    free(temp_string);
    return 0;
  }

  size = 1;
  while (strtok(NULL, NA_WHITE_SPACE) != NULL)
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

  ptr = strtok(both_string, NA_WHITE_SPACE);
  size = 0;
  name_array->names[size++] = strdup(ptr);

  while ((ptr = strtok(NULL, NA_WHITE_SPACE)) != NULL)
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
  for (int i = 0; i < name_array->num; i++)
  {
    if (!strcmp(name_array->names[i], name))
      return i;
  }

  return -1;
}

int NA_NameToIndexExitOnError(NameArray name_array, const char *name, const char* key)
{
  for (int i = 0; i < name_array->num; i++)
  {
    if (!strcmp(name_array->names[i], name))
      return i;
  }

  NA_InputError(name_array, name, key);

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

void NA_InputError(NameArray name_array, const char *switch_name, const char *key)
{
  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Error: invalid value <%s> for key <%s>\n", switch_name, key);
    amps_Printf("       Allowed values are:\n");

    for (int i = 0; i < name_array->num; i++)
    {
      amps_Printf("           %s\n", name_array->names[i]);
    }
  }

  exit(1);
}
