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
/****************************************************************************
 * Program to interactively read, print, etc. different file formats.
 *
 * (C) 1993 Regents of the University of California.
 *
 *----------------------------------------------------------------------------
 * $Revision: 1.48 $
 *
 *----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow_config.h"

#include <string.h>
#include <unistd.h>
#include <ctype.h>

#ifdef _WIN32
#define strdup _strdup
#endif

#include "pftools.h"

#include "grid.h"
#include "usergrid.h"
#include "file.h"

#include <tcl.h>

#include "readdatabox.h"
#include "printdatabox.h"
#include "solidtools.h"
#include "velocity.h"
#include "head.h"
#include "flux.h"
#include "stats.h"
#include "diff.h"
#include "error.h"
#include "getsubbox.h"
#include "enlargebox.h"
#include "file.h"
#include "load.h"
#include "top.h"
#include "compute_domain.h"
#include "water_table.h"
#include "water_balance.h"
#include "toposlopes.h"

#include "region.h"
#include "grid.h"
#include "usergrid.h"

#include "general.h"


/*-----------------------------------------------------------------------
 * Load:
 *   distribute the data
 *-----------------------------------------------------------------------*/

void           Load(
                    int           type,
                    char *        filename,
                    SubgridArray *all_subgrids,
                    Background *  background,
                    Databox *     databox)
{
  switch (type)
  {
    case ParflowB:
      LoadParflowB(filename, all_subgrids, background, databox);
      break;

    default:
      printf("Cannot load onto a file of that type\n");
  }
}

/*-----------------------------------------------------------------------
 * routine for `pfdist' command
 * Description: distributes the file to the virtual distributed file
 *              system.  Based on current database processor topology.
 *
 * Cmd. syntax: pfdist filename
 *-----------------------------------------------------------------------*/

int            PFDistCommand(
                             ClientData  clientData,
                             Tcl_Interp *interp,
                             int         argc,
                             char *      argv[])
{
  char *filename;
  char *filetype;

  int num_procs_x;
  int num_procs_y;
  int num_procs_z;
  int num_procs;
  int nz_manual = 0;

  Background    *background;
  Grid          *user_grid;
  SubgridArray  *all_subgrids;

  Databox *inbox;

  char command[1024];

  // Setup and error checking for manual nz spec
  if ((argc == 2) || (argc == 4))
  {
    if (argc == 4) /* Check that third argument is -nz */
    {
      if (strcmp(argv[1], "-nz") != 0)
      {
        printf("Error: Expected optional argument is: -nz \n");
        printf("  argument read as: %s \n", argv[1]);
        return TCL_ERROR;
      }
      nz_manual = atoi(argv[2]);
      if (nz_manual < 1)
      {
        printf("Error: -nz must be greater than 0 \n");
        return TCL_ERROR;
      }
    }
  }
  else
  {
    /*WrongNumArgsError(interp, LOADPFUSAGE); */
    printf("Error: Invalid number of arguments passed to pfdist \n");
    printf("       2 or 4 allowed, %d passed by user \n", argc);
    return TCL_ERROR;
  }

  if (argc > 2)
  {
    filename = argv[3];
  }
  else
  {
    filename = argv[1];
  }

  /* Make sure the file extension is valid */
  if ((filetype = GetValidFileExtension(filename)) == (char*)NULL)
  {
    InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
    return TCL_ERROR;
  }

  if (strcmp(filetype, "pfb") == 0)
  {
    /*--------------------------------------------------------------------
     * Get the processor topology from the database
     *--------------------------------------------------------------------*/
    num_procs_x = GetInt(interp, "Process.Topology.P");
    num_procs_y = GetInt(interp, "Process.Topology.Q");
    num_procs_z = GetInt(interp, "Process.Topology.R");

    num_procs = num_procs_x * num_procs_y * num_procs_z;


    /*--------------------------------------------------------------------
     * Get the initial grid info from the database
     *--------------------------------------------------------------------*/
    background = ReadBackground(interp);
    user_grid = ReadUserGrid(interp);

    int nz_in;
    Subgrid     *user_subgrid = GridSubgrid(user_grid, 0);
    if (nz_manual != 0)
    {
      nz_in = SubgridNZ(user_subgrid); // Save the correct nz
      SubgridNZ(user_subgrid) = nz_manual; // Set the manual nz
    }
    /*--------------------------------------------------------------------
     * Get inbox from input_filename
     *--------------------------------------------------------------------*/

    inbox = Read(ParflowB, filename);

    /*--------------------------------------------------------------------
     * Load the data
     *--------------------------------------------------------------------*/

    all_subgrids = DistributeUserGrid(user_grid, num_procs,
                                      num_procs_x, num_procs_y, num_procs_z);
    if (nz_manual != 0)
    {
      SubgridNZ(user_subgrid) = nz_in;  // Restore the correct nz
    }
    if (!all_subgrids)
    {
      printf("Incorrect process allocation input\n");
      exit(1);
    }

#ifdef _WIN32
    sprintf(command, "move %s %s.bak", filename, filename);
    system(command);
#else
    sprintf(command, "mv %s %s.bak", filename, filename);
    system(command);
#endif

    Load(ParflowB, filename, all_subgrids, background, inbox);

#ifdef _WIN32
    sprintf(command, "del %s.bak", filename);
    system(command);
#else
    sprintf(command, "%s.bak", filename);
    unlink(command);
#endif


    FreeBackground(background);
    FreeGrid(user_grid);
    FreeSubgridArray(all_subgrids);
    FreeDatabox(inbox);

    return TCL_OK;
  }
  else
  {
    InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
    return TCL_ERROR;
  }
}


/*-----------------------------------------------------------------------
 * routine for `pfdistondomain' command
 * Description: distributes the file to the virtual distributed file
 *              system.  Based on supplied domain.
 *
 * Cmd. syntax: pfdist filename domain
 *-----------------------------------------------------------------------*/

int            PFDistOnDomainCommand(
                                     ClientData  clientData,
                                     Tcl_Interp *interp,
                                     int         argc,
                                     char *      argv[])
{
  Data       *data = (Data*)clientData;

  char *filename;
  char *filetype;


  Databox *inbox;

  char command[1024];

  if (argc != 3)
  {
    WrongNumArgsError(interp, PFDISTONDOMAINUSAGE);
    return TCL_ERROR;
  }

  filename = argv[1];

  /* Make sure the file extension is valid */

  if ((filetype = GetValidFileExtension(filename)) == (char*)NULL)
  {
    InvalidFileExtensionError(interp, 1, PFDISTONDOMAINUSAGE);
    return TCL_ERROR;
  }

  if (strcmp(filetype, "pfb") == 0)
  {
    /*--------------------------------------------------------------------
     * Get the initial grid info from the database
     *--------------------------------------------------------------------*/
    Background    *background = ReadBackground(interp);

    /*--------------------------------------------------------------------
     * Get inbox from input_filename
     *--------------------------------------------------------------------*/

    inbox = Read(ParflowB, filename);

    /*--------------------------------------------------------------------
     * Get domain from user argument
     *--------------------------------------------------------------------*/
    char       *domain_hashkey = argv[2];
    SubgridArray *domain;
    Tcl_HashEntry *entryPtr;     /* Points to new hash table entry         */
    if ((domain = (SubgridArray*)DataMember(data, domain_hashkey, entryPtr)) == NULL)
    {
      SetNonExistantError(interp, domain_hashkey);
      return TCL_ERROR;
    }

    /*--------------------------------------------------------------------
     * Load the data
     *--------------------------------------------------------------------*/

    if (!domain)
    {
      printf("Incorrect process allocation input\n");
      exit(1);
    }

#ifdef _WIN32
    sprintf(command, "move %s %s.bak", filename, filename);
    system(command);
#else
    sprintf(command, "mv %s %s.bak", filename, filename);
    system(command);
#endif

    Load(ParflowB, filename, domain, background, inbox);

#ifdef _WIN32
    sprintf(command, "del %s.bak", filename);
    system(command);
#else
    sprintf(command, "%s.bak", filename);
    unlink(command);
#endif

    FreeBackground(background);
    FreeDatabox(inbox);

    return TCL_OK;
  }
  else
  {
    InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
    return TCL_ERROR;
  }
}


/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param interp TCL interpreter with the database
 * @param key The key to search for
 * @return The string which matches the search key
 */
char *GetString(Tcl_Interp *interp, char *key)
{
  Tcl_Obj *array_name;
  Tcl_Obj *key_name;
  Tcl_Obj *value;

  int length;

  array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
  key_name = Tcl_NewStringObj(key, strlen(key));

  if ((value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)))
  {
    return strdup(Tcl_GetStringFromObj(value, &length));
  }
  else
  {
    return NULL;
  }
}


/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param interp TCL interpreter with the database
 * @param key The key to search for
 * @return The string which matches the search key
 */
int GetInt(Tcl_Interp *interp, char *key)
{
  Tcl_Obj *array_name;
  Tcl_Obj *key_name;
  Tcl_Obj *value;

  int ret;

  array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
  key_name = Tcl_NewStringObj(key, strlen(key));

  if ((value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)))
  {
    Tcl_GetIntFromObj(interp, value, &ret);
    return ret;
  }
  else
  {
    return -99999999;
  }
}

/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param interp TCL interpreter with the database
 * @param key The key to search for
 * @return The string which matches the search key
 */
int GetIntDefault(Tcl_Interp *interp, char *key, int def)
{
  Tcl_Obj *array_name;
  Tcl_Obj *key_name;
  Tcl_Obj *value;

  int ret;

  array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
  key_name = Tcl_NewStringObj(key, strlen(key));

  if ((value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)))
  {
    Tcl_GetIntFromObj(interp, value, &ret);
    return ret;
  }
  else
  {
    return def;
  }
}

/**
 * Get an input string from the input database.  If the key is not
 * found print an error and exit.
 *
 * There is no checking on what the string contains, anything other than
 * NUL is allowed.
 *
 * @memo Get a string from the input database
 * @param interp TCL interpreter with the database
 * @param key The key to search for
 * @return The string which matches the search key
 */
double GetDouble(Tcl_Interp *interp, char *key)
{
  Tcl_Obj *array_name;
  Tcl_Obj *key_name;
  Tcl_Obj *value;

  double ret;

  array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
  key_name = Tcl_NewStringObj(key, strlen(key));

  if ((value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)))
  {
    Tcl_GetDoubleFromObj(interp, value, &ret);
    return ret;
  }
  else
  {
    return -99999999;
  }
}





/* Function InitPFToolsData - This function is used to allocate memory          */
/* for the structure used to store data set (databoxes).  Other values          */
/* used to keep track of the databoxes are also initialized here.               */
/*                                                                              */
/* Parameters - None                                                            */
/*                                                                              */
/* Return value - Data * - a pointer to the Data structure if one               */
/*                         could be allocated or null otherwise.                */

Data    *InitPFToolsData()
{
  Data *new_data;   /* Data structure used to hold data set hash table */

  if ((new_data = (Data*)calloc(1, sizeof(Data))) == NULL)
    return(NULL);

  Tcl_InitHashTable(&DataMembers(new_data), TCL_STRING_KEYS);

  DataGridType(new_data) = cell;
  DataTotalMem(new_data) = 0;
  DataNum(new_data) = 0;

  return new_data;
}

/* Function AddSubgridArray - This function adds a pointer to a new
 * subgrid array to the hash table of subgrid array pointers.  A
 * hash key used to access the pointer is generated automatically.
 *
 * Parameters
 * ----------
 * Data    *data    - The structure containing the hash table
 * SubgridArray *databox - Data set pointer to be stored int the hash table
 * char    *label   - Label of used to describe the data set
 * char    *hashkey - String used as the new data set's hash key
 *
 * Return value - int - Zero if the space could not be allocated for the
 *                      table entry.  One if the allocation was successful.
 */

int       AddSubgridArray(
                          Data *        data,
                          SubgridArray *subgrid_array,
                          char *        label,
                          char *        hashkey)
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  int new_data;                   /* 1 if the hashkey already exists        */
  int num;                   /* The number of the data set to be added */

  num = 0;

  /* Keep tring to find a unique hash key */
  do
  {
    sprintf(hashkey, "subgridarray%d", num);
    if ((entryPtr = Tcl_CreateHashEntry(&DataMembers(data), hashkey, &new_data))
        == NULL)
      return(0);

    num++;
  }
  while (!new_data);

  /* Truncate the label if it is too large */

  if ((strlen(label) + 1) > MAX_LABEL_SIZE)
    label[MAX_LABEL_SIZE - 1] = 0;

  Tcl_SetHashValue(entryPtr, subgrid_array);

  return(1);
}


/* Function AddData - This function adds a pointer to a new databox to the      */
/* hash table of data set pointers.  A hash key used to access the pointer is   */
/* generated automatically.  The label of the databox is then stored inside     */
/* the databox.                                                                 */
/*                                                                              */
/* Parameters                                                                   */
/* ----------                                                                   */
/* Data    *data    - The structure containing the hash table                   */
/* Databox *databox - Data set pointer to be stored int the hash table          */
/* char    *label   - Label of used to describe the data set                    */
/* char    *hashkey - String used as the new data set's hash key                */
/*                                                                              */
/* Return value - int - Zero if the space could not be allocated for the        */
/*                      table entry.  One if the allocation was successful.     */

int       AddData(
                  Data *   data,
                  Databox *databox,
                  char *   label,
                  char *   hashkey)
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  int new_data;              /* 1 if the hashkey already exists        */
  unsigned long num;          /* The number of the data set to be added */

  num = DataNum(data);

  /* Keep tring to find a unique hash key */
  do
  {
    sprintf(hashkey, "dataset%lu", num);
    if ((entryPtr = Tcl_CreateHashEntry(&DataMembers(data), hashkey, &new_data))
        == NULL)
      return(0);

    num++;
  }
  while (!new_data);

  /* Truncate the label if it is too large */

  if ((strlen(label) + 1) > MAX_LABEL_SIZE)
    label[MAX_LABEL_SIZE - 1] = 0;

  strcpy(DataboxLabel(databox), label);
  Tcl_SetHashValue(entryPtr, databox);

  DataNum(data)++;
  DataTotalMem(data)++;

  return(1);
}


/* Function ClientData - This function is an exit procedure the Tcl will        */
/* execute when the `exit' command is entered on the Tcl command line.  This    */
/* routine will deallocate the hash table and the data sets still in memory     */
/* at the time of exit.                                                         */
/*                                                                              */
/* Parameters                                                                   */
/* ----------                                                                   */
/* ClientData clientData - a pointer to the Data structure                      */
/*                                                                              */
/* Return value - void                                                          */
/*                                                                              */

void               PFTExitProc(
                               ClientData clientData)
{
  Tcl_HashSearch search;
  Tcl_HashEntry  *entryPtr;
  Databox        *databox;

  Data       *data = (Data*)clientData;

  entryPtr = Tcl_FirstHashEntry(&DataMembers(data), &search);

  /* Free the dynamic array in each data box */
  /* and free each data box.                 */

  while (entryPtr != NULL)
  {
    databox = (Databox*)Tcl_GetHashValue(entryPtr);
    FreeDatabox(databox);

    entryPtr = Tcl_NextHashEntry(&search);
  }

  /* Free the hash table */

  Tcl_DeleteHashTable(&DataMembers(data));

  /* Free the struct that was allocated during initialization */

  FreeData(data);
}


/* Function keycompare - This function is called by QuickSort to determine      */
/* which of two hashkeys is lexicographically greater than the other.  This     */
/*                                                                              */
/* Parameters                                                                   */
/* ----------                                                                   */
/* char *key1 - A hash key to be compared                                       */
/* char *key2 - A hash key to be compared                                       */
/*                                                                              */
/* Return value - int - Note: hashkeys have the form: datasetn where n is an    */
/*                            integer.                                          */
/*                      -1 if n1 in key1 is less than n2 in key2                */
/*                       1 if n1 in key1 is greater than n2 in key2             */
/*                       0 if they are equal                                    */

int keycompare(
               const void *key1,
               const void *key2)
{
  char *endnum1;            /* Points to the end of string key1 points to   */
  char *endnum2;            /* Points to the end of string key2 points to   */
  char *number1;            /* Points to the number substring in *key1      */
  char *number2;            /* Points to the number substring in *key2      */
  int num1, num2;           /* The numbers after they are converted to ints */
  int ret;                  /* The return value.                            */

  endnum1 = *(char**)key1;
  endnum2 = *(char**)key2;
  number1 = NULL;
  number2 = NULL;

  /* Find the end of the hash key.  It ends with a */
  /* space character, that separates the key from  */
  /* the datr set label.  This is why we look for  */
  /* the space.                                    */

  while (*endnum1 != ' ')
  {
    /* Point number1 to the begining of the number */
    /* substring.                                   */

    if (!number1 && isdigit(*endnum1))
      number1 = endnum1;

    endnum1++;
  }

  *endnum1 = '\0';

  /* Find the end of the second hash key */

  while (*endnum2 != ' ')
  {
    /* Point number2 to the begining of the number */

    if (!number2 && isdigit(*endnum2))
      number2 = endnum2;

    endnum2++;
  }

  *endnum2 = '\0';

  /* Convert the numbers here */

  num1 = atoi(number1);
  num2 = atoi(number2);

  if (num1 < num2)
    ret = -1;

  else if (num1 > num2)
    ret = 1;

  else
    ret = 0;

  /* Restore the key strings to their original state */

  *endnum1 = ' ';
  *endnum2 = ' ';

  return(ret);
}




/*************************************************************************/
/* Tcl Commands                                                          */
/*************************************************************************/

/* The procedures below have the parameters necessary to make them Tcl commands.*/
/* When a PFTools command is executed from within Tcl, Tcl will send four       */
/* arguments to the procedure which implements the command.  They are described */
/* here.                                                                        */
/*                                                                              */
/* ClientData clientData - Points to data that a Tcl command may need acess to. */
/*                         In the case of the PFTools commands, it will point   */
/*                         to the Data structure which contains the hash table  */
/*                         of data box pointers.                                */
/* Tcl_Interp *interp    - The interpreter being used to execute PFTools        */
/*                         commands.                                            */
/* int argc              - The number of agruments in the PFTools command       */
/* char *argv            - each of the arguments in the command                 */




/*-----------------------------------------------------------------------
 * routine for `pfgetsubbox' command
 *
 * Cmd. Syntax: pfgetsubbox dataset il jl kl iu ju ku
 *-----------------------------------------------------------------------*/

int               GetSubBoxCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;

  int il, jl, kl;
  int iu, ju, ku;
  char          *hashkey_in;
  Databox       *databox_in;
  Databox       *sub_box;

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];


  /* Three arguments must be given */

  if (argc != 8)
  {
    WrongNumArgsError(interp, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }

  hashkey_in = argv[1];

  /* Make sure data sets exist */

  if ((databox_in = DataMember(data, hashkey_in, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey_in);
    return TCL_ERROR;
  }

  /* Make sure il jl kl iu ju and ku are all integers */

  if (Tcl_GetInt(interp, argv[2], &il) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[3], &jl) == TCL_ERROR)
  {
    NotAnIntError(interp, 2, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[4], &kl) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }
  if (Tcl_GetInt(interp, argv[5], &iu) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[6], &ju) == TCL_ERROR)
  {
    NotAnIntError(interp, 2, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[7], &ku) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, GETSUBBOXUSAGE);
    return TCL_ERROR;
  }


  /* All three data sets must belong to grids of the */
  /* same dimensions.                                */


  sub_box = CompSubBox(databox_in, il, jl, kl, iu, ju, ku);

  /* Make sure no error occured while computing. */
  /* Also, make sure there were no memory allocation problems. */

  if (sub_box)
  {
    sprintf(label, "Sub Box %s", hashkey_in);

    /* Make sure the data set pointer was added */
    /* to the hash table successfully.          */

    if (!AddData(data, sub_box, label, newhashkey))
      FreeDatabox(sub_box);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfenlargebox' command
 *
 * Cmd. Syntax: pfenlargebox dataset sx sy sz
 *-----------------------------------------------------------------------*/

int               EnlargeBoxCommand(
                                    ClientData  clientData,
                                    Tcl_Interp *interp,
                                    int         argc,
                                    char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;

  int sx, sy, sz;
  char          *hashkey_in;
  Databox       *databox_in;
  Databox       *new_box;

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];


  /* Three arguments must be given */

  if (argc != 5)
  {
    WrongNumArgsError(interp, ENLARGEBOXUSAGE);
    return TCL_ERROR;
  }

  hashkey_in = argv[1];

  /* Make sure data sets exist */

  if ((databox_in = DataMember(data, hashkey_in, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey_in);
    return TCL_ERROR;
  }

  /* Make sure sx sy sz are all integers */

  if (Tcl_GetInt(interp, argv[2], &sx) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, ENLARGEBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[2], &sy) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, ENLARGEBOXUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[2], &sz) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, ENLARGEBOXUSAGE);
    return TCL_ERROR;
  }


  new_box = EnlargeBox(databox_in, sx, sy, sz);

  /* Make sure no error occured while computing. */
  /* Also, make sure there were no memory allocation problems. */

  if (new_box)
  {
    sprintf(label, "Enlarge Box %s", hashkey_in);

    /* Make sure the data set pointer was added */
    /* to the hash table successfully.          */

    if (!AddData(data, new_box, label, newhashkey))
      FreeDatabox(new_box);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfreload' command
 * Description: One arguments is required; the name of the dataset to
 *              reload.
 * Cmd. syntax: pfreload dataset
 *-----------------------------------------------------------------------*/

int            ReLoadPFCommand(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  char          *hashkey;
  Databox       *databox;

  char          *filename;
  char          *filetype;

  FILE           *fp;

  double default_value = 0.0;

  if (argc != 2)
  {
    WrongNumArgsError(interp, RELOADUSAGE);
    return TCL_ERROR;
  }


  hashkey = argv[1];

  /* Make sure the two data set given exits */
  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }


  /* Get the filename */
  filename = strdup(DataboxLabel(databox));

  if ((fp = fopen(filename, "rb")) == NULL)
  {
    printf("Warning: can't open %s, not reloading this file\n", filename);
    return 0;
  }
  else
  {
    fclose(fp);
  }

  /* Free the old databox */
  FreeDatabox(databox);

  /* Make sure the file extension is valid */

  if ((filetype = GetValidFileExtension(filename)) == (char*)NULL)
  {
    InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
    return TCL_ERROR;
  }

  if (strcmp(filetype, "pfb") == 0)
    databox = ReadParflowB(filename, default_value);
  else if (strcmp(filetype, "pfsb") == 0)
    databox = ReadParflowSB(filename, default_value);
  else if (strcmp(filetype, "sa") == 0)
    databox = ReadSimpleA(filename, default_value);
  else if (strcmp(filetype, "sb") == 0)
    databox = ReadSimpleB(filename, default_value);
  else if (strcmp(filetype, "silo") == 0)
    databox = ReadSilo(filename, default_value);
  else
    databox = ReadRealSA(filename, default_value);

  strcpy(DataboxLabel(databox), filename);
  Tcl_SetHashValue(entryPtr, databox);

  return 0;
}


/*-----------------------------------------------------------------------
 * routine for `pfload' command
 * Description: One or more arguments are required.  If the first
 *              argument is an option, then it tells what the format of
 *              the following filename is.  If no option is given, then
 *              the filename extension is used to determine the type of
 *              the file.
 * Cmd. syntax: pfload [-option] filename [default_value]
 *-----------------------------------------------------------------------*/

int            LoadPFCommand(
                             ClientData  clientData,
                             Tcl_Interp *interp,
                             int         argc,
                             char *      argv[])
{
  Data       *data = (Data*)clientData;

  Databox    *databox;

  char       *filetype, *filename;
  char newhashkey[MAX_KEY_SIZE];

  double default_value = 0.0;


  /* Check and see if there is at least one argument following  */
  /* the command.                                               */

  if (argc == 1)
  {
    WrongNumArgsError(interp, LOADPFUSAGE);
    return TCL_ERROR;
  }

  /* Options are preceeded by a dash.  Check to make sure the */
  /* option is valid.                                         */

  if (*argv[1] == '-')
  {
    /* Skip past the '-' before the file type option */
    filetype = argv[1] + 1;

    if (!IsValidFileType(filetype))
    {
      InvalidOptionError(interp, 1, LOADPFUSAGE);
      return TCL_ERROR;
    }

    /* Make sure a filename follows the option */
    // if (argc == 2 || argc == 3)
    if (argc == 2)
    {
      MissingFilenameError(interp, 1, LOADPFUSAGE);
      return TCL_ERROR;
    }
    else
      filename = argv[2];

    if (argc == 4)
    {
      if (Tcl_GetDouble(interp, argv[3], &default_value) == TCL_ERROR)
      {
        NotADoubleError(interp, 1, LOADPFUSAGE);
        return TCL_ERROR;
      }
    }
  }
  else
  {
    /* If no option is given, then check the extension of the   */
    /* filename.  If the extension on the filename is invalid,  */
    /* then give an error.                                      */

    filename = argv[1];

    /* Make sure the file extension is valid */

    if ((filetype = GetValidFileExtension(filename)) == (char*)NULL)
    {
      InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
      return TCL_ERROR;
    }

    if (argc == 3)
    {
      if (Tcl_GetDouble(interp, argv[2], &default_value) == TCL_ERROR)
      {
        NotADoubleError(interp, 1, LOADPFUSAGE);
        return TCL_ERROR;
      }
    }
  }

  if (strcmp(filetype, "pfb") == 0)
    databox = ReadParflowB(filename, default_value);
  else if (strcmp(filetype, "pfsb") == 0)
    databox = ReadParflowSB(filename, default_value);
  else if (strcmp(filetype, "sa") == 0)
    databox = ReadSimpleA(filename, default_value);
  else if (strcmp(filetype, "sb") == 0)
    databox = ReadSimpleB(filename, default_value);
  else if (strcmp(filetype, "fld") == 0)
    databox = ReadAVSField(filename, default_value);
  else if (strcmp(filetype, "silo") == 0)
    databox = ReadSilo(filename, default_value);
  else
    databox = ReadRealSA(filename, default_value);

  /* Make sure the memory for the data was allocated */

  if (databox)
  {
    /* Make sure the data set pointer was added to */
    /* the hash table successfully.                */

    if (!AddData(data, databox, filename, newhashkey))
      FreeDatabox(databox);
    else
    {
      Tcl_AppendElement(interp, newhashkey);
    }
  }
  else
  {
    ReadWriteError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}

#ifdef HAVE_HDF

/*-----------------------------------------------------------------------
 * routine for `pfloadsds' command
 * Description: The first argument must be the name of the file in HDF
 *              format.  The second argument is used to locate the data
 *              set to be loaded from within the HDF file.
 * Cmd. syntax: pfloadsds filename dsnum
 *-----------------------------------------------------------------------*/

int            LoadSDSCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data       *data = (Data*)clientData;

  Databox    *databox;
  char newhashkey[MAX_KEY_SIZE];

  char       *filename;
  char       *label;
  int ds_num;


  /* There must be at least two arguments */

  if (argc == 1)
  {
    WrongNumArgsError(interp, LOADSDSUSAGE);
    return TCL_ERROR;
  }

  /* Give an error if the ds number is missing */

  if (argc == 3)
  {
    /* The argument following the filename should be an integer */

    if (Tcl_GetInt(interp, argv[2], &ds_num) == TCL_ERROR)
    {
      NotAnIntError(interp, 2, LOADSDSUSAGE);
      return TCL_ERROR;
    }
  }

  else
    ds_num = 0;

  filename = argv[1];

  databox = ReadSDS(filename, ds_num);

  label = (char*)calloc(strlen(filename) + 20, sizeof(char));
  sprintf(label, "SDS# %d of HDF `%s'", ds_num, filename);

  /* Allocate data for the data sets and place */
  /* pointers to them in the Tcl hash table.   */

  if (databox)
  {
    if (!AddData(data, databox, label, newhashkey))
      FreeDatabox(databox);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    ReadWriteError(interp);
    return TCL_ERROR;
  }

  free((char*)label);
  return TCL_OK;
}

#endif


/*-----------------------------------------------------------------------
 * routine for `pfsave' command
 * Description: The first argument to this command is the hashkey of the
 *              dataset to be saved, the second is the format of the
 *              file the data is to be saved in.
 * Cmd. syntax: pfsave dataset -filetype filename
 *-----------------------------------------------------------------------*/

int               SavePFCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *filetype, *filename;
  FILE          *fp = NULL;

  char          *hashkey;
  Tcl_HashEntry *entryPtr;
  Databox       *databox;



  /* The command three arguments */

  if (argc != 4)
  {
    WrongNumArgsError(interp, SAVEPFUSAGE);
    return TCL_ERROR;
  }

  hashkey = argv[1];

  /* Make sure the dataset exists */

  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }

  /* Check for an option specifying the file type */

  if (*argv[2] != '-')
  {
    MissingOptionError(interp, 1, SAVEPFUSAGE);
    return TCL_ERROR;
  }

  filetype = argv[2] + 1;

  /* Validate the file type */

  if (!IsValidFileType(filetype))
  {
    InvalidOptionError(interp, 1, SAVEPFUSAGE);
    return TCL_ERROR;
  }

  filename = argv[3];

  /* Execute the appropriate file printing routine */
  if (strcmp(filetype, "pfb") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintParflowB(fp, databox);
  }
  else if (strcmp(filetype, "sa") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintSimpleA(fp, databox);
  }


  else if (strcmp(filetype, "sa2d") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintSimpleA2D(fp, databox);
  }


  else if (strcmp(filetype, "sb") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintSimpleB(fp, databox);
  }
  else if (strcmp(filetype, "fld") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintAVSField(fp, databox);
  }
  else if (strcmp(filetype, "vis") == 0)
  {
    /* Make sure the file could be opened */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    PrintVizamrai(fp, databox);
  }
#ifdef HAVE_SILO
  else if (strcmp(filetype, "silo") == 0)
  {
    PrintSilo(filename, databox);
  }
#endif

  /* Close the file, if opened */
  if (fp)
  {
    fclose(fp);
  }
  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfvtksave' command
 * Description: The first argument to this command is the hashkey of the
 *              dataset to be saved, the second is the format of the
 *              file the data is to be saved in. The remaining arguments
 *              are optional and can be in any order.
 * Cmd. syntax: pftkvsave dataset -filetype filename [-flt -dem demname -var variable name]
 * NBE: May 2015
 *-----------------------------------------------------------------------*/

int SavePFVTKCommand(
                     ClientData  clientData,
                     Tcl_Interp *interp,
                     int         argc,
                     char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *filename;
  char          *varname;
  int i, j, k;
  int flt = 0;
  char          *dzlist_in;
  char*         Endp1 = 0;
  int dz_els;

  FILE          *fp = NULL;

  char          *hashkey, *demhash;
  Tcl_HashEntry *entryPtr;
  Databox       *databox, *databox2;

// Fail based on number of arguments
  if ((argc < 4) || (argc > 11))
  {
    printf("ERROR: Wrong number of arguments");
    return TCL_ERROR;
  }

  if (strcmp(argv[2], "-vtk") == 0)
  {
    // Do nothing
  }
  else if (strcmp(argv[2], "–vtk") == 0)
  {
    argv[2] = "-vtk";     // Corrects long dash if present
  }
  else if (strcmp(argv[2], "-clmvtk") == 0)
  {
    // Do nothing
  }
  else if (strcmp(argv[2], "–clmvtk") == 0)
  {
    argv[2] = "-clmvtk";     // Corrects long dash if present
  }
  else
  {
    printf("ERROR: Invalid file type. Must be vtk or clmvtk");
    return TCL_ERROR;
  }

  filename = argv[3];

  // Initialize to stop the warning
  hashkey = argv[1];
  demhash = argv[2];

  varname = "variable"; // Initialize some defaults
  dzlist_in = "-999999";

  /* Scan through the argument list and match options */
  for (i = 4; i < argc; ++i)
  {
    if ((strcmp(argv[i], "-flt") == 0) || (strcmp(argv[i], "–flt") == 0))
    {
      flt = 1;
    }
    if ((strcmp(argv[i], "-dem") == 0) || (strcmp(argv[i], "–dem") == 0))
    {
      demhash = argv[i + 1];
    }
    if ((strcmp(argv[i], "-var") == 0) || (strcmp(argv[i], "–var") == 0))
    {
      varname = argv[i + 1];
    }
    if ((strcmp(argv[i], "-tfg") == 0) || (strcmp(argv[i], "–tfg") == 0))
    {
      dzlist_in = argv[i + 1];
    }                                                                                          // NBE 20150813

//        printf("Arg[%i]: %s \n",i,argv[i]); //NBE
  }

  if (strcmp(dzlist_in, "-999999") != 0)
  {
    dz_els = strtod(dzlist_in, &Endp1);
  }
  else
  {
    dz_els = 1;
  }

  if (dz_els == 0)
  {
    printf("ERROR: dz_list has length zero \n");
    return TCL_ERROR;
  }
  else if (dz_els < 0)
  {
    printf("ERROR: dz_list has length less than zero \n");
    return TCL_ERROR;
  }

  double dzlst[dz_els];

  if (strcmp(dzlist_in, "-999999") != 0)
  {
    for (i = 0; i < dz_els; ++i)
    {
      dzlst[i] = strtod(Endp1, &Endp1);
    }
  }
  else
  {
    dzlst[0] = 1.0;
  }

  /* Make sure the MAIN dataset exists */
  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }

  if (argc == 4)     //ARGC LOGICAL
  {     // Basic write, structured points
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }

    if ((strcmp(argv[2], "-vtk") == 0))
    {
      PrintVTK(fp, databox, varname, flt);
    }
    else if ((strcmp(argv[2], "-clmvtk") == 0))
    {
      PrintCLMVTK(fp, databox, varname, flt);
    }
    else
    {
      printf("ERROR: Invalid filetype \n");
      return TCL_ERROR;
    }
    fclose(fp);

    return TCL_OK;
  }
  else
  {
    if (strcmp(argv[2], demhash) == 0)  // Only a variable name was found
    {
      if ((fp = fopen(filename, "wb")) == NULL)
      {
        ReadWriteError(interp);
        return TCL_ERROR;
      }
      if ((strcmp(argv[2], "-vtk") == 0))
      {
        PrintVTK(fp, databox, varname, flt);
      }
      else if ((strcmp(argv[2], "-clmvtk") == 0))
      {
        PrintCLMVTK(fp, databox, varname, flt);
      }
      else
      {
        printf("ERROR: Invalid filetype \n");
        return TCL_ERROR;
      }
      fclose(fp);

      return TCL_OK;
    }

    if ((demhash == NULL) || ((databox2 = DataMember(data, demhash, entryPtr)) == NULL))
    {
      printf("WARNING: dem not found, reverting to structured point VTK \n");
      if ((fp = fopen(filename, "wb")) == NULL)
      {
        ReadWriteError(interp);
        return TCL_ERROR;
      }
      if ((strcmp(argv[2], "-vtk") == 0))
      {
        PrintVTK(fp, databox, varname, flt);
      }
      else if ((strcmp(argv[2], "-clmvtk") == 0))
      {
        PrintCLMVTK(fp, databox, varname, flt);
      }
      else
      {
        printf("ERROR: Invalid filetype \n");
        return TCL_ERROR;
      }
      fclose(fp);
      return TCL_OK;
    }

    /* Since the dem was found, use it to shift the vertical */

    // Make sure the grids are same size
    int nx = DataboxNx(databox);        // Data
    int ny = DataboxNy(databox);
    int nz = DataboxNz(databox);
    int nx2 = DataboxNx(databox2);        // DEM
    int ny2 = DataboxNy(databox2);
    int nz2 = DataboxNz(databox2);

    if ((nx != nx2) || (ny != ny2))
    {
      printf("ERROR: Grid dimensions do not match! \n");
      return TCL_ERROR;
    }

    double x = DataboxX(databox);
    double y = DataboxY(databox);

    double dx = DataboxDx(databox);
    double dy = DataboxDy(databox);
    double dz = DataboxDz(databox);

    int nxp = nx + 1;
    int nyp = ny + 1;
    int nzp = nz + 1;
    int nxyzp = nxp * nyp * nzp;
    int n = 0;

    double *Xp;

    double elev = 0;
    double zoffset = nz * dz;

    /* CLM mode uses 1-layer, adjust accordingly */
    int nzl = 0;

    int imn, jmn, imx, jmx;

    if ((strcmp(argv[2], "-clmvtk") == 0))
    {
      nzp = 2;
      nzl = 0;
      zoffset = dz;
      Xp = (double*)malloc(sizeof(double) * (nxp * nyp * 2) * 3);
      // Block for CLM case
      for (k = nzl; k < nzp; ++k)
      {
        for (j = 0; j < nyp; ++j)
        {
          for (i = 0; i < nxp; ++i)
          {
            Xp[n] = x + i * dx;
            Xp[n + 1] = y + j * dy;
            //
            if (1 == 0)
            {
              /* Simple method, use cell elevation, duplicate edge point */
              imn = 0;
              jmn = 0;
              if (i >= nx)
              {
                imn = -1;
              }
              if (j >= ny)
              {
                jmn = -1;
              }
              elev = *DataboxCoeff(databox2, i + imn, j + jmn, nz2 - 1);
              Xp[n + 2] = elev - zoffset + dz * (double)k;
            }
            else
            {
              /*  Simple interpolation */
              imn = -1;
              jmn = -1;
              imx = 0;
              jmx = 0;
              if (i == 0)
              {
                imn = 0;
              }
              if (i >= nx)
              {
                imx = -1;
              }
              if (j == 0)
              {
                jmn = 0;
              }
              if (j >= ny)
              {
                jmx = -1;
              }
              elev = (*DataboxCoeff(databox2, i + imn, j + jmn, nz2 - 1) + *DataboxCoeff(databox2, i + imx, j + jmn, nz2 - 1) +
                      *DataboxCoeff(databox2, i + imn, j + jmx, nz2 - 1) + *DataboxCoeff(databox2, i + imx, j + jmx, nz2 - 1)) / 4.0;
              Xp[n + 2] = elev - zoffset + dz * (double)k;
            }
            n = n + 3;
          }
        }
      }
      // End block for CLM case
    }
    else
    {
      Xp = (double*)malloc(sizeof(double) * nxyzp * 3);

      if ((strcmp(dzlist_in, "-999999") != 0) && (dz_els != (nzp - 1)))
      {
        printf("ERROR: Num els of Var_dz list not equal to nz!  \n");
        return TCL_ERROR;
      }

      // Block for normal datafiles
      double c_dz[nzp];
      c_dz[0] = 0.0;

      if (strcmp(dzlist_in, "-999999") != 0)
      {
        zoffset = 0.0;
        for (i = 0; i < dz_els; ++i)
        {
          zoffset = zoffset + dzlst[i];
        }

        for (i = 1; i < nzp; ++i)
        {
          c_dz[i] = c_dz[i - 1] + dzlst[i - 1];
        }
      }
      else
      {
        c_dz[0] = 0;
        for (i = 1; i < nzp; ++i)
        {
          c_dz[i] = c_dz[i - 1] + dz;
        }
      }

      /* compute point locations */
      for (k = nzl; k < nzp; ++k)
      {
        for (j = 0; j < nyp; ++j)
        {
          for (i = 0; i < nxp; ++i)
          {
            Xp[n] = x + i * dx;
            Xp[n + 1] = y + j * dy;
            //
            if (1 == 0)
            {
              /* Simple method, use cell elevation, duplicate edge point */
              imn = 0;
              jmn = 0;
              if (i >= nx)
              {
                imn = -1;
              }
              if (j >= ny)
              {
                jmn = -1;
              }
              elev = *DataboxCoeff(databox2, i + imn, j + jmn, nz2 - 1);
              //                        Xp[n+2] = elev - zoffset + dz*(double)k;  // NBE commented
              Xp[n + 2] = elev - zoffset + c_dz[k];
            }
            else
            {
              /*  Simple interpolation */
              imn = -1;
              jmn = -1;
              imx = 0;
              jmx = 0;
              if (i == 0)
              {
                imn = 0;
              }
              if (i >= nx)
              {
                imx = -1;
              }
              if (j == 0)
              {
                jmn = 0;
              }
              if (j >= ny)
              {
                jmx = -1;
              }
              elev = (*DataboxCoeff(databox2, i + imn, j + jmn, nz2 - 1) + *DataboxCoeff(databox2, i + imx, j + jmn, nz2 - 1) +
                      *DataboxCoeff(databox2, i + imn, j + jmx, nz2 - 1) + *DataboxCoeff(databox2, i + imx, j + jmx, nz2 - 1)) / 4.0;
              //                        Xp[n+2] = elev - zoffset + dz*(double)k; // NBE commented
              Xp[n + 2] = elev - zoffset + c_dz[k];
            }
            n = n + 3;
          }
        }
      }

      // End of normal block
    }

    /* Make sure the file could be opened, then write to it */
    if ((fp = fopen(filename, "wb")) == NULL)
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
    if ((strcmp(argv[2], "-vtk") == 0))
    {
      PrintTFG_VTK(fp, databox, Xp, varname, flt);
      return TCL_OK;
    }
    else if ((strcmp(argv[2], "-clmvtk") == 0))
    {
      PrintTFG_CLMVTK(fp, databox, Xp, varname, flt);
      return TCL_OK;
    }
    else
    {
      printf("ERROR: Invalid filetype \n");
      return TCL_ERROR;
    }
    fclose(fp);
    return TCL_OK;
  }   // END OF ARGC LOGICAL

  /* Close the file, if still opened */
  if (fp)
  {
    fclose(fp);
  }
  return TCL_OK;
}

// END of PFVsave
/* -------------------------------------------------------------------------------------- */

/*-----------------------------------------------------------------------
 * routine for `pfpatchysolid' command
 *-----------------------------------------------------------------------*/

int MakePatchySolidCommand(
                           ClientData  clientData,
                           Tcl_Interp *interp,
                           int         argc,
                           char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *filename, *vtk_filename;
  int i;

  FILE          *fp = NULL;
  FILE          *fp_vtk = NULL;
  char          *maskkey, *tophash, *bothash;
  Tcl_HashEntry *entryPtr;
  Databox       *databox, *top_databox, *bot_databox;

  // Fail based on an invalid number of arguments
  if ((argc < 5) || (argc > 13))
  {
    printf("\n ERROR (pfpatchysolid): Invalid number of arguments\n");
    return TCL_ERROR;
  }

  // Initialize
  filename = "SolidFile.pfsol";
  vtk_filename = NULL;
  maskkey = NULL;
  tophash = NULL;
  bothash = NULL;

  int msk = 0, top = 0, bot = 0, vtk = 0, sub_patch = 0, bin_out = 0;  //Initalize flags for the options
  // Note: bin_out is a place holder for a yet to be added BINARY solid file...
  //       vtk is a flag for writing a BINARY VTK of the solid file too

  // Scan through the argument list and match up the options
  for (i = 1; i < argc; ++i)
  {
    if ((strcmp(argv[i], "-msk") == 0) || (strcmp(argv[i], "–msk") == 0))
    {
      msk = 1;
      maskkey = argv[i + 1];
    }
    if ((strcmp(argv[i], "-top") == 0) || (strcmp(argv[i], "–top") == 0))
    {
      top = 1;
      tophash = argv[i + 1];
    }
    if ((strcmp(argv[i], "-bot") == 0) || (strcmp(argv[i], "–bot") == 0))
    {
      bot = 1;
      bothash = argv[i + 1];
    }
    if ((strcmp(argv[i], "-sub") == 0) || (strcmp(argv[i], "–sub") == 0))
    {
      /* Subdivide patches by face direction */
      sub_patch = 1;
    }

    char          *file1_ext;
    if ((strcmp(argv[i], "-vtk") == 0) || (strcmp(argv[i], "–vtk") == 0))
    {
      vtk = 1;
      vtk_filename = argv[i + 1];
      file1_ext = strrchr(vtk_filename, '.');
      if (strcmp(file1_ext + 1, "vtk") != 0)
      {
        printf("ERROR (pfpatchysolid): vtk file extension must be .vtk\n");
        printf("                       detected %s\n", file1_ext + 1);
        return TCL_ERROR;
      }
    }

    if ((strcmp(argv[i], "-pfsol") == 0) || (strcmp(argv[i], "–pfsol") == 0))
    {
      filename = argv[i + 1];
      file1_ext = strrchr(filename, '.');
      if (strcmp(file1_ext + 1, "pfsol") != 0)
      {
        printf("ERROR (pfpatchysolid): ASCII file extension must be .pfsol\n");
        printf("                       detected %s\n", file1_ext + 1);
        return TCL_ERROR;
      }
    }
    else if ((strcmp(argv[i], "-pfsolb") == 0) || (strcmp(argv[i], "–pfsolb") == 0))
    {
      bin_out = 1;
      filename = argv[i + 1];
      file1_ext = strrchr(filename, '.');
      if (strcmp(file1_ext + 1, "pfsolb") != 0)
      {
        printf("ERROR (pfpatchysolid): Binary file extension must be .pfsolb\n");
        printf("                       detected %s\n", file1_ext + 1);
        return TCL_ERROR;
      }
    }
  }

  if ((top == 0) || (bot == 0)) // (msk==0) ||
  {
    printf("\n ERROR (pfpatchysolid): Missing required arguments. Please add:\n");
    // if (msk==0)
    // {
    //   printf("      -msk <Mask_dataset_ID>\n");
    // }
    if (top == 0)
    {
      printf("      -top <Top_surface_dataset_ID>\n");
    }
    if (top == 0)
    {
      printf("      -bot <Bottom_surface_dataset_ID>\n");
    }
    return TCL_ERROR;
  }

  if (strcmp(filename, "SolidFile.pfsol") == 0)
  {
    printf("WARNING (pfpatchysolid): No solid file name specified, default is: SolidFile.pfsol\n");
  }

  /* Make sure the MAIN datasets exists */
  if ((top_databox = DataMember(data, tophash, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, maskkey);
    return TCL_ERROR;
  }
  if ((bot_databox = DataMember(data, bothash, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, maskkey);
    return TCL_ERROR;
  }

  if (msk == 1)
  {
    if ((databox = DataMember(data, maskkey, entryPtr)) == NULL)
    {
      SetNonExistantError(interp, maskkey);
      return TCL_ERROR;
    }
  }
  else
  {
    // No mask provided so make one of all ones matching the size of top
    int NX = DataboxNx(top_databox);
    int NY = DataboxNy(top_databox);
    int NZ = DataboxNz(top_databox);
    double X = DataboxX(top_databox);
    double Y = DataboxY(top_databox);
    double Z = DataboxZ(top_databox);
    double DX = DataboxDx(top_databox);
    double DY = DataboxDy(top_databox);
    double DZ = DataboxDz(top_databox);

    databox = NewDataboxDefault(NX, NY, NZ, X, Y, Z, DX, DY, DZ, 1.0);
  }

  if ((fp = fopen(filename, "wb")) == NULL)
  {
    printf("\n ERROR (pfpatchysolid): pfsol output file could not be opened\n");
    ReadWriteError(interp);
    return TCL_ERROR;
  }
  if (vtk == 1)
  {
    if ((fp_vtk = fopen(vtk_filename, "wb")) == NULL)
    {
      printf("\n ERROR (pfpatchysolid): vtk output file could not be opened\n");
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  // #include <time.h>
  // clock_t start_time, end_time;
  // start_time=clock();
  // double run_time;

  i = MakePatchySolid(fp, fp_vtk, databox, top_databox, bot_databox, sub_patch, bin_out);

  // end_time=clock();
  // run_time=(double)(end_time-start_time) / CLOCKS_PER_SEC;
  // printf("Elapsed time: %f\n", run_time);

  if (i != 0)
  {
    if (i == -2) // Flag for errors deriving from input issues
    {
      printf("\n ERROR (pfpatchysolid): Error with inputs\n");
    }
    else  // Everything else...
    {
      printf("\n ERROR (pfpatchysolid): Other internal error\n");
    }
  }

  /* Close the file, if still opened */
  if (fp)
  {
    fclose(fp);
  }
  if (fp_vtk)
  {
    fclose(fp_vtk);
  }
  return TCL_OK;
}

// END of PFPATCHYSOLID

// ----------------------------------------------------------------------------
//  Convert an ascii solid file (.pfsol) to a binary solid (.pfsolb)or vice versa
//   order of specified files determines function based on file extension. Converts
//   from first filename format to the second filename format
// ----------------------------------------------------------------------------
int pfsolFmtConvert(
                    ClientData  clientData,
                    Tcl_Interp *interp,
                    int         argc,
                    char *      argv[])
{
  char          *bin_filename, *ascii_filename;
  char          *file1_name, *file2_name, *file1_ext, *file2_ext;
  int bin2asc;

  FILE          *fp_bin = NULL;
  FILE          *fp_ascii = NULL;

  // Perform some checks before calling the appropriate routine in solidtools.c

  if (argc != 3)
  {
    printf("ERROR (pfsolidfmtconvert): Two input file names are required\n");
    return TCL_ERROR;
  }

  file1_name = argv[1];
  file2_name = argv[2];

  file1_ext = strrchr(file1_name, '.');
  file2_ext = strrchr(file2_name, '.');

  if (!file1_ext)
  {
    printf("ERROR (pfsolidfmtconvert): Missing extension on file 1, must be .pfsol or .pfsolb\n");
    return TCL_ERROR;
  }
  if (!file2_ext)
  {
    printf("ERROR (pfsolidfmtconvert): Missing extension on file 2, must be .pfsol or .pfsolb\n");
    return TCL_ERROR;
  }
  if ((strcmp(file1_ext + 1, "pfsol") != 0) && (strcmp(file1_ext + 1, "pfsolb") != 0))
  {
    printf("ERROR (pfsolidfmtconvert): File extension on file 1 must be .pfsol or .pfsolb\n");
    printf("                           detected .%s\n", file1_ext + 1);
    return TCL_ERROR;
  }
  if ((strcmp(file2_ext + 1, "pfsol") != 0) && (strcmp(file2_ext + 1, "pfsolb") != 0))
  {
    printf("ERROR (pfsolidfmtconvert): File extension on file 2 must be .pfsol or .pfsolb\n");
    printf("                           detected .%s\n", file2_ext + 1);
    return TCL_ERROR;
  }
  if (strcmp(file2_ext + 1, file1_ext + 1) == 0)
  {
    printf("ERROR (pfsolidfmtconvert): File extensions must be different\n");
    return TCL_ERROR;
  }

  if (strcmp(file1_ext + 1, "pfsol") == 0)
  {
    ascii_filename = file1_name;
    bin_filename = file2_name;
    bin2asc = 0;

    if ((fp_ascii = fopen(ascii_filename, "r")) == NULL)
    {
      printf("\n ERROR (pfsolidfmtconvert): ascii file file could not be opened\n");
      ReadWriteError(interp);
      return TCL_ERROR;
    }
    if ((fp_bin = fopen(bin_filename, "wb")) == NULL)
    {
      printf("\n ERROR (pfsolidfmtconvert): binary file file could not be opened\n");
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  else
  {
    ascii_filename = file2_name;
    bin_filename = file1_name;
    bin2asc = 1;

    if ((fp_ascii = fopen(ascii_filename, "w")) == NULL)
    {
      printf("\n ERROR (pfsolidfmtconvert): ascii file file could not be opened\n");
      ReadWriteError(interp);
      return TCL_ERROR;
    }
    if ((fp_bin = fopen(bin_filename, "rb")) == NULL)
    {
      printf("\n ERROR (pfsolidfmtconvert): binary file file could not be opened\n");
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  int out_status = 0;
  if (bin2asc == 0)
  {
    out_status = ConvertPfsolAscii2Bin(fp_ascii, fp_bin);
  }
  else
  {
    out_status = ConvertPfsolBin2Ascii(fp_bin, fp_ascii);
  }

  if (fp_ascii)
  {
    fclose(fp_ascii);
  }
  if (fp_bin)
  {
    fclose(fp_bin);
  }

  if (out_status != 0)
  {
    printf("\n ERROR (pfsolidfmtconvert): Problem during file conversion\n");
    return TCL_ERROR;
  }

  return TCL_OK;
}

/* -------------------------------------------------------------------------------------- */

#ifdef HAVE_HDF4

/*-----------------------------------------------------------------------
 * routine for `pfsavesds' command
 * Description: The first argument is the data set to be saved in HDF
 *              format, the second is the file type, and the third is the
 *              filename.
 * Cmd. syntax: pfsavesds dataset -filetype filename
 *-----------------------------------------------------------------------*/

int              SaveSDSCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  char          *hashkey;
  char          *printoption;
  char          *filename;
  int32 type;


  /* Make sure that there are enough arguments */

  if (argc != 4)
  {
    WrongNumArgsError(interp, SAVESDSUSAGE);
    return TCL_ERROR;
  }

  hashkey = argv[1];

  /* Make sure that the data set exists */

  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }

  /* Make sure that there is a file type specified */

  if (*argv[2] != '-')
  {
    MissingOptionError(interp, 2, SAVESDSUSAGE);
    return TCL_ERROR;
  }

  printoption = argv[2] + 1;
  filename = argv[3];

  /* Determine the format that the data will be saved in */

  if (strcmp(printoption, "float32") == 0)
    type = DFNT_FLOAT32;
  else if (strcmp(printoption, "float64") == 0)
    type = DFNT_FLOAT64;
  else if (strcmp(printoption, "int8") == 0)
    type = DFNT_INT8;
  else if (strcmp(printoption, "uint8") == 0)
    type = DFNT_UINT8;
  else if (strcmp(printoption, "int16") == 0)
    type = DFNT_INT16;
  else if (strcmp(printoption, "uint16") == 0)
    type = DFNT_UINT16;
  else if (strcmp(printoption, "int32") == 0)
    type = DFNT_INT32;
  else if (strcmp(printoption, "uint32") == 0)
    type = DFNT_UINT32;
  else if (strcmp(printoption, "int32") == 0)
    type = DFNT_INT32;
  else
  {
    InvalidOptionError(interp, 2, SAVESDSUSAGE);
    return TCL_ERROR;
  }

  /* Make sure the file could be written to */

  if (!PrintSDS(filename, type, databox))
    ReadWriteError(interp);

  else
    return TCL_OK;
}
#endif


/*-----------------------------------------------------------------------
 * list a range of data members in Data structure
 * Description: If an argument is given, the it should be the name of a
 *              loaded dataset.  The string returned should be the name
 *              of the dataset followed by its description.  If no
 *              argument is given, then all of the data set names followed
 *              by their description is returned to the TCL interpreter.
 * Cmd. Syntax: pfgetlist [dataset]
 *-----------------------------------------------------------------------*/

int                GetListCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Data           *data = (Data*)clientData;

  Tcl_HashEntry  *entryPtr;
  Tcl_HashSearch search;
  Databox        *databox;

  Tcl_DString dspair;
  Tcl_DString result;
  char           *pair;
  char           *hashkey;

  char           **list;
  int i;


  /* There must not be more than 2 arguments */

  if (argc > 2)
  {
    WrongNumArgsError(interp, GETLISTUSAGE);
    return TCL_ERROR;
  }

  Tcl_DStringInit(&dspair);
  Tcl_DStringInit(&result);

  /* Return a list of pairs where each pair contains a hashkey    */
  /* and a the dataset's label associated with the hashkey.       */

  if (argc == 1)
  {
    /* Create an array that all of the key-label) pairs        */
    /* will be placed.                                         */

    entryPtr = Tcl_FirstHashEntry(&DataMembers(data), &search);
    list = (char**)calloc(DataTotalMem(data), sizeof(char *));

    /* Copy the pairs from the hash table to the array */
    /* of strings.                                     */

    for (i = 0; i < DataTotalMem(data); i++)
    {
      databox = (Databox*)Tcl_GetHashValue(entryPtr);
      hashkey = Tcl_GetHashKey(&DataMembers(data), entryPtr);

      Tcl_DStringAppendElement(&dspair, hashkey);

      list[i] = (char*)calloc(MAX_LABEL_SIZE, sizeof(char));
      strcpy(list[i], Tcl_DStringAppendElement(&dspair,
                                               DataboxLabel(databox)));
      Tcl_DStringFree(&dspair);

      entryPtr = Tcl_NextHashEntry(&search);
    }

    qsort(list, DataTotalMem(data), sizeof(char *), keycompare);

    /* Append the sorted elements to the Tcl result string */

    for (i = 0; i < DataTotalMem(data); i++)
    {
      Tcl_DStringAppendElement(&result, list[i]);
      free((char*)list[i]);
    }

    free((char**)list);
  }

  /* Same as above, labels are only returned for each data set       */
  /* that is an argument to pflist.                                  */

  else if (argc == 2)
  {
    hashkey = argv[1];

    /* dspair will hold the hashkey and the description */

    Tcl_DStringAppendElement(&dspair, hashkey);

    /* The hash entry does not exist */

    if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
    {
      SetNonExistantError(interp, hashkey);
      Tcl_DStringFree(&dspair);
      Tcl_DStringFree(&result);
      return TCL_ERROR;
    }

    /* Create the key-lable pair */

    else
      pair = Tcl_DStringAppendElement(&dspair, DataboxLabel(databox));

    /* Append the key-label pair onto the result */

    Tcl_DStringAppendElement(&result, pair);
    Tcl_DStringFree(&dspair);
  }

  Tcl_DStringResult(interp, &result);
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfgetelt' command
 * Description: The i, j, and k coordinates are given first followed by
 *              a hash key representing any data set.  The element's value
 *              will be returned.
 * Cmd. syntax: pfgetelt dataset i j k
 *-----------------------------------------------------------------------*/

int               GetEltCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Data          *data = (Data*)clientData;

  int i, j, k;
  char          *hashkey;

  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  Tcl_Obj       *result;

  hashkey = argv[1];

  /* There must be four arguments given to pfgetelt */

  if (argc != 5)
  {
    WrongNumArgsError(interp, GETELTUSAGE);
    return TCL_ERROR;
  }

  /* Make sure the data set name(hashkey) exists */

  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, argv[1]);
    return TCL_ERROR;
  }

  /* Make sure i j and k are all integers */

  if (Tcl_GetInt(interp, argv[2], &i) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, GETELTUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[3], &j) == TCL_ERROR)
  {
    NotAnIntError(interp, 2, GETELTUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[4], &k) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, GETELTUSAGE);
    return TCL_ERROR;
  }

  /* The coordinates must be in range */

  if (!InRange(i, j, k, databox))
  {
    OutOfRangeError(interp, i, j, k);
    return TCL_ERROR;
  }

  result = Tcl_NewDoubleObj(*DataboxCoeff(databox, i, j, k));
  Tcl_SetObjResult(interp, result);

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfgetgrid' command
 * Description: The hash key for a data set is given as the argument and
 * a list of the following form is returned: {nx ny nz} {x y z} {dx dy dz}
 * The values nx, ny, and nz are the number of points in each direction.
 * Values x, y, and z describe the origin.  Values dx, dy, and dz are the
 * intervals between coordinates along each axis.
 *
 * Cmd. syntax: pfgetgrid dataset
 *-----------------------------------------------------------------------*/

int               GetGridCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  /* One argument must be passed to the pfgetgrid command */

  if (argc != 2)
  {
    WrongNumArgsError(interp, GETGRIDUSAGE);
    return TCL_ERROR;
  }

  /* Make sure that the dat set given exists */

  if ((databox = DataMember(data, argv[1], entryPtr)) == NULL)
  {
    SetNonExistantError(interp, argv[1]);
    return TCL_ERROR;
  }

  GetDataboxGrid(interp, databox);
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfgridtype' command
 * Description: The argument is either vertex or cell.  A vertex grid
 *              type means that the grid is vertex centerd and the cell
 *              type means that the grid is cell centered.  If no argument
 *              is given, then the current setting is returned as the TCL
 *              result.
 * Cmd. syntax: pfgridtype [vertex | cell]
 *-----------------------------------------------------------------------*/

int            GridTypeCommand(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Data       *data = (Data*)clientData;

  char       *newtype;


  /* There must be zero or one arguments */

  if (argc > 2)
  {
    WrongNumArgsError(interp, GRIDTYPEUSAGE);
    return TCL_ERROR;
  }

  /* If there is an argument, we must change the grid type */

  if (argc == 2)
  {
    newtype = argv[1];

    /* Determine what the new grid type is */

    if (strcmp(newtype, "vertex") == 0)
      DataGridType(data) = vertex;
    else if (strcmp(newtype, "cell") == 0)
      DataGridType(data) = cell;
    else
    {
      InvalidArgError(interp, 1, GRIDTYPEUSAGE);
      return TCL_ERROR;
    }
  }

  /* Append the new grid type to the Tcl result */

  if (DataGridType(data) == vertex)
    Tcl_SetResult(interp, "vertex", TCL_STATIC);
  else
    Tcl_SetResult(interp, "cell", TCL_STATIC);

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcvel' commands
 * Description: Two hash keys representing the conductivity and pressure
 *              head data sets are required.  three new data sets are
 *              created by using the conductivity and pressure head to
 *              compute the Darcy velocity in the cells.  The data sets
 *              representing the x, y, and z component of the velocity
 *              are returned to TCL upon successful completion.
 * Cmd. syntax: pfcvel conductivity phead
 *-----------------------------------------------------------------------*/

int               CVelCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *cond, *pHead;
  char newhashkey[96];
  Tcl_HashEntry *entryPtr;
  Databox       *databoxk, *databoxh;

  char label[MAX_LABEL_SIZE];
  char          *component_ptr;

  Databox      **vel;

  /* There must two data set names given */

  if (argc != 3)
  {
    WrongNumArgsError(interp, CVELUSAGE);
    return TCL_ERROR;
  }

  cond = argv[1];
  pHead = argv[2];

  /* Make sure the sets exist */

  if ((databoxk = DataMember(data, cond, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, cond);
    return TCL_ERROR;
  }

  if ((databoxh = DataMember(data, pHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pHead);
    return TCL_ERROR;
  }

  /* The sets must have the same dimensions */

  if (!SameDimensions(databoxk, databoxh))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  vel = CompCellVel(databoxk, databoxh);
  sprintf(label, "Darcy velocity in cells for `%s' and `%s'",
          cond, pHead);

  component_ptr = label + strlen(label);

  /* Add the velocity components to the hash table */

  if (vel)
  {
    /* Make sure each new data set pointer is added to */
    /* the data set successfully.                      */

    sprintf(component_ptr, " (x velocity)");

    if (!AddData(data, vel[0], label, newhashkey))
      FreeDatabox(vel[0]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (y velocity)");
    if (!AddData(data, vel[1], label, newhashkey))
      FreeDatabox(vel[1]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (z velocity)");
    if (!AddData(data, vel[2], label, newhashkey))
      FreeDatabox(vel[2]);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  free(vel);
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfvvel' commands
 * Description: Two hash keys for the conductivity and pressure head data sets
 *              are given as arguments.  Three hash keys for x, y, and z
 *              velocity components are appended to the Tcl result.
 *
 * Cmd. Syntax: pfvvel conductivity phead
 *-----------------------------------------------------------------------*/

int               VVelCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *cond, *pHead;
  char newhashkey[MAX_KEY_SIZE];
  Databox       *databoxk, *databoxh;
  Tcl_HashEntry *entryPtr;

  char label[MAX_LABEL_SIZE];
  char          *component_ptr;

  Databox      **vel;


  /* Two data set names must be given */

  if (argc != 3)
  {
    WrongNumArgsError(interp, VVELUSAGE);
    return TCL_ERROR;
  }

  cond = argv[1];
  pHead = argv[2];

  /* Make sure the sets exist */

  if ((databoxk = DataMember(data, cond, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, cond);
    return TCL_ERROR;
  }

  if ((databoxh = DataMember(data, pHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pHead);
    return TCL_ERROR;
  }

  /* The dimensions of the data sets should be compatible */

  if (!SameDimensions(databoxk, databoxh))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  vel = CompVertVel(databoxk, databoxh);
  sprintf(label, "Darcy velocity in vertices for `%s' and `%s'",
          cond, pHead);

  component_ptr = label + strlen(label);

  /* Add the components to the hash table */

  if (vel)
  {
    sprintf(component_ptr, " (x velocity)");
    if (!AddData(data, vel[0], label, newhashkey))
      FreeDatabox(vel[0]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (y velocity)");
    if (!AddData(data, vel[1], label, newhashkey))
      FreeDatabox(vel[1]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (z velocity)");
    if (!AddData(data, vel[2], label, newhashkey))
      FreeDatabox(vel[2]);
    else
      Tcl_AppendElement(interp, newhashkey);
  }

  /* An error has occured computing the velocity */

  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  free(vel);
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfbfcvel' commands
 * Description: Two hash keys for the conductivity and pressure head data sets
 *              are given as arguments.  Three hash keys for x, y, and z
 *              velocity components are appended to the Tcl result.
 *
 * Cmd. Syntax: pfbfcvel conductivity phead
 *-----------------------------------------------------------------------*/

int               BFCVelCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *cond, *pHead;
  char newhashkey[MAX_KEY_SIZE];
  Databox       *databoxk, *databoxh;
  Tcl_HashEntry *entryPtr;

  char label[MAX_LABEL_SIZE];
  char          *component_ptr;

  Databox      **vel;


  /* Two data set names must be given */

  if (argc != 3)
  {
    WrongNumArgsError(interp, BFCVELUSAGE);
    return TCL_ERROR;
  }

  cond = argv[1];
  pHead = argv[2];

  /* Make sure the sets exist */

  if ((databoxk = DataMember(data, cond, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, cond);
    return TCL_ERROR;
  }

  if ((databoxh = DataMember(data, pHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pHead);
    return TCL_ERROR;
  }

  /* The dimensions of the data sets should be compatible */

  if (!SameDimensions(databoxk, databoxh))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  vel = CompBFCVel(databoxk, databoxh);
  sprintf(label, "Darcy velocity in vertices for `%s' and `%s'",
          cond, pHead);

  component_ptr = label + strlen(label);

  /* Add the components to the hash table */

  if (vel)
  {
    sprintf(component_ptr, " (x velocity)");
    if (!AddData(data, vel[0], label, newhashkey))
      FreeDatabox(vel[0]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (y velocity)");
    if (!AddData(data, vel[1], label, newhashkey))
      FreeDatabox(vel[1]);
    else
      Tcl_AppendElement(interp, newhashkey);

    sprintf(component_ptr, " (z velocity)");
    if (!AddData(data, vel[2], label, newhashkey))
      FreeDatabox(vel[2]);
    else
      Tcl_AppendElement(interp, newhashkey);
  }

  /* An error has occured computing the velocity */

  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  free(vel);
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfvmag' command
 * Description: Three hash keys for three data sets are given as input to
 *              the command.  Each data set will be treated as a velocity
 *              component.  The components will be used to compute the
 *              velocity magnitude whose hash key will be returned as the
 *              Tcl result.
 *
 * Cmd. Syntax: pfvmag datasetx datasety datasetz
 *-----------------------------------------------------------------------*/

int               VMagCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;

  char          *hashkeyx, *hashkeyy, *hashkeyz;
  Databox       *databoxx, *databoxy, *databoxz;
  Databox       *vmag;

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];


  /* Three arguments must be given */

  if (argc != 4)
  {
    WrongNumArgsError(interp, VMAGUSEAGE);
    return TCL_ERROR;
  }

  hashkeyx = argv[1];
  hashkeyy = argv[2];
  hashkeyz = argv[3];

  /* Make sure the three data sets exist */

  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  if ((databoxz = DataMember(data, hashkeyz, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyz);
    return TCL_ERROR;
  }

  /* All three data sets must belong to grids of the */
  /* same dimensions.                                */

  if (!SameDimensions(databoxx, databoxy) ||
      !SameDimensions(databoxy, databoxz))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  vmag = CompVMag(databoxx, databoxy, databoxz);

  /* Make sure no error occured while computing the magnitude. */
  /* Also, make sure there were no memory allocation problems. */

  if (vmag)
  {
    sprintf(label, "Velocity magnitude of `%s', `%s', and `%s'",
            hashkeyx, hashkeyy, hashkeyz);

    /* Make sure the data set pointer was added */
    /* to the hash table successfully.          */

    if (!AddData(data, vmag, label, newhashkey))
      FreeDatabox(vmag);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfhhead' command
 * Description: This command computes the hydraulic head from the pressure
 *              head.  The hash key for a pressure head must be passed as
 *              an argument.  The hash key for the pressure head is
 *              returned as the Tcl result.
 *
 * Cmd. Syntax: pfhhead phead
 *-----------------------------------------------------------------------*/

int               HHeadCommand(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *pHead;
  char newhashkey[MAX_KEY_SIZE];

  Tcl_HashEntry *entryPtr;
  Databox       *databoxh;
  Databox       *hHead;

  char label[MAX_LABEL_SIZE];


  /* One argument must be given */

  if (argc != 2)
  {
    WrongNumArgsError(interp, HHEADUSAGE);
    return TCL_ERROR;
  }

  pHead = argv[1];

  /* Make sure the set exists */

  if ((databoxh = DataMember(data, pHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pHead);
    return TCL_ERROR;
  }

  hHead = HHead(databoxh, DataGridType(data));
  sprintf(label, "Hydraulic head of `%s'", pHead);

  /* Make sure the hydraulic head could be allocated and */
  /* computed successfuly.                               */

  if (hHead)
  {
    /* Make sure the data set pointer was added to */
    /* the hash table successfully.                */

    if (!AddData(data, hHead, label, newhashkey))
      FreeDatabox(hHead);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfphead' command
 * Description: This command computes the pressure head from the hydraulic
 *              head.  A hash key for the hydraulic head must be passed as
 *              an argument.  The hash key for the pressure head is returned
 *              as the Tcl result.
 *
 * Cmd. Syntax: pfphead hhead
 *-----------------------------------------------------------------------*/

int               PHeadCommand(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *hHead;
  char newhashkey[MAX_KEY_SIZE];

  Tcl_HashEntry *entryPtr;
  Databox       *databoxh;
  Databox       *pHead;

  char label[MAX_LABEL_SIZE];


  /* One argument must be given */

  if (argc != 2)
  {
    WrongNumArgsError(interp, PHEADUSAGE);
    return TCL_ERROR;
  }

  hHead = argv[1];

  /* Check and make sure that the data set exists */

  if ((databoxh = DataMember(data, hHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hHead);
    return TCL_ERROR;
  }

  pHead = PHead(databoxh, DataGridType(data));
  sprintf(label, "Pressure head of `%s'", hHead);

  /* Make sure the pressure head was allocated and */
  /* computed successfuly.                         */

  if (pHead)
  {
    /* Make sure the data set pointer was added to */
    /* the hash table successfully.                */

    if (!AddData(data, pHead, label, newhashkey))
      FreeDatabox(pHead);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfflux' command
 * Description: The hash keys for conductivity and hydraulic head data sets
 *              must be passed as arguments to this command so the flux
 *              can be computed.  A string containing the hash key for
 *              the flux is appended to the Tcl result.
 * Cmd. Syntax: pfflux conductivity hhead
 *-----------------------------------------------------------------------*/

int               FluxCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *cond, *hHead;
  Tcl_HashEntry *entryPtr;

  Databox       *databoxk, *databoxh;
  Databox       *flux;
  char newhashkey[MAX_KEY_SIZE];

  char label[MAX_LABEL_SIZE];


  /* Two arguments must be given */

  if (argc != 3)
  {
    WrongNumArgsError(interp, FLUXUSAGE);
    return TCL_ERROR;
  }

  cond = argv[1];
  hHead = argv[2];

  /* Make sure the sets given exist */

  if ((databoxk = DataMember(data, cond, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, cond);
    return TCL_ERROR;
  }

  if ((databoxh = DataMember(data, hHead, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hHead);
    return TCL_ERROR;
  }

  /* Make sure the data sets have the same dimensions */

  if (!SameDimensions(databoxk, databoxh))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  flux = CompFlux(databoxk, databoxh);

  /* Make sure that memory for the flux was allocated */
  /* and the flux was computed successfuly.           */

  if (flux)
  {
    sprintf(label, "Flux of `%s' and `%s'", cond, hHead);

    /* Make sure the data set pointer could be added */
    /* to the hash table successfully.               */

    if (!AddData(data, flux, label, newhashkey))
      FreeDatabox(flux);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfnewgrid' command
 * Description: Create a new data set whose grid is described by passing
 *              three lists and a label as arguments.  The first list
 *              will be the number of coordinates in the x, y, and z
 *              directions.  The second list will describe the origin.
 *              The third contains the intervals between coordinates
 *              along each axis.  The hash key for the data set created
 *              will be appended to the Tcl result.
 *
 * Cmd. Syntax: pfnewgrid {nx ny nz} {x y z} {dx dy dz} label
 *-----------------------------------------------------------------------*/

int               NewGridCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *npoints, *origin, *intervals;
  char          *label;
  Databox       *newdatabox;

  char          *num;
  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  char newhashkey[MAX_KEY_SIZE];

  /* Two arguments must be given */

  if (argc != 5)
  {
    WrongNumArgsError(interp, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  npoints = argv[1];
  origin = argv[2];
  intervals = argv[3];
  label = argv[4];

  /* Make sure that the number of points along each axis are all */
  /* integers.                                                   */

  if (!(num = strtok(npoints, WS)) || (sscanf(num, "%d", &nx) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%d", &ny) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%d", &nz) != 1))
  {
    NotAnIntError(interp, 1, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  /* there should only be three numbers in the npoints list */

  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 1, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  /* Make sure that the origin is given in floating point numbers */

  if (!(num = strtok(origin, WS)) || (sscanf(num, "%lf", &x) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &y) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &z) != 1))
  {
    NotADoubleError(interp, 2, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  /* There should only be three numbers in the origin list */

  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 2, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  /* Make sure that the intervals are given in floating point numbers */

  if (!(num = strtok(intervals, WS)) || (sscanf(num, "%lf", &dx) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &dy) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &dz) != 1))
  {
    NotADoubleError(interp, 3, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  /* There should only be three numbers in the intervals list */

  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 3, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  newdatabox = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz);

  /* Make sure the new data set could be allocated */

  if (newdatabox)
  {
    /* Make sure the data set pointer was added to the */
    /* hash table successfully.                        */

    if (!AddData(data, newdatabox, label, newhashkey))
      FreeDatabox(newdatabox);
    else
      Tcl_AppendElement(interp, newhashkey);
  }
  else
  {
    MemoryError(interp);
    return TCL_ERROR;
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfsetgrid' command
 * Description: Reset the grid information on an existing grid (e.g., for
 *              a variable read in from simple ascii, which ends up with
 *              default values for x,y,z and dx,dy,dz). Arguments are three
 *              the label of the given variable and three lists.  The first
 *              list is the number of coordinates in the x, y, and z
 *              directions.  The second list will describe the origin.
 *              The third contains the intervals between coordinates
 *              along each axis.  The hash key for the data set created
 *              will be appended to the Tcl result.
 *
 * Cmd. Syntax: pfsetgrid {nx ny nz} {x y z} {dx dy dz} dataset
 *-----------------------------------------------------------------------*/

int               SetGridCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Data          *data = (Data*)clientData;

  Databox       *databox;
  char          *hashkey;
  Tcl_HashEntry *entryPtr;

  char          *npoints, *origin, *intervals;
  char          *num;
  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  int return_code = TCL_ERROR;

  /* Five arguments must be given */
  if (argc != 5)
  {
    WrongNumArgsError(interp, NEWGRIDUSAGE);
    return TCL_ERROR;
  }

  npoints = strdup(argv[1]);
  origin = strdup(argv[2]);
  intervals = strdup(argv[3]);
  hashkey = argv[4];

  /* Make sure that the number of points along each axis are all */
  /* integers.                                                   */
  if (!(num = strtok(npoints, WS)) || (sscanf(num, "%d", &nx) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%d", &ny) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%d", &nz) != 1))
  {
    NotAnIntError(interp, 1, SETGRIDUSAGE);
    goto exit;
  }

  /* there should only be three numbers in the npoints list */
  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 1, SETGRIDUSAGE);
    goto exit;
  }

  /* Make sure that the origin is given in floating point numbers */
  if (!(num = strtok(origin, WS)) || (sscanf(num, "%lf", &x) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &y) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &z) != 1))
  {
    NotADoubleError(interp, 2, SETGRIDUSAGE);
    goto exit;
  }

  /* There should only be three numbers in the origin list */
  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 2, SETGRIDUSAGE);
    return TCL_ERROR;
  }

  /* Make sure that the intervals are given in floating point numbers */
  if (!(num = strtok(intervals, WS)) || (sscanf(num, "%lf", &dx) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &dy) != 1) ||
      !(num = strtok(NULL, WS)) || (sscanf(num, "%lf", &dz) != 1))
  {
    NotADoubleError(interp, 3, SETGRIDUSAGE);
    goto exit;
  }

  /* There should only be three numbers in the intervals list */
  if (strtok(NULL, WS))
  {
    InvalidArgError(interp, 3, SETGRIDUSAGE);
    goto exit;
  }

  /* Make sure dataset exists */
  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    goto exit;
  }

  /* Swap grid values */
  SetDataboxGrid(databox, nx, ny, nz, x, y, z, dx, dy, dz);

  return_code = TCL_OK;

exit:

  free(npoints);
  free(origin);
  free(intervals);

  return return_code;
}


/*-----------------------------------------------------------------------
 * routine for `pfnewlabel' command
 * Description: A data set hash key and a label are passed as arguments.
 *              the data set cooresponding to the hash key will have
 *              it's label changed.  Nothing will be appended to the
 *              Tcl result upon success.
 *
 * Cmd. Syntax: pfnewlabel dataset newlabel
 *-----------------------------------------------------------------------*/
int                NewLabelCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Data          *data = (Data*)clientData;

  char           *hashkey, *newlabel;
  Tcl_HashEntry  *entryPtr;
  Databox        *databox;

  /* Two arguments must be given */

  if (argc != 3)
  {
    WrongNumArgsError(interp, NEWLABELUSAGE);
    return TCL_ERROR;
  }

  hashkey = argv[1];

  /* The set must exist */

  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }

  newlabel = argv[2];

  /* Truncate the new label if it is too long */

  if ((strlen(newlabel) + 1) > MAX_LABEL_SIZE)
    newlabel[MAX_LABEL_SIZE - 1] = 0;

  strcpy(DataboxLabel(databox), newlabel);

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfaxpy' command
 * Description: The arguments are a floating point number alpha, and
 *              two data set hash keys.  The operation:
 *              datasety=alpha*datasetx+datasety
 *              is perfomed.  The hash key `datasety' will be appended
 *              to the Tcl result upon successful completion.
 *
 * Cmd. Syntax: pfaxpy alpha datasetx datasety
 *-----------------------------------------------------------------------*/

int               AxpyCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *hashkeyx, *hashkeyy;
  Tcl_HashEntry *entryPtr;
  Databox       *databoxx, *databoxy;
  double alpha;


  /* Three arguments are needed */

  if (argc != 4)
  {
    WrongNumArgsError(interp, AXPYUSAGE);
    return TCL_ERROR;
  }

  /* The first argument should be a double */

  if (Tcl_GetDouble(interp, argv[1], &alpha) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, AXPYUSAGE);
    return TCL_ERROR;
  }

  hashkeyx = argv[2];
  hashkeyy = argv[3];

  /* Make sure the sets exist */

  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  /* Make sure the grids the sets belong to have the */
  /* same dimensions.                                */

  if (!SameDimensions(databoxx, databoxy))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  /* Make sure axpy can be computed from the data sets given */

  Axpy(alpha, databoxx, databoxy);

  Tcl_AppendElement(interp, hashkeyy);
  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfsum' command
 * Description: The arguments are data set hash key.  The operation:
 *              sum = sum of all elements of datasetx is perfomed.
 *
 * Cmd. Syntax: pfsum datasetx
 *-----------------------------------------------------------------------*/

int               SumCommand(
                             ClientData  clientData,
                             Tcl_Interp *interp,
                             int         argc,
                             char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *hashkeyx;
  Tcl_HashEntry *entryPtr;
  Databox       *databoxx;
  double sum;

  Tcl_Obj       *result;

  /* Three arguments are needed */

  if (argc != 2)
  {
    WrongNumArgsError(interp, SUMUSAGE);
    return TCL_ERROR;
  }

  /* The first argument should be a double */

  hashkeyx = argv[1];

  /* Make sure the sets exist */

  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  Sum(databoxx, &sum);

  result = Tcl_NewDoubleObj(sum);
  Tcl_SetObjResult(interp, result);
  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfcellsum' command
 * Description: The arguments are data set hash key.  The operation:
 *              sum = sum of elements of datasetx and datasety.
 *
 * Cmd. Syntax: pfcellsum datasetx datasety mask
 *-----------------------------------------------------------------------*/

int               CellSumCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *databoxy, *mask;
  Databox       *cellsum;
  char          *hashkeyx, *hashkeyy, *mask_hashkey;
  char cellsum_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise sum";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLSUMUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, datay, mask */
  hashkeyx = argv[1];
  hashkeyy = argv[2];
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  /* Datasets must belong to grids of the same dimensions */
  if (!SameDimensions(databoxx, databoxy))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((cellsum = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, cellsum, filename, cellsum_hashkey))
        FreeDatabox(cellsum);
      else
      {
        Tcl_AppendElement(interp, cellsum_hashkey);
      }

      CellSum(databoxx, databoxy, mask, cellsum);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcelldiff' command
 * Description: The arguments are data set hash key.  The operation:
 *              diff = difference of elements of datasetx and datasety.
 *
 * Cmd. Syntax: pfcelldiff datasetx datasety mask
 *-----------------------------------------------------------------------*/

int               CellDiffCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *databoxy, *mask;
  Databox       *celldiff;
  char          *hashkeyx, *hashkeyy, *mask_hashkey;
  char celldiff_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise difference";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLDIFFUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, datay, mask */
  hashkeyx = argv[1];
  hashkeyy = argv[2];
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  /* Datasets must belong to grids of the same dimensions */
  if (!SameDimensions(databoxx, databoxy))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((celldiff = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, celldiff, filename, celldiff_hashkey))
        FreeDatabox(celldiff);
      else
      {
        Tcl_AppendElement(interp, celldiff_hashkey);
      }

      CellDiff(databoxx, databoxy, mask, celldiff);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcellmult' command
 * Description: The arguments are data set hash key.  The operation:
 *              prod = product of elements of datasetx and datasety.
 *
 * Cmd. Syntax: pfcellmult datasetx datasety mask
 *-----------------------------------------------------------------------*/

int               CellMultCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *databoxy, *mask;
  Databox       *cellmult;
  char          *hashkeyx, *hashkeyy, *mask_hashkey;
  char cellmult_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise product";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLMULTUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, datay, mask */
  hashkeyx = argv[1];
  hashkeyy = argv[2];
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  /* Datasets must belong to grids of the same dimensions */
  if (!SameDimensions(databoxx, databoxy))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((cellmult = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, cellmult, filename, cellmult_hashkey))
        FreeDatabox(cellmult);
      else
      {
        Tcl_AppendElement(interp, cellmult_hashkey);
      }

      CellMult(databoxx, databoxy, mask, cellmult);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcelldiv' command
 * Description: The arguments are data set hash key.  The operation:
 *              div = quotient of elements of datasetx divided by datasety.
 *
 * Cmd. Syntax: pfcelldiv datasetx datasety mask
 *-----------------------------------------------------------------------*/

int               CellDivCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *databoxy, *mask;
  Databox       *celldiv;
  char          *hashkeyx, *hashkeyy, *mask_hashkey;
  char celldiv_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise quotient";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLDIVUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, datay, mask */
  hashkeyx = argv[1];
  hashkeyy = argv[2];
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((databoxy = DataMember(data, hashkeyy, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyy);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  /* Datasets must belong to grids of the same dimensions */
  if (!SameDimensions(databoxx, databoxy))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((celldiv = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, celldiv, filename, celldiv_hashkey))
        FreeDatabox(celldiv);
      else
      {
        Tcl_AppendElement(interp, celldiv_hashkey);
      }

      CellDiv(databoxx, databoxy, mask, celldiv);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcellsumconst' command
 * Description: The arguments are data set hash key.  The operation:
 *              sum = sum of all elements of datasetx is perfomed.
 *
 * Cmd. Syntax: pfcellsumconst datasetx const mask
 *-----------------------------------------------------------------------*/

int               CellSumConstCommand(
                                      ClientData  clientData,
                                      Tcl_Interp *interp,
                                      int         argc,
                                      char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *mask;
  Databox       *cellsum;
  double val;
  char          *hashkeyx, *mask_hashkey;
  char cellsum_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise sum";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLSUMCONSTUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, mask */
  hashkeyx = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &val) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, CELLSUMCONSTUSAGE);
    return TCL_ERROR;
  }
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((cellsum = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, cellsum, filename, cellsum_hashkey))
        FreeDatabox(cellsum);
      else
      {
        Tcl_AppendElement(interp, cellsum_hashkey);
      }

      CellSumConst(databoxx, val, mask, cellsum);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcelldiffconst' command
 * Description: The arguments are data set hash key.  The operation:
 *              diff = diff of elements of datasetx and <constant>.
 *
 * Cmd. Syntax: pfcelldiffconst datasetx const mask
 *-----------------------------------------------------------------------*/

int               CellDiffConstCommand(
                                       ClientData  clientData,
                                       Tcl_Interp *interp,
                                       int         argc,
                                       char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *mask;
  Databox       *celldiff;
  double val;
  char          *hashkeyx, *mask_hashkey;
  char celldiff_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise difference";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLDIFFCONSTUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, mask */
  hashkeyx = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &val) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, CELLDIFFCONSTUSAGE);
    return TCL_ERROR;
  }
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((celldiff = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, celldiff, filename, celldiff_hashkey))
        FreeDatabox(celldiff);
      else
      {
        Tcl_AppendElement(interp, celldiff_hashkey);
      }

      CellDiffConst(databoxx, val, mask, celldiff);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcellmultconst' command
 * Description: The arguments are data set hash key.  The operation:
 *              mult = product of elements of datasetx and <constant>.
 *
 * Cmd. Syntax: pfcellmultconst datasetx const mask
 *-----------------------------------------------------------------------*/

int               CellMultConstCommand(
                                       ClientData  clientData,
                                       Tcl_Interp *interp,
                                       int         argc,
                                       char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *mask;
  Databox       *cellmult;
  double val;
  char          *hashkeyx, *mask_hashkey;
  char cellmult_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise product";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLMULTCONSTUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, mask */
  hashkeyx = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &val) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, CELLMULTCONSTUSAGE);
    return TCL_ERROR;
  }
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((cellmult = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, cellmult, filename, cellmult_hashkey))
        FreeDatabox(cellmult);
      else
      {
        Tcl_AppendElement(interp, cellmult_hashkey);
      }

      CellMultConst(databoxx, val, mask, cellmult);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfcelldivconst' command
 * Description: The arguments are data set hash key.  The operation:
 *              div = quotient of elements of datasetx and <constant>.
 *
 * Cmd. Syntax: pfcelldivconst datasetx const mask
 *-----------------------------------------------------------------------*/

int               CellDivConstCommand(
                                      ClientData  clientData,
                                      Tcl_Interp *interp,
                                      int         argc,
                                      char *      argv[])
{
  Tcl_HashEntry *entryPtr;
  Data          *data = (Data*)clientData;
  Databox       *databoxx, *mask;
  Databox       *celldiv;
  double val;
  char          *hashkeyx, *mask_hashkey;
  char celldiv_hashkey[MAX_KEY_SIZE];
  char          *filename = "cellwise quotient";

  /* Check that three arguments are given */
  if (argc == 3)
  {
    WrongNumArgsError(interp, CELLDIVCONSTUSAGE);
    return TCL_ERROR;
  }

  /* Get arguments -- datax, mask */
  hashkeyx = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &val) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, CELLDIVCONSTUSAGE);
    return TCL_ERROR;
  }
  mask_hashkey = argv[3];

  /* Make sure the datasets exist */
  if ((databoxx = DataMember(data, hashkeyx, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyx);
    return TCL_ERROR;
  }

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    /* Read array size/shape */
    int nx = DataboxNx(databoxx);
    int ny = DataboxNy(databoxx);
    int nz = DataboxNz(databoxx);

    double x = DataboxX(databoxx);
    double y = DataboxY(databoxx);
    double z = DataboxZ(databoxx);

    double dx = DataboxDx(databoxx);
    double dy = DataboxDy(databoxx);
    double dz = DataboxDz(databoxx);

    /* create the new databox structure for surface storage  */
    if ((celldiv = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure data set pointer was added to hash table successfully */
      if (!AddData(data, celldiv, filename, celldiv_hashkey))
        FreeDatabox(celldiv);
      else
      {
        Tcl_AppendElement(interp, celldiv_hashkey);
      }

      CellDivConst(databoxx, val, mask, celldiv);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfstats' command
 * Description: A data set hash key is given and a list of statistics is
 *              appended to the Tcl result.  The data returned is the
 *              minimum, maximum, mean, sum, variance, and the standard
 *              deviation of the data contained in the set.
 *
 * Cmd. Syntax: pfstats dataset
 *-----------------------------------------------------------------------*/

int               GetStatsCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Data          *data = (Data*)clientData;

  char          *hashkey;
  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  double min, max, mean, sum, variance, stdev;

  Tcl_Obj     *result = Tcl_GetObjResult(interp);
  Tcl_Obj     *double_obj;

  /* One argument must be given */

  if (argc != 2)
  {
    WrongNumArgsError(interp, GETSTATSUSAGE);
    return TCL_ERROR;
  }

  hashkey = argv[1];

  /* Make sure the data set exists */

  if ((databox = DataMember(data, hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkey);
    return TCL_ERROR;
  }

  Stats(databox, &min, &max, &mean, &sum, &variance, &stdev);

  double_obj = Tcl_NewDoubleObj(min);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  double_obj = Tcl_NewDoubleObj(max);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  double_obj = Tcl_NewDoubleObj(mean);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  double_obj = Tcl_NewDoubleObj(sum);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  double_obj = Tcl_NewDoubleObj(variance);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  double_obj = Tcl_NewDoubleObj(stdev);
  Tcl_ListObjAppendElement(interp, result, double_obj);

  return TCL_OK;
}



/*-----------------------------------------------------------------------
 * routine for `pfmdiff' command
 * Description: Two data set hash keys are given as the first two arguments.
 *              The grid point at which the number of digits in agreement
 *              (significant digits) is fewest is determined.  If sig_digs
 *              is >= 0 then the coordinate whose two values differ in
 *              more than m significant digits will be computed.  If
 *              `sig_digs' is < 0 then the coordinate whose values have
 *              a minimum number of significant digits will be computed.
 *              The number of the fewest significant digits is determined,
 *              and the maximum absolute difference is computed.  A list of the
 *              The only coordintes that will be considered will be those
 *              whose differences are greater than absolute zero.
 *              following form is appended to the Tcl result upon success:
 *
 *              {i j k sd} max_adiff
 *
 *              where i, j, and k are the coordinates computed, sd is the
 *              minimum number of significant digits computed, and max_adiff
 *              is the maximum absolute difference computed.
 *
 * Cmd. Syntax: pfmdiff hashkeyp hashkeyq sig_digs [abs_zero]
 *-----------------------------------------------------------------------*/

int               MDiffCommand(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  char          *hashkeyp, *hashkeyq;
  Databox       *databoxp, *databoxq;

  int sd;
  double abs_zero;

  Tcl_Obj       *result;

  /* Three or four arguments may be given */

  if ((argc < 4) || (argc > 5))
  {
    WrongNumArgsError(interp, MDIFFUSAGE);
    return TCL_ERROR;
  }

  hashkeyp = argv[1];
  hashkeyq = argv[2];

  /* Make sure the two data set given exits */

  if ((databoxp = DataMember(data, hashkeyp, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyp);
    return TCL_ERROR;
  }

  if ((databoxq = DataMember(data, hashkeyq, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyq);
    return TCL_ERROR;
  }

  /* Make sure the grids for each data set are compatible */

  if (!SameDimensions(databoxp, databoxq))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  /* Determine if a number for the absolute zero has been given */

  if (Tcl_GetInt(interp, argv[3], &sd) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, MDIFFUSAGE);
    return TCL_ERROR;
  }

  /* The default value for absolute zero is zero */

  if (argc == 4)
    abs_zero = 0.0;
  else
  {
    /* The number for the absolute zero should be a double */

    if (Tcl_GetDouble(interp, argv[4], &abs_zero) == TCL_ERROR)
    {
      NotADoubleError(interp, 4, MDIFFUSAGE);
      return TCL_ERROR;
    }

    /* It should also be positive */

    if (abs_zero < 0)
    {
      NumberNotPositiveError(interp, 4);
      return TCL_ERROR;
    }
  }

  /* The data sets should be compatible */

  result = Tcl_GetObjResult(interp);

  MSigDiff(interp, databoxp, databoxq, sd, abs_zero, result);

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfsavediff' command
 * Description: Two data set hash keys are passed as the first two
 *              arguments.  Coordinates whose values differ in
 *              sig_digs or more agreeing significant digits will be determined.
 *              The absolute difference of each of the above mentioned
 *              coordinates must also be greater than abs_zero before it
 *              is saved to file.  The coordinate at which
 *
 * Cmd. Syntax: pfsavediff datasetp datasetq sig_digs [abs_zero] -file filename
 *-----------------------------------------------------------------------*/

int               SaveDiffCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  char          *hashkeyp, *hashkeyq;
  Databox       *databoxp, *databoxq;

  int sd;
  FILE           *fp = NULL;

  int filearg;
  double abs_zero = 0.0;


  /* Four, five or seven arguments may be given */

  if ((argc < 6) || (argc > 7))
  {
    WrongNumArgsError(interp, SAVEDIFFUSAGE);
    return TCL_ERROR;
  }

  hashkeyp = argv[1];
  hashkeyq = argv[2];

  /* Make sure the sets exist */

  if ((databoxp = DataMember(data, hashkeyp, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyp);
    return TCL_ERROR;
  }

  if ((databoxq = DataMember(data, hashkeyq, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyq);
    return TCL_ERROR;
  }

  /* Make sure the data sets are on grids of the */
  /* same dimensions.                            */

  if (!SameDimensions(databoxp, databoxq))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  /* The third argument must be an integer */

  if (Tcl_GetInt(interp, argv[3], &sd) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, SAVEDIFFUSAGE);
    return TCL_ERROR;
  }

  /* The number of significant digits must be positive */

  if (sd < 0)
  {
    NumberNotPositiveError(interp, 3);
    return TCL_ERROR;
  }

  /* Check for additional arguments */

  if (argc > 4)
  {
    /* Check for an absolute zero value */

    if (*argv[4] != '-')
    {
      /* The argument should be a double */

      if (Tcl_GetDouble(interp, argv[4], &abs_zero) == TCL_ERROR)
      {
        NotADoubleError(interp, 4, SAVEDIFFUSAGE);
        return TCL_ERROR;
      }

      /* The absolute zero should be positive */

      if (abs_zero < 0)
      {
        NumberNotPositiveError(interp, 4);
        return TCL_ERROR;
      }

      filearg = 5;
    }
    else
      filearg = 4;


    /* Check for a file option argument */

    if ((argc == (filearg + 2)) && (strcmp(argv[filearg], "-file") == 0))
    {
      /* The file option argument should be followed by a file name */

      if (argc != (filearg + 2))
      {
        MissingFilenameError(interp, filearg, SAVEDIFFUSAGE);
        return TCL_ERROR;
      }

      /* Make sure the file was opened successfuly */

      if ((fp = fopen(argv[filearg + 1], "wb")) == NULL)
      {
        ReadWriteError(interp);
        return TCL_ERROR;
      }
    }

    /* report an error if the option was invalid */

    else if (argc == (filearg + 2))
    {
      InvalidOptionError(interp, 5, SAVEDIFFUSAGE);
      return TCL_ERROR;
    }

    else
    {
      MissingFilenameError(interp, filearg, SAVEDIFFUSAGE);
      return TCL_ERROR;
    }
  }

  SigDiff(databoxp, databoxq, sd, abs_zero, fp);

  fflush(fp);
  fclose(fp);

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfdiffelt' command
 * Description: Two data set hash keys are passed as the first two
 *              arguments.  Coordinates for the element to be diffed are
 *              then passed as three seperate arguments: i, j, and k.
 *              If the values at (i, j, k) differ by more than sig_digs
 *              significant digits and the difference is greater than
 *              absolute zero, then the differnce will be appended to the
 *              Tcl result.
 *
 * Cmd. Syntax: pfdiffelt datasetp datasetq i j k sig_digs [abs_zero]
 *-----------------------------------------------------------------------*/

int               DiffEltCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  char          *hashkeyp, *hashkeyq;
  Databox       *databoxp, *databoxq;

  int sd;
  int i, j, k;
  double diff;
  double abs_zero = 0;

  Tcl_Obj       *result;

  /* Four, five or seven arguments may be given */

  if ((argc < 7) || (argc > 8))
  {
    WrongNumArgsError(interp, DIFFELTUSAGE);
    return TCL_ERROR;
  }

  hashkeyp = argv[1];
  hashkeyq = argv[2];

  /* Make sure the sets exist */

  if ((databoxp = DataMember(data, hashkeyp, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyp);
    return TCL_ERROR;
  }

  if ((databoxq = DataMember(data, hashkeyq, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, hashkeyq);
    return TCL_ERROR;
  }

  /* Make sure the data sets are on compatible */
  /* grids.                                    */

  if (!SameDimensions(databoxp, databoxq))
  {
    DimensionError(interp);
    return TCL_ERROR;
  }

  /* The third, fourth, and fifth arguments must be */
  /* integer coordinates.                           */

  if (Tcl_GetInt(interp, argv[3], &i) == TCL_ERROR)
  {
    NotAnIntError(interp, 3, DIFFELTUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[4], &j) == TCL_ERROR)
  {
    NotAnIntError(interp, 4, DIFFELTUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[5], &k) == TCL_ERROR)
  {
    NotAnIntError(interp, 5, DIFFELTUSAGE);
    return TCL_ERROR;
  }

  /* Make sure the coordinates are in range of */
  /* the grid.                                 */

  if (!InRange(i, j, k, databoxp))
  {
    OutOfRangeError(interp, i, j, k);
    return TCL_ERROR;
  }

  /* The number of significant digits must be an integer */

  if (Tcl_GetInt(interp, argv[6], &sd) == TCL_ERROR)
  {
    NotAnIntError(interp, 6, DIFFELTUSAGE);
    return TCL_ERROR;
  }


  /* The number of significant digits must be positive */

  if (sd < 0)
  {
    NumberNotPositiveError(interp, 6);
    return TCL_ERROR;
  }

  /* Check for additional arguments */

  if (argc > 7)
  {
    /* Check for an absolute zero value */

    /* The argument should be a double */

    if (Tcl_GetDouble(interp, argv[7], &abs_zero) == TCL_ERROR)
    {
      NotADoubleError(interp, 7, DIFFELTUSAGE);
      return TCL_ERROR;
    }

    /* The absolute zero should be positive */

    if (abs_zero < 0)
    {
      NumberNotPositiveError(interp, 7);
      return TCL_ERROR;
    }
  }

  diff = DiffElt(databoxp, databoxq, i, j, k, sd, abs_zero);

  /* If the values and difference at the given coordinate */
  /* met the given criteria, then append the difference   */
  /* to the Tcl string.                                   */

  if (diff >= 0.0)
  {
    result = Tcl_NewDoubleObj(diff);
    Tcl_SetObjResult(interp, result);
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfdelete' command
 * Description: A data set hash key is given and the data set is
 *              deleted from memory.
 * Cmd. Syntax: pfdelete dataset
 *-----------------------------------------------------------------------*/

int               DeleteCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Data          *data = (Data*)clientData;

  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  /* One or more arguments must be given */

  if (argc != 2)
  {
    WrongNumArgsError(interp, DELETEUSAGE);
    return TCL_ERROR;
  }

  /* Make sure the data set exists */

  if ((databox = DataMember(data, argv[1], entryPtr)) == NULL)
  {
    SetNonExistantError(interp, argv[1]);
    return TCL_ERROR;
  }

  FreeDatabox(databox);
  Tcl_DeleteHashEntry(entryPtr);

  DataTotalMem(data)--;

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfcomputetop' command
 * Description: One argument of a Databox containing the mask is required.
 *
 * Cmd. syntax: pfcomputetop databox
 *-----------------------------------------------------------------------*/
int            ComputeTopCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *mask;
  Databox    *top;

  char       *filename = "top";
  char       *mask_hashkey;

  char newhashkey[MAX_KEY_SIZE];

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFCOMPUTETOPUSAGE);
    return TCL_ERROR;
  }

  mask_hashkey = argv[1];

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(mask);
    int ny = DataboxNy(mask);

    double x = DataboxX(mask);
    double y = DataboxY(mask);
    double z = DataboxZ(mask);

    double dx = DataboxDx(mask);
    double dy = DataboxDy(mask);
    double dz = DataboxDz(mask);

    /* create the new databox structure for top */
    if ((top = NewDatabox(nx, ny, 1, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, top, filename, newhashkey))
        FreeDatabox(top);
      else
      {
        Tcl_AppendElement(interp, newhashkey);
      }

      ComputeTop(mask, top);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfcomputebottom' command
 * Description: One argument of a Databox containing the mask is required.
 *
 * Cmd. syntax: pfcomputebottom databox
 *-----------------------------------------------------------------------*/
int            ComputeBottomCommand(
                                    ClientData  clientData,
                                    Tcl_Interp *interp,
                                    int         argc,
                                    char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *mask;
  Databox    *bottom;

  char       *filename = "bottom";
  char       *mask_hashkey;

  char newhashkey[MAX_KEY_SIZE];

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFCOMPUTEBOTTOMUSAGE);
    return TCL_ERROR;
  }

  mask_hashkey = argv[1];

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(mask);
    int ny = DataboxNy(mask);

    double x = DataboxX(mask);
    double y = DataboxY(mask);
    double z = DataboxZ(mask);

    double dx = DataboxDx(mask);
    double dy = DataboxDy(mask);
    double dz = DataboxDz(mask);

    /* create the new databox structure for bottom */
    if ((bottom = NewDatabox(nx, ny, 1, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, bottom, filename, newhashkey))
        FreeDatabox(bottom);
      else
      {
        Tcl_AppendElement(interp, newhashkey);
      }

      ComputeBottom(mask, bottom);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfextracttop' command
 * Description: Extract the top cells of a dataset.
 *
 * Cmd. syntax: pfcomputetop top data
 *-----------------------------------------------------------------------*/
int            ExtractTopCommand(
                                 ClientData  clientData,
                                 Tcl_Interp *interp,
                                 int         argc,
                                 char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *top;
  Databox    *databox;
  Databox    *top_values;

  char       *filename = "top values";
  char       *top_hashkey;
  char       *data_hashkey;

  char newhashkey[MAX_KEY_SIZE];

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 2)
  {
    WrongNumArgsError(interp, PFEXTRACTTOPUSAGE);
    return TCL_ERROR;
  }

  top_hashkey = argv[1];
  data_hashkey = argv[2];

  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  if ((databox = DataMember(data, data_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, data_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(databox);
    int ny = DataboxNy(databox);

    double x = DataboxX(databox);
    double y = DataboxY(databox);
    double z = DataboxZ(databox);

    double dx = DataboxDx(databox);
    double dy = DataboxDy(databox);
    double dz = DataboxDz(databox);

    /* create the new databox structure for top */
    if ((top_values = NewDatabox(nx, ny, 1, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, top_values, filename, newhashkey))
        FreeDatabox(top);
      else
      {
        Tcl_AppendElement(interp, newhashkey);
      }

      ExtractTop(top, databox, top_values);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfcomputedomain' command
 * Description: Compute terrain following domain
 *
 * Cmd. syntax: pfcomputedomain top bottom
 *-----------------------------------------------------------------------*/
int            ComputeDomainCommand(
                                    ClientData  clientData,
                                    Tcl_Interp *interp,
                                    int         argc,
                                    char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 2)
  {
    WrongNumArgsError(interp, PFCOMPUTEDOMAINUSAGE);
    return TCL_ERROR;
  }

  char       *top_hashkey = argv[1];
  Databox    *top;
  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  char       *bottom_hashkey = argv[2];
  Databox    *bottom;
  if ((bottom = DataMember(data, bottom_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, bottom_hashkey);
    return TCL_ERROR;
  }

  /*--------------------------------------------------------------------
   * Get the processor topology from the database
   *--------------------------------------------------------------------*/
  int num_procs_x = GetInt(interp, "Process.Topology.P");
  int num_procs_y = GetInt(interp, "Process.Topology.Q");
  int num_procs_z = GetInt(interp, "Process.Topology.R");

  if (num_procs_z > 1)
  {
    fprintf(stderr, "Error: Process.Topology.R must be 1 for pfcomputedomain to work\n");
    return TCL_ERROR;
  }

  int num_procs = num_procs_x * num_procs_y * num_procs_z;

  /*--------------------------------------------------------------------
   * Get the initial grid info from the database
   *--------------------------------------------------------------------*/
  Grid          *user_grid = ReadUserGrid(interp);

  /*--------------------------------------------------------------------
   * Load the data
   *--------------------------------------------------------------------*/

  SubgridArray  *all_subgrids = DistributeUserGrid(user_grid, num_procs,
                                                   num_procs_x, num_procs_y, num_procs_z);

  if (!all_subgrids)
  {
    printf("Incorrect process allocation input\n");
    return TCL_ERROR;
  }

  ComputeDomain(all_subgrids, top, bottom, num_procs_x, num_procs_y, num_procs_z);

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];

  sprintf(label, "Subgrid Array");

  if (!AddSubgridArray(data, all_subgrids, label, newhashkey))
    FreeSubgridArray(all_subgrids);
  else
    Tcl_AppendElement(interp, newhashkey);

  return TCL_OK;
}


int PrintDomainCommand(
                       ClientData  clientData,
                       Tcl_Interp *interp,
                       int         argc,
                       char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFPRINTDOMAINUSAGE);
    return TCL_ERROR;
  }

  char       *subgrid_array_hashkey = argv[1];
  SubgridArray *subgrid_array;
  if ((subgrid_array = (SubgridArray*)DataMember(data, subgrid_array_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, subgrid_array_hashkey);
    return TCL_ERROR;
  }


  Tcl_DString result;
  Tcl_DStringInit(&result);

  char line[2048];

  /*--------------------------------------------------------------------
   * Get the processor topology from the database
   *--------------------------------------------------------------------*/
  int P = GetInt(interp, "Process.Topology.P");
  int Q = GetInt(interp, "Process.Topology.Q");
  int R = GetInt(interp, "Process.Topology.R");
  int num_procs = P * Q * R;

  int p;


  sprintf(line, "pfset ProcessGrid.NumSubgrids %d\n", subgrid_array->size);
  Tcl_DStringAppend(&result, line, strlen(line));

  for (p = 0; p < num_procs; p++)
  {
    int s_i;
    ForSubgridI(s_i, subgrid_array)
    {
      Subgrid* subgrid = SubgridArraySubgrid(subgrid_array, s_i);

      int process = SubgridProcess(subgrid);

      if (process == p)
      {
        int ix = SubgridIX(subgrid);
        int iy = SubgridIY(subgrid);
        int iz = SubgridIZ(subgrid);

        int nx = SubgridNX(subgrid);
        int ny = SubgridNY(subgrid);
        int nz = SubgridNZ(subgrid);

        sprintf(line, "pfset ProcessGrid.%d.P %d\n", s_i, process);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.IX %d\n", s_i, ix);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.IY %d\n", s_i, iy);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.IZ %d\n", s_i, iz);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.NX %d\n", s_i, nx);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.NY %d\n", s_i, ny);
        Tcl_DStringAppend(&result, line, strlen(line));

        sprintf(line, "pfset ProcessGrid.%d.NZ %d\n", s_i, nz);
        Tcl_DStringAppend(&result, line, strlen(line));
      }
    }
  }


  Tcl_DStringResult(interp, &result);
  return TCL_OK;
}


/*
 * Builds a subgrid array from the current Parflow database key/values
 * that specify the processor topology.
 */
int BuildDomainCommand(
                       ClientData  clientData,
                       Tcl_Interp *interp,
                       int         argc,
                       char *      argv[])
{
  Data       *data = (Data*)clientData;
  SubgridArray *subgrid_array;

  Tcl_DString result;

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 0)
  {
    WrongNumArgsError(interp, PFBUILDDOMAINUSAGE);
    return TCL_ERROR;
  }

  Tcl_DStringInit(&result);

  subgrid_array = NewSubgridArray();

  /*--------------------------------------------------------------------
   * Get the processor topology from the database
   *--------------------------------------------------------------------*/
  int s_i;
  int size = GetInt(interp, "ProcessGrid.NumSubgrids");

  for (s_i = 0; s_i < size; s_i++)
  {
    char key[1024];
    int p;

    int ix;
    int iy;
    int iz;

    int nx;
    int ny;
    int nz;

    int rx = 0;
    int ry = 0;
    int rz = 0;

    sprintf(key, "ProcessGrid.%d.P", s_i);
    p = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.IX", s_i);
    ix = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.IY", s_i);
    iy = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.IZ", s_i);
    iz = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.NX", s_i);
    nx = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.NY", s_i);
    ny = GetInt(interp, key);

    sprintf(key, "ProcessGrid.%d.NZ", s_i);
    nz = GetInt(interp, key);

    AppendSubgrid(NewSubgrid(ix, iy, iz,
                             nx, ny, nz,
                             rx, ry, rz,
                             p),
                  &subgrid_array);
  }

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];

  sprintf(label, "Subgrid Array");

  if (!AddSubgridArray(data, subgrid_array, label, newhashkey))
    FreeSubgridArray(subgrid_array);
  else
    Tcl_AppendElement(interp, newhashkey);

  return TCL_OK;
}

int Extract2DDomainCommand(
                           ClientData  clientData,
                           Tcl_Interp *interp,
                           int         argc,
                           char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFEXTRACT2DDOMAINUSAGE);
    return TCL_ERROR;
  }

  char       *subgrid_array_hashkey = argv[1];
  SubgridArray *subgrid_array;

  if ((subgrid_array = (SubgridArray*)DataMember(data, subgrid_array_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, subgrid_array_hashkey);
    return TCL_ERROR;
  }

  SubgridArray  *all_subgrids = Extract2DDomain(subgrid_array);

  if (!all_subgrids)
  {
    printf("Extract 2D Domain failed\n");
    return TCL_ERROR;
  }

  char newhashkey[32];
  char label[MAX_LABEL_SIZE];

  sprintf(label, "Subgrid2D Array");

  if (!AddSubgridArray(data, all_subgrids, label, newhashkey))
    FreeSubgridArray(all_subgrids);
  else
    Tcl_AppendElement(interp, newhashkey);

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfsurfacestorage' command
 * Description: Compute the surface storage
 *
 * Cmd. syntax: pfsufacestorage top pressure
 *-----------------------------------------------------------------------*/
int            SurfaceStorageCommand(
                                     ClientData  clientData,
                                     Tcl_Interp *interp,
                                     int         argc,
                                     char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *top;
  Databox    *pressure;
  Databox    *surface_storage;

  char       *filename = "surface storage";
  char       *top_hashkey;
  char       *pressure_hashkey;

  char surface_storage_hashkey[MAX_KEY_SIZE];

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 2)
  {
    WrongNumArgsError(interp, PFSURFACESTORAGEUSAGE);
    return TCL_ERROR;
  }

  top_hashkey = argv[1];
  pressure_hashkey = argv[2];

  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  if ((pressure = DataMember(data, pressure_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pressure_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(pressure);
    int ny = DataboxNy(pressure);

    double x = DataboxX(pressure);
    double y = DataboxY(pressure);
    double z = DataboxZ(pressure);

    double dx = DataboxDx(pressure);
    double dy = DataboxDy(pressure);
    double dz = DataboxDz(pressure);

    /* create the new databox structure for surface storage  */
    if ((surface_storage = NewDatabox(nx, ny, 1, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, surface_storage, filename, surface_storage_hashkey))
        FreeDatabox(surface_storage);
      else
      {
        Tcl_AppendElement(interp, surface_storage_hashkey);
      }

      ComputeSurfaceStorage(top, pressure, surface_storage);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfsubsurfacestorage' command
 * Description: Compute the subsurface storage
 *
 * Cmd. syntax: pfsubsurfacestorage
 *-----------------------------------------------------------------------*/
int            SubsurfaceStorageCommand(
                                        ClientData  clientData,
                                        Tcl_Interp *interp,
                                        int         argc,
                                        char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *mask;
  Databox    *porosity;
  Databox    *saturation;
  Databox    *pressure;
  Databox    *specific_storage;
  Databox    *subsurface_storage;

  char    *mask_hashkey;
  char    *porosity_hashkey;
  char    *saturation_hashkey;
  char    *pressure_hashkey;
  char    *specific_storage_hashkey;
  char subsurface_storage_hashkey[MAX_KEY_SIZE];

  char       *filename = "subsurface storage";

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 5)
  {
    WrongNumArgsError(interp, PFSUBSURFACESTORAGEUSAGE);
    return TCL_ERROR;
  }

  mask_hashkey = argv[1];
  porosity_hashkey = argv[2];
  pressure_hashkey = argv[3];
  saturation_hashkey = argv[4];
  specific_storage_hashkey = argv[5];

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }
  if ((porosity = DataMember(data, porosity_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, porosity_hashkey);
    return TCL_ERROR;
  }

  if ((pressure = DataMember(data, pressure_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pressure_hashkey);
    return TCL_ERROR;
  }

  if ((saturation = DataMember(data, saturation_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, saturation_hashkey);
    return TCL_ERROR;
  }

  if ((specific_storage = DataMember(data, specific_storage_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, specific_storage_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(pressure);
    int ny = DataboxNy(pressure);
    int nz = DataboxNz(pressure);

    double x = DataboxX(pressure);
    double y = DataboxY(pressure);
    double z = DataboxZ(pressure);

    double dx = DataboxDx(pressure);
    double dy = DataboxDy(pressure);
    double dz = DataboxDz(pressure);

    /* create the new databox structure for surface storage  */
    if ((subsurface_storage = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, subsurface_storage, filename, subsurface_storage_hashkey))
        FreeDatabox(subsurface_storage);
      else
      {
        Tcl_AppendElement(interp, subsurface_storage_hashkey);
      }

      ComputeSubsurfaceStorage(mask, porosity, pressure, saturation, specific_storage, subsurface_storage);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfgwstorage' command
 * Description: Compute the subsurface storage **saturated cells only**
 *
 * Cmd. syntax: pfgwstorage
 *-----------------------------------------------------------------------*/
int            GWStorageCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *mask;
  Databox    *porosity;
  Databox    *saturation;
  Databox    *pressure;
  Databox    *specific_storage;
  Databox    *gw_storage;

  char    *mask_hashkey;
  char    *porosity_hashkey;
  char    *saturation_hashkey;
  char    *pressure_hashkey;
  char    *specific_storage_hashkey;
  char gw_storage_hashkey[MAX_KEY_SIZE];

  char    *filename = "groundwater storage";

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 5)
  {
    WrongNumArgsError(interp, PFGWSTORAGEUSAGE);
    return TCL_ERROR;
  }

  mask_hashkey = argv[1];
  porosity_hashkey = argv[2];
  pressure_hashkey = argv[3];
  saturation_hashkey = argv[4];
  specific_storage_hashkey = argv[5];

  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }
  if ((porosity = DataMember(data, porosity_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, porosity_hashkey);
    return TCL_ERROR;
  }

  if ((pressure = DataMember(data, pressure_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pressure_hashkey);
    return TCL_ERROR;
  }

  if ((saturation = DataMember(data, saturation_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, saturation_hashkey);
    return TCL_ERROR;
  }

  if ((specific_storage = DataMember(data, specific_storage_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, specific_storage_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(pressure);
    int ny = DataboxNy(pressure);
    int nz = DataboxNz(pressure);

    double x = DataboxX(pressure);
    double y = DataboxY(pressure);
    double z = DataboxZ(pressure);

    double dx = DataboxDx(pressure);
    double dy = DataboxDy(pressure);
    double dz = DataboxDz(pressure);

    /* create the new databox structure for surface storage  */
    if ((gw_storage = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, gw_storage, filename, gw_storage_hashkey))
        FreeDatabox(gw_storage);
      else
      {
        Tcl_AppendElement(interp, gw_storage_hashkey);
      }

      ComputeGWStorage(mask, porosity, pressure, saturation, specific_storage, gw_storage);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfsurfacerunoff' command
 * Description: Compute the surface runoff
 *
 * Cmd. syntax: pfsurfacerunoff
 *-----------------------------------------------------------------------*/
int            SurfaceRunoffCommand(
                                    ClientData  clientData,
                                    Tcl_Interp *interp,
                                    int         argc,
                                    char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *top;
  Databox    *slope_x;
  Databox    *slope_y;
  Databox    *mannings;
  Databox    *pressure;
  Databox    *surface_runoff;

  char    *top_hashkey;
  char    *slope_x_hashkey;
  char    *slope_y_hashkey;
  char    *mannings_hashkey;
  char    *pressure_hashkey;
  char surface_runoff_hashkey[MAX_KEY_SIZE];

  char       *filename = "surface runoff";

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 5)
  {
    WrongNumArgsError(interp, PFSURFACERUNOFFUSAGE);
    return TCL_ERROR;
  }

  top_hashkey = argv[1];
  slope_x_hashkey = argv[2];
  slope_y_hashkey = argv[3];
  mannings_hashkey = argv[4];
  pressure_hashkey = argv[5];

  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  if ((slope_x = DataMember(data, slope_x_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, slope_x_hashkey);
    return TCL_ERROR;
  }

  if ((slope_y = DataMember(data, slope_y_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, slope_y_hashkey);
    return TCL_ERROR;
  }

  if ((mannings = DataMember(data, mannings_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mannings_hashkey);
    return TCL_ERROR;
  }

  if ((pressure = DataMember(data, pressure_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, pressure_hashkey);
    return TCL_ERROR;
  }


  {
    int nx = DataboxNx(pressure);
    int ny = DataboxNy(pressure);

    double x = DataboxX(pressure);
    double y = DataboxY(pressure);
    double z = DataboxZ(pressure);

    double dx = DataboxDx(pressure);
    double dy = DataboxDy(pressure);
    double dz = DataboxDz(pressure);

    /* create the new databox structure for surface storage  */
    if ((surface_runoff = NewDatabox(nx, ny, 1, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */

      if (!AddData(data, surface_runoff, filename, surface_runoff_hashkey))
        FreeDatabox(surface_runoff);
      else
      {
        Tcl_AppendElement(interp, surface_runoff_hashkey);
      }

      ComputeSurfaceRunoff(top,
                           slope_x,
                           slope_y,
                           mannings,
                           pressure,
                           surface_runoff);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfwatertabledepth' command
 * Description: Compute the water depth
 *
 * Cmd. syntax: pfwatertabledepth top saturation
 *-----------------------------------------------------------------------*/
int            WaterTableDepthCommand(
                                      ClientData  clientData,
                                      Tcl_Interp *interp,
                                      int         argc,
                                      char *      argv[])
{
  Tcl_HashEntry *entryPtr;   /* Points to new hash table entry         */
  Data       *data = (Data*)clientData;

  Databox    *top;
  Databox    *saturation;
  Databox    *water_table_depth;

  char       *filename = "water table depth";
  char       *top_hashkey;
  char       *saturation_hashkey;

  char water_table_depth_hashkey[MAX_KEY_SIZE];

  /* Check and see if there is at least one argument following  */
  /* the command.                                               */
  if (argc == 2)
  {
    WrongNumArgsError(interp, PFWATERTABLEDEPTHUSAGE);
    return TCL_ERROR;
  }

  top_hashkey = argv[1];
  saturation_hashkey = argv[2];

  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  if ((saturation = DataMember(data, saturation_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, saturation_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(saturation);
    int ny = DataboxNy(saturation);
    int nz = 1;

    double x = DataboxX(saturation);
    double y = DataboxY(saturation);
    double z = DataboxZ(saturation);

    double dx = DataboxDx(saturation);
    double dy = DataboxDy(saturation);
    double dz = DataboxDz(saturation);

    /* create the new databox structure for the water table depth  */
    if ((water_table_depth = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */
      if (!AddData(data, water_table_depth, filename, water_table_depth_hashkey))
        FreeDatabox(water_table_depth);
      else
      {
        Tcl_AppendElement(interp, water_table_depth_hashkey);
      }

      ComputeWaterTableDepth(top, saturation, water_table_depth);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfslopex' command
 * Description: Compute slopes in x-direction at all [i,j] using 1st order
 *              upwind finite difference scheme
 *
 * Notes:       local maxima: slope set to max downward gradient
 *              local minima: slope set to zero (no drainage in x-dir)
 *              otherwise:    1st order upwind finite difference
 *
 * Cmd. syntax: pfslopex dem
 *-----------------------------------------------------------------------*/
int            SlopeXUpwindCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *sx;
  char sx_hashkey[MAX_KEY_SIZE];
  char          *filename = "slope_x";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSLOPEXUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for slope values (sx) */
    if ((sx = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, sx, filename, sx_hashkey))
        FreeDatabox(sx);
      else
      {
        Tcl_AppendElement(interp, sx_hashkey);
      }
      /* Compute slopex */
      ComputeSlopeXUpwind(dem, dx, sx);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfslopey' command
 * Description: Compute slopes in y-direction at all [i,j] using 1st order
 *              upwind finite difference scheme
 *
 * Notes:       local maxima: slope set to max downward gradient
 *              local minima: slope set to zero (no drainage in y-dir)
 *              otherwise:    1st order upwind finite difference
 *
 * Cmd. syntax: pfslopey dem
 *-----------------------------------------------------------------------*/
int            SlopeYUpwindCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *sy;
  char sy_hashkey[MAX_KEY_SIZE];
  char          *filename = "slope_y";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSLOPEYUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for slope values (sy) */
    if ((sy = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, sy, filename, sy_hashkey))
        FreeDatabox(sy);
      else
      {
        Tcl_AppendElement(interp, sy_hashkey);
      }
      /* Compute slopey */
      ComputeSlopeYUpwind(dem, dy, sy);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfupstreamarea' command
 * Description: Compute upstream contributing area for each point [i,j]
 *              based on neighboring slopes (sx and sy).
 *
 * Notes:       loops of over all [i,j]
 *              at each [i,j], loops over neighbors to find parent(s)
 *              adds parent area to area[i,j]
 *              recursively loops over each parent's neighbors, calculates
 *                area over ALL upstream cells until reaches a divide
 *              returns values as NUMBER OF CELLS (not actual area)
 *                to calculate areas, simply multiply area*dx*dy
 *
 * Cmd. syntax: pfupstreamarea sx sy
 *-----------------------------------------------------------------------*/
int            UpstreamAreaCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;
  Databox       *sx;
  char          *sx_hashkey;
  Databox       *sy;
  char          *sy_hashkey;

  // Output
  Databox       *area;
  char area_hashkey[MAX_KEY_SIZE];
  char          *filename = "upstream contributing area";

  /* Check if two arguments following command  */
  if (argc == 3)
  {
    WrongNumArgsError(interp, PFUPSTREAMAREAUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  sx_hashkey = argv[2];
  sy_hashkey = argv[3];

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  if ((sx = DataMember(data, sx_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sx_hashkey);
    return TCL_ERROR;
  }

  if ((sy = DataMember(data, sy_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sy_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(sx);
    int ny = DataboxNy(sx);
    int nz = 1;

    double x = DataboxX(sx);
    double y = DataboxY(sx);
    double z = DataboxZ(sx);

    double dx = DataboxDx(sx);
    double dy = DataboxDy(sx);
    double dz = DataboxDz(sx);

    /* create the new databox structure for area values (area) */
    if ((area = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, area, filename, area_hashkey))
        FreeDatabox(area);
      else
      {
        Tcl_AppendElement(interp, area_hashkey);
      }

      /* Compute areas */
      ComputeUpstreamArea(dem, sx, sy, area);
    }

    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pffillflats' command
 * Description: Find flat regions in DEM, eliminat flats by bilinearly
 *              interpolating elevations across flat region.
 *
 * Cmd. syntax: pffillflats dem
 *-----------------------------------------------------------------------*/
int            FillFlatsCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *newdem;
  char newdem_hashkey[MAX_KEY_SIZE];
  char          *filename = "DEM with flats filled";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFFILLFLATSUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for new dem values (newdem) */
    if ((newdem = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, newdem, filename, newdem_hashkey))
      {
        FreeDatabox(newdem);
      }
      else
      {
        Tcl_AppendElement(interp, newdem_hashkey);
      }

      /* Compute DEM w/o flat regions */
      ComputeFillFlats(dem, newdem);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfpitfilldem' command
 * Description: Iterative pit-fill routine to calculate upstream  [i,j]
 *              based on neighboring slopes (sx and sy).
 *
 * Notes:       Assumes that user specifies dpit in same units as DEM
 *
 * Cmd. syntax: pfpitfilldem dem dpit maxiter
 *-----------------------------------------------------------------------*/
int            PitFillCommand(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Inputs
  Databox       *dem;
  char          *dem_hashkey;
  double dpit;
  int maxiter;

  // Output
  Databox       *newdem;
  char          *filename = "Pit-Filled DEM";
  char newdem_hashkey[MAX_KEY_SIZE];

  // Local
  int iter;
  int nsink;
  int i, j;
  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  /* Check if three arguments following command  */
  if (argc == 3)
  {
    WrongNumArgsError(interp, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &dpit) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }
  if (Tcl_GetInt(interp, argv[3], &maxiter) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    nx = DataboxNx(dem);
    ny = DataboxNy(dem);
    nz = 1;

    x = DataboxX(dem);
    y = DataboxY(dem);
    z = DataboxZ(dem);

    dx = DataboxDx(dem);
    dy = DataboxDy(dem);
    dz = DataboxDz(dem);

    /* create the new databox structure for pit-filled dem  */
    if ((newdem = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */
      if (!AddData(data, newdem, filename, newdem_hashkey))
        FreeDatabox(newdem);
      else
      {
        Tcl_AppendElement(interp, newdem_hashkey);
      }

      // Set values of newdem to values of original dem
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < nx; i++)
        {
          *DataboxCoeff(newdem, i, j, 0) = *DataboxCoeff(dem, i, j, 0);
        }
      }

      // Iterate to fill pits...
      iter = 0;
      nsink = 9999;
      while ((iter < maxiter) && (nsink > 0))
      {
        nsink = ComputePitFill(newdem, dpit);
        iter = iter + 1;
      }

      // Print summary...
      printf("*******************************************************\n");
      printf("SUMMARY: pfpitfilldem  \n");
      printf("*******************************************************\n");
      printf("ITERATIONS: \t\t %d \n", iter);
      printf("REMAINING SINKS: \t %d \n", nsink);
      printf("   \n");
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfmovingavgdem' command
 * Description: Iterative moving average routine to fill sinks in DEM
 *              using average of neighboring cells.
 *
 * Notes:       Parameter wsize is given in number of cells
 *
 * Cmd. syntax: pfmovingavgdem dem wsize maxiter
 *-----------------------------------------------------------------------*/
int            MovingAvgCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Inputs
  Databox       *dem;
  char          *dem_hashkey;
  double wsize;
  int maxiter;

  // Output
  Databox       *newdem;
  char          *filename = "Moving-Avg Sink-Filled DEM";
  char newdem_hashkey[MAX_KEY_SIZE];

  // Local
  int iter;
  int nsink;
  int i, j;
  int nx, ny, nz;
  double x, y, z;
  double dx, dy, dz;

  /* Check if three arguments following command  */
  if (argc == 3)
  {
    WrongNumArgsError(interp, PFMOVINGAVGDEMUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if (Tcl_GetDouble(interp, argv[2], &wsize) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFMOVINGAVGDEMUSAGE);
    return TCL_ERROR;
  }
  if (Tcl_GetInt(interp, argv[3], &maxiter) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFMOVINGAVGDEMUSAGE);
    return TCL_ERROR;
  }

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    nx = DataboxNx(dem);
    ny = DataboxNy(dem);
    nz = 1;

    x = DataboxX(dem);
    y = DataboxY(dem);
    z = DataboxZ(dem);

    dx = DataboxDx(dem);
    dy = DataboxDy(dem);
    dz = DataboxDz(dem);

    /* create the new databox structure for moving avg filled dem  */
    if ((newdem = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure the data set pointer was added to */
      /* the hash table successfully.                */
      if (!AddData(data, newdem, filename, newdem_hashkey))
        FreeDatabox(newdem);
      else
      {
        Tcl_AppendElement(interp, newdem_hashkey);
      }

      // Set values of newdem to values of original dem
      for (j = 0; j < ny; j++)
      {
        for (i = 0; i < nx; i++)
        {
          *DataboxCoeff(newdem, i, j, 0) = *DataboxCoeff(dem, i, j, 0);
        }
      }

      // Iterate to fill pits...
      iter = 0;
      nsink = 9999;
      while ((iter < maxiter) && (nsink > 0))
      {
        nsink = ComputeMovingAvg(newdem, wsize);
        iter = iter + 1;
      }

      // Print summary...
      printf("*******************************************************\n");
      printf("SUMMARY: pfmovingavgdem  \n");
      printf("*******************************************************\n");
      printf("ITERATIONS: \t\t %d \n", iter);
      printf("REMAINING SINKS: \t %d \n", nsink);
      printf("   \n");
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfslopexD4' command
 * Description: Compute D4 slope x at all [i,j].
 *
 * Notes:       local minima: slope set to zero (no drainage)
 *              otherwise:    1st order maximum downward slope to lowest
 *                            neighbor (adjacent only!)
 *
 * Cmd. syntax: pfslopexD4 dem
 *-----------------------------------------------------------------------*/
int            SlopeXD4Command(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Outputs
  Databox       *sx;
  char sx_hashkey[MAX_KEY_SIZE];
  char          *filename = "slope-x (D4)";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSLOPEXD4USAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for slope values (slope) */
    if ((sx = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, sx, filename, sx_hashkey))
        FreeDatabox(sx);
      else
      {
        Tcl_AppendElement(interp, sx_hashkey);
      }

      /* Compute sx */
      ComputeSlopeXD4(dem, sx);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfslopeyD4' command
 * Description: Compute D4 slope y at all [i,j].
 *
 * Notes:       local minima: slope set to zero (no drainage)
 *              otherwise:    1st order maximum downward slope to lowest
 *                            neighbor (adjacent only!)
 *
 * Cmd. syntax: pfslopeyD4 dem
 *-----------------------------------------------------------------------*/
int            SlopeYD4Command(
                               ClientData  clientData,
                               Tcl_Interp *interp,
                               int         argc,
                               char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Outputs
  Databox       *sy;
  char sy_hashkey[MAX_KEY_SIZE];
  char          *filename = "slope-y (D4)";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSLOPEYD4USAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for slope values (slope) */
    if ((sy = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, sy, filename, sy_hashkey))
        FreeDatabox(sy);
      else
      {
        Tcl_AppendElement(interp, sy_hashkey);
      }

      /* Compute sy */
      ComputeSlopeYD4(dem, sy);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfslopeD8' command
 * Description: Compute D8 slopes all [i,j].
 *
 * Notes:       local minima: slope set to zero (no drainage)
 *              otherwise:    1st order maximum downward slope to lowest
 *                            neighbor (adjacent or diagonal)
 *
 * Cmd. syntax: pfslopeD8 dem
 *-----------------------------------------------------------------------*/
int            SlopeD8Command(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *slope;
  char slope_hashkey[MAX_KEY_SIZE];
  char          *filename = "slope (D8)";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSLOPED8USAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for slope values (slope) */
    if ((slope = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, slope, filename, slope_hashkey))
        FreeDatabox(slope);
      else
      {
        Tcl_AppendElement(interp, slope_hashkey);
      }
      /* Compute slopex */
      ComputeSlopeD8(dem, slope);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfsegmentD8' command
 * Description: Compute D8 segment lengths for all [i,j].
 *
 * Notes:       local minima: segment length set to zero (as for slope)
 *              otherwise:    segment length is distance between parent and
 *                            child cell centers
 *
 * Cmd. syntax: pfsegmentD8 dem
 *-----------------------------------------------------------------------*/
int            SegmentD8Command(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *ds;
  char ds_hashkey[MAX_KEY_SIZE];
  char          *filename = "ds (D8)";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFSEGMENTD8USAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for ds values (ds) */
    if ((ds = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, ds, filename, ds_hashkey))
        FreeDatabox(ds);
      else
      {
        Tcl_AppendElement(interp, ds_hashkey);
      }
      /* Compute ds */
      ComputeSegmentD8(dem, ds);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfchildD8' command
 * Description: Compute unique D8 childe for all cells;
 *              child[i,j] is elevation of cell to which [i,j] drains
 *              (i.e., elevation of [i,j]'s child)
 *
 * Notes:       local minima: child elevation set to own elevation
 *
 * Cmd. syntax: pfchildD8 dem
 *-----------------------------------------------------------------------*/
int            ChildD8Command(
                              ClientData  clientData,
                              Tcl_Interp *interp,
                              int         argc,
                              char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *child;
  char child_hashkey[MAX_KEY_SIZE];
  char          *filename = "child elevation (D8)";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFCHILDD8USAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for child values (child) */
    if ((child = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, child, filename, child_hashkey))
        FreeDatabox(child);
      else
      {
        Tcl_AppendElement(interp, child_hashkey);
      }
      /* Compute child */
      ComputeChildD8(dem, child);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfflintslaw' command
 * Description: Compute DEM elevations using Flint's Law based on user
 *              provided DEM and parameters (c,p). Computed elevations
 *              returned as new databox.
 *
 * Cmd. syntax: pfflintslaw dem c p
 *-----------------------------------------------------------------------*/
int            FlintsLawCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;
  double c;
  double p;

  // Output
  Databox       *flint;
  char flint_hashkey[MAX_KEY_SIZE];
  char          *filename = "Flint's Law Elevations";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFFLINTSLAWDEMUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[2], &c) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFFLINTSLAWDEMUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[3], &p) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFFLINTSLAWDEMUSAGE);
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for flint values (flint) */
    if ((flint = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, flint, filename, flint_hashkey))
        FreeDatabox(flint);
      else
      {
        Tcl_AppendElement(interp, flint_hashkey);
      }
      /* Compute elevations */
      ComputeFlintsLaw(dem, c, p, flint);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfflintslawfit' command
 * Description: Compute DEM elevations using Flint's Law based on user
 *              provided DEM, with parameters fit based on non-linear
 *              least squares minimization (i.e., Flints law is fit to data)
 *
 * NOTES:       Fitting uses D8 slopes, D8 child elevations, and bi-directional
 *              upstream area.
 *
 * Cmd. syntax: pfflintslawfit dem c0 p0 maxiter
 *-----------------------------------------------------------------------*/
int            FlintsLawFitCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;
  double c, p;
  int maxiter;

  // Output
  Databox       *flint;
  char flint_hashkey[MAX_KEY_SIZE];
  char          *filename = "Flint's Law Elevations";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFFLINTSLAWFITUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[2], &c) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[3], &p) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[4], &maxiter) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFFLINTSLAWDEMUSAGE);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for flint values (flint) */
    if ((flint = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, flint, filename, flint_hashkey))
        FreeDatabox(flint);
      else
      {
        Tcl_AppendElement(interp, flint_hashkey);
      }
      /* Compute elevations */
      ComputeFlintsLawFit(dem, c, p, maxiter, flint);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}

/*-----------------------------------------------------------------------
 * routine for `pfflintslawbybasin' command
 * Description: Compute DEM elevations using Flint's Law based on user
 *              provided DEM, with parameters fit based on non-linear
 *              least squares minimization (i.e., Flints law is fit to data)
 *
 * NOTES:       Fitting uses D8 slopes, D8 child elevations, and bi-directional
 *              upstream area.
 *              Flint's Law is fit for each basin separately...
 *              This is the only difference w/ FlintsLawFitCommand
 *
 * Cmd. syntax: pfflintslawbybasin dem c0 p0 maxiter
 *-----------------------------------------------------------------------*/
int            FlintsLawByBasinCommand(
                                       ClientData  clientData,
                                       Tcl_Interp *interp,
                                       int         argc,
                                       char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;
  double c, p;
  int maxiter;

  // Output
  Databox       *flint;
  char flint_hashkey[MAX_KEY_SIZE];
  char          *filename = "Flint's Law Elevations";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFFLINTSLAWBYBASINUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[2], &c) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetDouble(interp, argv[3], &p) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFPITFILLDEMUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[4], &maxiter) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFFLINTSLAWDEMUSAGE);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for flint values (flint) */
    if ((flint = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, flint, filename, flint_hashkey))
        FreeDatabox(flint);
      else
      {
        Tcl_AppendElement(interp, flint_hashkey);
      }
      /* Compute elevations */
      ComputeFlintsLawByBasin(dem, c, p, maxiter, flint);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfsattrans' command
 * Description: Compute saturated transmissivity for all [i,j].
 *
 * Notes:       See detailed notes in toposlopes.c regarding assumptions,
 *              methods, etc.
 *
 * Cmd. syntax: pfsattrans mask perm
 *-----------------------------------------------------------------------*/
int            SatTransmissivityCommand(
                                        ClientData  clientData,
                                        Tcl_Interp *interp,
                                        int         argc,
                                        char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  int nlayers;
  Databox       *mask;
  char          *mask_hashkey;
  Databox       *perm;
  char          *perm_hashkey;

  // Output
  Databox       *trans;
  char trans_hashkey[MAX_KEY_SIZE];
  char          *filename = "saturated transmissivity";

  /* Check if three arguments following command  */
  if (argc != 4)
  {
    WrongNumArgsError(interp, PFSATTRANSUSAGE);
    return TCL_ERROR;
  }

  if (Tcl_GetInt(interp, argv[1], &nlayers) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFSATTRANSUSAGE);
    return TCL_ERROR;
  }

  printf("TESTING: nlayers=%d n", nlayers);

  mask_hashkey = argv[2];
  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  perm_hashkey = argv[3];
  if ((perm = DataMember(data, perm_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, perm_hashkey);
    return TCL_ERROR;
  }

  {
    // grab databox info for output (trans)
    int nx = DataboxNx(mask);
    int ny = DataboxNy(mask);
    int nz = 1;

    double x = DataboxX(mask);
    double y = DataboxY(mask);
    double z = DataboxZ(mask);

    double dx = DataboxDx(mask);
    double dy = DataboxDy(mask);
    double dz = DataboxDz(mask);

    /* create the new databox structure for topoindex values (topoindex) */
    if ((trans = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, trans, filename, trans_hashkey))
        FreeDatabox(trans);
      else
      {
        Tcl_AppendElement(interp, trans_hashkey);
      }
      /* Compute saturated transmissivity */
      ComputeSatTransmissivity(nlayers, mask, perm, trans);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pftopoindex' command
 * Description: Compute topographic index for all [i,j].
 *
 * Notes:       See detailed notes in toposlopes.c regarding assumptions,
 *              methods, etc.
 *
 * Cmd. syntax: pftopoindex dem sx sy
 *-----------------------------------------------------------------------*/
int            TopoIndexCommand(
                                ClientData  clientData,
                                Tcl_Interp *interp,
                                int         argc,
                                char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *dem;
  char          *dem_hashkey;
  Databox       *sx;
  char          *sx_hashkey;
  Databox       *sy;
  char          *sy_hashkey;

  // Output
  Databox       *topoindex;
  char topoindex_hashkey[MAX_KEY_SIZE];
  char          *filename = "topographic index";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFTOPOINDEXUSAGE);
    return TCL_ERROR;
  }

  dem_hashkey = argv[1];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  sx_hashkey = argv[2];
  if ((sx = DataMember(data, sx_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sx_hashkey);
    return TCL_ERROR;
  }

  sy_hashkey = argv[3];
  if ((sy = DataMember(data, sy_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sy_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for topoindex values (topoindex) */
    if ((topoindex = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, topoindex, filename, topoindex_hashkey))
        FreeDatabox(topoindex);
      else
      {
        Tcl_AppendElement(interp, topoindex_hashkey);
      }
      /* Compute topoindex */
      ComputeTopoIndex(dem, sx, sy, topoindex);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pftoporecharge' command
 * Description: Compute effective recharge at all [i,j] over upstream area
 *              based on topmodel assumptions and given list of river points.
 *
 * Notes:       See detailed notes in toposlopes.c regarding assumptions,
 *              methods, etc.
 *
 * Cmd. syntax: pftopore riverfile nriver  trans dem sx sy
 *
 * Input Notes: nriver is an integer (number of river points)
 *              river  is an array of integers [nriver][2]
 *                     (list of river indices, ordered from outlet to headwaters)
 *              trans  is a Databox of saturated transmissivity
 *              dem    is a Databox of elevations at each cell
 *              sx     is a Databox of slopes (x-dir) -- lets you use processed slopes!
 *              sy     is a Databox of slopes (y-dir) -- lets you use processed slopes!
 *-----------------------------------------------------------------------*/
int            TopoRechargeCommand(
                                   ClientData  clientData,
                                   Tcl_Interp *interp,
                                   int         argc,
                                   char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFTOPORECHARGEUSAGE);
    return TCL_ERROR;
  }

  // Grab inputs
  int nriver;
  if (Tcl_GetInt(interp, argv[1], &nriver) == TCL_ERROR)
  {
    NotAnIntError(interp, 1, PFTOPORECHARGEUSAGE);
    return TCL_ERROR;
  }

  char          *inputfilename;
  inputfilename = argv[2];

  Databox       *trans;
  char          *trans_hashkey;
  trans_hashkey = argv[3];

  Databox       *dem;
  char          *dem_hashkey;
  dem_hashkey = argv[4];

  Databox       *sx;
  char          *sx_hashkey;
  sx_hashkey = argv[5];

  Databox       *sy;
  char          *sy_hashkey;
  sy_hashkey = argv[6];

  // Create output databox
  Databox       *recharge;
  char recharge_hashkey[MAX_KEY_SIZE];
  char          *filename = "effective recharge upstream of cell, based on TOPMODEL";

  // Read river points from input file (inputfilename)
  int line;
  int river[nriver][2];
  FILE          *inputfile;
  if ((inputfile = fopen(inputfilename, "r")) == NULL)
  {
    printf("Input File %s does not seem to exist. \n", inputfilename);
    WrongNumArgsError(interp, PFTOPORECHARGEUSAGE);
  }
  for (line = 0; line < nriver; line++)
  {
    fscanf(inputfile, "%d %d\n", &river[line][0], &river[line][1]);
  }

  // Test datasets
  if ((trans = DataMember(data, trans_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, trans_hashkey);
    return TCL_ERROR;
  }

  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  if ((sx = DataMember(data, sx_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sx_hashkey);
    return TCL_ERROR;
  }

  if ((sy = DataMember(data, sy_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sy_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for reldef values (reldef) */
    if ((recharge = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, recharge, filename, recharge_hashkey))
        FreeDatabox(recharge);
      else
      {
        Tcl_AppendElement(interp, recharge_hashkey);
      }

      // TESTING
      // int i, j;
      // printf( "TESTING INPUT DATA:/n" );
      // for (j = 0; j < ny; j++ ) {
      //  for (i = 0; i < ny; i++ ) {
      //   printf( "%d\t%d\t%10.5f\t%10.5f\t%10.5f\t%10.5f\n", i, j,
      //           *DataboxCoeff(trans,i,j,0), *DataboxCoeff(dem,i,j,0), *DataboxCoeff(sx,i,j,0), *DataboxCoeff(sy,i,j,0) );
      //  }
      // }

      /* Compute recharge */
      ComputeTopoRecharge(river, nriver, trans, dem, sx, sy, recharge);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfeffectiverecharge' command
 * Description: Compute effective recharge at all [i,j] based on total P-ET
 *              over upstream area. Effective recharge is consistent with
 *              TOPMODEL definition, NOT local P-ET.
 *
 * Cmd. syntax: pfeffective recharge precip et sx sy dem
 *
 * Input Notes: precip is total annual (or average annual) precip at each point
 *              et     is total annual (or average annual) ET at each point
 *              sx, sy, and dem are needed to compute upstream areas
 *-----------------------------------------------------------------------*/
int            EffectiveRechargeCommand(
                                        ClientData  clientData,
                                        Tcl_Interp *interp,
                                        int         argc,
                                        char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFEFFRECHARGEUSAGE);
    return TCL_ERROR;
  }

  // Input
  Databox       *precip;
  char          *precip_hashkey;
  Databox       *et;
  char          *et_hashkey;
  Databox       *runoff;
  char          *runoff_hashkey;
  Databox       *sx;
  char          *sx_hashkey;
  Databox       *sy;
  char          *sy_hashkey;
  Databox       *dem;
  char          *dem_hashkey;

  // Output
  Databox       *recharge;
  char recharge_hashkey[MAX_KEY_SIZE];
  char          *filename = "effective recharge over upstream area";

  precip_hashkey = argv[1];
  if ((precip = DataMember(data, precip_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, precip_hashkey);
    return TCL_ERROR;
  }

  et_hashkey = argv[2];
  if ((et = DataMember(data, et_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, et_hashkey);
    return TCL_ERROR;
  }

  runoff_hashkey = argv[3];
  if ((runoff = DataMember(data, runoff_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, runoff_hashkey);
    return TCL_ERROR;
  }

  sx_hashkey = argv[4];
  if ((sx = DataMember(data, sx_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sx_hashkey);
    return TCL_ERROR;
  }

  sy_hashkey = argv[5];
  if ((sy = DataMember(data, sy_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sy_hashkey);
    return TCL_ERROR;
  }

  dem_hashkey = argv[6];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(sx);
    int ny = DataboxNy(sx);
    int nz = 1;

    double x = DataboxX(sx);
    double y = DataboxY(sx);
    double z = DataboxZ(sx);

    double dx = DataboxDx(sx);
    double dy = DataboxDy(sx);
    double dz = DataboxDz(sx);

    /* create the new databox structure for area values (area) */
    if ((recharge = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, recharge, filename, recharge_hashkey))
        FreeDatabox(recharge);
      else
      {
        Tcl_AppendElement(interp, recharge_hashkey);
      }

      /* Compute effective recharge */
      ComputeEffectiveRecharge(precip, et, runoff, sx, sy, dem, recharge);
    }

    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }

  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pftopodeficit' command
 * Description: Compute water deficit for all [i,j]
 *              based on TOPMODEL/topographic index.
 *
 * Cmd. syntax: pftopodeficit profile m trans dem slopex slopey recharge ssat sres porosity mask
 *-----------------------------------------------------------------------*/
int            TopoDeficitCommand(
                                  ClientData  clientData,
                                  Tcl_Interp *interp,
                                  int         argc,
                                  char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  int profile;
  char          *profilename;
  double m;
  Databox       *trans;
  char          *trans_hashkey;
  Databox       *dem;
  char          *dem_hashkey;
  Databox       *sx;
  char          *sx_hashkey;
  Databox       *sy;
  char          *sy_hashkey;
  Databox       *recharge;
  char          *recharge_hashkey;
  Databox       *ssat;
  char          *ssat_hashkey;
  Databox       *sres;
  char          *sres_hashkey;
  Databox       *porosity;
  char          *porosity_hashkey;
  Databox       *mask;
  char          *mask_hashkey;

  // Output
  Databox       *deficit;
  char deficit_hashkey[MAX_KEY_SIZE];
  char          *filename = "water deficit based on TOPMODEL [m]";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFTOPODEFUSAGE);
    return TCL_ERROR;
  }

  profilename = argv[1];
  profile = 0;    // Fixes compiler warning
  if (strcmp(profilename, "Exponential") == 0)
  {
    profile = 0;
    printf("Profile = %s (case = %d) \n", profilename, profile);
  }
  else if (strcmp(profilename, "Linear") == 0)
  {
    profile = 1;
    printf("Profile = %s (case = %d) \n", profilename, profile);
  }
  else
  {
    printf("Profile = %s -- NOT A VALID OPTION! \n", profilename);
    printf("(valid options: Exponential, Linear)\n");
    InvalidOptionError(interp, 1, PFTOPODEFUSAGE);
  }

  if (Tcl_GetDouble(interp, argv[2], &m) == TCL_ERROR)
  {
    NotADoubleError(interp, 1, PFTOPODEFUSAGE);
    return TCL_ERROR;
  }

  trans_hashkey = argv[3];
  if ((trans = DataMember(data, trans_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, trans_hashkey);
    return TCL_ERROR;
  }

  dem_hashkey = argv[4];
  if ((dem = DataMember(data, dem_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, dem_hashkey);
    return TCL_ERROR;
  }

  sx_hashkey = argv[5];
  if ((sx = DataMember(data, sx_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sx_hashkey);
    return TCL_ERROR;
  }

  sy_hashkey = argv[6];
  if ((sy = DataMember(data, sy_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sy_hashkey);
    return TCL_ERROR;
  }

  recharge_hashkey = argv[7];
  if ((recharge = DataMember(data, recharge_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, recharge_hashkey);
    return TCL_ERROR;
  }

  ssat_hashkey = argv[8];
  if ((ssat = DataMember(data, ssat_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, ssat_hashkey);
    return TCL_ERROR;
  }

  sres_hashkey = argv[9];
  if ((sres = DataMember(data, sres_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sres_hashkey);
    return TCL_ERROR;
  }

  porosity_hashkey = argv[10];
  if ((porosity = DataMember(data, porosity_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, porosity_hashkey);
    return TCL_ERROR;
  }

  mask_hashkey = argv[11];
  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(dem);
    int ny = DataboxNy(dem);
    int nz = 1;

    double x = DataboxX(dem);
    double y = DataboxY(dem);
    double z = DataboxZ(dem);

    double dx = DataboxDx(dem);
    double dy = DataboxDy(dem);
    double dz = DataboxDz(dem);

    /* create the new databox structure for deficit values */
    if ((deficit = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, deficit, filename, deficit_hashkey))
        FreeDatabox(deficit);
      else
      {
        Tcl_AppendElement(interp, deficit_hashkey);
      }
      /* Compute deficit */
      ComputeTopoDeficit(profile, m, trans, dem, sx, sy, recharge, ssat, sres, porosity, mask, deficit);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pftopowt' command
 * Description: Compute water depth from column water deficit for all [i,j]
 *              based on TOPMODEL/topographic index.
 *
 * Cmd. syntax: pftopowt deficit porosity ssat sres mask top wtdepth
 *-----------------------------------------------------------------------*/
int            TopoDeficitToWTCommand(
                                      ClientData  clientData,
                                      Tcl_Interp *interp,
                                      int         argc,
                                      char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *deficit;
  char          *deficit_hashkey;
  Databox       *porosity;
  char          *porosity_hashkey;
  Databox       *ssat;
  char          *ssat_hashkey;
  Databox       *sres;
  char          *sres_hashkey;
  Databox       *mask;
  char          *mask_hashkey;
  Databox       *top;
  char          *top_hashkey;

  // Output
  Databox       *wtdepth;
  char wtdepth_hashkey[MAX_KEY_SIZE];
  char          *filename = "water table depth based on TOPMODEL water deficit [m]";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFTOPODEFTOWTUSAGE);
    return TCL_ERROR;
  }

  /* Grab inputs */
  deficit_hashkey = argv[1];
  if ((deficit = DataMember(data, deficit_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, deficit_hashkey);
    return TCL_ERROR;
  }

  porosity_hashkey = argv[2];
  if ((porosity = DataMember(data, porosity_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, porosity_hashkey);
    return TCL_ERROR;
  }

  ssat_hashkey = argv[3];
  if ((ssat = DataMember(data, ssat_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, ssat_hashkey);
    return TCL_ERROR;
  }

  sres_hashkey = argv[4];
  if ((sres = DataMember(data, sres_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, sres_hashkey);
    return TCL_ERROR;
  }

  mask_hashkey = argv[5];
  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  top_hashkey = argv[6];
  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(top);
    int ny = DataboxNy(top);
    int nz = 1;

    double x = DataboxX(top);
    double y = DataboxY(top);
    double z = DataboxZ(top);

    double dx = DataboxDx(top);
    double dy = DataboxDy(top);
    double dz = DataboxDz(top);

    /* create the new databox structure for deficit values */
    if ((wtdepth = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, wtdepth, filename, wtdepth_hashkey))
        FreeDatabox(wtdepth);
      else
      {
        Tcl_AppendElement(interp, wtdepth_hashkey);
      }
      /* Compute wt depth */
      ComputeTopoDeficitToWT(deficit, porosity, ssat, sres, mask, top, wtdepth);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}


/*-----------------------------------------------------------------------
 * routine for `pfwttopress' command
 * Description: Compute hydrostratic pressure field from water table depth
 *
 * Cmd. syntax: pfhydrostatic wtdepth top mask
 *-----------------------------------------------------------------------*/
int            HydroStatFromWTCommand(
                                      ClientData  clientData,
                                      Tcl_Interp *interp,
                                      int         argc,
                                      char *      argv[])
{
  Tcl_HashEntry *entryPtr;    // Points to new hash table entry
  Data          *data = (Data*)clientData;

  // Input
  Databox       *wtdepth;
  char          *wtdepth_hashkey;
  Databox       *mask;
  char          *mask_hashkey;
  Databox       *top;
  char          *top_hashkey;

  // Output
  Databox       *press;
  char press_hashkey[MAX_KEY_SIZE];
  char          *filename = "hydrostatic pressure field with respec to water table depths provided [m]";

  /* Check if one argument following command  */
  if (argc == 1)
  {
    WrongNumArgsError(interp, PFHYDROSTATUSAGE);
    return TCL_ERROR;
  }

  /* Grab inputs */
  wtdepth_hashkey = argv[1];
  if ((wtdepth = DataMember(data, wtdepth_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, wtdepth_hashkey);
    return TCL_ERROR;
  }

  mask_hashkey = argv[2];
  if ((mask = DataMember(data, mask_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, mask_hashkey);
    return TCL_ERROR;
  }

  top_hashkey = argv[3];
  if ((top = DataMember(data, top_hashkey, entryPtr)) == NULL)
  {
    SetNonExistantError(interp, top_hashkey);
    return TCL_ERROR;
  }

  {
    int nx = DataboxNx(mask);
    int ny = DataboxNy(mask);
    int nz = DataboxNz(mask);

    double x = DataboxX(mask);
    double y = DataboxY(mask);
    double z = DataboxZ(mask);

    double dx = DataboxDx(mask);
    double dy = DataboxDy(mask);
    double dz = DataboxDz(mask);

    /* create the new databox structure for deficit values */
    if ((press = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)))
    {
      /* Make sure dataset pointer was added to hash table   */
      if (!AddData(data, press, filename, press_hashkey))
        FreeDatabox(press);
      else
      {
        Tcl_AppendElement(interp, press_hashkey);
      }

      /* Compute pressure field */
      ComputeHydroStatFromWT(wtdepth, top, mask, press);
    }
    else
    {
      ReadWriteError(interp);
      return TCL_ERROR;
    }
  }
  return TCL_OK;
}
