 /*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.48 $
 *********************************************************************EHEADER*/
/*****************************************************************************
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


/*-----------------------------------------------------------------------
 * Load:
 *   distribute the data
 *-----------------------------------------------------------------------*/

void           Load(type, filename, all_subgrids, background, databox)
int            type;
char          *filename;
SubgridArray  *all_subgrids;
Background    *background;
Databox       *databox;
{
   switch(type)
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

int            PFDistCommand(clientData, interp, argc, argv)
ClientData     clientData;
Tcl_Interp    *interp;
int            argc;
char          *argv[];
{

   char *filename;
   char *filetype;

   int num_procs_x;
   int num_procs_y;
   int num_procs_z;
   int num_procs;

   Background    *background;
   Grid          *user_grid;
   SubgridArray  *all_subgrids;

   Databox *inbox;

   char command[1024];

   if (argc != 2)
   {
      WrongNumArgsError(interp, LOADPFUSAGE);
      return TCL_ERROR;
   }

    filename = argv[1];

    /* Make sure the file extension is valid */
    
    if ((filetype = GetValidFileExtension(filename)) == (char *)NULL)
    {
       InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
       return TCL_ERROR;
    }

    if (strcmp (filetype, "pfb") == 0)
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
       user_grid  = ReadUserGrid(interp);

       /*--------------------------------------------------------------------
	* Get inbox from input_filename
	*--------------------------------------------------------------------*/

       inbox = Read(ParflowB, filename);

       /*--------------------------------------------------------------------
	* Load the data
	*--------------------------------------------------------------------*/

       all_subgrids = DistributeUserGrid(user_grid, num_procs,
				 num_procs_x, num_procs_y, num_procs_z);

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


/**
Get an input string from the input database.  If the key is not
found print an error and exit.

There is no checking on what the string contains, anything other than
NUL is allowed. 

@memo Get a string from the input database
@param interp TCL interpreter with the database
@param key The key to search for
@return The string which matches the search key
*/
char *GetString(Tcl_Interp *interp, char *key)
{
   Tcl_Obj *array_name;
   Tcl_Obj *key_name;
   Tcl_Obj *value;

   int length;

   array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
   key_name = Tcl_NewStringObj(key, strlen(key));

   if ( (value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)) )
   {
      return strdup(Tcl_GetStringFromObj(value, &length));
   }
   else
   {
      return NULL;
   }
}


/**
Get an input string from the input database.  If the key is not
found print an error and exit.

There is no checking on what the string contains, anything other than
NUL is allowed. 

@memo Get a string from the input database
@param interp TCL interpreter with the database
@param key The key to search for
@return The string which matches the search key
*/
int GetInt(Tcl_Interp *interp, char *key)
{
   Tcl_Obj *array_name;
   Tcl_Obj *key_name;
   Tcl_Obj *value;

   int ret;

   array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
   key_name = Tcl_NewStringObj(key, strlen(key));

   if ( (value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)) )
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
Get an input string from the input database.  If the key is not
found print an error and exit.

There is no checking on what the string contains, anything other than
NUL is allowed. 

@memo Get a string from the input database
@param interp TCL interpreter with the database
@param key The key to search for
@return The string which matches the search key
*/
int GetIntDefault(Tcl_Interp *interp, char *key, int def)
{
   Tcl_Obj *array_name;
   Tcl_Obj *key_name;
   Tcl_Obj *value;

   int ret;

   array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
   key_name = Tcl_NewStringObj(key, strlen(key));

   if ( (value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)) )
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
Get an input string from the input database.  If the key is not
found print an error and exit.

There is no checking on what the string contains, anything other than
NUL is allowed. 

@memo Get a string from the input database
@param interp TCL interpreter with the database
@param key The key to search for
@return The string which matches the search key
*/
double GetDouble(Tcl_Interp *interp, char *key)
{
   Tcl_Obj *array_name;
   Tcl_Obj *key_name;
   Tcl_Obj *value;

   double ret;

   array_name = Tcl_NewStringObj("Parflow::PFDB", 13);
   key_name = Tcl_NewStringObj(key, strlen(key));

   if ( (value = Tcl_ObjGetVar2(interp, array_name, key_name, 0)) )
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
/*										*/
/* Parameters - None                                                            */
/*                                                                              */
/* Return value - Data * - a pointer to the Data structure if one               */
/*                         could be allocated or null otherwise.                */

Data    *InitPFToolsData()
{
   Data *new;  /* Data structure used to hold data set hash table */

   if ((new = calloc(1, sizeof (Data))) == NULL)
      return (NULL);

   Tcl_InitHashTable(&DataMembers(new), TCL_STRING_KEYS);

   DataGridType(new) = cell;
   DataTotalMem(new) = 0;
   DataNum(new) = 0;

   return new;
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

int       AddData(data, databox, label, hashkey)
Data     *data;
Databox  *databox;
char     *label;
char     *hashkey;
{
   Tcl_HashEntry *entryPtr;  /* Points to new hash table entry         */
   int            new;       /* 1 if the hashkey already exists        */
   int            num;       /* The number of the data set to be added */

   num = DataNum(data);

   /* Keep tring to find a unique hash key */

   do
   {
      sprintf(hashkey, "dataset%d", num); 
      if ((entryPtr = Tcl_CreateHashEntry(&DataMembers(data), hashkey, &new))
          == NULL)
         return (0);
      
      num++;

   } while (!new);

   /* Truncate the label if it is too large */

   if ((strlen(label) + 1) > MAX_LABEL_SIZE)
      label[MAX_LABEL_SIZE - 1] = 0; 
      
   strcpy(DataboxLabel(databox), label);
   Tcl_SetHashValue(entryPtr, databox);

   DataNum(data)++;
   DataTotalMem(data)++;

   return (1);

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
/* Return value - void								*/
/*								                */

void               PFTExitProc(clientData)
ClientData clientData;
{
   Tcl_HashSearch  search;
   Tcl_HashEntry  *entryPtr;
   Databox        *databox;

   Data       *data = (Data *)clientData;

   entryPtr = Tcl_FirstHashEntry(&DataMembers(data), &search);

   /* Free the dynamic array in each data box */
   /* and free each data box.                 */

   while (entryPtr != NULL)
   {
      databox = (Databox *)Tcl_GetHashValue(entryPtr);
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
/* char *key1 - A hash key to be compared					*/
/* char *key2 - A hash key to be compared					*/
/*										*/
/* Return value - int - Note: hashkeys have the form: datasetn where n is an    */
/*                            integer.                                          */
/*                      -1 if n1 in key1 is less than n2 in key2		*/
/*                       1 if n1 in key1 is greater than n2 in key2		*/
/*                       0 if they are equal    				*/

int keycompare (key1, key2)
const void *key1;
const void *key2;
{
   char *endnum1;           /* Points to the end of string key1 points to   */
   char *endnum2;           /* Points to the end of string key2 points to   */
   char *number1;           /* Points to the number substring in *key1      */
   char *number2;           /* Points to the number substring in *key2      */
   int num1, num2;          /* The numbers after they are converted to ints */
   int ret;                 /* The return value.                            */

   endnum1 = *(char **)key1;
   endnum2 = *(char **)key2;
   number1 = NULL;
   number2 = NULL;

   /* Find the end of the hash key.  It ends with a */
   /* space character, that separates the key from  */
   /* the datr set label.  This is why we look for  */
   /* the space.                                    */

   while (*endnum1 != ' ') {

     /* Point number1 to the begining of the number */
     /* substring.				    */

     if (!number1 && isdigit(*endnum1))
        number1 = endnum1;

     endnum1++;

   }

   *endnum1 = '\0';

   /* Find the end of the second hash key */

   while (*endnum2 != ' ') {

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

   return (ret);

}
   
   
   

/*************************************************************************/
/* Tcl Commands                                                          */
/*************************************************************************/

/* The procedures below have the parameters necessary to make them Tcl commands.*/
/* When a PFTools command is executed from within Tcl, Tcl will send four       */
/* arguments to the procedure which implements the command.  They are described */
/* here.									*/
/*										*/
/* ClientData clientData - Points to data that a Tcl command may need acess to. */
/*                         In the case of the PFTools commands, it will point   */
/*                         to the Data structure which contains the hash table  */
/*                         of data box pointers.                                */
/* Tcl_Interp *interp    - The interpreter being used to execute PFTools 	*/
/*                         commands.						*/
/* int argc              - The number of agruments in the PFTools command       */
/* char *argv            - each of the arguments in the command                 */




/*-----------------------------------------------------------------------
 * routine for `pfgetsubbox' command
 *
 * Cmd. Syntax: pfgetsubbox dataset il jl kl iu ju ku
 *-----------------------------------------------------------------------*/

int               GetSubBoxCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;

   int            il, jl, kl;
   int            iu, ju, ku;
   char          *hashkey_in;
   Databox       *databox_in;
   Databox       *sub_box;

   char         newhashkey[32];
   char         label[MAX_LABEL_SIZE];
   
   
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

int               EnlargeBoxCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;

   int            sx, sy, sz;
   char          *hashkey_in;
   Databox       *databox_in;
   Databox       *new_box;

   char         newhashkey[32];
   char         label[MAX_LABEL_SIZE];
   
   
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
 * Cmd. syntax: pfload dataset
 *-----------------------------------------------------------------------*/

int            ReLoadPFCommand(clientData, interp, argc, argv)
ClientData     clientData;
Tcl_Interp    *interp;
int            argc;
char          *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;
   char          *hashkey;
   Databox       *databox;

   char          *filename;
   char          *filetype;

   FILE           *fp;

   if ( argc != 2 )
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
   
   if ((filetype = GetValidFileExtension(filename)) == (char *)NULL)
   {
      InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
      return TCL_ERROR;
   }

   if (strcmp (filetype, "pfb") == 0)
      databox = ReadParflowB(filename);
   else if (strcmp(filetype, "pfsb") == 0)
      databox = ReadParflowSB(filename);
   else if (strcmp(filetype, "sa") == 0)
      databox = ReadSimpleA(filename);
   else if (strcmp(filetype, "sb") == 0)
      databox = ReadSimpleB(filename);
   else
      databox = ReadRealSA(filename);

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
 * Cmd. syntax: pfload [-option] filename
 *-----------------------------------------------------------------------*/

int            LoadPFCommand(clientData, interp, argc, argv)
ClientData     clientData;
Tcl_Interp    *interp;
int            argc;
char          *argv[];
{
   Data       *data = (Data *)clientData;

   Databox    *databox;

   char       *filetype, *filename;
   char        newhashkey[MAX_KEY_SIZE];


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

      if (argc == 2)
      {
         MissingFilenameError(interp, 1, LOADPFUSAGE);
         return TCL_ERROR;
      }
      else
        filename = argv[2];
   }

   /* If no option is given, then check the extension of the   */
   /* filename.  If the extension on the filename is invalid,  */
   /* then give an error.                                      */

   else
   {
      filename = argv[1];

      /* Make sure the file extension is valid */

      if ((filetype = GetValidFileExtension(filename)) == (char *)NULL)
      {
         InvalidFileExtensionError(interp, 1, LOADPFUSAGE);
         return TCL_ERROR;
      }

   }

   if (strcmp (filetype, "pfb") == 0)
      databox = ReadParflowB(filename);
   else if (strcmp(filetype, "pfsb") == 0)
      databox = ReadParflowSB(filename);
   else if (strcmp(filetype, "sa") == 0)
      databox = ReadSimpleA(filename);
   else if (strcmp(filetype, "sb") == 0)
      databox = ReadSimpleB(filename);
   else if (strcmp(filetype, "fld") == 0)
      databox = ReadAVSField(filename);
   else
      databox = ReadRealSA(filename);

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

int            LoadSDSCommand(clientData, interp, argc, argv)
ClientData     clientData;
Tcl_Interp    *interp;
int            argc;
char          *argv[];
{
   Data       *data = (Data *)clientData;

   Databox    *databox;
   char        newhashkey[MAX_KEY_SIZE];

   char       *filename;
   char       *label;
   int         ds_num;
   

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

      if (Tcl_GetInt(interp, argv[2], &ds_num) ==  TCL_ERROR)
      {
         NotAnIntError(interp, 2, LOADSDSUSAGE);
         return TCL_ERROR;
      }

   }

   else
      ds_num = 0;

   filename = argv[1];
   
   databox = ReadSDS(filename, ds_num);

   label = (char *)calloc(strlen(filename) + 20, sizeof(char));
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

   free((char *)label);
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

int               SavePFCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *filetype, *filename;
   FILE          *fp;

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
   if (strcmp(filetype, "pfb") == 0) {
      /* Make sure the file could be opened */
      if ((fp = fopen(filename, "wb")) == NULL)
      {
	 ReadWriteError(interp);
	 return TCL_ERROR;
      }
      
      PrintParflowB(fp, databox);
   }
   else if (strcmp(filetype, "sa") == 0) {
      /* Make sure the file could be opened */
      if ((fp = fopen(filename, "wb")) == NULL)
      {
	 ReadWriteError(interp);
	 return TCL_ERROR;
      }
      
      PrintSimpleA(fp, databox);
   }
   else if (strcmp(filetype, "sb") == 0) {
      /* Make sure the file could be opened */
      if ((fp = fopen(filename, "wb")) == NULL)
      {
	 ReadWriteError(interp);
	 return TCL_ERROR;
      }
      
      PrintSimpleB(fp, databox);
   }
   else if (strcmp(filetype, "fld") == 0) {
      /* Make sure the file could be opened */
      if ((fp = fopen(filename, "wb")) == NULL)
      {
	 ReadWriteError(interp);
	 return TCL_ERROR;
      }
      
      PrintAVSField(fp, databox);
   }
   else if (strcmp(filetype, "vis") == 0) {
      /* Make sure the file could be opened */
      if ((fp = fopen(filename, "wb")) == NULL)
      {
	 ReadWriteError(interp);
	 return TCL_ERROR;
      }
      
      PrintVizamrai(fp, databox);
   }
#ifdef HAVE_SILO
   else if (strcmp(filetype, "silo") == 0) {
      PrintSilo(filename, databox);
   }
#endif

   /* Close the file, if opened */
   if(fp) {
      fclose(fp);
   }
   return TCL_OK;
}


#ifdef HAVE_HDF4

/*-----------------------------------------------------------------------
 * routine for `pfsavesds' command
 * Description: The first argument is the data set to be saved in HDF
 *              format, the second is the file type, and the third is the
 *              filename.
 * Cmd. syntax: pfsavesds dataset -filetype filename 
 *-----------------------------------------------------------------------*/

int              SaveSDSCommand(clientData, interp, argc, argv)
ClientData       clientData;
Tcl_Interp      *interp;
int              argc;
char            *argv[];
{
  Data          *data = (Data *)clientData;

  Tcl_HashEntry *entryPtr;
  Databox       *databox;

  char          *hashkey;
  char          *printoption;
  char          *filename;
  int32          type;

  
  /* Make sure that there are enough arguments */

  if(argc != 4)
  {
     WrongNumArgsError(interp, SAVESDSUSAGE);
     return TCL_ERROR;
  }

  hashkey = argv[1];

  /* Make sure that the data set exists */

  if((databox = DataMember(data, hashkey, entryPtr)) == NULL)
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

  if (strcmp(printoption, "float32") == 0 )
    type = DFNT_FLOAT32;
  else if (strcmp(printoption, "float64") == 0 )
    type = DFNT_FLOAT64;
  else if (strcmp(printoption, "int8") == 0 )
    type = DFNT_INT8;
  else if (strcmp(printoption, "uint8") == 0 )
    type = DFNT_UINT8;
  else if (strcmp(printoption, "int16") == 0 )
    type = DFNT_INT16;
  else if (strcmp(printoption, "uint16") == 0 )
    type = DFNT_UINT16;
  else if (strcmp(printoption, "int32") == 0 )
    type = DFNT_INT32;
  else if (strcmp(printoption, "uint32") == 0 )
    type = DFNT_UINT32;
  else if (strcmp(printoption, "int32") == 0 )
    type = DFNT_INT32;
  else
    {
      InvalidOptionError(interp, 2, SAVESDSUSAGE);
      return TCL_ERROR;
    }

  /* Make sure the file could be written to */

  if(!PrintSDS(filename, type, databox))
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

int                GetListCommand(clientData, interp, argc, argv)
ClientData         clientData;
Tcl_Interp        *interp;
int                argc;
char              *argv[];
{
   Data           *data = (Data *)clientData;

   Tcl_HashEntry  *entryPtr;
   Tcl_HashSearch  search;
   Databox        *databox;

   Tcl_DString     dspair;
   Tcl_DString     result;
   char           *pair;
   char           *hashkey;

   char           **list;
   int             i;


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
      list = (char **)calloc(DataTotalMem(data), sizeof(char *));

      /* Copy the pairs from the hash table to the array */
      /* of strings.                                     */

      for (i = 0; i < DataTotalMem(data); i++)
      {
         databox = (Databox *)Tcl_GetHashValue(entryPtr);
         hashkey = Tcl_GetHashKey(&DataMembers(data), entryPtr);

         (void *) Tcl_DStringAppendElement(&dspair, hashkey);

         list[i] = (char *)calloc(MAX_LABEL_SIZE, sizeof(char));
         strcpy(list[i], Tcl_DStringAppendElement(&dspair,
                DataboxLabel(databox)));
         Tcl_DStringFree(&dspair);

         entryPtr = Tcl_NextHashEntry(&search);
      }

      qsort(list, DataTotalMem(data), sizeof(char *), keycompare);

      /* Append the sorted elements to the Tcl result string */

      for (i = 0; i < DataTotalMem(data); i++)
      {
         (void *) Tcl_DStringAppendElement(&result, list[i]); 
         free((char *) list[i]);
      }

      free((char **)list);
 
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

int               GetEltCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   int            i, j, k;
   char          *hashkey;
   char           result[128];

   Tcl_HashEntry *entryPtr;
   Databox       *databox;
   

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

   sprintf(result, "%e", *DataboxCoeff(databox, i, j, k));
   Tcl_AppendElement(interp, result);
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

int               GetGridCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

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

int            GridTypeCommand(clientData, interp, argc, argv)
ClientData     clientData;
Tcl_Interp    *interp;
int            argc;
char          *argv[];
{
   Data       *data = (Data *)clientData;

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

int               CVelCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;
   
   char          *cond, *pHead;
   char           newhashkey[96];
   Tcl_HashEntry *entryPtr;
   Databox       *databoxk, *databoxh;

   char           label[MAX_LABEL_SIZE];
   char          *component_ptr;
   
   Databox      **vel;
   
   /* There must two data set names given */

   if (argc != 3)
   {
      WrongNumArgsError(interp, CVELUSAGE);
      return TCL_ERROR;
   }

   cond  = argv[1];
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

int               VVelCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *cond, *pHead;
   char           newhashkey[MAX_KEY_SIZE];
   Databox       *databoxk, *databoxh;
   Tcl_HashEntry *entryPtr;

   char           label[MAX_LABEL_SIZE];
   char          *component_ptr;
   
   Databox      **vel;
   

   /* Two data set names must be given */

   if (argc != 3)
   {
      WrongNumArgsError(interp, VVELUSAGE);
      return TCL_ERROR;
   }

   cond  = argv[1];
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

int               BFCVelCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *cond, *pHead;
   char           newhashkey[MAX_KEY_SIZE];
   Databox       *databoxk, *databoxh;
   Tcl_HashEntry *entryPtr;

   char           label[MAX_LABEL_SIZE];
   char          *component_ptr;
   
   Databox      **vel;
   

   /* Two data set names must be given */

   if (argc != 3)
   {
      WrongNumArgsError(interp, BFCVELUSAGE);
      return TCL_ERROR;
   }

   cond  = argv[1];
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

int               VMagCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;

   char          *hashkeyx, *hashkeyy, *hashkeyz;
   Databox       *databoxx, *databoxy, *databoxz;
   Databox       *vmag;

   char         newhashkey[32];
   char         label[MAX_LABEL_SIZE];
   
   
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

int               HHeadCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *pHead;
   char           newhashkey[MAX_KEY_SIZE];

   Tcl_HashEntry *entryPtr;
   Databox       *databoxh;
   Databox       *hHead;

   char           label[MAX_LABEL_SIZE];
   
   
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

int               PHeadCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *hHead;
   char           newhashkey[MAX_KEY_SIZE];

   Tcl_HashEntry *entryPtr;
   Databox       *databoxh;
   Databox       *pHead;

   char           label[MAX_LABEL_SIZE];
   
   
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

int               FluxCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *cond, *hHead;
   Tcl_HashEntry *entryPtr;

   Databox       *databoxk, *databoxh;
   Databox       *flux;
   char           newhashkey[MAX_KEY_SIZE];

   char           label[MAX_LABEL_SIZE];


   /* Two arguments must be given */

   if (argc != 3)
   {
      WrongNumArgsError(interp, FLUXUSAGE);
      return TCL_ERROR;
   }

   cond  = argv[1];
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

int               NewGridCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *npoints, *origin, *intervals;
   char          *label;
   Databox       *newdatabox;

   char          *num;
   int            nx, ny, nz;
   double         x,  y,  z;
   double         dx, dy, dz;
   
   char           newhashkey[MAX_KEY_SIZE];

   /* Two arguments must be given */

   if (argc != 5)
   {
      WrongNumArgsError(interp, NEWGRIDUSAGE);
      return TCL_ERROR;
   }

   npoints     = argv[1];
   origin      = argv[2];
   intervals   = argv[3];
   label       = argv[4];

   /* Make sure that the number of points along each axis are all */
   /* integers.                                                   */
 
   if (!(num = strtok(npoints, WS)) || (sscanf(num, "%d", &nx) != 1) ||
       !(num = strtok(NULL, WS))    || (sscanf(num, "%d", &ny) != 1) ||
       !(num = strtok(NULL, WS))    || (sscanf(num, "%d", &nz) != 1))
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
       !(num = strtok(NULL, WS))   || (sscanf(num, "%lf", &y) != 1) ||
       !(num = strtok(NULL, WS))       || (sscanf(num, "%lf", &z) != 1))
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
       !(num = strtok(NULL, WS))      || (sscanf(num, "%lf", &dy) != 1) ||
       !(num = strtok(NULL, WS))      || (sscanf(num, "%lf", &dz) != 1))
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

   newdatabox = NewDatabox(nx,ny,nz, x, y, z, dx, dy, dz);

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
 * routine for `pfnewlabel' command
 * Description: A data set hash key and a label are passed as arguments.
 *              the data set cooresponding to the hash key will have
 *              it's label changed.  Nothing will be appended to the
 *              Tcl result upon success.
 *
 * Cmd. Syntax: pfnewlabel dataset newlabel
 *-----------------------------------------------------------------------*/


int                NewLabelCommand(clientData, interp, argc, argv)
ClientData         clientData;
Tcl_Interp        *interp;
int                argc;
char              *argv[];
{
   Data          *data = (Data *)clientData;

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

int               AxpyCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *hashkeyx, *hashkeyy;
   Tcl_HashEntry *entryPtr;
   Databox       *databoxx, *databoxy;
   double         alpha;

   
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
 * routine for `pfstats' command
 * Description: A data set hash key is given and a list of statistics is
 *              appended to the Tcl result.  The data returned is the
 *              minimum, maximum, mean, sum, variance, and the standard
 *              deviation of the data contained in the set.
 *
 * Cmd. Syntax: pfstats dataset
 *-----------------------------------------------------------------------*/

int               GetStatsCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   char          *hashkey;
   Tcl_HashEntry *entryPtr;
   Databox       *databox;
   
   double         min, max, mean, sum, variance, stdev;
   char           num[32];

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

   sprintf(num, "%e", min);
   Tcl_AppendElement(interp, num);
   sprintf(num, "%e", max);
   Tcl_AppendElement(interp, num);
   sprintf(num, "%e", mean);
   Tcl_AppendElement(interp, num);
   sprintf(num, "%e", sum);
   Tcl_AppendElement(interp, num);
   sprintf(num, "%e", variance);
   Tcl_AppendElement(interp, num);
   sprintf(num, "%e", stdev);
   Tcl_AppendElement(interp, num);

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

int               MDiffCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;
   char          *hashkeyp, *hashkeyq;
   Databox       *databoxp, *databoxq;

   int            sd;
   double         abs_zero;

   Tcl_DString    result;

   
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

   MSigDiff(databoxp, databoxq, sd, abs_zero, &result);

   Tcl_DStringResult(interp, &result);
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

int               SaveDiffCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;
   char          *hashkeyp, *hashkeyq;
   Databox       *databoxp, *databoxq;

   int            sd;
   FILE           *fp = NULL;

   Tcl_DString    result;

   int            filearg;
   double         abs_zero = 0.0;
   
   
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
   
   SigDiff(databoxp, databoxq, sd, abs_zero, &result, fp);

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

int               DiffEltCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{

   Data          *data = (Data *)clientData;

   Tcl_HashEntry *entryPtr;
   char          *hashkeyp, *hashkeyq;
   Databox       *databoxp, *databoxq;

   int            sd;
   int            i, j, k;
   double         diff;
   char           diff_str[64];

   double         abs_zero;
   
   
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
      sprintf(diff_str, "%e", diff);
      Tcl_AppendElement(interp, diff_str);
   }

   return TCL_OK;

}


/*-----------------------------------------------------------------------
 * routine for `pfdelete' command
 * Description: A data set hash key is given and the data set is
 *              deleted from memory.
 * Cmd. Syntax: pfdelete dataset
 *-----------------------------------------------------------------------*/

int               DeleteCommand(clientData, interp, argc, argv)
ClientData        clientData;
Tcl_Interp       *interp;
int               argc;
char             *argv[];
{
   Data          *data = (Data *)clientData;

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
