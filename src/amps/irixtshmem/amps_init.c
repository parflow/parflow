/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <stdio.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/unistd.h>
#include <sys/times.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/prctl.h>

#include "amps.h"


/* SGS Why are these here ? */
char *getenv(char *);
extern char *optarg;
extern int optind;


long AMPS_CPU_TICKS_PER_SEC;

int amps_size;

amps_Comm amps_CommWorld = 21;

amps_Buffer **amps_PtrBufferList = NULL;
amps_Buffer **amps_PtrBufferListEnd = NULL;
amps_Buffer **amps_PtrBufferFreeList = NULL;
amps_Invoice *amps_PtrSyncInvoice = NULL;

int amps_syncinvoice_value;

char amps_malloclog[MAXPATHLEN];

char *amps_arena_name;
usptr_t *amps_arena;
AMPS_ShMemInfo *amps_shmem_info;

#undef main

void amps_main(arg)
void *arg;
{
   char *env_value;

   char **argv = arg;
   int argc;

   (PRDA -> usr_prda.fill[0]) = (int)argv[0];

   argc = (int)argv[1];

#if 0

   SGS THis is not working for some reason, can we do this here?

   if ( (env_value = getenv("AMPS_SIGNAL_DEBUG")) )
   {
      printf( "New signal handlers being installed.\n" );
      Fsignal( );
      free( env_value );
   }
#endif

   amps_SyncInvoice = amps_NewInvoice("%d", &amps_syncinvoice_value);

   amps_Sync(amps_CommWorld);

   /* Invoke users main here */
   AMPS_USERS_MAIN(argc, &argv[2]);
   return;
}


int main(argc, argv)
int argc;
char *argv[];
{
   int i, j;

   char *letter;
   char *env_value;
   char **pass_argv;

   char filename[MAXPATHLEN];
   char numprocs[10];
   char proc_num[10];

   pid_t  pid;

   char *env_string;
   int amps_shared_mem_size;

   int status;

#if 0
      _utrace=1;
      _uerror=1;
#endif

   if( getopt(argc, argv, "n:" ) == -1 )
   {
      printf("AMPS ERROR: can't determine number for nodes\n");
      exit(1);
   }
   else
   {

      /* Set up some global (all threads) items */
      AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
      amps_size = atoi( optarg );

      if ( (amps_PtrBufferList = calloc(amps_size,sizeof(amps_Buffer *))) == NULL)
      {
	 amps_Printf("AMPS Error: malloc of BufferList array failed\n");
	 exit(1);
      }

      if ( (amps_PtrBufferListEnd = calloc(amps_size, sizeof(amps_Buffer *) )) == NULL)
      {
	 amps_Printf("AMPS Error: malloc of BufferListEnd array failed\n");
	 exit(1);
      }

      if ( (amps_PtrBufferFreeList = calloc(amps_size, sizeof(amps_Buffer *) )) == NULL)
      {
	 amps_Printf("AMPS Error: malloc of BufferFreeList array failed\n");
	 exit(1);
      }


      if ( (amps_PtrSyncInvoice = calloc(amps_size, sizeof(amps_Invoice) )) == NULL)
      {
	 amps_Printf("AMPS Error: malloc of amps_SyncInvoice array failed\n");
	 exit(1);
      }

      (PRDA -> usr_prda.fill[0]) = 0;
   
      /*====================================================================*/
      /* Set up the shared memory arena                                     */
      /*====================================================================*/
      
      if( (env_string = getenv("AMPS_SHMEM_FILEDIR")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_FILEDIR\n");
	 exit(1);
      }

      amps_arena_name = tempnam( env_string, "amps");
      
      if( (env_string = getenv("AMPS_SHMEM_SIZE")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_SIZE\n");
	 exit(1);
      }
      amps_shared_mem_size = atoi(env_string);
      
      
      usconfig(CONF_INITUSERS, amps_size+1);
      
      usconfig(CONF_INITSIZE, amps_shared_mem_size);
      
      if( (amps_arena = usinit(amps_arena_name))== NULL)
      {
	 printf("AMPS Error: Can't open shared memory arena\n");
	 exit(1);
      }
      
      if( (amps_shmem_info = (AMPS_ShMemInfo *)uscalloc(
							sizeof(AMPS_ShMemInfo), 1, amps_arena)) == NULL)
      {
	 printf("AMPS Error: Failed to allocate shared memory info block\n");
	 exit(1);
      }
      
      if( (amps_shmem_info -> sync = new_barrier(amps_arena)) == NULL)
      {
	 printf("AMPS Error: Failed to allocate barrier\n");
	 exit(1);
      }
      
      if( (amps_shmem_info -> bcast_lock = usnewlock(amps_arena)) == NULL)
      {
	 printf("AMPS Error: Failed to allocate barrier\n");
	 exit(1);
      }
      
      /* Set up global arrays */
      
      for(i = 0; i < amps_size; i++)
      {
	 if( (amps_shmem_info -> locks[i] = usnewlock(amps_arena)) == NULL)
	 {
	    printf("AMPS Error: Failed to allocate lock\n");
	    exit(1);
	 }
	 
	 if( (amps_shmem_info -> sema[i] = usnewsema(amps_arena,0) ) 
	    == NULL)
	 {
	    printf("AMPS Error: Failed to allocate send/recv sema\n");
	    exit(1);
	 }
	 
	 
	 if( (amps_shmem_info -> bcast_locks[i] = usnewlock(amps_arena)) == NULL)
	 {
	    printf("AMPS Error: Failed to allocate lock\n");
	    exit(1);
	 }
	 
	 if( (amps_shmem_info -> bcast_sema[i] = usnewsema(amps_arena,0) ) 
	    == NULL)
	 {
	    printf("AMPS Error: Failed to allocate bcast sema\n");
	    exit(1);
	 }
	 
      }
      
      /* put the structure so other process can get it */
      usputinfo(amps_arena, amps_shmem_info);
      

      /* initialize the clock */
      amps_clock_init();

      /* Start up the nodes */
      for(i=0; i<amps_size; i++)
      {
	 pass_argv = (char **)malloc((argc)*sizeof(char *));

	 pass_argv[0] = (char*)i;      
	 pass_argv[1] = (char*)argc-2;
	 pass_argv[2] = argv[0];

	 for(j = 3; j < argc; j++)
	    pass_argv[j] = argv[j];

	 if( (pid = sproc(amps_main, PR_SALL,pass_argv) < 0 ))
	 {
	    printf("AMPS_ERROR: sproc Error\n");
	    exit(1);
	 }
      }

      /* Wait for all pid's here SGS */

      for(i = 0; i < amps_size; i++)
	 wait(&status);

      for(i = 0; i < amps_size; i++)
      {
	 usfreelock( amps_shmem_info -> locks[i], amps_arena);
	 usfreesema( amps_shmem_info -> sema[i], amps_arena);
	 usfreelock( amps_shmem_info -> bcast_locks[i], amps_arena);
	 usfreesema( amps_shmem_info -> bcast_sema[i], amps_arena);
      }
       
      usfreelock(amps_shmem_info -> bcast_lock, amps_arena);
      free_barrier(amps_shmem_info -> sync);
      usfree(amps_shmem_info, amps_arena);

#ifdef AMPS_MALLOC_DEBUG
      amps_memcheck();
      /* check out the heap and shut everything down if we are in debug mode */
#endif
	 
      usdetach(amps_arena);
      unlink(amps_arena_name);
      free(amps_arena_name);
   }

   return 0;
}

int amps_Init(argc, argv)
int *argc;
char **argv[];
{
   return 0;
}

void *_amps_CTAlloc(int count, char *filename, int line)
{
   void *ptr;

   if(count)
      if((ptr = uscalloc(count, 1, amps_arena)) == NULL )
      {
	 amps_Printf("Error: out of memory in <%s> at line %d\n",
		     filename, line);
	 exit(1);
      }
      else
	 return ptr;
   else
      return NULL;
}

void *_amps_TAlloc(int count, char *filename, int line)
{
   void *ptr;

   if(count)
      if((ptr = usmalloc(count, amps_arena)) == NULL )
      {
	 amps_Printf("Error: out of memory in <%s> at line %d\n",
		     filename, line);
	 exit(1);
      }
      else
	 return ptr;
   else
      return NULL;
}
