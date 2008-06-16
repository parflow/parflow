/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

#include <strings.h>
#include <sys/param.h>
#include <sys/unistd.h>
#include <sys/times.h>
#include <sys/wait.h>
#include <sys/types.h>

#ifdef AMPS_SYSMP
#include <sys/sysmp.h>
#endif

long AMPS_CPU_TICKS_PER_SEC;

char *getenv(char *);
extern char *optarg;
extern int optind;

int amps_tid;
int amps_rank;
int amps_size;

amps_Comm amps_CommWorld = 21;

amps_Buffer *amps_BufferList = NULL;
amps_Buffer *amps_BufferListEnd = NULL;
amps_Buffer *amps_BufferFreeList = NULL;
amps_Invoice amps_SyncInvoice = NULL;

int amps_syncinvoice_value;

char amps_malloclog[MAXPATHLEN];

char amps_arena_name[MAXPATHLEN];
usptr_t *amps_arena;
AMPS_ShMemInfo *amps_shmem_info;

int amps_Init(argc, argv)
int *argc;
char **argv[];
{
   int i;


   char *letter;
   char *env_value;

   /*========================================================================*/
   /* host specific                                                          */
   /*========================================================================*/

   char filename[MAXPATHLEN];
   char numprocs[10];
   char proc_num[10];

   pid_t  pid;

   char *env_string;
   int amps_shared_mem_size;

#if 0
      _utrace=1;
      _uerror=1;
#endif

   
   if( getopt(*argc, *argv, "n:" ) == -1 )
   {
      amps_size = atoi((*argv)[1]);
      amps_rank = atoi((*argv)[2]);

#ifdef AMPS_SYSMP
     sysmp(MP_MUSTRUN, amps_rank+1);
#endif

      if( (env_string = getenv("AMPS_SHMEM_FILEDIR")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_FILEDIR\n");
	 exit(1);
      }
      sprintf(amps_arena_name, "%s/amps_shmem_%d", env_string, getuid());

      if( (env_string = getenv("AMPS_SHMEM_SIZE")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_SIZE\n");
	 exit(1);
      }
      amps_shared_mem_size = atoi(env_string);

      
      usconfig(CONF_INITUSERS, amps_size);
   
      usconfig(CONF_INITSIZE, amps_shared_mem_size);
      
      if( (amps_arena = usinit(amps_arena_name))== NULL)
      {
	 printf("AMPS Error: Can't open shared memory arena %s\n",
		amps_arena_name);
	 exit(1);
      }
      
      if( (amps_shmem_info = usgetinfo(amps_arena)) == NULL)
      {
	 printf("AMPS Error: Failed to get access to shared memory info");
	 exit(1);
      }
      
   }
   else
   {
      amps_size = atoi( optarg );

      amps_rank = 0;

      /*====================================================================*/
      /* Set up the shared memory arena                                     */
      /*====================================================================*/

      if( (env_string = getenv("AMPS_SHMEM_FILEDIR")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_FILEDIR\n");
	 exit(1);
      }
      sprintf(amps_arena_name, "%s/amps_shmem_%d", env_string, getuid());

      if( (env_string = getenv("AMPS_SHMEM_SIZE")) == NULL)
      {
	 amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_SIZE\n");
	 exit(1);
      }
      amps_shared_mem_size = atoi(env_string);


      usconfig(CONF_INITUSERS, amps_size);
      
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
      
      /* Start up the nodes */
      
      sprintf(numprocs, "%i", amps_size);
      (*argv)[1] = numprocs;

#ifdef AMPS_SYSMP
     sysmp(MP_MUSTRUN, amps_rank+1);
#endif
      
      for(i=amps_size-1; i; i--)
      {
	 sprintf(proc_num, "%i", i);
	 (*argv)[2] = proc_num;

	 if( (pid = fork()) < 0 )
	 {
	    printf("Fork Error\n");
	    exit(1);
	 }
	 else if (pid == 0)
	 {
	    if( execvp((*argv[0]), *argv) < 0 )
	       printf("Exec Error %i\n", i-1);
	 }
      }
   }

#ifdef AMPS_MALLOC_DEBUG
   malloc_logpath = amps_malloclog;
   sprintf(malloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif


   if ( (env_value = getenv("AMPS_SIGNAL_DEBUG")) )
   {
      printf( "New signal handlers being installed.\n" );
      Fsignal( );
      free( env_value );
   }

   /* Warning:                              */
   /* Hacking in progress                   */
   /* Possibly loosing a little memory here */
   /* This could potentially cause problems */
   (*argv)[2] = (*argv)[0];
   *argv += 2;
   *argc -= 2;
      
   AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
   
   amps_SyncInvoice = amps_NewInvoice("%d", &amps_syncinvoice_value);

   amps_Sync(amps_CommWorld);

   amps_clock_init();

   /* unlink the arena since everyone has attached by this point */
   if(!amps_rank)
      unlink(amps_arena_name);

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

