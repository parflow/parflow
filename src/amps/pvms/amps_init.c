/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <stdio.h>
#include <sys/param.h>
#include <sys/param.h>
#include <sys/times.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "amps.h"

int amps_tid;
int *amps_tids;
int amps_rank;
int amps_size;

char amps_malloclog[MAXPATHLEN];

amps_Comm amps_CommWorld = "amps_CommWorld";

long AMPS_CPU_TICKS_PER_SEC;

extern char *optarg;
extern int optind;

int amps_Init(argc, argv)
int *argc;
char **argv[];
{
    int path_length;
    
    char amps_path[MAXPATHLEN];
    char *temp_path;
    
    char *progname;

    int i;

    amps_tid = pvm_mytid();

    if(pvm_parent() == PvmNoParent)
    {

       if( getopt(*argc, *argv, "n:" ) == -1 )
       {
	  printf("Please specify number of nodes with -n #\n");
	  exit(1);
       }
       else
       {
	  amps_size = atoi( optarg );
	  printf("Running on %d \n", amps_size);
       }
       
       if(amps_size > 1)
       {
	  if( !(amps_tids = (int *)malloc(sizeof(int)*amps_size)) )
	  {
	     printf("AMPS Error: can't malloc memory for tids array\n");
	     exit(1);
	  }
	  
	  if( (temp_path = getenv("AMPS_EXE_DIR")) == NULL)
	  {
	     printf("AMPS Error: can't get AMPS_EXE_DIR envirnment variabl\n");
	     exit(1);
	  }
	  
	  /* PVM gets path from PVM_ROOT */
	  if((progname = strrchr( (*argv)[0], '/')))
	     progname++;
	  else
	      progname = (*argv)[0];
	   
	  if ( pvm_spawn(progname, &(*argv)[3], 0 & PvmTaskDebug, 
			 "", amps_size-1, &amps_tids[1]) !=
	      (amps_size -1))
	  {
	     printf("AMPS Error: Spawn failed\n");
	     exit(1);
	  }

	   amps_rank = pvm_joingroup(amps_CommWorld);
	   
	   pvm_initsend( PvmDataDefault );
#if SGS_PRINT
	   printf("SGS: allocating in init %d\n", bufid);
#endif

	   pvm_pkint(&amps_size, 1, 1);
	   path_length = strlen(temp_path) + 1;
	   pvm_pkint(&path_length, 1, 1);
	   pvm_pkbyte(temp_path, path_length, 1);
	   pvm_mcast(&amps_tids[1], amps_size-1, amps_MsgTag);
	}

       /* Warning:                              */
       /* Hacking in progress                   */
       /* Possibly loosing a little memory here */
       /* This could potentially cause problems */
       (*argv)[2] = (*argv)[0];
       *argv += 2;
       *argc -= 2;

    }
    else
    {
       pvm_recv(pvm_parent(), amps_MsgTag);
       pvm_upkint( &amps_size, 1, 1);
       pvm_upkint( &path_length, 1, 1);
       pvm_upkbyte(amps_path, path_length, 1);

       if(chdir(amps_path))
	  printf("AMPS Error: can't set working directory to %s", amps_path);
	
       amps_rank = pvm_joingroup(amps_CommWorld);

       if( !(amps_tids = (int *)malloc(sizeof(int)*amps_size)) )
       {
	  printf("AMPS Error: can't malloc memory for tids array\n");
	  exit(1);
       }
    }

    if(amps_size > 1)
    {    
       pvm_barrier(amps_CommWorld, amps_size);

       for(i=0; i< amps_size; i++)
	  amps_tids[i] = pvm_gettid(amps_CommWorld, i);
    }

#ifdef AMPS_BSD_TIME
    amps_clock_init();
#endif

#ifdef AMPS_MALLOC_DEBUG
    malloc_logpath = amps_malloclog;
    sprintf(malloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif

#ifdef TIMING
   AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
#endif

    return 0;
}  


