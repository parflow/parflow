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
/*****************************************************************************
*
* The main routine
*
*****************************************************************************/
#include "parflow.h"
#include "pfversion.h"
#include "amps.h"
#include "fgetopt.h"  /* getopt replacement since getopt is not available on Windows */

#ifdef HAVE_SAMRAI
#include "SAMRAI/SAMRAI_config.h"

// Headers for basic SAMRAI classes
#include "SAMRAI/appu/VisItDataWriter.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/tbox/SAMRAIManager.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"
#include "SAMRAI/tbox/Utilities.h"
#include "SAMRAI/tbox/SAMRAI_MPI.h"
#include "SAMRAI/tbox/RestartManager.h"

using namespace SAMRAI;

#endif

#include "Parflow.hxx"

#if defined(PARFLOW_HAVE_CUDA) || defined(PARFLOW_HAVE_KOKKOS)
#include "pf_devices.h"
#endif

#ifdef PARFLOW_HAVE_ETRACE
#include "ptrace.h"
#endif

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv [])
{
  FILE *file = NULL;

  FILE *log_file = NULL;

  amps_Clock_t wall_clock_time;

  /*-----------------------------------------------------------------------
   * Initialize tbox::MPI and SAMRAI, enable logging, and process
   * command line.
   *-----------------------------------------------------------------------*/

#ifdef HAVE_SAMRAI
  tbox::SAMRAI_MPI::init(&argc, &argv);
  tbox::SAMRAI_MPI::setCallAbortInParallelInsteadOfMPIAbort(true);
  tbox::SAMRAIManager::initialize();
  tbox::SAMRAIManager::startup();
#endif

  // SGS FIXME remove this
  // tbox::SAMRAIManager::setMaxNumberPatchDataEntries(2048);
  {
    /*-----------------------------------------------------------------------
     * Initialize AMPS from existing MPI state initialized by SAMRAI
     *-----------------------------------------------------------------------*/
#ifdef HAVE_SAMRAI
    if (amps_EmbeddedInit())
    {
      amps_Printf("Error: amps_EmbeddedInit initalization failed\n");
      exit(1);
    }
#else
    if (amps_Init(&argc, &argv))
    {
      amps_Printf("Error: amps_Init initalization failed\n");
      exit(1);
    }
#endif

    /* Set the destination stream for PF output/logging */
    amps_SetConsole(stdout);

#ifndef NDEBUG
    /*-----------------------------------------------------------------------
     * Wait for debugger if PARFLOW_DEBUG_RANK environment variable is set
     *-----------------------------------------------------------------------*/
    if (getenv("PARFLOW_DEBUG_RANK") != NULL)
    {
      const int mpi_debug = atoi(getenv("PARFLOW_DEBUG_RANK"));
      if (mpi_debug == amps_Rank(amps_CommWorld))
      {
        volatile int i = 0;
        amps_Printf("PARFLOW_DEBUG_RANK environment variable found.\n");
        amps_Printf("Attach debugger to PID %ld (MPI rank %d) and set var i = 1 to continue\n", (long)getpid(), mpi_debug);
        while (i == 0)
        {              /*  change 'i' in the  debugger  */
        }
      }
      amps_Sync(amps_CommWorld);
    }
#endif // !NDEBUG

    /*-----------------------------------------------------------------------
     * Initialize acceleration architectures
     *-----------------------------------------------------------------------*/
#if defined(PARFLOW_HAVE_KOKKOS)
    kokkosInit();
#elif defined(PARFLOW_HAVE_CUDA)
    /*-----------------------------------------------------------------------
     * Check CUDA compute capability, set device, and initialize RMM allocator
     *-----------------------------------------------------------------------*/
    {
      // CUDA
      if (!amps_Rank(amps_CommWorld))
      {
        CUDA_ERR(cudaSetDevice(0));
      }
      else
      {
        int num_devices = 0;
        CUDA_ERR(cudaGetDeviceCount(&num_devices));
        CUDA_ERR(cudaSetDevice(amps_node_rank % num_devices));
      }

      int device;
      CUDA_ERR(cudaGetDevice(&device));

      struct cudaDeviceProp props;
      CUDA_ERR(cudaGetDeviceProperties(&props, device));

      // int value;
      // CUDA_ERR(cudaDeviceGetAttribute(&value, cudaDevAttrCanUseHostPointerForRegisteredMem, device));
      // printf("cudaDevAttrCanUseHostPointerForRegisteredMem: %d\n", value);

      if (props.major < 6)
      {
        amps_Printf("\nError: The GPU compute capability %d.%d of %s is not sufficient.\n", props.major, props.minor, props.name);
        amps_Printf("\nThe minimum required GPU compute capability is 6.0.\n");
        exit(1);
      }
    }
#endif // PARFLOW_HAVE_KOKKOS

    /*-----------------------------------------------------------------------
     * Initialize RMM pool allocator
     *-----------------------------------------------------------------------*/
#ifdef PARFLOW_HAVE_RMM
    // RMM
    rmmOptions_t rmmOptions;
    rmmOptions.allocation_mode = (rmmAllocationMode_t)(PoolAllocation | CudaManagedMemory);
    rmmOptions.initial_pool_size = 1;   // size = 0 initializes half the device memory
    rmmOptions.enable_logging = false;
    RMM_ERR(rmmInitialize(&rmmOptions));
#endif // PARFLOW_HAVE_RMM

    wall_clock_time = amps_Clock();

    /*-----------------------------------------------------------------------
     * Command line arguments
     *-----------------------------------------------------------------------*/

    char *restart_read_dirname = NULL;
    int is_from_restart = FALSE;
    int restore_num = 0;
    int c;
    char * input_name = NULL;

    opterr = 0;
    while ((c = getopt(argc, argv, "v")) != -1)
      switch (c)
      {
        case 'v':
          PrintVersionInfo(stdout);
          return 0;
          break;

        case '?':
          if (isprint(optopt))
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
          else
            fprintf(stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
          return 1;

        default:
          abort();
      }

    int non_opt_argc = argc - optind;

    if ((non_opt_argc != 1) && (non_opt_argc != 3))
    {
      fprintf(stderr, "USAGE: %s <input pfidb filename> <restart dir> <restore number>\n",
              argv[0]);
      return(-1);
    }
    else
    {
      input_name = argv[optind];

      if (non_opt_argc == 3)
      {
        restart_read_dirname = strdup(argv[optind + 1]);
        restore_num = atoi(argv[optind + 2]);

        is_from_restart = TRUE;
      }
    }

#ifdef PARFLOW_HAVE_ETRACE
    {
      char filename[2048];
      sprintf(filename, "%s.%06d.etrace", input_name, amps_Rank(MPI_CommWorld));
      init_tracefile(filename);
    }
#endif

#ifdef HAVE_SAMRAI
    /*-----------------------------------------------------------------------
     * SAMRAI initialization.
     *-----------------------------------------------------------------------*/

    /*-----------------------------------------------------------------------
     * Create input database and parse all data in input file.
     *-----------------------------------------------------------------------*/

    std::string input_filename("samrai.input");

    tbox::Dimension dim(3);

    tbox::Pointer < tbox::Database > input_db(new tbox::InputDatabase("input_db"));
    tbox::InputManager::getManager()->parseInputFile(input_filename, input_db);

    /*
     * Parse input for options to control logging, visualization and restart.
     */
    tbox::Pointer < tbox::Database > main_db = input_db->getDatabase("Main");

    std::string log_file_name = "life.log";
    if (main_db->keyExists("log_file_name"))
    {
      log_file_name = main_db->getString("log_file_name");
    }
    bool log_all_nodes = false;
    if (main_db->keyExists("log_all_nodes"))
    {
      log_all_nodes = main_db->getBool("log_all_nodes");
    }
    if (log_all_nodes)
    {
      tbox::PIO::logAllNodes(log_file_name);
    }
    else
    {
      tbox::PIO::logOnlyNodeZero(log_file_name);
    }

    int viz_dump_interval = 0;
    if (main_db->keyExists("viz_dump_interval"))
    {
      viz_dump_interval = main_db->getInteger("viz_dump_interval");
    }

    tbox::Array < std::string > viz_writer(1);
    std::string viz_dump_dirname;
    if (viz_dump_interval > 0)
    {
      if (main_db->keyExists("viz_dump_dirname"))
      {
        viz_dump_dirname = main_db->getStringWithDefault(
                                                         "viz_dump_dirname", "./visit");
      }
    }

    int restart_interval = 0;
    if (main_db->keyExists("restart_interval"))
    {
      restart_interval = main_db->getInteger("restart_interval");
    }

    std::string restart_write_dirname;
    if (restart_interval > 0)
    {
      if (main_db->keyExists("restart_write_dirname"))
      {
        restart_write_dirname = main_db->getString("restart_write_dirname");
      }
      else
      {
        TBOX_ERROR("restart_interval > 0, but key `restart_write_dirname'"
                   << " not specifed in input file");
      }
    }

    /*-----------------------------------------------------------------------
     * Initial logging info on what was run.
     *-----------------------------------------------------------------------*/
    tbox::plog << "input_filename = " << input_filename << std::endl;
    tbox::plog << "restart_read_dirname = " << restart_read_dirname << std::endl;
    tbox::plog << "restore_num = " << restore_num << std::endl;

    /*-----------------------------------------------------------------------
     * If run is from restart then open the restart file.
     *-----------------------------------------------------------------------*/
    tbox::RestartManager* restart_manager = tbox::RestartManager::getManager();

    if (is_from_restart)
    {
      std::string restart_dir(restart_read_dirname);
      restart_manager->
      openRestartFile(restart_dir, restore_num,
                      amps_Size());
    }
#else
    PF_UNUSED(restore_num);
    PF_UNUSED(is_from_restart);
    PF_UNUSED(restart_read_dirname);
#endif

    /*-----------------------------------------------------------------------
     * Set up globals structure
     *-----------------------------------------------------------------------*/

    NewGlobals(input_name);

#ifdef HAVE_SAMRAI
    GlobalsParflowSimulation = new Parflow("Parflow",
                                           input_db->getDatabase("Parflow"));
#endif

    /*-----------------------------------------------------------------------
     * Read the Users Input Deck
     *-----------------------------------------------------------------------*/

    amps_ThreadLocal(input_database) = IDB_NewDB(GlobalsInFileName);

    /*-----------------------------------------------------------------------
     * Setup log printing
     *-----------------------------------------------------------------------*/

    NewLogging();

    /*-----------------------------------------------------------------------
     * Setup timing table
     *-----------------------------------------------------------------------*/

    NewTiming();

    /*-----------------------------------------------------------------------
     * Solve the problem
     *-----------------------------------------------------------------------*/
    Solve();

    if (!amps_Rank(amps_CommWorld))
    {
      amps_Printf("Problem solved \n");
      fflush(NULL);
    }

    /*-----------------------------------------------------------------------
     * Log global information
     *-----------------------------------------------------------------------*/

    LogGlobals();

    /*-----------------------------------------------------------------------
     * Print timing results
     *-----------------------------------------------------------------------*/

    PrintTiming();

    /*-----------------------------------------------------------------------
     * Clean up
     *-----------------------------------------------------------------------*/

    FreeLogging();

    FreeTiming();

    /*-----------------------------------------------------------------------
     * Finalize AMPS and exit
     *-----------------------------------------------------------------------*/

    wall_clock_time = amps_Clock() - wall_clock_time;

    IfLogging(0)
    {
      if (!amps_Rank(amps_CommWorld))
      {
        log_file = OpenLogFile("ParFlow Total Time");

        fprintf(log_file, "Total Run Time: %f seconds\n\n",
                (double)wall_clock_time / (double)AMPS_TICKS_PER_SEC);


        {
          char filename[2048];
          sprintf(filename, "%s.timing.csv", GlobalsOutFileName);

          if ((file = fopen(filename, "a")) == NULL)
          {
            InputError("Error: can't open output file %s%s\n", filename, "");
          }

          fprintf(file, "%s,%f,%s,%s\n", "Total Runtime",
                  (double)wall_clock_time / (double)AMPS_TICKS_PER_SEC,
                  "-nan", "0");
        }

        fclose(file);
      }
    }

    printMaxMemory(log_file);

    IfLogging(0)
    {
      fprintf(log_file, "\n");

      if (!amps_Rank(amps_CommWorld))
      {
        printMemoryInfo(log_file);
        fprintf(log_file, "\n");

        CloseLogFile(log_file);
      }
    }

    if (!amps_Rank(amps_CommWorld))
    {
      char filename[2048];
      sprintf(filename, "%s.pftcl", GlobalsOutFileName);

      file = fopen(filename, "w");

      IDB_PrintUsage(file, amps_ThreadLocal(input_database));

      fclose(file);
    }

    IDB_FreeDB(amps_ThreadLocal(input_database));

    FreeGlobals();

    /*-----------------------------------------------------------------------
     * Shutdown AMPS
     *-----------------------------------------------------------------------*/
    amps_Finalize();
  }

  /*-----------------------------------------------------------------------
   * Shutdown SAMRAI and MPI.
   *-----------------------------------------------------------------------*/
#ifdef HAVE_SAMRAI
  tbox::SAMRAIManager::shutdown();
  tbox::SAMRAIManager::finalize();
  tbox::SAMRAI_MPI::finalize();
#endif

  /*-----------------------------------------------------------------------
   * Shutdown Kokkos
   *-----------------------------------------------------------------------*/
#ifdef PARFLOW_HAVE_KOKKOS
  kokkosFinalize();
#endif

  /*-----------------------------------------------------------------------
   * Shutdown RMM pool allocator
   *-----------------------------------------------------------------------*/
#ifdef PARFLOW_HAVE_RMM
  RMM_ERR(rmmFinalize());
#endif

  return 0;
}
