#include "amps.h"

#ifdef PARFLOW_HAVE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

extern "C"{
#include "pf_cudamain.h"

void init_acc_arch(){
#if PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA

#ifndef NDEBUG
    /*-----------------------------------------------------------------------
    * Wait for debugger if PARFLOW_DEBUG_RANK environment variable is set
    *-----------------------------------------------------------------------*/
    if(getenv("PARFLOW_DEBUG_RANK") != NULL) {
      const int mpi_debug = atoi(getenv("PARFLOW_DEBUG_RANK"));
      if(mpi_debug == amps_Rank(amps_CommWorld)){
        volatile int i = 0;
        printf("PARFLOW_DEBUG_RANK environment variable found.\n");
        printf("Attach debugger to PID %ld (MPI rank %d) and set var i = 1 to continue\n", (long)getpid(), mpi_debug);
        while(i == 0) {/*  change 'i' in the  debugger  */}
      }
      amps_Sync(amps_CommWorld);
    }
#endif // !NDEBUG

    /*-----------------------------------------------------------------------
    * Check CUDA compute capability, set device, and initialize RMM allocator
    *-----------------------------------------------------------------------*/
    {
      // CUDA
      if (!amps_Rank(amps_CommWorld))
      {
        CUDA_ERR(cudaSetDevice(0));  
      }else{
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
        printf("\nError: The GPU compute capability %d.%d of %s is not sufficient.\n",props.major,props.minor,props.name);
        printf("\nThe minimum required GPU compute capability is 6.0.\n");
        exit(1);
      }

#ifdef PARFLOW_HAVE_KOKKOS
      Kokkos::InitArguments args;
      args.ndevices = 1;
      Kokkos::initialize(args);    
#endif // PARFLOW_HAVE_KOKKOS
    }
#endif // PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA
}
}
