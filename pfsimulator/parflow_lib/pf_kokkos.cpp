#include "amps.h"
#include <Kokkos_Core.hpp>

extern "C"{
#include "pf_devices.h"

/**
 * @brief Allocate memory with Kokkos
 *
 * @param size The size of the allocation in bytes
 */
void* kokkosAlloc(size_t size){
#ifdef __CUDACC__
  return Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(size);
#else
  return Kokkos::kokkos_malloc<Kokkos::HostSpace>(size);
#endif
}

/**
 * @brief Free memory with Kokkos
 *
 * @param ptr Freed pointer
 */
void kokkosFree(void *ptr){
#ifdef __CUDACC__
  Kokkos::kokkos_free<Kokkos::CudaUVMSpace>(ptr);
#else
  Kokkos::kokkos_free<Kokkos::HostSpace>(ptr);
#endif
}

/**
 * @brief Kokkos memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpy(char *dest, char *src, size_t size){
#ifdef __CUDACC__
  Kokkos::View<char*, Kokkos::CudaUVMSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaUVMSpace> src_view(src, size);
#else
  Kokkos::View<char*, Kokkos::HostSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::HostSpace> src_view(src, size);
#endif
  Kokkos::deep_copy(dest_view, src_view);
}

/**
 * @brief Kokkos memset
 *
 * @param ptr Pointer for the data to be set to zero
 * @param size Bytes to be zeroed
 */
void kokkosMemSet(char *ptr, size_t size){
  /* Deep-copy style initialization (may become faster in the future) */
  Kokkos::View<char*, Kokkos::CudaUVMSpace> ptr_view(ptr, size);
  Kokkos::deep_copy(ptr_view, 0);
  /* Loop style initialization */
  // Kokkos::parallel_for(size, KOKKOS_LAMBDA(int i){ptr[i] = 0;});
  // Kokkos::fence(); 
}

/**
 * @brief Initialize Kokkos
 */
void kokkosInit(){
  Kokkos::InitArguments args;
#ifdef __CUDACC__
  args.device_id = amps_node_rank % Kokkos::Cuda::detect_device_count();
  args.ndevices = 1;
#endif
  Kokkos::initialize(args);    
}

/**
 * @brief Finalize Kokkos
 */
void kokkosFinalize(){
  if(Kokkos::is_initialized) Kokkos::finalize();
}

}
