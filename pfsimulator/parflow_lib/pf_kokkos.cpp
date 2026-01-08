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
  return Kokkos::kokkos_malloc(size);
}

/**
 * @brief Free memory with Kokkos
 *
 * @param ptr Freed pointer
 */
void kokkosFree(void *ptr){
  Kokkos::kokkos_free(ptr);
}

/**
 * @brief Kokkos memcopy
 *
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Bytes to be copied
 */
void kokkosMemCpy(char *dest, char *src, size_t size){
  Kokkos::View<char*> dest_view(dest, size);
  Kokkos::View<char*> src_view(src, size);
  Kokkos::deep_copy(dest_view, src_view);
}

/**
 * @brief Kokkos memset
 *
 * @param ptr Pointer for the data to be set to zero
 * @param size Bytes to be zeroed
 */
void kokkosMemSet(char* ptr, size_t size) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;

  // Create an unmanaged view over the raw memory
  Kokkos::View<char*, Kokkos::MemoryUnmanaged> view(ptr, size);

  // Perform efficient parallel zero-initialization
  Kokkos::deep_copy(ExecSpace(), view, (char)0);

  // Synchronize only the current execution space (e.g., CUDA, HIP, etc.)
  ExecSpace().fence();
}

/**
 * @brief Initialize Kokkos
 */
void kokkosInit(){
  if(!Kokkos::is_initialized()) Kokkos::initialize();
}

/**
 * @brief Finalize Kokkos
 */
void kokkosFinalize(){
  if(Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

}
