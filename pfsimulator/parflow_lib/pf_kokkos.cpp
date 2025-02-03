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
void kokkosMemSet(char *ptr, size_t size){
  /* Loop style initialization */
  if(size % sizeof(int))
  {
    /* Writing 1 byte / thread is slow */
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(int i){ptr[i] = 0;});
  }
  else
  {
    /* Writing 4 bytes / thread is fast */
    Kokkos::parallel_for(size / sizeof(int), KOKKOS_LAMBDA(int i){((int*)ptr)[i] = 0;});
  }
  Kokkos::fence(); 

  /* Deep_copy style initialization for char* should be fast in future Kokkos releases */
  // Kokkos::View<char*> ptr_view(ptr, size);
  // Kokkos::deep_copy(ptr_view, 0);
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
