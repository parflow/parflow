#include "amps_rmm_wrapper.h"
#ifdef PARFLOW_HAVE_RMM

/* Notes on the transition from the RMM C API to the C++ API:
 *
 * The RMM C++ API does not support the old rmmAlloc() and rmmFree() functions.
 * Instead, they have implemented resource classes for the  different memory allocation
 * algorithms. One consequence is that the deallocate() method of the pooled memory 
 * allocator needs to know how many bytes to free. The RMM team encourages the use of
 * containers in memory allocation in order to achieve that (such as the RMM-provided
 * device_buffer).
 *
 * This is a good strategy but unfortunately, the adaptor layer of ParFlow makes this
 * nearly impossible to implement. The scientific code is agnostic to the backend that
 * ParFlow uses to run and makes alloc() and free() calls expecting and providing void
 * pointers to the backend. If we started to use containers for memory allocation, we
 * would have to either distinguish memory requests based on the backend, or modify all
 * backends to use containers and modify the scientific code as well to work with those
 * containers.
 *
 * The solution below creates those containers implicitly, by asking for PADDING more data
 * and storing the size of the allocation in the beginning of the allocated block. PADDING
 * is chosen so that memory alignment is not affected (CUDA memory allocation routines
 * always return an address aligned to at least 256 bytes). For a ParFlow simulation, the
 * amount of GPU memory used for padding is just a rounding error.
 * 
 */

#define PADDING 256

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

static rmm::mr::managed_memory_resource cuda_mr;
static rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>* pool_mr = nullptr;

extern "C" {
  void amps_rmmInit() {
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially all of available device memory
    auto initial_size = rmm::percent_of_free_device_memory(100);
    pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>(&cuda_mr, initial_size);
    rmm::mr::set_current_device_resource(pool_mr); // Updates the current device resource pointer to `pool_mr`
  }

  void* amps_rmmAlloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    size_t total_bytes = bytes + PADDING;
    if (total_bytes < bytes) {
      throw std::runtime_error("Unsigned overflow, cannot allocate memory.");
    }
    void* container = pool_mr->allocate(total_bytes);
    void* data = (void*)((char*)container + PADDING);
    *(size_t*)container = total_bytes;
    return data;
  }

  void amps_rmmFree(void *data) {
    if (!data) return;
    void* container = (void*)((char*)data - PADDING);
    size_t total_bytes = *(size_t*)container;
    pool_mr->deallocate(container, total_bytes);
  }

  void amps_rmmFinalize() {
    delete pool_mr;
  }
}

#endif
