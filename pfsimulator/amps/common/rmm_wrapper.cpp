#include "rmm_wrapper.h"
#ifdef PARFLOW_HAVE_RMM

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

static rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>* pool_mr = nullptr;

extern "C" {
  void rmmInit() {
    rmm::mr::managed_memory_resource cuda_mr;
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially all of available device memory
    auto initial_size = rmm::percent_of_free_device_memory(100);
    pool_mr = new rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>(&cuda_mr, initial_size);
    rmm::mr::set_current_device_resource(pool_mr); // Updates the current device resource pointer to `pool_mr`
  }

  void* rmmAlloc(size_t bytes) {
    return pool_mr->allocate(bytes);
  }

  void rmmFree(void *p) {
    pool_mr->deallocate(p, 0);
  }
}

#endif
