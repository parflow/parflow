#include "amps_rmm_wrapper.h"
#ifdef PARFLOW_HAVE_RMM

#define PADDING 128

#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

static rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>* pool_mr = nullptr;

extern "C" {
  void amps_rmmInit() {
    rmm::mr::managed_memory_resource cuda_mr;
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
}

#endif
