#include "rmm_wrapper.h"
#ifdef PARFLOW_HAVE_RMM

#include <pool_memory_resource.hpp>

extern "C" {
  void rmmInit() {
    rmm::mr::cuda_memory_resource cuda_mr;
    // Construct a resource that uses a coalescing best-fit pool allocator
    // With the pool initially all of available device memory
    auto initial_size = rmm::percent_of_free_device_memory(100);
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, initial_size};
    rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
  }

  void* rmmAlloc(size_t bytes) {
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
    return mr.allocate(bytes, cuda_stream_view(0));
  }

  void rmmFree(void *p, size_t bytes) {
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(); // Points to `pool_mr`
    mr.deallocate(p, bytes, cuda_stream_view(0));
  }
}

#endif
