#include "amps_umpire_wrapper.h"
#ifdef PARFLOW_HAVE_UMPIRE

#include <cstddef>
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

extern "C" {
  void amps_umpireInit() {
    // Create a pool allocator
    // Initial size (default: 512MB) and pool increase size (1MB) can be tuned in the constructor.
    auto& rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.getAllocator("UM");
    auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>("UM_pool", allocator);    
  }

  void* amps_umpireAlloc(std::size_t bytes) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.getAllocator("UM_pool");
    return allocator.allocate(bytes);
  }

  void amps_umpireFree(void *p) {
    auto& rm = umpire::ResourceManager::getInstance();
    auto allocator = rm.getAllocator("UM_pool");
    allocator.deallocate(p);
  }
}

#endif
