#include "amps_umpire_wrapper.h"
#ifdef PARFLOW_HAVE_UMPIRE

#include <cstddef>
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

extern "C" {

void amps_umpireInit()
{
    auto& rm = umpire::ResourceManager::getInstance();
    std::string base_allocator_name;

#if defined(PARFLOW_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA) 
    // Native CUDA or Kokkos-enabled CUDA
    base_allocator_name = "UM";
#elif defined(KOKKOS_ENABLE_HIP) 
    // Kokkos-enabled HIP
    base_allocator_name = "DEVICE";
#else
    // CPU-only fallback
    base_allocator_name = "HOST";
#endif

    auto base_alloc = rm.getAllocator(base_allocator_name);
    auto pooled_alloc =
        rm.makeAllocator<umpire::strategy::DynamicPoolList>(
            base_allocator_name + "_pool", base_alloc);

}

void* amps_umpireAlloc(std::size_t bytes)
{
    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc;

#if defined(PARFLOW_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA) 
    alloc = rm.getAllocator("UM_pool");
#elif defined(KOKKOS_ENABLE_HIP)
    alloc = rm.getAllocator("DEVICE_pool");
#else
    alloc = rm.getAllocator("HOST_pool");
#endif

    return alloc.allocate(bytes);
}

void amps_umpireFree(void* p)
{
    auto& rm = umpire::ResourceManager::getInstance();
    umpire::Allocator alloc;

#if defined(PARFLOW_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA) 
    alloc = rm.getAllocator("UM_pool");
#elif defined(KOKKOS_ENABLE_HIP)
    alloc = rm.getAllocator("DEVICE_pool");
#else
    alloc = rm.getAllocator("HOST_pool");
#endif

    alloc.deallocate(p);
}

} // extern "C"



#endif

