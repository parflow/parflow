#include "amps_umpire_wrapper.h"

#ifdef PARFLOW_HAVE_UMPIRE

#include <cstddef>
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

extern "C" {
// -----------------------------------------------------------------------------
// Helper: Select correct Umpire memory resource based on active backend
// -----------------------------------------------------------------------------
static std::string get_resource_name()
{
#if defined(PARFLOW_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA)
  return "UM";
#elif defined(KOKKOS_ENABLE_HIP)
  return "DEVICE";
#elif defined(KOKKOS_ENABLE_SYCL)
  return "UM";
#elif defined(KOKKOS_ENABLE_OPENMP_TARGET)
  return "UM";
#else  // CPU-only fallback
  return "HOST";
#endif
}

// -----------------------------------------------------------------------------
// Helper: Name of the pooled allocator constructed from the resource
// -----------------------------------------------------------------------------
static std::string get_pool_name()
{
  return get_resource_name() + "_pool";
}

// -----------------------------------------------------------------------------
// Initialize Umpire allocators and pool strategy.
// -----------------------------------------------------------------------------
void amps_umpireInit()
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource_name = get_resource_name();
  auto resource_alloc = rm.getAllocator(resource_name);

  // Construct a DynamicPoolList allocator with tunable sizes:
  // - Initial pool size: 512 MB
  // - Growth size: 1 MB
  //
  // Other parameters use defaults (but can also be tuned in the constructor):
  //   * Allocation alignment: 16 bytes
  //   * Coalescing heuristic: 100% releasable
  //     (automatically coalesces when all bytes in the pool are releasable
  //      and there is more than one block)
  auto pooled_alloc =
    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                                                        get_pool_name(),
                                                        resource_alloc,
                                                        512 * 1024 * 1024, // initial size: 512 MB
                                                        1 * 1024 * 1024 // growth size: 1 MB
                                                        );
}

// -----------------------------------------------------------------------------
// Alloc/Free from pool
// -----------------------------------------------------------------------------
void* amps_umpireAlloc(std::size_t bytes)
{
  auto alloc = umpire::ResourceManager::getInstance().getAllocator(get_pool_name());

  return alloc.allocate(bytes);
}

void amps_umpireFree(void* p)
{
  auto alloc = umpire::ResourceManager::getInstance().getAllocator(get_pool_name());

  alloc.deallocate(p);
}
} // extern "C"

#endif // PARFLOW_HAVE_UMPIRE