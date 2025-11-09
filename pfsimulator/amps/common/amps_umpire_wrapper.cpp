#include "amps_umpire_wrapper.h"

#ifdef PARFLOW_HAVE_UMPIRE

#include <cstddef>
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"

extern "C" {
// -----------------------------------------------------------------------------
// Helper function: determine the correct base allocator name based on backend.
// -----------------------------------------------------------------------------
static std::string get_base_allocator_name()
{
#if defined(PARFLOW_HAVE_CUDA) || defined(KOKKOS_ENABLE_CUDA)
  // Native CUDA or Kokkos-enabled CUDA
  return "UM";  // Unified Memory
#elif defined(KOKKOS_ENABLE_HIP)
  // Kokkos-enabled HIP (AMD)
  return "DEVICE";
#elif defined(KOKKOS_ENABLE_SYCL)
  // Kokkos-enabled SYCL (Intel/oneAPI or other SYCL devices)
  return "UM";
#elif defined(KOKKOS_ENABLE_OPENMP_TARGET)
  // OpenMP offload targets (NVIDIA/AMD/Intel GPUs)
  return "UM";
#else
  // CPU-only fallback
  return "HOST";
#endif
}

// -----------------------------------------------------------------------------
// Initialize Umpire allocators and dynamic memory pools
// -----------------------------------------------------------------------------
void amps_umpireInit()
{
  // Create a pool allocator
  // Initial size (default: 512MB) and pool increase size (1MB) can be tuned in the constructor.
  auto& rm = umpire::ResourceManager::getInstance();

  std::string base_allocator_name = get_base_allocator_name();
  auto base_alloc = rm.getAllocator(base_allocator_name);
  auto pooled_alloc =
    rm.makeAllocator<umpire::strategy::DynamicPoolList>(
                                                        base_allocator_name + "_pool", base_alloc);
}

// -----------------------------------------------------------------------------
// Allocate memory using the correct pooled allocator
// -----------------------------------------------------------------------------
void* amps_umpireAlloc(std::size_t bytes)
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string pool_name = get_base_allocator_name() + "_pool";

  auto alloc = rm.getAllocator(pool_name);

  return alloc.allocate(bytes);
}

// -----------------------------------------------------------------------------
// Free memory using the corresponding pooled allocator
// -----------------------------------------------------------------------------
void amps_umpireFree(void* p)
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string pool_name = get_base_allocator_name() + "_pool";

  auto alloc = rm.getAllocator(pool_name);

  alloc.deallocate(p);
}
} // extern "C"

#endif // PARFLOW_HAVE_UMPIRE
