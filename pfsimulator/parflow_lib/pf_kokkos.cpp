#include "amps.h"
#include <Kokkos_Core.hpp>

extern "C"{
#include "pf_devices.h"

void* kokkosAlloc(size_t size){
  return Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(size);
}

void kokkosFree(void *ptr){
  Kokkos::kokkos_free<Kokkos::CudaUVMSpace>(ptr);
}

void kokkosMemCpy(char *dest, char *src, size_t size){
  Kokkos::View<char*, Kokkos::CudaUVMSpace> dest_view(dest, size);
  Kokkos::View<char*, Kokkos::CudaUVMSpace> src_view(src, size);
  Kokkos::deep_copy(dest_view, src_view);
}

void kokkosInit(){
  Kokkos::InitArguments args;
  args.device_id = amps_node_rank % Kokkos::Cuda::detect_device_count();
  args.ndevices = 1;
  Kokkos::initialize(args);    
}

void kokkosFinalize(){
  if(Kokkos::is_initialized) Kokkos::finalize();
}

}
