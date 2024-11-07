#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <iostream>

extern "C" {
  void* create_random_tensor(int rows, int cols) {
    torch::Tensor* tensor = new torch::Tensor(torch::rand({rows, cols}));
    return static_cast<void*>(tensor);
  }

  void print_tensor(void* tensor_ptr) {
    torch::Tensor* tensor = static_cast<torch::Tensor*>(tensor_ptr);
    std::cout << *tensor << std::endl;
  }

  void free_tensor(void* tensor_ptr) {
    torch::Tensor* tensor = static_cast<torch::Tensor*>(tensor_ptr);
    delete tensor;
  }
}

#endif
