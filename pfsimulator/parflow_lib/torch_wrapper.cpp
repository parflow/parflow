#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

extern "C" {
  double* predict_next_pressure_step(double* pp, int nx, int ny, int nz) {
    std::string model_path = "/home/ga6/saved_model.pth";
    torch::jit::script::Module model;
    try {
      model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
      std::cerr << "Error loading the model\n";
      // raise error here
    }
    std::cout << "Model loaded successfully\n";

    torch::Tensor input_tensor = torch::from_blob(pp, {nx, ny, nz}, torch::kDouble);
    std::cout << "Model input: " << input_tensor << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::Tensor output = model.forward(inputs).toTensor();
    std::cout << "Model output: " << output << std::endl;

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }
    double* c_array = output.data_ptr<double>();
    return c_array;
  }
  
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
