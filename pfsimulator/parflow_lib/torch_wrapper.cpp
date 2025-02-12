#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

static torch::jit::script::Module model;

extern "C" {
  void init_torch_model(char* model_filepath) {
    std::string model_path = std::string(model_filepath);
    try {
      model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
      std::cerr << "Error loading the model\n";
      // raise error here
    }

    // also call scale statics and store the result in a global variable.
  }
  
  double* predict_next_pressure_step(double* pp, int nx, int ny, int nz) {

    torch::Tensor input_tensor = torch::from_blob(pp, {nx, ny, nz}, torch::kDouble);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    at::Tensor output = model.forward(inputs).toTensor();

    if (!output.is_contiguous()) {
        output = output.contiguous();
    }
    double* predicted_pressure = output.data_ptr<double>();

    // Copy pressure data back to the pressure field
    if (predicted_pressure != pp) {
      std::size_t sz = nx * ny * nz;
      std::copy(predicted_pressure, predicted_pressure + sz, pp);
    }
    return pp;
  }
}

#endif
