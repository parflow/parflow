#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

static torch::jit::script::Module model;


extern "C" {
  void init_torch_model(char* model_filepath, int nx, int ny, int nz, double *po_dat,
			double *mann_dat, double *slopex_dat, double *slopey_dat, double *permx_dat,
			double *permy_dat, double *permz_dat, double *sres_dat, double *ssat_dat,
			double *fbz_dat, double *specific_storage_dat, double *alpha_dat, double *n_dat) {                           
    std::string model_path = std::string(model_filepath);
    try {
      model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
      throw std::runtime_error(std::string("Failed to load the Torch model:\n") + e.what());
    }

    std::unordered_map<std::string, torch::Tensor> statics;
    torch::Tensor porosity = torch::from_blob(po_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["porosity"] = porosity;
    torch::Tensor mannings = torch::from_blob(mann_dat, {nx, ny}, torch::kDouble).clone();
    statics["mannings"] = mannings;
    torch::Tensor slope_x = torch::from_blob(slopex_dat, {nx, ny}, torch::kDouble).clone();
    statics["slope_x"] = slope_x;
    torch::Tensor slope_y = torch::from_blob(slopey_dat, {nx, ny}, torch::kDouble).clone();
    statics["slope_y"] = slope_y;
    torch::Tensor perm_x = torch::from_blob(permx_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["permeability_x"] = perm_x;
    torch::Tensor perm_y = torch::from_blob(permy_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["permeability_y"] = perm_y;
    torch::Tensor perm_z = torch::from_blob(permz_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["permeability_z"] = perm_z;
    torch::Tensor sres = torch::from_blob(sres_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["sres"] = sres;
    torch::Tensor ssat = torch::from_blob(ssat_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["ssat"] = ssat;
    torch::Tensor fbz = torch::from_blob(fbz_dat, {nx, ny, nz}, torch::kDouble).clone();
    statics["pf_flowbarrier"] = fbz;
    std::cout << ">>>>>>>>>>>>> FBz: " << fbz << std::endl;
    // also call scale statics and store the result in a global variable.
  }
  
  double* predict_next_pressure_step(double* pp, double* et, int nx, int ny, int nz) {
    torch::Tensor press = torch::from_blob(pp, {nx, ny, nz}, torch::kDouble);
    torch::Tensor evap_trans = torch::from_blob(et, {nx, ny}, torch::kDouble);    

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(press);

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
