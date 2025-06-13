#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

using namespace torch::indexing;

static torch::jit::script::Module model;
static torch::Tensor statics;

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

    // Get the true fields without the ghost nodes
    std::unordered_map<std::string, torch::Tensor> statics_map;
    torch::Tensor porosity = torch::from_blob(po_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["porosity"] = porosity;
    torch::Tensor mannings = torch::from_blob(mann_dat, {3, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["mannings"] = mannings;
    torch::Tensor slope_x = torch::from_blob(slopex_dat, {3, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["slope_x"] = slope_x;
    torch::Tensor slope_y = torch::from_blob(slopey_dat, {3, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["slope_y"] = slope_y;
    torch::Tensor perm_x = torch::from_blob(permx_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["perm_x"] = perm_x;
    torch::Tensor perm_y = torch::from_blob(permy_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["perm_y"] = perm_y;
    torch::Tensor perm_z = torch::from_blob(permz_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["perm_z"] = perm_z;
    torch::Tensor sres = torch::from_blob(sres_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["sres"] = sres;
    torch::Tensor ssat = torch::from_blob(ssat_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["ssat"] = ssat;
    torch::Tensor fbz = torch::from_blob(fbz_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["pf_flowbarrier"] = fbz;
    torch::Tensor specific_storage = torch::from_blob(specific_storage_dat,
						      {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["specific_storage"] = specific_storage;
    torch::Tensor alpha = torch::from_blob(alpha_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["alpha"] = alpha;
    torch::Tensor n = torch::from_blob(n_dat, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    statics_map["n"] = n;

    statics = model.run_method("get_parflow_statics", statics_map).toTensor();
  }
  
  double* predict_next_pressure_step(double* pp, double* et, int nx, int ny, int nz) {
    torch::Tensor press = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)});
    torch::Tensor evaptrans = torch::from_blob(et, {nz, ny, nx}, torch::kDouble).index({Slice(1, -1), Slice(1, -1), Slice(1, -1)}).clone();
    press = model.run_method("get_parflow_pressure", press).toTensor();
    evaptrans = model.run_method("get_parflow_evaptrans", evaptrans).toTensor();
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(press);
    inputs.push_back(evaptrans);
    inputs.push_back(statics);
    torch::Tensor output = model.forward(inputs).toTensor();
    torch::Tensor model_output = model.run_method("get_predicted_pressure", output).toTensor();
    torch::Tensor predicted_pressure = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble);
    predicted_pressure.index_put_({Slice(1, nz-1), Slice(1, ny-1), Slice(1, nx-1)}, model_output);
    
    if (!predicted_pressure.is_contiguous()) {
      predicted_pressure = predicted_pressure.contiguous();
    }

    double* predicted_pressure_array = predicted_pressure.data_ptr<double>();

    // Copy pressure data back to the pressure field
    if (predicted_pressure_array != pp) {
      std::size_t sz = nx * ny * nz;
      std::copy(predicted_pressure_array, predicted_pressure_array + sz, pp);
    }
    return pp;
  }
}

#endif
