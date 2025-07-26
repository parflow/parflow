#include "torch_wrapper.h"
#ifdef PARFLOW_HAVE_TORCH

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>

using namespace torch::indexing;

static torch::jit::script::Module model;
static torch::Tensor statics;
static torch::Device device = torch::kCPU;

extern "C" {
  void init_torch_model(char* model_filepath, int nx, int ny, int nz, double *po_dat,
                        double *mann_dat, double *slopex_dat, double *slopey_dat, double *permx_dat,
                        double *permy_dat, double *permz_dat, double *sres_dat, double *ssat_dat,
                        double *fbz_dat, double *specific_storage_dat, double *alpha_dat, double *n_dat,
                        int torch_debug, char* torch_device) {
    // Determine device from string
    if (std::string(torch_device) == "cuda") {
      device = torch::kCUDA;
    } else {
      device = torch::kCPU;
    }

    std::string model_path = std::string(model_filepath);
    try {
      model = torch::jit::load(model_path);
      model.to(device);
      model.eval();
    }
    catch (const c10::Error& e) {
      throw std::runtime_error(std::string("Failed to load the Torch model:\n") + e.what());
    }

    std::unordered_map<std::string, torch::Tensor> statics_map;
    auto interior = Slice(1, -1);

    statics_map["porosity"] = torch::from_blob(po_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["mannings"] = torch::from_blob(mann_dat, {3, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["slope_x"] = torch::from_blob(slopex_dat, {3, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["slope_y"] = torch::from_blob(slopey_dat, {3, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["perm_x"] = torch::from_blob(permx_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["perm_y"] = torch::from_blob(permy_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["perm_z"] = torch::from_blob(permz_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["sres"] = torch::from_blob(sres_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["ssat"] = torch::from_blob(ssat_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["pf_flowbarrier"] = torch::from_blob(fbz_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["specific_storage"] = torch::from_blob(specific_storage_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["alpha"] = torch::from_blob(alpha_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);
    statics_map["n"] = torch::from_blob(n_dat, {nz, ny, nx}, torch::kDouble)
                              .index({interior, interior, interior}).clone().to(device);

    statics = model.run_method("get_parflow_statics", statics_map).toTensor();
    if (torch_debug) {
      torch::save(statics, "scaled_statics.pt");
    }
  }

  double* predict_next_pressure_step(double* pp, double* et, int nx, int ny, int nz, int file_number, int torch_debug) {
    auto interior = Slice(1, -1);
    torch::Tensor press = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble)
                            .index({interior, interior, interior}).clone().to(device);
    torch::Tensor evaptrans = torch::from_blob(et, {nz, ny, nx}, torch::kDouble)
                            .index({interior, interior, interior}).clone().to(device);

    press = model.run_method("get_parflow_pressure", press).toTensor();
    evaptrans = model.run_method("get_parflow_evaptrans", evaptrans).toTensor();

    if (torch_debug) {
      char filename[64];
      std::snprintf(filename, sizeof(filename), "scaled_pressure_%05d.pt", file_number);
      torch::save(press, filename);
      std::snprintf(filename, sizeof(filename), "scaled_evaptrans_%05d.pt", file_number);
      torch::save(evaptrans, filename);
    }

    std::vector<torch::jit::IValue> inputs = {press, evaptrans, statics};
    torch::Tensor output = model.forward(inputs).toTensor();
    torch::Tensor model_output = model.run_method("get_predicted_pressure", output).toTensor().to(torch::kCPU);

    torch::Tensor predicted_pressure = torch::from_blob(pp, {nz, ny, nx}, torch::kDouble);
    predicted_pressure.index_put_({Slice(1, nz-1), Slice(1, ny-1), Slice(1, nx-1)}, model_output);

    if (!predicted_pressure.is_contiguous()) {
      predicted_pressure = predicted_pressure.contiguous();
    }

    double* predicted_pressure_array = predicted_pressure.data_ptr<double>();
    if (predicted_pressure_array != pp) {
      std::size_t sz = nx * ny * nz;
      std::copy(predicted_pressure_array, predicted_pressure_array + sz, pp);
    }

    return pp;
  }
}

#endif
