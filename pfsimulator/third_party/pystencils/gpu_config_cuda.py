import pystencils as ps

from pystencilssfg import SfgConfig


def configure_sfg(cfg: SfgConfig):
    cfg.extensions.header = "h"
    cfg.extensions.impl = "cu"
    cfg.c_interfacing = True


def project_info():
    return {
        "project_name": "pystencils_coupling",
        "default_dtype": "float64",
        "use_cpu": False,
        "use_cuda": True,
        "default_block_size": (256, 1, 1),  # sets block size for the CUDA execution configuration
        "use_warp_reductions" : False,      # use warp-level reductions for speedup
        "use_shared_mem_reductions": False, # extend warp-level reductions with shared memory reduction for more speedup
        "use_cub_reductions": False,        # use CUB backend for fast reductions
        "gpu_indexing_scheme": "linear3d",  # determines how kernels are launched
        "use_manual_exec_cfg_cuda": False,  # exec cfg is set manually with `default_block_size` and `manual_grid_size`
        "manual_grid_size": (32, 32, 32),   # only effective if the flag above is set
        "use_openmp": False,
        "target": ps.Target.CUDA,
    }
