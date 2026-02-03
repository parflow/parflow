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
        "use_openmp": False,
        "target": ps.Target.CUDA,
        # sets block size for the CUDA execution configuration
        "default_block_size": (32, 8, 4),
        # use warp-level reductions for speedup (does not need shared memory, but warp-divisible block size)
        "use_warp_reductions" : False,
        # extend warp-level reductions with shared memory reduction for even more speedup
        "use_shared_mem_reductions": False,
        # use CUB backend for fast reductions (uses shared memory)
        "use_cub_reductions": False,
        # determines how kernels are launched
        "gpu_indexing_scheme": "linear3d",
        # launch configuration is set manually with `default_block_size` and `manual_grid_size`
        "use_manual_exec_cfg_cuda": False,
        # only effective if `use_manual_exec_cfg_cuda` is set
        "manual_grid_size": (32, 32, 32),
    }
