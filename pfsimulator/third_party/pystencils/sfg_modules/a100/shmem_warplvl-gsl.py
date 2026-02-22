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
        "default_block_size": (4, 8, 32),
        "use_shared_mem_reductions": True,
        "use_manual_exec_cfg_cuda": True,
        "manual_grid_size": (12, 9, 8),
        "gpu_indexing_scheme": "gridstrided_linear3d",
    }
