import xarray as xr
import dask
import numpy as np
import odc.geo.xr # Needed for reprojection
from dask.diagnostics import ProgressBar

from parflow.tools.backend.xarray_utils import xr_pf_wtd, xr_pf_evaptrans, xr_pf_flow, xr_pf_surface_storage, \
    xr_pf_subsurface_storage, xr_pf_trend, xr_pf_recharge, add_decade, xr_pf_regression

pbar = ProgressBar()
pbar.register()

# Dask config
dask.config.set(scheduler='single-threaded')
# dask.config.set({"scheduler": "threads", "num_workers": 4})

# Parflow simulation path
simu_path = "/home/.../projects/results_fullsim2/"

# Reprojection
src_crs = dict(proj="aea", lat_1=16.12, lat_2=6.58, lat_0=11.35, lon_0=-4.3, x_0=0, y_0=0,
                   datum="WGS84", units="m", no_defs=None)
dst_crs = dict(proj="latlong")

# Export parameters
encoding = dict(time=dict(dtype="float64", units="seconds since 1970-01-01 00:00:00"))
unlimited_dims = "time"

# CLM layers
clm = ['eflx_lh_tot', 'eflx_lwrad_out', 'eflx_sh_tot', 'eflx_soil_grnd', 'qflx_evap_tot', 'qflx_evap_grnd',
       'qflx_evap_soi', 'qflx_evap_veg', 'qflx_tran_veg', 'qflx_infl', 'swe_out', 't_grnd', 'qflx_qirr', 't_soil']

index_clm = {v: i for i, v in enumerate(clm)}

# Reading parameters
kwargs = dict(
    start_date="2015-01-01",
    time_label="left",
    default_forcing_ts=10800
)


def export_debug():
    ds = xr.open_dataset(simu_path, engine="pf_ige", **kwargs)
    print(ds)

def export_prcp():
    path = "/home/.../projects/pfb_files/Forc_3h/APCP"
    ds = xr.open_dataset(path, engine="pf_ige", **kwargs).astype("float32")
    ds = ds.rename({"forc_time": "time"})

    # ds_d_sum = ds.resample(dict(time="D")).sum()

    ds.to_netcdf("res/prcp_3h.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_press():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_variables="press", **kwargs).astype("float32")
    ds.to_netcdf("res/press.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_prcp_decade():
    path = "/home/.../projects/pfb_files/Forc_3h/APCP"
    ds = xr.open_dataset(path, engine="pf_ige", **kwargs)
    ds = ds.rename({"forc_time": "time"})

    ds = add_decade(ds)
    ds = ds.groupby('decade').sum('time').rename({'decade': 'time'})

    ds = ds.odc.assign_crs(src_crs)
    ds = ds.odc.reproject(dst_crs)

    ds.to_netcdf("res/prcp_decade_latlon.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_instant_evap():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_types="clm", select_sim="WA_vd_KWE_RST", **kwargs).astype("float32")
    ds = ds.rename({"clm_time": "time"})
    ds = ds["clm"][:, index_clm["qflx_evap_tot"]]
    ds = ds.resample(time='1D').sum(dim='time')
    ds.to_netcdf("res/instant_evap_daily.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_wtd():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_variables=["press", "satur", "porosity"], **kwargs).astype("float32")

    ds_final = xr.Dataset()
    ds_final["wtd"] = xr_pf_wtd(ds["press"], ds["satur"])

    ds_final["wtd_mean"] = ds_final["wtd"].mean(dim="time")

    ds_final["wtd_regression"] = xr_pf_regression(ds_final["wtd"], ds_final["wtd_mean"])

    ds_final["wtdn"] = ds_final["wtd"] - ds_final["wtd_regression"]

    ds_final["amplitude"] = ds_final["wtd"].max(axis=0) - ds_final["wtd"].min(axis=0)

    ds_final["quality"] = np.abs(ds_final["amplitude"] - xr_pf_trend(ds_final["wtd"], ds_final["wtd_mean"])) / np.where(ds_final["amplitude"] > 0, ds_final["amplitude"], np.nan).astype("float32")

    ds_final["recharge"] = xr_pf_recharge(ds_final["wtd"], ds["porosity"], effective=True)

    ds_final["recharge_emptying"] = xr_pf_recharge(ds_final["wtd"], ds["porosity"], effective=True, emptying=True)

    ds_final.to_netcdf("res/wtd.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_params():
    units = {
        "input_porosity": "%",
        "input_Ksat": "m/s",
        "input_alpha": "m-1",
        "input_Ssat": "%",
        "input_n": "",
        "input_Sr": "%"
    }

    ds = xr.open_dataset(simu_path, engine="pf_ige", select_types="input", **kwargs).astype("float32")

    for var, unit in units.items():
        ds[var].attrs["units"] = unit

    # https://odc-geo.readthedocs.io/en/latest/_api/odc.geo.xr.ODCExtensionDa.reproject.html#odc.geo.xr.ODCExtensionDa.reproject
    ds = ds.odc.assign_crs(src_crs)
    ds = ds.odc.reproject(dst_crs)
    ds.to_netcdf("res/input.nc")

    # for name in ds.keys():
    #     print(name, end=" ")
    #     variable = ds[name]
    #     print(variable.shape)
    #
    #     variable = variable.odc.assign_crs(src_crs)
    #     variable = variable.odc.reproject(dst_crs)
    #     variable.to_netcdf(f"res/{name}.nc")


def export_evaptrans():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_variables="evaptrans", **kwargs).astype("float32")

    et = xr_pf_evaptrans(ds["evaptrans"])

    et.to_netcdf("res/evaptrans.nc", encoding=encoding, unlimited_dims=unlimited_dims)

def export_storage():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_types=["output", "temporal_output"], **kwargs).astype("float32")

    subsurface_storage = xr_pf_subsurface_storage(ds["press"], ds["satur"], ds["specific_storage"], ds["porosity"])

    subsurface_storage.to_netcdf("res/subsurface_storage.nc", encoding=encoding, unlimited_dims=unlimited_dims)

    surface_storage = xr_pf_surface_storage(ds["press"])

    surface_storage.to_netcdf("res/surface_storage.nc", encoding=encoding, unlimited_dims=unlimited_dims)


def export_flow():
    ds = xr.open_dataset(simu_path, engine="pf_ige", select_variables=["press", "slopex", "slopey", "mannings"], **kwargs).astype("float32")

    flow = xr_pf_flow(ds["press"], ds["input_slopex"], ds["input_slopey"], ds["mannings"])

    flow.to_netcdf("res/flow.nc", encoding=encoding, unlimited_dims=unlimited_dims)


export_debug()
# export_press()
# export_prcp()
# export_prcp_decade()
# export_instant_evap()
# export_wtd()
# export_params()
# export_evaptrans()
# export_storage()
# export_flow()
