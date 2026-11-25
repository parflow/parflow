from typing import Optional
from numbers import Number

import xarray as xr
import pandas as pd
import numpy as np


def get_dz(da: xr.DataArray) -> np.array:
    """Computes dz from da"""
    # alternative: z = da.coords["z"].data; return -np.diff(z, append=z[-1])
    depth = np.array(da.coords["z"].data)[::-1]
    return np.diff(depth, prepend=depth[0])[::-1]


def get_dx_dy(da: xr.DataArray) -> tuple[int, int]:
    """Computes dx and dy from da"""
    dx = da.coords["x"].values
    dx = dx[1] - dx[0]
    dy = da.coords["y"].values
    dy = dy[1] - dy[0]
    return dx, dy


def add_decade(ds: xr.Dataset, time_name="time") -> xr.Dataset:
    """Adds a decade coordinate to the variable."""
    dt = ds[time_name].dt
    decade = pd.to_datetime(
        {"year": dt.year, "month": dt.month, "day": (dt.day // 10).clip(max=2) + 1}
    )
    return ds.assign_coords(decade=(time_name, decade))


def xr_pf_wtd(
    press: xr.DataArray,
    satur: xr.DataArray,
    ssat: Optional[Number | xr.DataArray] = None,
    epsilon: float = 0.001,
) -> xr.DataArray:
    """Xarray/Dask wrapper for pftools.hydrology.calculate_water_table_depth"""
    from parflow.tools.hydrology import calculate_water_table_depth

    def wtd_fct(press, satur, dz, ssat, epsilon):
        if isinstance(ssat, xr.DataArray):
            ssat = ssat.data
        # squeeze from t, z, x, y to z, x, y
        res = calculate_water_table_depth(
            press.data.squeeze(), satur.data.squeeze(), dz, ssat, epsilon
        )
        res = np.expand_dims(res, axis=0)  # from x, y to t, x, y

        coords = press.coords.copy()
        del coords["z"]

        dims = [dim for dim in press.dims if dim != "z"]

        wtd_da = xr.DataArray(res, dims=dims, coords=coords)

        return wtd_da

    press = press.chunk(dict(time=1))
    satur = satur.chunk(dict(time=1))

    if ssat is None:
        ssat = np.nanmax(satur[0])
        print(f"Automatic ssat value retrieved from saturation: {ssat}")

    dz = get_dz(press)

    res = xr.map_blocks(
        wtd_fct,
        press,
        (satur,),
        kwargs={"dz": dz, "ssat": ssat, "epsilon": epsilon},
        template=press.isel(z=0).drop_vars("z"),
    )

    return res.rename("wtd").assign_attrs(dict(units="m"))


def xr_pf_flow(
    press: xr.DataArray,
    slopex: xr.DataArray,
    slopey: xr.DataArray,
    mannings: xr.DataArray,
) -> xr.DataArray:
    """Xarray/Dask wrapper for pftools.hydrology.calculate_overland_flow_grid"""
    from parflow.tools.hydrology import calculate_overland_flow_grid

    def flow_fct(press, slopex, slopey, mannings, dx, dy):
        press_data = press.data.squeeze()
        # squeeze from t, z, x, y to z, x, y
        mask = np.ones_like(press_data)

        res = calculate_overland_flow_grid(
            press_data,
            slopex.data.squeeze(),
            slopey.data.squeeze(),
            mannings=mannings.data.squeeze(),
            dx=dx,
            dy=dy,
            mask=mask,
        )
        res = np.expand_dims(res, axis=0)  # from x, y to t, x, y

        coords = press.coords.copy()
        del coords["z"]

        dims = [dim for dim in press.dims if dim != "z"]

        flow_da = xr.DataArray(res, dims=dims, coords=coords)

        return flow_da

    dx, dy = get_dx_dy(press)

    press = press.chunk(dict(time=1))

    res = xr.map_blocks(
        flow_fct,
        press,
        (slopex, slopey, mannings),
        kwargs={"dx": dx, "dy": dy},
        template=press.isel(z=0).drop_vars("z"),
    )

    return res.rename("flow").assign_attrs(dict(units="m3/s"))


def xr_pf_evaptrans(evaptrans: xr.DataArray) -> xr.DataArray:
    """Xarray/Dask wrapper for pftools.hydrology.calculate_evapotranspiration"""
    from parflow.tools.hydrology import calculate_evapotranspiration

    def evap_fct(evaptrans, dz, dx, dy):
        evaptrans_data = evaptrans.data.squeeze()
        res = calculate_evapotranspiration(evaptrans_data, dx=dx, dy=dy, dz=dz)
        res = np.expand_dims(res, axis=0)

        coords = evaptrans.coords.copy()
        dims = tuple(evaptrans.dims)

        et_da = xr.DataArray(res, dims=dims, coords=coords)

        return et_da

    dx, dy = get_dx_dy(evaptrans)
    dz = get_dz(evaptrans)

    evaptrans = evaptrans.chunk(dict(time=1))

    res = xr.map_blocks(
        evap_fct, evaptrans, kwargs={"dx": dx, "dy": dy, "dz": dz}, template=evaptrans
    )

    return res.rename("evaptrans").assign_attrs(dict(units="L^3/T"))


def xr_pf_subsurface_storage(
    press: xr.DataArray,
    satur: xr.DataArray,
    specific_storage: xr.DataArray,
    porosity: xr.DataArray,
) -> xr.DataArray:
    """Xarray/Dask wrapper for pftools.hydrology.calculate_subsurface_storage"""
    from parflow.tools.hydrology import calculate_subsurface_storage

    def subsurface_storage_fct(press, satur, specific_storage, porosity, dx, dy, dz):
        # squeeze from t, z, x, y to z, x, y
        res = calculate_subsurface_storage(
            porosity.data.squeeze(),
            press.data.squeeze(),
            satur.data.squeeze(),
            specific_storage.data.squeeze(),
            dx=dx,
            dy=dy,
            dz=dz,
        )
        res = np.expand_dims(res, axis=0)  # from x, y to t, x, y

        coords = press.coords.copy()
        dims = tuple(press.dims)

        wtd_da = xr.DataArray(res, dims=dims, coords=coords)

        return wtd_da

    press = press.chunk(dict(time=1))
    satur = satur.chunk(dict(time=1))

    dz = get_dz(press)
    dx, dy = get_dx_dy(press)

    res = xr.map_blocks(
        subsurface_storage_fct,
        press,
        (satur, specific_storage, porosity),
        kwargs={"dx": dx, "dy": dy, "dz": dz},
        template=press,
    )

    return res.rename("subsurface_storage").assign_attrs(dict(units="m^3"))


def xr_pf_surface_storage(press: xr.DataArray) -> xr.DataArray:
    """Xarray/Dask wrapper for pftools.hydrology.calculate_surface_storage"""
    from parflow.tools.hydrology import calculate_surface_storage

    def surface_storage_fct(press, dx, dy):
        # squeeze from t, z, x, y to z, x, y
        res = calculate_surface_storage(press.data.squeeze(), dx=dx, dy=dy)
        res = np.expand_dims(res, axis=0)  # from x, y to t, x, y

        coords = press.coords.copy()
        del coords["z"]

        dims = [dim for dim in press.dims if dim != "z"]

        wtd_da = xr.DataArray(res, dims=dims, coords=coords)

        return wtd_da

    press = press.chunk(dict(time=1))

    dx, dy = get_dx_dy(press)

    res = xr.map_blocks(
        surface_storage_fct,
        press,
        kwargs={"dx": dx, "dy": dy},
        template=press.isel(z=0).drop_vars("z"),
    )

    return res.rename("surface_storage").assign_attrs(dict(units="m^3"))


def xr_pf_trend(
    var: xr.DataArray, var_mean: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Computes the trend of the ridge regression for the given variable 3d variable (time, x, y)."""
    # a = cov(t_mean,y_mean)/var(t_mean) and b=y_mean−at_mean
    t_num = var.coords["time"].data.astype("int64")
    t_mean = t_num.mean()
    if var_mean is None:
        var_mean = var.mean("time")

    a = ((t_num[:, np.newaxis, np.newaxis] - t_mean) * (var - var_mean)).mean(
        "time"
    ) / ((t_num - t_mean) ** 2).mean()

    name = str(var.name) + "_trend" if var.name else "trend"

    coords = var.coords.copy()
    del coords["time"]

    dims = [dim for dim in var.dims if dim != "time"]

    unit = var.attrs.get("units", "") + "/s"

    trend_da = xr.DataArray(
        a, name=name, dims=dims, coords=coords, attrs=dict(units=unit)
    )

    return trend_da


def xr_pf_regression(
    var: xr.DataArray, var_mean: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """Computes the ridge regression over time for the given variable."""
    # a = cov(t_mean,y_mean)/var(t_mean) and b=y_mean−at_mean
    t_num = var.coords["time"].data.astype("int64")
    t_mean = t_num.mean()
    if var_mean is None:
        var_mean = var.mean("time")

    a = ((t_num[:, np.newaxis, np.newaxis] - t_mean) * (var - var_mean)).mean(
        "time"
    ) / ((t_num - t_mean) ** 2).mean()
    b = var_mean - a * t_mean
    trend = np.expand_dims(a, axis=0) * t_num[
        :, np.newaxis, np.newaxis
    ] + np.expand_dims(b, axis=0)

    name = str(var.name) + "_trend" if var.name else "trend"

    trend_da = xr.DataArray(
        trend,
        name=name,
        dims=tuple(var.dims),
        coords=var.coords.copy(),
        attrs=var.attrs.copy(),
    )

    return trend_da


def xr_pf_recharge(
    wtd: xr.DataArray,
    porosity: Optional[xr.DataArray],
    effective: bool = False,
    emptying: bool = False,
):
    """Computes the recharge from the wtd

    :param wtd: wtd DataArray
    :param porosity: porosity DataArray
    :param effective: If true, use porosity to calculate the effective variation.
    :param emptying: if true, estimates the average loss over time, and adds this loss back to the recharge.
    :return: the estimated recharge.
    """

    coords = wtd.coords.copy()
    dims = [d for d in wtd.dims]

    wtd = wtd.data

    if effective:
        if porosity is None:
            print(
                "Unable to calculate effective recharge without porosity (porosity=None), skipping."
            )
        else:
            """Idea :
            wtd       8
            z         [100 30  10  5   3   1   1  ]
            dz        [70  20  5   2   2   0   0  ]
            z - wtd   [92  22  2  -3  -5  -7  -7  ]
            minimum   [70  20  2  -3  -5  -7  -7  ]
            maximum   [70  20  2   0   0   0   0  ]
            """
            z = porosity.coords["z"].data
            dz = get_dz(porosity)

            z = z[:, np.newaxis, np.newaxis, np.newaxis]  # len(z) -> len(z), 1, 1, 1
            dz = dz[:, np.newaxis, np.newaxis, np.newaxis]  # len(z) -> len(z), 1, 1, 1
            wtd = np.expand_dims(wtd, axis=0)  # t, x, y -> 1, t, x, y
            porosity_data = np.expand_dims(
                porosity.data, axis=1
            )  # len(z), x, y -> len(z), 1, x, y

            wtd = np.sum(np.maximum(np.minimum(z - wtd, dz), 0) * porosity_data, axis=0)

    recharge = wtd[:-1] - wtd[1:]

    if emptying:
        emptying_da = np.where(recharge < 0, recharge, 0)
        emptying_mean = emptying_da.sum(axis=0) / np.maximum(
            np.count_nonzero(emptying_da, axis=0), 1
        )
        recharge = recharge - emptying_mean
        recharge = np.where(recharge > 0, recharge, 0)

    coords["time"] = ("time", coords["time"].data[:-1])

    recharge_da = xr.DataArray(
        name="recharge", data=recharge, dims=dims, coords=coords, attrs=dict(units="m")
    )

    return recharge_da
