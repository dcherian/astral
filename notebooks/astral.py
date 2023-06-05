import dcpy
from pathlib import Path
import pandas as pd
import xarray as xr
import holoviews as hv

# Mooring locations from INCOIS map
# https://incois.gov.in/portal/datainfo/buoys.jsp
LOCS = {
    "AD06": {"latitude": 18.5, "longitude": 67.45},
    "AD07": {"latitude": 14.93, "longitude": 68.98},
    "AD08": {"latitude": 12.07, "longitude": 68.63},
}


def preprocess(ds0):
    init = ds0.Time.attrs["units"].split(" ")[2]
    ds0.coords["init"] = pd.Timestamp(init)
    ds0["init"].attrs["standard_name"] = "forecast_reference_time"
    ds0 = ds0.expand_dims("init")

    ds0 = ds0.rename({"Time": "lead"})
    ds0["lead"].attrs = {"standard_name": "forecast_period"}
    ds0["lead"] = ds0.lead.astype("timedelta64[h]")

    ds0.coords["time"] = ds0.init + ds0.lead
    ds0.time.attrs.update({"axis": "T", "standard_name": "time"})

    ds0["hmxl"] /= 100
    ds0.hmxl.attrs["units"] = "m"

    return ds0


def read_forecast_data():
    folder = Path("/glade/scratch/acsubram/EKAMSAT/")
    files = [
        file
        for file in sorted(folder.glob("*.nc"))
        if "20230601.nc" not in file.name  # incomplete file
    ]

    ds = xr.open_mfdataset(
        files,
        preprocess=preprocess,
        decode_times=False,
        combine="nested",
        concat_dim="init",
        coords="minimal",
        join="exact",
        compat="override",
    )
    return ds


def plot_facetgrid(da, decimated, cbar_location="top", pad=0.05):
    if "Z" in da.cf:
        da = da.cf.sel(Z=[0, 25, 50], method="nearest")
        kwargs = {"row": "zt_k"}
        extra_loc = {}
    else:
        kwargs = {}
        extra_loc = {"zt_k": 0, "method": "nearest"}

    fg = da.isel(init=-1).plot(
        x="xt_i",
        y="yt_j",
        **kwargs,
        col="lead",
        robust=True,
        cbar_kwargs={
            "shrink": 0.6,
            "aspect": 40,
            "orientation": "horizontal",
            "location": cbar_location,
            "pad": pad,
            "label": (
                f"{da.attrs['long_name']} [{da.attrs['units']}]"
                f"| initialized: {da.init[-1].dt.strftime('%Y-%m-%d').data}"
            ),
        },
        cmap="coolwarm",
        aspect=1 / 1.5,
        size=4,
    )
    for ax, loc in zip(fg.axs.ravel(), fg.name_dicts.ravel()):
        (
            (decimated[["u", "v"]] / 100)
            .cf.sel(**loc, **extra_loc)
            .cf.plot.quiver(
                x="X",
                y="Y",
                u="u",
                v="v",
                ax=ax,
                add_guide=False,
                scale=4,
            )
        )
        for name, moor in LOCS.items():
            ax.plot(
                moor["longitude"],
                moor["latitude"],
                marker="o",
                color="w",
                markersize=10,
            )
            ax.plot(
                moor["longitude"], moor["latitude"], marker="o", color="k", markersize=6
            )
            ax.text(moor["longitude"], moor["latitude"], name[-2:], fontsize="large")

    dcpy.plots.clean_axes(fg.axs)
    fg.set_titles()
    return fg


def plot_profile(da, **kwargs):
    return (
        da.cf.sel(Z=slice(500))
        .isel(init=-1)
        .swap_dims({"lead": "time"})
        .hvplot.quadmesh(
            x="time",
            y="zt_k",
            cmap="coolwarm",
            **kwargs,
        )
        .opts(
            title=(
                f"{da.moor.data} - "
                f"initialized {da.init[-1].dt.strftime('%Y-%m-%d').data}"
            )
        )
    )


def plot_surface_vars(da):
    surf = da.cf.sel(Z=0, method="nearest")
    surf["init"] = surf.init.dt.strftime("%Y-%m-%d")
    return (
        (
            surf.temp.hvplot.line(x="time", by="init", title="SST", ylim=(29.5, 31.5))
            + surf.salinity.hvplot.line(
                x="time",
                by="init",
                title="SSS",  # ylim=(34, 36.3)
            )
            + surf.hmxl.hvplot.line(
                x="time", by="init", title="HMXL", flip_yaxis=True, ylim=(0, 30)
            )
        )
        .opts(hv.opts.Overlay(width=800, aspect=4, legend_position="bottom_right"))
        .cols(1)
    ).opts(title=da.moor.data.item())


def plot_ts_profiles_line(da):
    def plot(da_):
        return (
            da_.cf.sel(Z=slice(50))
            .isel(init=-1)
            .hvplot.line(by="zt_k", height=600, width=900)
            .opts(hv.opts.Overlay(legend_position="bottom", legend_cols=2))
        )

    return (plot(da.temp) + plot(da.salinity)).cols(1)


def offset(da, x, y, offset=0, remove_mean=False):
    import numpy as np

    assert da[y].ndim == 1

    off = xr.DataArray(np.arange(da.sizes[y], dtype=np.float64), dims=y)
    off *= offset

    # remove mean and add offset
    if remove_mean:
        daoffset = da.groupby(y) - da.groupby(y).mean()
    else:
        daoffset = da

    daoffset = daoffset + off

    return daoffset


def plot_ts_profiles_quadmesh(da):
    return (
        (
            plot_profile(da.temp, clim=(19, 31))
            + plot_profile(da.salinity, clim=(34, 37))
        )
        .opts(hv.opts.QuadMesh(invert_yaxis=True, ylim=(160, 0)))
        .cols(1)
    )


def offset_line_plot(da):
    return (
        offset(
            da.cf.sel(Z=slice(150)).isel(init=-1),
            offset=0.5,
            remove_mean=False,
            x="zt_k",
            y="lead",
        )
        .rename(da.name)
        .hvplot.line(
            y="zt_k",
            by="time",
            color="b",
            legend=False,
            flip_yaxis=True,
            width=400,
            height=400,
            xlabel="offset by 0.5",
            title=da.moor.data.item(),
        )
    )


def plot_forecast_offset_lines(moors):
    return (
        hv.Layout([offset_line_plot(moors.sel(moor=name)) for name in LOCS])
        .cols(3)
        .opts(
            hv.opts.Layout(
                title=(
                    f"{moors.attrs['long_name']} | "
                    f"initialized : "
                    f"{moors.init[-1].dt.strftime('%Y-%m-%d').data.item()}"
                )
            )
        )
    )
