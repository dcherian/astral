import dcpy
from pathlib import Path
import pandas as pd
import xarray as xr

# Mooring locations from INCOIS map: https://incois.gov.in/portal/datainfo/buoys.jsp
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
            title=f"{da.moor.data} - initialized {da.init[-1].dt.strftime('%Y-%m-%d').data}"
        )
    )
