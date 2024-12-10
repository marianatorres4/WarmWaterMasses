print('load libraries')
import datetime
import xarray as xr
import numpy as np
import dask
from tqdm import tqdm
import pandas as pd
import xwmb
import xwmt
import xgcm
import warnings
warnings.filterwarnings('ignore')
import cftime
print(xwmb.__version__, xwmt.__version__, xgcm.__version__)

## Loading data and realigning cell center/corner coordianates
print("Loading ds")
hfdrake_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/data_daily/"
ds = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily.*.nc", chunks={"time":1})
ds = ds.isel(yh=slice(1, None), yq=slice(None, -1), xh=slice(1,None), xq=slice(None, -1)) # realign cell center/corner coordinates
print("...loaded and realigned ds")

print("Loading snap")
snap = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily_snap*.nc", chunks={"time":1})
# Rename snapshot time coordinates to time_bounds so they can later be merged with ds
snap = snap.rename({
    **{'time': 'time_bounds'},
    **{v: f"{v}_bounds" for v in snap.data_vars}
    })
print("...loaded and renamed time coordinates to time_bounds so they can later be merged with ds")

print("Loading static")
static = xr.open_dataset("/pub/hfdrake/datasets/CM4_MHW_blobs/data/WMT_monthly/ocean_month_rho2.static.nc")
print("...loaded static")
print("...loaded all data, moving onto labels")

for year in np.unique(ds.time.dt.year.values):
    print("loading labels")
    mt_path = "/pub/mariant3/WarmWaterMasses/data/"
    labels = xr.open_dataset(f"{mt_path}ocetracv6/ocetrac-v6-processed/ocetrac-v6-blobs-tos-t1-r1-msq0-01860315-01891214-manso.nc").sel(time=slice(f"0{year}",f"0{year}")).blobs.rename('event_mask')
    print("loaded labels and renamed to event_mask")
    
    ids = np.unique(labels.values[~np.isnan(labels.values)])
    #Because we want to run the budgets for all individual events, we will not get rid of 1 day events. Instead, we sort the events based on their area (largest to smallest)
    area_array = [(labels == id).any("time").sum(["xh", "yh"]).values for id in ids]
    ids = [id for _, id in sorted(zip(area_array, ids), reverse=True)]

    for mhw in ids:
        print(f'Working on MHW: {mhw}')
    
        print('----------------- creating a 3-day dynamic mask -----------------')
    
        # Create centered 3-day rolling cumulative mask
        event_mask = (labels == mhw).rolling({"time":3}, center=True).max("time").fillna(0.).astype("float64")

        print('----------------- zooming into the each event -----------------')
    
        # Zoom in on time period of event (+ 1 day before and 1 day after)
        event_times = event_mask.time[event_mask.any(["xh", "yh"]).compute()]
        event_mask = event_mask.sel(time=slice(
            event_times[ 0] + datetime.timedelta(days = -1),
            event_times[-1] + datetime.timedelta(days =  1),
        ))

        print('----------------- merging and realigning-----------------')
        
        # Merge budget diagnostics with the event mask
        ds_event = xr.merge([ds, event_mask], join='inner')
        
        # Get snapshots that bound the event
        snap_event = snap.sel(
            time_bounds=slice(
                ds_event.time[0].values + datetime.timedelta(days = -1),
                ds_event.time[-1].values + datetime.timedelta(days = 1)
            )
        )
        
        # Merge budget diagnostics, snapshots, and static grid
        ds_event = xr.merge([ds_event, snap_event, static], join='inner')
    
        # Zoom in on region of the actual event
        xh_event = ds_event.xh[ds_event.event_mask.any(["time", "yh"])]
        yh_event = ds_event.yh[ds_event.event_mask.any(["time", "xh"])]
        ds_event = ds_event.sel(xh=xh_event, yh=yh_event)
    
        # Realign tracer center/corner coordinates because inner "join" only shrinks the ("xh", "yh") dimensions!
        xq_inner = ds_event.xq.sel(xq=slice(ds_event.xh[0], ds_event.xh[-1]))
        xq_islice = (np.abs(ds_event.xq - xq_inner[0]).argmin().values - 1, np.abs(ds_event.xq - xq_inner[-1]).argmin().values + 2)
    
        yq_inner = ds_event.yq.sel(yq=slice(ds_event.yh[0], ds_event.yh[-1]))
        yq_islice = (np.abs(ds_event.yq - yq_inner[0]).argmin().values - 1, np.abs(ds_event.yq - yq_inner[-1]).argmin().values + 2)
        
        ds_event = ds_event.isel(xq=slice(*xq_islice), yq=slice(*yq_islice))
    
        print('----------------- Adding core coordinates of static to ds_event -----------------')
    
        def add_estimated_layer_interfaces(ds):
            return ds.assign_coords({"zi": xr.DataArray(
                np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6000]]),
                dims=('zi',)
            )})
                
        ds_event = add_estimated_layer_interfaces(ds_event)
        
        ds_event = ds_event.assign_coords({
            "areacello": xr.DataArray(ds_event["areacello"].values, dims=('yh', 'xh',)), # Required for area-integration
            "lon": xr.DataArray(ds_event["geolon"].values, dims=('yh', 'xh',)), # Required for calculating density if not already provided!
            "lat": xr.DataArray(ds_event["geolat"].values, dims=('yh', 'xh',)), # Required for calculating density if not already provided!
            "xq": xr.DataArray(ds_event["xq"].values, dims=('xq',)),
            "yq": xr.DataArray(ds_event["yq"].values, dims=('yq',)),
            "deptho": xr.DataArray(ds_event["deptho"].values, dims=('yh', 'xh',)),
            "geolon": xr.DataArray(ds_event["geolon"].values, dims=('yh', 'xh',)),
            "geolat": xr.DataArray(ds_event["geolat"].values, dims=('yh', 'xh',)),
            "geolon_c": xr.DataArray(ds_event["geolon_c"].values, dims=('yq', 'xq',)),
            "geolat_c": xr.DataArray(ds_event["geolat_c"].values, dims=('yq', 'xq',)),
        })
        coords = {
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
        'Z': {'center': 'zl', 'outer': 'zi'}
        }
        metrics = {
            ('X','Y'): "areacello", # Required for area-integration
        }
        
        ds_event['tos'] = ds_event['thetao'].isel(zl=0)
    
        import numpy as np
        import regionate
        import warnings

        print('----------------- creating _calc_temperature_wmt function -----------------')
    
        def _calc_temperature_wmt(ds_event):
            lam = "heat"
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                    
                grid = xgcm.Grid(ds_event.copy(), coords=coords, metrics=metrics, boundary={'X':'extend', 'Y':'extend', 'Z':'extend'},
                                 autoparse_metadata=False)
                wm = xwmt.WaterMass(grid)
                
                import xbudget
                budgets_dict = xbudget.load_preset_budget(model="MOM6_3Donly").copy()
                del budgets_dict['salt']['lhs']
                del budgets_dict['salt']['rhs']
                
                xbudget.collect_budgets(grid, budgets_dict)
                
                wmb = xwmb.WaterMassBudget(
                    grid,
                    budgets_dict,
                    ds_event.event_mask.squeeze() if ds_event.event_mask.squeeze().any() else None
                )
                wmb.mass_budget(lam, greater_than=True, default_bins=True)
                
            return wmb.wmt
        
        def sel_times(ds, t):
            return ds.sel(
                time = t.expand_dims("time"),
                time_bounds = slice(
                    t + datetime.timedelta(days = -1),
                    t + datetime.timedelta(days = 1)
                )
            )

        print('----------------- getting ready to define wmt -----------------')

        wmt = xr.concat([
            _calc_temperature_wmt(sel_times(ds_event, t)).drop_dims(["time_bounds"])
            for t in ds_event.time
        ], dim="time")
        
        start = wmt.time.values.astype('datetime64[D]')[0]
        end = wmt.time.values.astype('datetime64[D]')[-1]
        print(f'Event {mhw} starts on {start} and ends on {end}')
        print(wmt.time.values)
        
        print('----------------- getting ready to load wmt -----------------')
        
        wmt.load()
        
        print('-----------------loaded wmt -----------------')

        print('saving nc file...')
        print(f"wmt_mhw_event_full.to_netcdf(f'/pub/mariant3/WarmWaterMasses/data/ocetracv6/ocetrac-v6-processed/event-budget-id{int(mhw)}-{start}-{end}-ocetrac-v6.nc', mode='w'")
        wmt.to_netcdf(f'/pub/mariant3/WarmWaterMasses/data/ocetracv6/ocetrac-v6-processed/event-budget-id{int(mhw)}-{start}-{end}-ocetrac-v6.nc', mode='w')
        print(f'...saved nc file for mhw{mhw}!')
        