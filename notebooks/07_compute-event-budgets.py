print('load libraries')
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
xwmb.__version__, xwmt.__version__, xgcm.__version__

print("loading ds, snap, static...")
hfdrake_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/data_daily/"
ds = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily.*.nc", chunks={"time":1})
snap = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily_snap*.nc", chunks={"time":1})
static = xr.open_dataset("/pub/hfdrake/datasets/CM4_MHW_blobs/data/WMT_monthly/ocean_month_rho2.static.nc")#chunks={'time':1})
print("...loaded ds, snap, static")

print("loading labels,df...")
mt_path = "/pub/mariant3/WarmWaterMasses/data/"
labels = xr.open_dataset(f"{mt_path}ocetracv5/ocetrac-v5-processed/ocetrac-v5-blobs-tos-t1-r3-msq75-01860315-01891101-manso.nc").blobs.rename('event_mask')
df = pd.read_pickle(f"{mt_path}ocetracv5/ocetrac-v5-processed/ocetrac-v5-blobs-tos-t1-df-r3-msq75-01860315-01891101-manso.pkl")
print("...loaded labels,df")

#Merge snapshots with time-averages
snap = snap.rename({
    **{'time': 'time_bounds'},
    **{v: f"{v}_bounds" for v in snap.data_vars}
    })

ids = np.unique(labels)
ids = np.array([id for id in ids if ~np.isnan(id)])
one_day_ids = df[df['duration'] == 1]['id'].tolist()
# Remove ids from the original labels array
ids = np.array([id for id in ids if id not in one_day_ids])

#for all events
for mhw in ids[18:]:
    print(f'...Working on event {mhw}')
    mhw_df = df.loc[df.id==mhw]
    print('----------------- starting cumulative mask part -----------------')
    
    event = (labels == mhw).any("time")
    ds_event = xr.merge([ds, event], join="inner")
    ds_event = xr.merge([ds_event.sel(time=ds_event.time[1:]), snap])
    ds_event = xr.merge([static, ds_event], join='inner')

    def add_estimated_layer_interfaces(ds):
        return ds.assign_coords({"zi": xr.DataArray(
            np.concatenate([[0], 0.5*(ds.zl.values[1:]+ds.zl.values[0:-1]), [6000]]),
            dims=('zi',)
        )})
    
    ds_event = add_estimated_layer_interfaces(ds_event)
    
    print('----------------- Adding core coordinates of static to ds_event -----------------')

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
    grid = xgcm.Grid(ds_event.copy(), coords=coords, metrics=metrics, boundary={'X':'extend', 'Y':'extend', 'Z':'extend'}, autoparse_metadata=False)
    wm = xwmt.WaterMass(grid)

    print('----------------- importing xbudget -----------------')
    
    import xbudget
    budgets_dict = xbudget.load_preset_budget(model="MOM6_3Donly").copy()
    del budgets_dict['salt']['lhs']
    del budgets_dict['salt']['rhs']

    xbudget.collect_budgets(grid, budgets_dict)

    #print(budgets_dict)
    
    print('----------------- MANSO Region part, regionate -----------------')
    
    import numpy as np
    import regionate
    import matplotlib.pyplot as plt
    print(' manso_region part')
    name = "MANSO"
    lons = np.array([-138.,-120.,-100., -70., -70., -100., -120., -138.])
    lats = np.array([10., 10., 10., 10., 38., 38., 38., 38.])
    manso_region = regionate.GriddedRegion(name, lons, lats, grid)

    import warnings
    
    print('----------------- getting ready for xwmb.WaterMassBudget calculation -----------------')
    
    lam = "heat"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        wmb = xwmb.WaterMassBudget(
            grid,
            budgets_dict,
            ds_event.event_mask
        )
        wmb.mass_budget(lam, greater_than=True, default_bins=True)
        wmtcalc = wmb.wmt
        wmtcalc
    
    start_event_date = str(mhw_df['date'].values[0][0])
    start_event = wmtcalc.get_index("time").get_loc(f"{start_event_date}").start
    end_event_date = str(mhw_df['date'].values[0][-1])
    end_event = wmtcalc.get_index("time").get_loc(f"{end_event_date}").start
    
    print(f'Event {mhw} starts on {start_event_date} index={start_event} and ends on {end_event_date},index={end_event}')
    
    #wmt_event = wmtcalc.where(
    #(wmtcalc.time > cftime.datetime(186, 7, 16, calendar="noleap")) &
   # (wmtcalc.time < cftime.datetime(186, 10, 1, calendar="noleap")), 0
    #   )
    wmt_event = wmtcalc.isel(time=slice(start_event-15, end_event+15))
    #wmt_event = wmtcalc.sel(time=slice(start_date_mhw,end_date_mhw))
    print(wmt_event.time)
    
    print('----------------- loading xwmb.WaterMassBudget calculation -----------------')
    wmt_mhw_event_full = wmt_event.load()
    print('done with wmt_event.load()')
    
    print('saving...')
    wmt_mhw_event_full.to_netcdf(f'/pub/mariant3/WarmWaterMasses/data/ocetracv5/ocetrac-v5-processed/ocetrac-v5-t1-v2-ind-event-budget-{int(mhw)}-{start_event_date}-{end_event_date}.nc', mode='w')
    print(f'Saved mhw #{int(mhw)}!!')
