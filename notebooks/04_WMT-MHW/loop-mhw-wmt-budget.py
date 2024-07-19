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

print(xwmb.__version__, xwmt.__version__, xgcm.__version__)

hfdrake_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/data_daily/"
ds = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily.*.nc", chunks={"time":1})

snap = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily_snap*.nc", chunks={"time":1})
static = xr.open_dataset("/pub/hfdrake/datasets/CM4_MHW_blobs/data/WMT_monthly/ocean_month_rho2.static.nc")#chunks={'time':1})

mt_path = "/pub/mariant3/WarmWaterMasses/notebooks/02_mhw_metrics/data_ocetrac-labels/"
blobs = xr.open_dataset(f"{mt_path}/01860503-01901020_ocetrac-labels-region.nc")
labels = blobs.blobs
df = pd.read_pickle(f"{mt_path}/01860503-01901020_mhw-metrics-region.pkl")

ids = np.unique(labels)
ids = np.array([id for id in ids if ~np.isnan(id)])
print(ids)

one_day_ids = df[df['duration'] == 1]['id'].tolist()
# Remove ids from the original labels array
ids = np.array([id for id in ids if id not in one_day_ids])
print(ids)

#Merge snapshots with time-averages
snap = snap.rename({
    **{'time': 'time_bounds'},
    **{v: f"{v}_bounds" for v in snap.data_vars}
    })

for mhw in ids[12:]:
    print(mhw)
    #mhw = int(mhw)
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

    print(' Add core coordinates of static to ds_event')

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

    # xgcm grid for dataset
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

    import xbudget
    budgets_dict = xbudget.load_preset_budget(model="MOM6_3Donly").copy()
    del budgets_dict['salt']['lhs']
    del budgets_dict['salt']['rhs']

    xbudget.collect_budgets(grid, budgets_dict)

    budgets_dict;

    import numpy as np
    import regionate
    import matplotlib.pyplot as plt
    print(' manso_region part')
    name = "MANSO"
    lons = np.array([-138.,-120.,-100., -70., -70., -100., -120., -138.])
    lats = np.array([10., 10., 10., 10., 38., 38., 38., 38.])
    manso_region = regionate.GriddedRegion(name, lons, lats, grid)

    import warnings

    lam = "heat"
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        wmb = xwmb.WaterMassBudget(
            grid,
            budgets_dict,
            ds_event.blobs #instead of manso_region.mask
        )
        wmb.mass_budget(lam, greater_than=True, default_bins=True)
        wmtcalc = wmb.wmt
        wmtcalc
    print('done with wmtcalc part')
    wmt_event = wmtcalc.where(
        (wmtcalc.time > cftime.datetime(186, 7, 16, calendar="noleap")) &
        (wmtcalc.time < cftime.datetime(186, 10, 1, calendar="noleap")), 0
    )
    print('starting wmt_event.load()')
    wmt_mhw_event_full = wmt_event.load()
    print('done with wmt_event.load()')

    print('saving...')

    wmt_mhw_event_full.to_netcdf(f'/pub/mariant3/WarmWaterMasses/notebooks/04_WMT-MHW/WMT_data/wmt_mhw_{int(mhw)}-full.nc', mode='w')
    print(f'saved!{int(mhw)} mhw')





