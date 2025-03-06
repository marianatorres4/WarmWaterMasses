import xarray as xr
import glob
import os
import re

import dask
import warnings
warnings.filterwarnings('ignore')

wmt = xr.open_dataset(
    '/pub/mariant3/WarmWaterMasses/data/WMT_data/0186-0189_wmt-daily.nc'
).sel(time=slice("0186","0189"))

dsnovar = xr.Dataset(wmt.coords)
path = glob.glob('/pub/mariant3/WarmWaterMasses/data/budgets/7day_rolling_budgets/event-7d-roll-budget_id-*.nc')
path = sorted(path, key=lambda p: int(re.search(r'id-(\d+)_', p).group(1)))

mhwdataset = []

for files in path:
    match = re.search(r"id-(\d+)_", files)
    if match:
        var_name = int(match.group(1))
    
    dsog = xr.open_dataset(files)
    dsog = dsog.expand_dims({'mhw': [var_name]})
    dsog = dsog.sel(time=slice(dsog.time[0].values, dsog.time[-1].values))

    ds = xr.combine_by_coords([dsnovar, dsog]).fillna(0)
    mhwdataset.append(ds)
    
mhwdataset = xr.concat(mhwdataset, dim='mhw')

mhwdataset.to_netcdf("/pub/mariant3/WarmWaterMasses/data/budgets/7day_rolling_budgets/yearly-mhw-wmt-budgets-7d-rolling-mask_0186-0189.nc", mode='w')