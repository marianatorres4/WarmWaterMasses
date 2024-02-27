import dask

import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import ocetrac


import warnings
warnings.filterwarnings('ignore')
from datetime import date

print('loaded libraries')


dir_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/"
ds = xr.open_dataset(f"{dir_path}/data/ocean_daily_cmip.01860101-01901231.tos.nc", chunks={'time':100})
mt_path = "/pub/mariant3/WarmWaterMasses/02_tracking-mhws"

print('loaded data')

start = 111
end = 300

print('defined start & end')

len_ds = len(np.unique(ds.time.dt.year))
len_ds =np.array(len_ds)

for i in range(1, len_ds+1):
    print(i)
    binary_out = ds['tos'].isel(time = slice(start,end)) > 29
    mask = xr.ones_like(binary_out.isel(time=0))
    print('binary_out & mask')
    Tracker = ocetrac.Tracker(binary_out, mask, radius=2, min_size_quartile=0.75, timedim='time', xdim='xh', ydim='yh', positive=True)
    blobs = Tracker.track()
    d = ds['tos'].isel(time = start).time.dt
    e = ds['tos'].isel(time = end).time.dt
    date_d = f"{d.year.values:0004}-{d.month.values:02}-{d.day.values:02}"
    date_e = f"{e.year.values:0004}-{e.month.values:02}-{e.day.values:02}"
    print(date_d)
    print(date_e)
    blobs.to_netcdf(f'{mt_path}/{date_d}_{date_e}_blobs.nc', mode = 'w')
    print('saved netcdf')
    start+=365
    end+=365



