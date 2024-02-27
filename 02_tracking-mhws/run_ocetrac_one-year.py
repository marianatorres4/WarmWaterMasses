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
mt_path = "/pub/mariant3/VeryWarmWaterMasses/saved_blob_output_ncfiles"

print('loaded data')
#start = 123
#end = 330

start = 111
end = 341
print('defined start & end')


binary_out = ds['tos'].sel(xh = slice(-138, 0), yh = slice(8, 49)).isel(time = slice(start,end)) > 29
mask = xr.ones_like(binary_out.isel(time=0))
print('binary_out & mask')
Tracker = ocetrac.Tracker(binary_out, mask, radius=2, min_size_quartile=0.75, timedim='time', xdim='xh', ydim='yh', positive=True)
print('Tracker')
blobs = Tracker.track()
print('blobs')
d = ds['tos'].isel(time = start).time.dt
e = ds['tos'].isel(time = end).time.dt
date_d = f"{d.year.values:0004}-{d.month.values:02}-{d.day.values:02}"
date_e = f"{e.year.values:0004}-{e.month.values:02}-{e.day.values:02}"
print(date_d)
print(date_e)
blobs.to_netcdf(f'{mt_path}/blobs_{date_d}_{date_e}.nc', mode = 'w')
print('saved netcdf')

