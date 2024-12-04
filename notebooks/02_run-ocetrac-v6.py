#ocetrac-v6, script for running ocetrac using tos output and climatology and changing the radius and size threshold configurations
import dask
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import ocetrac
import warnings
warnings.filterwarnings('ignore')
from datetime import date

# setting paths
dir_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/"
mt_path = "/pub/mariant3/WarmWaterMasses/"
save_path = "/pub/mariant3/WarmWaterMasses/data"
# data
ds = xr.open_dataset(f"{dir_path}/data/ocean_daily_cmip.01860101-01901231.tos.nc", chunks={'time': 100}).tos
climatology = xr.open_dataarray(f"{mt_path}/data/climatology/climatology-manso-0186-01-01-0189-12-31.nc")

initial_start = ds.get_index("time").get_loc("0186-03-15").start
initial_end = ds.get_index("time").get_loc("0186-12-15").start

start = initial_start
end = initial_end

len_ds = len(np.unique(ds.time.dt.year))

task = 1
#change below if needed for more tasks
while task <=2:
    
    if task == 1:
        data = ds
        radius = 1
        size_thresh = 0.0
        msq = str(size_thresh).split('.')[1]
        data_file_name = "tos"
        
    elif task == 2:
        data = climatology
        radius = 1
        size_thresh = 0.0
        msq = str(size_thresh).split('.')[1]
        data_file_name = "clim"

    # elif task == 3:
    #     data = ds
    #     radius = 3
    #     size_thresh = 0.25
    #     msq = str(size_thresh).split('.')[1]
    #     data_file_name = "tos"

    # elif task == 4:
    #     data = climatology
    #     radius = 3
    #     size_thresh = 0.75
    #     msq = str(size_thresh).split('.')[1]
    #     data_file_name = "clim"

    # elif task == 5:
    #     data = climatology
    #     radius = 1
    #     size_thresh = 0.75
    #     msq = str(size_thresh).split('.')[1]
    #     data_file_name = "clim"

    # elif task == 6:
    #     data = climatology
    #     radius = 3
    #     size_thresh = 0.25
    #     msq = str(size_thresh).split('.')[1]
    #     data_file_name = "clim"

    for i in range(1, len_ds):
        print(f" TASK: {task}, iterations of each task: {i}/4, radius={radius}, min_size_quartile={size_thresh}, pulling data from {data_file_name}")

        hot_water = data.isel(time=slice(start, end)) > 29
        print('defined hot_water')
        
        mask_ocean = 1 * np.ones(data.shape[1:]) * np.isfinite(data.isel(time=0))
        mask_land = 0 * np.ones(data.shape[1:]) * np.isnan(data.isel(time=0))
        mask = mask_ocean + mask_land
        print('defined mask')

        print(f"Tracker = ocetrac.Tracker(hot_water, mask, radius={radius}, min_size_quartile={size_thresh},timedim='time', xdim='xh', ydim='yh', positive=True)")
        Tracker = ocetrac.Tracker(hot_water, mask, radius=radius, min_size_quartile=size_thresh, timedim='time', xdim='xh', ydim='yh', positive=True)                     
        
        blobs = Tracker.track()
        print('Blobs tracked')

        d = data['time'].isel(time=start).time.dt
        e = data['time'].isel(time=end).time.dt
        date_d = f"{d.year.values:0004}-{d.month.values:02}-{d.day.values:02}"
        date_e = f"{e.year.values:0004}-{e.month.values:02}-{e.day.values:02}"
        print(date_d)
        print(date_e)

        print(f"blobs.to_netcdf(f'{save_path}/ocetracv6/ocetrac-v6-blobs-{data_file_name}-t{task}-i{i}-r{radius}-msq{msq}-{date_d}-{date_e}.nc', mode='w')")
        blobs.to_netcdf(f'{save_path}/ocetracv6/ocetrac-v6-blobs-{data_file_name}-t{task}-i{i}-r{radius}-msq{msq}-{date_d}-{date_e}.nc', mode='w')
        print('Saved NetCDF')
        
        #break here if only want one task
        start += 365
        end += 365

    start = initial_start
    print(start)
    end = initial_end
    print(start)
    #break here 
    task += 1