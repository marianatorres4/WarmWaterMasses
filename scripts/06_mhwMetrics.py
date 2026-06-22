print('load libraries')
import datetime
import xarray as xr
import numpy as np
import dask
import pandas as pd
import xwmb
import xwmt
import xgcm
import warnings
warnings.filterwarnings('ignore')
import cftime

path = "/pub/hfdrake/datasets/CM4_MHW_blobs"
hfdrake_path = f"{path}/data_daily"
mt_path = "/pub/mariant3/WarmWaterMasses"
output_path = f"{mt_path}/data"

ds = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily.*.nc", chunks={"time":1})
ds = ds.isel(yh=slice(1, None), yq=slice(None, -1), xh=slice(1,None), xq=slice(None, -1))

snap = xr.open_mfdataset(f"{hfdrake_path}/*.ocean_daily_snap*.nc", chunks={"time":1})
snap = snap.rename({
    'time': 'time_bounds',
    **{v: f"{v}_bounds" for v in snap.data_vars}
})

static = xr.open_dataset(f"{path}/data/WMT_monthly/ocean_month_rho2.static.nc")

ocetrac_labels = xr.open_dataset(f"{mt_path}/data/ocetracv9/ocetrac-v9-blobs-tos-t1-r1-msq0-01860315-01891214-region.nc", chunks={'time':1})

labels = ocetrac_labels.blobs.rename('event_mask').load()
wmt = xr.open_dataset(f'{mt_path}/data/WMT_data/0186-0189_wmt-daily.nc').sel(time=slice("0186","0189"))

xh_min, xh_max = ocetrac_labels.xh.min().values, ocetrac_labels.xh.max().values
yh_min, yh_max = ocetrac_labels.yh.min().values, ocetrac_labels.yh.max().values

ds_region = xr.open_dataset(f"{path}/data/ocean_daily_cmip.01860101-01901231.tos.nc", 
                           chunks={'time':100}).sel(xh=slice(xh_min, xh_max),
                                                  yh=slice(yh_min, yh_max))

static_region = xr.open_dataset(f"{path}/data/ocean_daily_cmip.static.nc").sel(
    xh=slice(xh_min, xh_max), 
    yh=slice(yh_min, yh_max)
)

ds_labels_region = xr.merge([ds_region, labels], join='inner')
ds_labels_static_region = xr.merge([static_region, ds_labels_region], join='inner')
areacello = ds_labels_static_region.areacello

ids = np.unique(labels)
ids = [int(id) for id in ids if ~np.isnan(id)]

wmt_og = wmt.copy()
wmt = wmt['time']
dsnovar = xr.Dataset(wmt.coords)

mhwdataset = []
for mhw in ids:
    print(mhw)
    event_mask = labels.where(labels == mhw, drop=True) == mhw
    event_time = event_mask.time
    
    ds_thetao = ds.thetao.sel(time=slice(event_time[0], event_time[-1]))
    thetao_mask = ds_thetao > 29
    event_thetao_mask = event_mask * thetao_mask
    
    volcello_event = event_thetao_mask * ds.volcello
    event_volume = volcello_event.sum(["xh", "yh", "zl"])
    thetao_vol_weight = volcello_event * ds_thetao
    volcello_weighted_thetao = thetao_vol_weight.sum(["xh", "yh", "zl"]) / volcello_event.sum(["xh", "yh", "zl"])
    
    event_mask_tos = ds_region.tos.where(event_mask > 0, drop=True)
    event_mask_tos = event_mask_tos == event_mask_tos
    event_area = areacello.where(event_mask_tos).sum(['xh', 'yh']).load()
    event_thickness = event_volume/event_area
    
    duration = event_time.shape[0]
    
    centroid_array = np.full((len(event_time), 2), np.nan)
    event_times = event_time.values
    
    for i, t in enumerate(event_times):
        event_slice = volcello_event.sel(time=t).sum(dim='zl')
        
        xcoords = event_slice.xh.values
        ycoords = event_slice.yh.values
        lon_grid, lat_grid = np.meshgrid(xcoords, ycoords)
        
        data_flat = event_slice.values.flatten()
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        
        if np.nansum(data_flat) == 0:
            centroid_x, centroid_y = np.nan, np.nan
        else:
            centroid_x = np.nansum(lon_flat * data_flat) / np.nansum(data_flat)
            centroid_y = np.nansum(lat_flat * data_flat) / np.nansum(data_flat)
            
        centroid_array[i, :] = [centroid_x, centroid_y]
    
    ds_weighted_thetao = xr.Dataset({
        'volcello_weighted_thetao': (['time'], np.full(len(wmt.time), np.nan)),
        'event_volume': (['time'], np.full(len(wmt.time), np.nan)),
        'event_area': (['time'], np.full(len(wmt.time), np.nan)),
        'event_thickness': (['time'], np.full(len(wmt.time), np.nan)),
        'centroid': (['time', 'xy'], np.full((len(wmt.time), 2), np.nan), 
                    {'xy': ['xh', 'yh']}),
        'duration': ([], duration)
    })
    
    start_idx = np.where(wmt.time == event_time[0])[0][0]
    ds_weighted_thetao['volcello_weighted_thetao'][start_idx:start_idx + len(volcello_weighted_thetao)] = volcello_weighted_thetao
    ds_weighted_thetao['event_volume'][start_idx:start_idx + len(event_volume)] = event_volume
    ds_weighted_thetao['event_area'][start_idx:start_idx + len(event_area)] = event_area
    ds_weighted_thetao['event_thickness'][start_idx:start_idx + len(event_thickness)] = event_thickness
    ds_weighted_thetao['centroid'][start_idx:start_idx + len(event_time), :] = centroid_array
    
    ds_weighted_thetao = ds_weighted_thetao.assign_coords(time=wmt.time)
    ds_weighted_thetao = ds_weighted_thetao.expand_dims({'mhw': [mhw]})
    ds_combined = xr.combine_by_coords([dsnovar, ds_weighted_thetao])
    mhwdataset.append(ds_combined)

mhw_weighted_thetao_dataset = xr.concat(mhwdataset, dim='mhw')
output_file = f'{output_path}/mhwMetrics.nc'
print(f"Saving output to {output_file}")
mhw_weighted_thetao_dataset.to_netcdf(output_file, mode='w')
print("Processing complete!")