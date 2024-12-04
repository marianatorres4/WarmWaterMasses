import dask
import pandas as pd
import netCDF4
import xarray as xr
import numpy as np
import dask.array as da
import warnings
warnings.filterwarnings('ignore')
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print('loaded libraries')

dir_path = "/pub/hfdrake/datasets/CM4_MHW_blobs/"
mt_path = "/pub/mariant3/WarmWaterMasses/data/"
ds = xr.open_dataset(f"{dir_path}/data/ocean_daily_cmip.01860101-01901231.tos.nc", chunks={'time':100}).sel(xh=slice(-130, -70), yh=slice(8, 38))
#ds_region = ds.sel(xh=slice(-130, -70), yh=slice(8, 38)) #.sel(xh = slice(-138, 0), yh = slice(8, 49))
static = xr.open_dataset(f"{dir_path}/data/ocean_daily_cmip.static.nc").sel(xh=slice(-130, -70), yh=slice(8, 38))
#static_region = static.sel(xh=slice(-130, -70), yh=slice(8, 38))
#ds_static_region = xr.merge([static_region,ds_region])
#labels = xr.open_dataset("/pub/mariant3/WarmWaterMasses/data/01860315-01891101-ocetrac-v3-labels-region.nc")
labels = xr.open_mfdataset(f"{mt_path}/ocetracv6/ocetrac-v6-processed/ocetrac-v6-blobs-tos-t1-*manso.nc", chunks={'time':1}).blobs.load()

ds_labels_region = xr.merge([ds,labels],join='inner')
ds_labels_static_region = xr.merge([static,ds_labels_region],join='inner') #previously ds_static_region

ids = np.unique(labels)
ids = np.array([id for id in ids if ~np.isnan(id)])

num_events = len(ids)
num_events

for id in ids:
    #print(i)
    event = labels.where(labels==id, drop=True)
    #print(event)
    event_mask = event == event

ssta = ds_labels_static_region.tos

import os, glob
import xarray as xr
import numpy as np
import pandas as pd
import warnings
from skimage.measure import label as label_np
from skimage.measure import regionprops 

try:
    from tqdm.auto import tqdm
    tqdm_avail = True
except:
    warnings.warn(
        "Optional dependency `tqdm` not found. This will make progressbars a lot nicer. \
    Install with `conda install -c conda-forge tqdm`"
    )
    tqdm_avail = False
    
def _wrap(labels):
    ''' Impose periodic boundary and wrap labels, then reorder the labels'''
    first_column = labels[..., 0]
    last_column = labels[..., -1]

    stacked = first_column.stack(z=['time','yh'])
    unique_first = np.unique(stacked[stacked.notnull()])

    # This loop iterates over the unique values in the first column, finds the location of those values in 
    # the first columnm and then uses that index to replace the values in the last column with the first column value
    for i in enumerate(unique_first):
        first = np.where(first_column == i[1])
        last = last_column[first[0], first[1]]
        stacked = last.stack(z=['time','yh'])
        bad_labels = np.unique(stacked[stacked.notnull()])
        replace = np.isin(labels, bad_labels)
        labels = labels.where(replace==False, other=i[1])
    
    labels = labels.fillna(0)
    labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
    labels_wrapped = xr.DataArray(labels_wrapped, dims=labels.dims, coords=labels.coords)

    return labels_wrapped

def _get_labels(binary_images):
    '''function used to label binary images at each time step using skimage.measure.label'''
    blobs_labels = label_np(binary_images, background=0)
    return blobs_labels
    
def _get_centroids(sub_labels):
    '''This function uses skimage.measure.regionprops to find the centroids of objects assigned 
    to each unique label'''
    props = regionprops(sub_labels.astype('int'))
    centroids = [(float(sub_labels.yh[round(p.centroid[0])].values),
                  float(sub_labels.xh[round(p.centroid[1])].values)) for p in props]
    for i in range(0,len(centroids)):
        if centroids[i][1] >= 359.875:
            centroids[i] = (centroids[i][0], list(centroids[i])[1] - 359.875)
    
    return centroids

def _get_intensity_area(event, ssta, mhw):
    '''Calculates event intensities and area at each time interval using anomaly data and coordinates 
    cooresponding to the event.'''
    
    event_ssta = ssta.where(event>0, drop=True)        
    event_ssta_mask = event_ssta == event_ssta
    #mhw['intensity_mean'].append(event_ssta.mean(('yh','xh')).values)
    #mhw['intensity_max'].append(event_ssta.max(('yh','xh')).values) 
    #mhw['intensity_min'].append(event_ssta.min(('yh','xh')).values)
    mhw['intensity_mean'].append((ds_labels_static_region.tos).where(event_ssta_mask).mean(['xh','yh']).values)
    mhw['intensity_min'].append((ds_labels_static_region.tos).where(event_ssta_mask).min(['xh','yh']).values)
    mhw['intensity_max'].append((ds_labels_static_region.tos).where(event_ssta_mask).max(['xh','yh']).values)
   
    #event_temperature_max = (ds_static_region.tos).where(event_mask).sel(xh = slice(-138, 0), yh = slice(8, 49)).max(['xh','yh'])
 
    mhw['intensity_cumulative'].append(np.nansum(event_ssta))
    coords = event.stack(z=('yh','xh'))
    coord_pairs = [(coords.isel(time=t[0]).dropna(dim='z', how='any').z.yh.values, 
                      coords.isel(time=t[0]).dropna(dim='z', how='any').z.xh.values) for t in enumerate(event.time)]

    mhw['coords'].append(coord_pairs)
    
    # Calculate weighted cell area assuming 0.25º resolution data and 111 km per degree of latitude
    
    #for id in ids:
        #print(id)
        #print(i)
        #events = labels.where(labels==id, drop=True)
        #event_mask = events == id
   # event_area
    #event_mask = events == event
    event_ssta_mask = event_ssta == event_ssta
    #print(event_ssta == event_ssta)
    #print(event_ssta.areacello.sum(('yh','xh')).values)
    event_area = (ds_labels_static_region.areacello).where(event_ssta_mask).sum(['xh','yh'])
    # event_area = (ds_static_region.areacello).where(event_ssta_mask).sel(xh = slice(-138, 0), yh = slice(8, 49)).sum(['xh','yh'])
    #print(event_area.values)
    cell_area = event_area.values
    mhw['area'].append(cell_area)

    #y, x = zip(*coord_pairs)
    #dlon = [np.cos(y[c]*np.pi/180)*(111*.25) for c in np.arange(0, len(coord_pairs))]
    #dlat = (111*.25) * np.ones(len(dlon))
    #cell_area = [np.sum(dlon[c]*dlat[c]) for c in np.arange(0, len(coord_pairs))]
    #mhw['area'].append(cell_area)

    
    return mhw

def _get_metrics(event, ssta):
    '''
    Creates a Pandas DataFrame of event attributes.
    
    Parameters
    ----------
      event : xarray.DataArray   
              Image set containing only objects corresponding to the event of interest. 
              Dimensions should be ('time', 'lat', 'lon')
              
      ssta  : xarray.DataArray
              Temperature vector [1D numpy array of length T]
    
    Returns
    -------
    
    mhw : pandas.DataFrame
          Marine heat wave event attributes. The keys listed below are 
          are contained within the dataset.
 
        'id'                     Unique label given to the MHW [int]
        'date'                   Dates corresponding to the event [datetime format]
        'coords'                 Latitude and longitude of all points contained in the event [tuple(lat,lon)]
        'centroid'               Center of each object contained in the event [tuple(lat,lon)]
        'duration'               Duration of event [months]
        'intensity_max'          Maximum intensity at each time interval [degC]
        'intensity_mean'         Mean intensity at each time interval [degC]
        'intensity_min'          Minimum intensity at each time interval [degC]
        'intensity_cumulative'   Cumulated intensity over the entire event [degC months]
        'area'                   Area of the event at each time interval [km2]
        
    '''
    
    # Initialize dictionary 
    mhw = {}
    mhw['id'] = [] # event label
    mhw['date'] = [] # datetime format
    mhw['coords'] = [] # (lat, lon)
    mhw['centroid'] = []  # (lat, lon)
    mhw['duration'] = [] # [months]
    mhw['intensity_max'] = [] # [deg C]
    mhw['intensity_mean'] = [] # [deg C]
    mhw['intensity_min'] = [] # [deg C]
    mhw['intensity_cumulative'] = [] # [deg C]
    mhw['area'] = [] # [km2]

    # TO ADD:
    # mhw['rate_onset'] = [] # [deg C / month]
    # mhw['rate_decline'] = [] # [deg C / month]

    mhw['id'].append(int(np.nanmedian(event.values)))
    mhw['date'].append(event.time.values.astype('datetime64[D]'))
    mhw['duration'].append(event.time.shape[0])

    # Turn images into binary
    binary_event = event.where(event>=0, other=0)
    binary_event = binary_event.where(binary_event==0, other=1)
      
    sub_labels = xr.apply_ufunc(_get_labels, binary_event,
                                input_core_dims=[['yh', 'xh']],
                                output_core_dims=[['yh', 'xh']],
                                output_dtypes=[binary_event.dtype],
                                vectorize=True,
                                dask='parallelized')
    
    # Turn background to NaNs
    sub_labels = xr.DataArray(sub_labels, dims=binary_event.dims, coords=binary_event.coords)
    sub_labels = sub_labels.where(sub_labels>0, drop=False, other=np.nan) 

    # The labels are repeated each time step, therefore we relabel them to be consecutive
    for p in range(1, sub_labels.shape[0]):
        sub_labels[p,:,:] = sub_labels[p,:,:].values + sub_labels[p-1,:,:].max().values
    
    sub_labels_wrapped = _wrap(sub_labels)
    
    mhw = _get_intensity_area(event, ssta, mhw)
    
 # '  centroid = []
 #   for s in np.arange(0, sub_labels_wrapped.shape[0]):
 #       lx = sub_labels_wrapped.isel(time=s)
 #       east = lx.where(lx.xh < 180, drop=True)
 #       east['xh'] = labels.xh
 #       append_east = xr.concat([lx.where(lx.xh >= 180, drop=True), east], dim="xh")
#        centroid.append(_get_centroids(append_east))
#    mhw['centroid'].append(centroid)'
    
    mhw = pd.DataFrame(dict([(name, pd.Series(data)) for name,data in mhw.items()]))
#     mhw.to_csv('df_'+str(mhw['id'].values[0]).zfill(4)+'.csv', index=False)
    return mhw

dataframes = []  
for i in range(0,num_events):
    #print(ids[i])
    event = labels.where(labels==ids[i], drop=True)
    df = _get_metrics(event, ssta)
    dataframes.append(df)

dff = pd.concat(dataframes, ignore_index=True)
dff.to_pickle(f"{mt_path}ocetracv6/ocetrac-v6-processed/ocetrac-v6-blobs-tos-t1-df-r1-msq0-0186-03-15-0189-12-14.pkl")
