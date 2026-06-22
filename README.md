# The Warm Water Mass Transformation Recipe: 

This repo contains the code to compute high-frequency water mass transformation (WMT) budgets for individually tracked marine heatwaves.

![Warm Water Mass Transformation Workflow](figures/WWMWFv11.png)

-  [01_eulerainDetection.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/01_eulerainDetection.ipynb): We start here by load the CM4 (NOAA GFDL) data from a 100-year pre-industrial experiment. Initial MHW detection is using an Eulerian approach. 
- [02_runOcetracv9.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/scripts/02_runOcetracv9.py): is a script used to run the tracker for the four years. The script is set up to run different configurations of ocetrac for our sensitivity analysis. The final analysis uses the ocetrac output with `min_size_quartile` and `radius` set to `0`. 
	- A substep here is needing to relabel the ocetrac output using [03_relabelOcetracv9.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/03_relabelOcetracv9.ipynb) since it was run only for a subperiod of each year (for computational/storage efficiency).
- [04_visualizeOcetrac](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/04_visualizeOcetrac.ipynb): here we visualize the labels or individual MHWs tracked. These blobs are the individual MHWs tracked. A gif if each blob and their SST evolution is available [here](https://github.com/marianatorres4/WarmWaterMasses/blob/main/figures/full_mhwv3.mp4)
- [05_visualizeSubsurfaceMHWs.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/05_visualizeSubsurfaceMHWs.ipynb): After tracking, we project each heatwave and see their depth-time evolution. You can also check out this [movie](https://github.com/marianatorres4/WarmWaterMasses/blob/main/figures/warm_layer_8fps.gif) to see the the depth extent evolution for each MHW tracked. 
- [06_mhwMetrics.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/scripts/06_mhwMetrics.py): Here we use this script to build  `mhwMetrics.nc` which contains the surface and subsurface statistics for each heatwave. 
- [07_plotDuration-intesity.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/07_plotDuration-intesity.ipynb) and [08_plotMetrics-timeseries.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/08_plotMetrics-timeseries.ipynb)both plot the metrics. 
- [09_compute-wmt-budget-7day-rolling-mask.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/scripts/09_compute-wmt-budget-7day-rolling-mask.py): is used to compute 7-day dynamic budgets for each MHW tracked. 
- Finally we use [13_plotWWMT.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/13_plotWWMT.ipynb) for the WMT budget analysis.
