# WarmWaterMasses

## the MHW-WMT workflow 
last update: 12/24

The MHW-WMT project starts in notebooks:

[01_compute-climatology-compare-ocetrac.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/01_compute-climatology-compare-ocetrac.ipynb "01_compute-climatology-compare-ocetrac.ipynb") : calculates and saves out the climatology, percentiles and maximum using the full ~60 year run. There are also some exploratory plots of ocetrac sensitivity to radius and min_size_quartile

[01a_load-climatology-make-plot.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/01a_load-climatology-make-plot.ipynb "01a_load-climatology-make-plot.ipynb"): This makes the plot of the manso region climatologies and the fixed-location climatology for the paper

[02_run-ocetrac-v6.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/02_run-ocetrac-v6.py "02_run-ocetrac-v6.py") : Runs ocetrac v6 (min_size_quartile=0, and radius=1) and can be modified to run different tasks

[02_test-ocetrac-v5.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/02_test-ocetrac-v5.py "02_test-ocetrac-v5.py"): Pretty much the same as 02_run_ocetrac-v5.py but with a bunch of print statements to verify. It doesn't actually pass the Tracker.track() method

[03_relabel-ocetrac-v6-output.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/03_relabel-ocetrac-v6-output.ipynb "03_relabel-ocetrac-v6-output.ipynb"): Here the ocetrac output is relabeled and saved out as an .nc file. 

[04_visualize-ocetrac-blobs.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/04_visualize-ocetrac-blobs.ipynb "04_visualize-ocetrac-blobs.ipynb"): This makes plots with ocetrac blobs on different days for the paper

[04a_visualize-SST-evolution.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/04a_visualize-SST-evolution.ipynb "04a_visualize-SST-evolution.ipynb"): Makes plots of SST evolution and closes in on the GoM events

[05_make-dataframe-v6-script.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/05_make-dataframe-v6-script.py "05_make-dataframe-v6-script.py"): this script makes a dataframe with MHW metrics

[05a_collect_surf_stats.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/05a_collect_surf_stats.ipynb "05a_collect_surf_stats.ipynb"): here I am testing a built-in method in ocetrac (collect_surface_stats) to pass after running ocetrac and output a dataframe

[05b_make-dataframe-ocetrac-v5-t1.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/05b_make-dataframe-ocetrac-v5-t1.ipynb "05b_make-dataframe-ocetrac-v5-t1.ipynb"): This is the same as 05_make-dataframe-v6-script but as a notebook and using the ocetrac-v5 output

[06_mhw-intensity-duration-area-plot.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/06_mhw-intensity-duration-area-plot.ipynb "06_mhw-intensity-duration-area-plot.ipynb"): Makes area evolution and duration, intensity, area plots. I also started exploring some individual events that were long lasting but very small. 

[07_Z-Section-plots.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/07_Z-Section-plots.ipynb "07_Z-Section-plots.ipynb"): Makes subsurface plots for the paper. Now some things are messy since I am testing out different ways of visualizing the 3D thetao evolution

[08_compute-event-budgets.py](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/08_compute-event-budgets.py "08_compute-event-budgets.py"): Script to compute the individual MHW event budgets

[09_merge-mhw-budgets.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/09_merge-mhw-budgets.ipynb "09_merge-mhw-budgets.ipynb"): Merges the individual budgets to have a xr.Dataarray with the mhw id and its budgets

[09a_merge-mhw-budgets-mean-subtraction-method.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/09a_merge-mhw-budgets-mean-subtraction-method.ipynb "09a_merge-mhw-budgets-mean-subtraction-method.ipynb"):  Merges the individual budgets to have a xr.Dataarray with the mhw id and its budgets but using the anomaly method (did this before checking in with the UCI Ocean and Climate Dynamics group and realized we could use the time_bounds method instead)

[10_ReadWMTBudgets_Check.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/10_ReadWMTBudgets_Check.ipynb "10_ReadWMTBudgets_Check.ipynb"): this checks that the budgets were saved correctly. (Can skip)

[10a_exploring-budget-one-event.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/10a_exploring-budget-one-event.ipynb "10a_exploring-budget-one-event.ipynb"): Exploring just one event with ocetrac-v5 output(can skip)

[10b_explore-MHW25-budget-output.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/10b_explore-MHW25-budget-output.ipynb "10b_explore-MHW25-budget-output.ipynb"): Exploring just one event with ocetrac-v1 output (can skip, old)

[11_budgets-analysis-time-bounds-method.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/11_budgets-analysis-time-bounds-method.ipynb "11_budgets-analysis-time-bounds-method.ipynb"): The mhw vs manso budget analysis (have not used since MHW workshop because we don't have new budgets yet)

[11a_budgets-analysis-mean-subtraction-method.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/11a_budgets-analysis-mean-subtraction-method.ipynb "11a_budgets-analysis-mean-subtraction-method.ipynb"): The mhw vs manso budget analysis with the mean-subtraction method (not using anymore)

[11b_cumulative_mass_budget_plot-GRC.ipynb](https://github.com/marianatorres4/WarmWaterMasses/blob/main/notebooks/11b_cumulative_mass_budget_plot-GRC.ipynb "11b_cumulative_mass_budget_plot-GRC.ipynb"): code for the GRC M(t) plot
