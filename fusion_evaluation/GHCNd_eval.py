import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
# import geopandas as gpd
import os
# from shapely.geometry import Point
# from mpl_toolkits.basemap import Basemap, addcyclic
import matplotlib.pyplot as plt
# import xarray as xr
import json
from statsmodels.tsa.stattools import acf
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


folder_path = '/data/kas7897/camels_shapefiles_new'

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Load data from CSVs A and B into pandas DataFrames
crd_camels = pd.read_csv("/data/kas7897/dPLHBVrelease/hydroDL-dev/gages_list.csv")

crd_GHCN = pd.read_csv("/data/kas7897/GHCN_Data/Station_Data/Station_Metadata.csv")

lossfactor=0
smoothfactor=0
eva0 = np.load(f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/'
               f'allprcp_33withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Train19801001_19951001Test19951001_20081001Buff5478Staind5477/Eva50.npy', allow_pickle=True)

def trend_error(df_predicted, df_observed, column):
    # Resample to annual total precipitation
    annual_observed = df_observed.resample('A', on='Date').sum()
    annual_predicted = df_predicted.resample('A', on='Date').sum()

    # Calculate annual anomalies by subtracting the mean annual precipitation
    observed_mean = annual_observed['avg_prcp'].mean()
    predicted_mean = annual_predicted[column].mean()

    years = df_observed['Year'].unique()
    observed_anomalies = annual_observed['avg_prcp'] - observed_mean
    predicted_anomalies = annual_predicted[column] - predicted_mean
    slope_obs, intercept_obs, r_value_obs, p_value_obs, std_err_obs = linregress(years, observed_anomalies)
    slope_pred, intercept_pred, r_value_pred, p_value_pred, std_err_pred = linregress(years, predicted_anomalies)
    trend_error = slope_obs - slope_pred
    trend_error_percent_per_year = trend_error * 100

    return trend_error_percent_per_year

# names = [][]
# for filename in os.listdir(folder_path):
#     if filename.endswith('.shp'):
#         # Read the shapefile
#         shapefile_path = os.path.join(folder_path, filename)
#         gdf_shapefile = gpd.read_file(shapefile_path)
#         # crd_GHCN_geo = gpd.GeoDataFrame(crd_GHCN, crs=gdf_shapefile.crs, geometry=geometry)
#
#         gdf_shapefile = gdf_shapefile.to_crs(crd_GHCN_geo.crs)
#         # Perform spatial join
#         # joined_data = gpd.sjoin(crd_GHCN_geo, gdf_shapefile, how='left', op='within')
#
#         # Create a key for the dictionary using the shapefile name (without the '.shp' extension)
#         shapefile_name = os.path.splitext(filename)[0]
#         # names.append(os.path.splitext(filename)[0])
#         merged_polygon = gdf_shapefile.unary_union
#
#         # names.append(os.path.splitext(filename)[0])
#         grouped_coordinates[shapefile_name] = []
#
#         # Loop through each coordinate in df_coordinates
#         for idx, coord in crd_GHCN_geo.iterrows():
#             # Check if the coordinate falls within any polygon of the shapefile
#             if coord['geometry'].within(merged_polygon):
#                 grouped_coordinates[shapefile_name].append(coord['Station_ID'])
#                 print(coord['Station_ID'])
#         k=1

output_file_path = 'grouped_GHCN_crd.json'

# # Save the dictionary to a JSON file
# with open(output_file_path, 'w') as f:
#     json.dump(grouped_coordinates, f)

with open(output_file_path, 'r') as f:
    grouped_dict = json.load(f)

keys_to_remove = [key for key, value in grouped_dict.items() if isinstance(value, list) and len(value) == 0]

# Remove the keys with empty lists
for key in keys_to_remove:
    del grouped_dict[key]
stainds = list(grouped_dict.keys())

# staint = list(map(int, stainds))
# nse = eva0[0]['NSE'][crd_camels[crd_camels['gage'].isin(staint)].index]
# # high_NSE_indices = [index for index, value in enumerate(nse) if value > 0.7]
# low_NSE_indices = [index for index, value in enumerate(nse) if value < 0.7]
#
# low_stainds = [stainds[i] for i in low_NSE_indices]
# for key in low_stainds:
#     del grouped_dict[key]
# stainds = list(grouped_dict.keys())
# Define categories and datasets

lossfactor=0
smoothfactor=0
# expno=11
wghts_sdate = '1980-10-01'
def bias_meanflowratio_calc(pred,target):
    # ngrid,nt = pred.shape
    # Bias = np.full(ngrid, np.nan)
    # meanflowratio = np.full(ngrid, np.nan)
    # for k in range(0, ngrid):
    x = pred.reset_index(drop=True)
    y = target.reset_index(drop=True)
    ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
    if ind.shape[0] > 0:
        xx = x[ind]
        yy = y[ind]
        Bias = (np.sum(xx)-np.sum(yy))/(np.sum(yy)+0.00001)
        meanflowratio  = np.sum(xx)/(np.sum(yy)+0.00001)
    else:
        Bias=np.nan
        meanflowratio=np.nan

    return Bias, meanflowratio

def corr(pred, target):
    pred = pred.reset_index(drop=True)
    target = target.reset_index(drop=True)

    ind = np.where(np.logical_and(~np.isnan(pred), ~np.isnan(target)))[0]
    xx = pred[ind]
    yy = target[ind]
    return np.corrcoef(xx, yy)[0, 1]

def annual_bias_meanflowratio_calc(pred,target, yearstart, yearsend, time_allyear):
    Bias_ = 0
    mean_ = 0
    corr_ = 0
    time_allyear = pd.Index(time_allyear)
    nanyear=0
    for year in range(yearstart,yearsend):
        time_year = pd.date_range(f'{year}-10-01', f'{year+1}-09-30', freq='d')
        idx_start = time_allyear.get_loc(time_year[0])
        idx_end = time_allyear.get_loc(time_year[-1])

        year_Bias_,year_mean_ = bias_meanflowratio_calc(pred[idx_start:idx_end+1],target[idx_start:idx_end+1])
        year_corr = corr(pred[idx_start:idx_end+1],target[idx_start:idx_end+1])

        if np.isnan(year_Bias_):
            nanyear+=1
        else:
            Bias_ = Bias_ + year_Bias_
            corr_ = corr_+ year_corr
            mean_ = mean_+year_mean_

    nyear = yearsend-yearstart-nanyear
    Bias_ = Bias_/nyear
    mean_ = mean_/nyear
    corr_ = corr_/nyear
    if nanyear!=0:
        print(nanyear)
    return Bias_,mean_, corr_



w0 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_33withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts1_old.csv',
    header=None)[:9131]
# w0 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/daymet32withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts1.csv',
#     header=None)[:9131]
# w0 =  pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-SACSMA/"
#                   "ALLallprcp_withloss23smooth0/BuffOpt0/NSE_para0.25/111111/Fold1/"
#                   "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_18_Buff_365_Mul_16/prcp_wghts1.csv", header=None)[:9131]
# w0 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-Ensmble/"
#                  "ALLallprcp_withloss23smooth0/BuffOpt0/NSE/111111/Fold1/"
#                  "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_ensemble_Buff_365_Mul_16/prcp_wghts1.csv", header=None)[:9131]
# w0 = pd.read_csv('/data/kas7897/dPLHBVrelease/ensemble_fusion_results/lstm_sacsma_prms_hbv27/prcp_wghts1.csv', header=None)[:9131]
wghts_days = len(w0)
date_range_wghts = pd.date_range(start=wghts_sdate, periods=wghts_days)
w0['dates'] = date_range_wghts
w0['Day'] = w0['dates'].dt.day
w0['Month'] = w0['dates'].dt.month
w0['Year'] = w0['dates'].dt.year
w0.drop(columns=['dates'], inplace=True)


w1 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_33withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts2_old.csv',
    header=None)[:9131]
# w1 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/maurer_extended32withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts1.csv',
#     header=None)[:9131]

# w1 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-SACSMA/"
#                   "ALLallprcp_withloss23smooth0/BuffOpt0/NSE_para0.25/111111/Fold1/"
#                   "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_18_Buff_365_Mul_16/prcp_wghts2.csv", header=None)[:9131]
# w1 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-LSTM/"
#                "allprcp_withloss23smooth0/BuffOpt0/NSE/111111/Fold1/"
#                "T_19801001_19951001_BS_100_HS_256_RHO_365_Buff_365/prcp_wghts2.csv", header=None)[:9131]
# w1 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-Ensmble/"
#                  "ALLallprcp_withloss23smooth0/BuffOpt0/NSE/111111/Fold1/"
#                  "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_ensemble_Buff_365_Mul_16/prcp_wghts2.csv", header=None)[:9131]
# w1 = pd.read_csv('/data/kas7897/dPLHBVrelease/ensemble_fusion_results/lstm_sacsma_prms_hbv27/prcp_wghts2.csv', header=None)[:9131]

w1['dates'] = date_range_wghts
w1['Day'] = w1['dates'].dt.day
w1['Month'] = w1['dates'].dt.month
w1['Year'] = w1['dates'].dt.year
w1.drop(columns=['dates'], inplace=True)

# w2 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-LSTM/"
#                "allprcp_withloss23smooth0/BuffOpt0/NSE/111111/Fold1/"
#                "T_19801001_19951001_BS_100_HS_256_RHO_365_Buff_365/prcp_wghts3.csv", header=None)[:9131]
# w2 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-SACSMA/"
#                   "ALLallprcp_withloss23smooth0/BuffOpt0/NSE_para0.25/111111/Fold1/"
#                   "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_18_Buff_365_Mul_16/prcp_wghts3.csv", header=None)[:9131]
w2 = pd.read_csv(
    f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/allprcp_33withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts3_old.csv',
    header=None)[:9131]
# w2 = pd.read_csv(
#     f'/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-dPLHBV/ALL/TDTestforc/TD1_13/nldas_extended32withloss{lossfactor}smooth{smoothfactor}/BuffOpt0/RMSE_para0.25/111111/Fold1/T_19801001_19951001_BS_100_HS_256_RHO_365_NF_14_Buff_365_Mul_16/prcp_wghts1.csv',
#     header=None)[:9131]
# w2 = pd.read_csv("/data/kas7897/dPLHBVrelease/output/CAMELSDemo/LSTM-Ensmble/"
#                  "ALLallprcp_withloss23smooth0/BuffOpt0/NSE/111111/Fold1/"
#                  "T_19801001_19951001_BS_100_HS_256_RHO_365_NF_ensemble_Buff_365_Mul_16/prcp_wghts3.csv", header=None)[:9131]
# w2 = pd.read_csv('/data/kas7897/dPLHBVrelease/ensemble_fusion_results/lstm_sacsma_prms_hbv27/prcp_wghts3.csv', header=None)[:9131]


w2['dates'] = date_range_wghts
w2['Day'] = w2['dates'].dt.day
w2['Month'] = w2['dates'].dt.month
w2['Year'] = w2['dates'].dt.year
w2.drop(columns=['dates'], inplace=True)


daymet_test = np.empty((9131,4))
ghcn_test = np.empty((9131, 4))
multi_test = np.empty((9131, 4))
wdaymet_test = np.empty((9131, 4))
wmaurer_test = np.empty((9131, 4))
wnldas_test = np.empty((9131, 4))
# daymet_test = np.empty((8561,4))
# ghcn_test = np.empty((8561, 4))
# multi_test = np.empty((8561, 4))
basin_test = []

# Initialize dictionary for storing metrics
metrics_dict = {
    "rmse": {}, "acf": {}, "hrmse": {}, "lrmse": {}, "bias": {}, "abias": {}, 
    "corr": {}, "rmse_w": {}, "corr_w": {}, "trend": {}, "mean": {}, "std": {}
}

datasets = ["daymet", "maurer", "nldas", "multi", "avg", "ghcn"]
time_scales = ["", "_3day", "_month", "_yearly", "_high"]

for dataset in datasets:
    for scale in time_scales:
        for metric in metrics_dict:
            metrics_dict[metric][f"{dataset}{scale}"] = np.empty(len(grouped_dict))

# Helper function to load precipitation data
def load_prcp_data(dataset, huc, st_ind):
    base_path = "/scratch/Camels/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing"
    file_path = f"{base_path}/{dataset}_extended/{huc}/{st_ind}_lump_{dataset}_forcing_leap.txt"
    prcp_df = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=4)
    prcp_df.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 5: f'precip_{dataset}'}, inplace=True)
    prcp_df.drop(columns=[3, 4, 6, 7, 8, 9, 10], inplace=True)
    return prcp_df

# Processing Loop
for i, st_ind in enumerate(stainds):
    camels_index = crd_camels[crd_camels['gage'] == int(st_ind)].index
    huc = str(crd_camels.loc[crd_camels['gage'] == int(st_ind), 'huc'].values[0]).zfill(2)

    # Load precipitation datasets
    prcp_daymet = load_prcp_data("daymet", huc, st_ind)
    prcp_maurer = load_prcp_data("maurer", huc, st_ind)
    prcp_nldas = load_prcp_data("nldas", huc, st_ind)

    # Merge with weighted station data
    w_stations = {"daymet": w0, "maurer": w1, "nldas": w2}
    for dataset, w_data in w_stations.items():
        prcp = eval(f"prcp_{dataset}")
        prcp = prcp.merge(w_data[[camels_index[0], 'Year', 'Month', 'Day']], on=['Year', 'Month', 'Day'], how='inner')
        exec(f"prcp_{dataset} = prcp")

    # Merge with GHCN station data
    ghcn_stations = grouped_dict[st_ind]
    for j in ghcn_stations:
        st_data = pd.read_csv(f"/data/kas7897/GHCN_Data/Station_Data/{j}.csv")
        for dataset in ["daymet", "maurer", "nldas"]:
            prcp = eval(f"prcp_{dataset}")
            prcp = prcp.merge(st_data, on=['Year', 'Month', 'Day'], how='left')
            prcp.drop(columns=['TMin', 'TMax'], inplace=True)
            prcp.rename(columns={'Precip': j}, inplace=True)
            prcp.replace(-9999, np.nan, inplace=True)
            exec(f"prcp_{dataset} = prcp")

    # Compute avg precipitation
    prcp_daymet['avg_prcp'] = prcp_daymet.iloc[:, -len(ghcn_stations):].mean(axis=1, skipna=True)

    # Compute weighted precipitation
    for dataset in ["daymet", "maurer", "nldas"]:
        prcp = eval(f"prcp_{dataset}")
        prcp[f'wght_prcp'] = prcp[f'precip_{dataset}'] * prcp[camels_index[0]]
        exec(f"prcp_{dataset} = prcp")

    # Multi-model precipitation
    prcp_avg = (prcp_daymet['precip_daymet'] + prcp_maurer['precip_maurer'] + prcp_nldas['precip_nldas']) / 3
    multi_prcp = prcp_daymet['wght_prcp'] + prcp_maurer['wght_prcp'] + prcp_nldas['wght_prcp']

    # Store values
    prcp_daymet['multi_prcp'] = multi_prcp
    prcp_daymet['precip_average'] = prcp_avg

    # Compute Monthly Aggregations
    prcp_daymet['Date'] = pd.to_datetime(prcp_daymet[['Year', 'Month', 'Day']])
    true_prcp_month = prcp_daymet.groupby(pd.PeriodIndex(prcp_daymet['Date'], freq="M"))['avg_prcp'].mean().dropna()
    fusion_prcp_month = prcp_daymet.groupby(pd.PeriodIndex(prcp_daymet['Date'], freq="M"))['multi_prcp'].mean()[true_prcp_month.index]

    # Compute Bias and Correlation Metrics
    for dataset in ["daymet", "maurer", "nldas", "avg"]:
        prcp_month = prcp_daymet.groupby(pd.PeriodIndex(prcp_daymet['Date'], freq="M"))[f'precip_{dataset}'].mean()[true_prcp_month.index]
        metrics_dict["bias"][f"{dataset}_month"][i] = np.nanmean(prcp_month - true_prcp_month)
        metrics_dict["abias"][f"{dataset}_month"][i] = np.nanmean(np.abs(prcp_month - true_prcp_month))
        metrics_dict["corr"][f"{dataset}_month"][i] = np.corrcoef(prcp_month, true_prcp_month)[0, 1]

    # Compute Rolling 3-Day Mean
    true_prcp_3day = prcp_daymet['avg_prcp'].rolling(3).mean().dropna()
    multi_prcp_3day = multi_prcp.rolling(3).mean()[true_prcp_3day.index]
    prcp_avg_3day = prcp_avg.rolling(3).mean()[true_prcp_3day.index]

    # Compute RMSE, Bias, and Correlation for 3-Day Mean
    for dataset in ["multi", "avg", "daymet", "maurer", "nldas"]:
        prcp_3day = eval(f"prcp_{dataset}_3day")
        metrics_dict["rmse"][f"{dataset}_3day"][i] = np.sqrt(np.nanmean((prcp_3day - true_prcp_3day) ** 2))
        metrics_dict["corr"][f"{dataset}_3day"][i] = np.corrcoef(prcp_3day, true_prcp_3day)[0, 1]
        metrics_dict["bias"][f"{dataset}_3day"][i] = np.nanmean(prcp_3day - true_prcp_3day)
        metrics_dict["abias"][f"{dataset}_3day"][i] = np.nanmean(np.abs(prcp_3day - true_prcp_3day))

    # Compute Mean and Standard Deviation for 3-Day Mean
    for dataset in ["multi", "daymet", "maurer", "nldas", "avg"]:
        prcp_3day = eval(f"prcp_{dataset}_3day")
        metrics_dict["mean"][f"{dataset}_3day"][i] = np.nanmean(prcp_3day)
        metrics_dict["std"][f"{dataset}_3day"][i] = np.nanstd(prcp_3day)

    # Compute ACF
    for dataset in ["multi", "daymet", "maurer", "nldas", "avg"]:
        prcp = eval(f"prcp_{dataset}")
        metrics_dict["acf"][dataset][i] = acf(prcp[f'precip_{dataset}'], nlags=1)[1]


# Performance evaluation: RMSE, Bias, and Correlation
a = np.mean(metrics_dict["rmse"]["multi_3day"] < metrics_dict["rmse"]["daymet_3day"])
print(f'3D MA Fused better for {a:.2%} RMSE')

b = np.mean(metrics_dict["abias"]["multi_3day"] < metrics_dict["abias"]["daymet_3day"])
print(f'3D MA Fused better for {b:.2%} ABias')

c = np.mean(metrics_dict["corr"]["multi_3day"] > metrics_dict["corr"]["daymet_3day"])
print(f'3D MA Fused better for {c:.2%} Correlation')

# Create correlation DataFrame
corr_df = pd.DataFrame({
    'Daymet': metrics_dict["corr"]["daymet_3day"],
    'Maurer': metrics_dict["corr"]["maurer_3day"],
    'NLDAS': metrics_dict["corr"]["nldas_3day"],
    'Average P': metrics_dict["corr"]["avg_3day"],
    'Fused P': metrics_dict["corr"]["multi_3day"]
})

# Identify which source has the highest correlation for each basin
best_sources = corr_df.idxmax(axis=1)

# Calculate the percentage of basins where each source has the highest correlation
best_percentage = best_sources.value_counts(normalize=True) * 100

# Plot correlation comparison
best_percentage.plot(kind='bar', alpha=0.7)
plt.xlabel('Source', fontsize=16)
plt.ylabel('Percentage of Basins with Best Correlation', fontsize=16)
plt.title('(a) Best Correlation by Source Across Basins', fontsize=18)
plt.xticks(rotation=0, fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('/home/kas7897/final_plots_fusion_paper/corr_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplot for Bias Comparisons
fig, ax = plt.subplots(figsize=(12, 6))
data = [
    [metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["daymet_3day"], metrics_dict["bias"]["multi_3day"]],
    [metrics_dict["bias"]["avg_high"], metrics_dict["bias"]["daymet_high"], metrics_dict["bias"]["multi_high"]]
]
positions = [1, 2, 3, 5, 6, 7]
batch_labels = ['Average', 'Daymet', 'Fused'] * 2
super_labels = ['0-3mm/day', '3-5mm/day']

ax.boxplot(sum(data, []), positions=positions, widths=0.6, showfliers=False)
ax.set_xticks(positions)
ax.set_xticklabels(batch_labels)

for i, label in enumerate(super_labels):
    ax.text(i * 4 + 2, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, label,
            horizontalalignment='center', fontsize=12, weight='bold')

ax.axhline(y=0, color='gray', linestyle='--')
ax.set_xlabel('Precipitation Categories')
ax.set_ylabel('Bias (mm/day)')
ax.set_title('Bias Boxplots for Different Precipitation Ranges')
plt.tight_layout()
plt.show()

# Scatter Plot Comparisons
def ghcn_comparison_plot(metric1, metric2, metric3, title, statfunc):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"][metric1], marker='o', color='red',
               label=f'Daymet ({statfunc.__name__} {metric3}: {statfunc(metrics_dict["abias"][metric1]):.3f}, '
                     f'{statfunc.__name__} {metric2}: {statfunc(metrics_dict["corr"][metric1]):.3f})')

    ax.scatter(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"][metric2], marker='s', color='green',
               label=f'Maurer ({statfunc.__name__} {metric3}: {statfunc(metrics_dict["abias"][metric2]):.3f}, '
                     f'{statfunc.__name__} {metric2}: {statfunc(metrics_dict["corr"][metric2]):.3f})')

    ax.scatter(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["multi_3day"], marker='*', color='yellow',
               edgecolors='black', linewidths=0.8, s=100,
               label=f'Fused P ({statfunc.__name__} {metric3}: {statfunc(metrics_dict["abias"]["multi_3day"]):.3f}, '
                     f'{statfunc.__name__} {metric2}: {statfunc(metrics_dict["corr"]["multi_3day"]):.3f})')

    ax.plot(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["avg_3day"], color='gray', linestyle='--',
            label=f'Avg P ({statfunc.__name__} {metric3}: {statfunc(metrics_dict["abias"]["avg_3day"]):.3f}, '
                  f'{statfunc.__name__} {metric2}: {statfunc(metrics_dict["corr"]["avg_3day"]):.3f})')

    ax.set_xlabel(f'GHCN {metric2} with Avg P', size=16)
    ax.set_ylabel(f'GHCN {metric2} with P datasets', size=16)
    ax.set_title(f'GHCN {metric2} Scatter Plot', size=18)
    ax.legend(loc='best', prop={'size': 10})

    fig.tight_layout()
    plt.savefig(f'/home/kas7897/final_plots_fusion_paper/final_{metric1}_{metric2}_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate comparison plots
ghcn_comparison_plot("daymet_3day", "maurer_3day", "abias_3day", "GHCN Comparison (3-Day Moving Average)", np.median)
ghcn_comparison_plot("daymet_high", "maurer_high", "abias_high", "High Precipitation GHCN Comparison", np.mean)
ghcn_comparison_plot("daymet_month", "maurer_month", "abias_month", "GHCN Comparison Monthly", np.median)
ghcn_comparison_plot("daymet_yearly", "maurer_yearly", "abias_yearly", "GHCN Comparison Yearly", np.median)



# **Function for Plotting Residuals**
def plot_residuals(metric_avg, metric, label, color, ax):
    model = LinearRegression()
    metric_avg = np.array(metric_avg).reshape(-1, 1)
    model.fit(metric_avg, metric)
    trend = model.predict(metric_avg)
    residuals = metric - trend
    ax.scatter(metric_avg, residuals, label=label, color=color, edgecolors='black')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Metric Avg')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residuals after Removing Linear Trend ({label})')
    ax.legend()

# **Plot Residuals for Absolute Bias**
fig, ax = plt.subplots(figsize=(10, 6))
plot_residuals(metrics_dict["abias"]["avg_3day"], metrics_dict["abias"]["daymet_3day"], 'Daymet', 'red', ax)
plot_residuals(metrics_dict["abias"]["avg_3day"], metrics_dict["abias"]["maurer_3day"], 'Maurer', 'green', ax)
plot_residuals(metrics_dict["abias"]["avg_3day"], metrics_dict["abias"]["nldas_3day"], 'NLDAS', 'blue', ax)
plot_residuals(metrics_dict["abias"]["avg_3day"], metrics_dict["abias"]["multi_3day"], 'Fused P', 'yellow', ax)
plt.title('Absolute Bias')
fig.tight_layout()
plt.show()

# **Plot Residuals for Bias**
fig, ax = plt.subplots(figsize=(10, 6))
plot_residuals(metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["daymet_3day"], 'Daymet', 'red', ax)
plot_residuals(metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["maurer_3day"], 'Maurer', 'green', ax)
plot_residuals(metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["nldas_3day"], 'NLDAS', 'blue', ax)
plot_residuals(metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["multi_3day"], 'Fused P', 'yellow', ax)
plt.title('Bias')
fig.tight_layout()
plt.show()

# **Plot Residuals for Correlation**
fig, ax = plt.subplots(figsize=(10, 6))
plot_residuals(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["daymet_3day"], 'Daymet', 'red', ax)
plot_residuals(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["maurer_3day"], 'Maurer', 'green', ax)
plot_residuals(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["nldas_3day"], 'NLDAS', 'blue', ax)
plot_residuals(metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["multi_3day"], 'Fused P', 'yellow', ax)
plt.title('Correlation')
fig.tight_layout()
plt.show()

# **Boxplot Comparisons for Correlation and Bias**
group1 = [
    metrics_dict["corr"]["avg_3day"], metrics_dict["corr"]["daymet_3day"],
    metrics_dict["corr"]["maurer_3day"], metrics_dict["corr"]["nldas_3day"], metrics_dict["corr"]["multi_3day"]
]
group2 = [
    metrics_dict["bias"]["avg_3day"], metrics_dict["bias"]["daymet_3day"],
    metrics_dict["bias"]["maurer_3day"], metrics_dict["bias"]["nldas_3day"], metrics_dict["bias"]["multi_3day"]
]

plt.figure(figsize=(10, 5))
plt.boxplot(group1, labels=['Average', 'Daymet', 'Maurer', 'NLDAS', 'Fusion'], showfliers=False)
plt.title('Correlation')
plt.ylabel('Value')
plt.xlabel('Datasets')
plt.show()

plt.figure(figsize=(10, 5))
plt.boxplot(group2, labels=['Average', 'Daymet', 'Maurer', 'NLDAS', 'Fusion'], showfliers=False)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Bias')
plt.ylabel('Value')
plt.xlabel('Datasets')
plt.show()

# **Boxplots for Mean, Standard Deviation, and Autocorrelation**
group1 = [
    metrics_dict["mean"]["avg_3day"], metrics_dict["mean"]["multi_3day"],
    metrics_dict["mean"]["daymet_3day"], metrics_dict["mean"]["maurer_3day"], metrics_dict["mean"]["nldas_3day"]
]
group2 = [
    metrics_dict["std"]["avg_3day"], metrics_dict["std"]["multi_3day"],
    metrics_dict["std"]["daymet_3day"], metrics_dict["std"]["maurer_3day"], metrics_dict["std"]["nldas_3day"]
]
group3 = [
    metrics_dict["acf"]["avg"], metrics_dict["acf"]["multi"],
    metrics_dict["acf"]["daymet"], metrics_dict["acf"]["maurer"], metrics_dict["acf"]["nldas"]
]

fig, ax = plt.subplots(3, 1, figsize=(5, 12))
fig.subplots_adjust(hspace=0.4)
colors = ['red', 'blue', 'green', 'yellow', 'violet']

# **Mean Boxplot**
ax[0].boxplot(group1, patch_artist=True)
for patch, color in zip(ax[0].artists, colors):
    patch.set_facecolor(color)
ax[0].set_xticklabels(["Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"])
ax[0].set_title("(a) Mean (3-day moving average)", size=12)

# **Standard Deviation Boxplot**
ax[1].boxplot(group2, patch_artist=True)
for patch, color in zip(ax[1].artists, colors):
    patch.set_facecolor(color)
ax[1].set_xticklabels(["Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"])
ax[1].set_title("(b) Standard Deviation (3-day moving average)", size=12)

# **Autocorrelation Boxplot**
ax[2].boxplot(group3, patch_artist=True)
for patch, color in zip(ax[2].artists, colors):
    patch.set_facecolor(color)
ax[2].set_xticklabels(["Avg P", "Fused P", "Daymet", "Maurer", "NLDAS"])
ax[2].set_title("(c) Autocorrelation (lag=1)", size=12)

plt.show()


