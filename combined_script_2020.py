#!/usr/bin/env python3
# -*- coding: utf-8 -*-


####################################################################
# RANDOM FOREST ALGORITHM

import pandas as pd
import numpy as np
import torch
from darts.models import RandomForest  # Changed from TFTModel to RandomForest
from concurrent.futures import ThreadPoolExecutor, as_completed  
from darts import TimeSeries
from darts.metrics import rmse, mae
import pickle  
import statistics
from collections import ChainMap

# Set float32 matmul precision to 'medium'
torch.set_float32_matmul_precision('medium')  

torch.manual_seed(1)
np.random.seed(1)

# Step 1: Load the Dataset
print("Step 1: Loading the dataset.")
df = pd.read_csv('combined_output_1degree.csv')
df = df.fillna(method='ffill')
df['time'] = pd.to_datetime(df['time'])  

# Step 2: Apply Dynamic Data Types
print("Step 2: Applying dynamic data types.")
float_dtype = 'float32'
columns_to_convert = ['tmpsf','wnd10']
for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(float_dtype)
    else:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

# Step 3: Split the DataFrame
print("Step 3: Splitting the DataFrame.")
df_split = np.array_split(df, 252)

# Step 4: Create Multivariate TimeSeries
print("Step 4: Creating multivariate TimeSeries.")
series_list = []
columns_to_include = ['tmpsf','wnd10']
for i in range(len(df_split)):
    series_list.append(TimeSeries.from_dataframe(df_split[i], 'time', columns_to_include))

# Step 5: Split into Training and Testing Sets
print("Step 5: Splitting into training and testing sets.")
train_list = []
test_list = []
valid_list = []
for i in range(len(df_split)):
    series = series_list[i]
    train_series = series.slice(pd.Timestamp('1989-01-01'), pd.Timestamp('2018-12-31'))
    test_series = series.slice(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    valid_series = series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    train_list.append(train_series)
    test_list.append(test_series)
    valid_list.append(valid_series) 

# Step 6: Initialize the Model
print("Step 6: Initializing the model.")
model = RandomForest(
    lags=365,    
    n_estimators=100,
    n_jobs=50
)

# Step 7: Train the Model
print("Step 7: Training the model.")
model.fit(series=train_list)

# Step 8: Save the Trained Model
print("Step 8: Saving the trained model.")
pickle.dump(model, open('SIO_2020_model_RF.pkl', 'wb'))  # Changed filename

# Step 9: Load the Trained Model
print("Step 9: Loading the trained model.")
model = pickle.load(open('SIO_2020_model_RF.pkl', 'rb'))  # Changed filename

# Step 10: Define Prediction Function
print("Step 10: Defining the prediction function.")
def predict_wa(args):
    index, n = args
    try:
        val = model.predict(n=n, series=train_list[index])
        return {f'{index}': val}
    except Exception:
        return None  

# Step 11: Predict for 2019 (Testing Data)
print("Step 11: Predicting for 2019.")
results_2019 = []
all_args_2019 = [(index, 365) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2019}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2019.append(result)
        except Exception:
            pass 

new_res_dict_2019 = dict(ChainMap(*results_2019))
SIO_2019_predicted = []
for x in range(len(df_split)):
    SIO_2019_predicted.append(new_res_dict_2019[f'{x}'])  

# Step 12: Evaluate Metrics for 2019
print("Step 12: Evaluating metrics for 2019.")
rmse_dict_2019 = {var: [] for var in columns_to_include}
mae_dict_2019 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(test_list[i][var], SIO_2019_predicted[i][var])
            mae_value = mae(test_list[i][var], SIO_2019_predicted[i][var])
            rmse_dict_2019[var].append(rmse_value)
            mae_dict_2019[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2019 = {var: statistics.mean(rmse_dict_2019[var]) for var in columns_to_include}
mean_mae_2019 = {var: statistics.mean(mae_dict_2019[var]) for var in columns_to_include}

# Step 13: Predict for Unseen Data (2020)
print("Step 13: Predicting for 2020.")
results_2020 = []
all_args_2020 = [(index, 731) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2020}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2020.append(result)
        except Exception:
            pass

new_res_dict_2020 = dict(ChainMap(*results_2020))
SIO_2020_predicted = []
for x in range(len(df_split)):
    full_pred_series = new_res_dict_2020[f'{x}']
    pred_2020 = full_pred_series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    SIO_2020_predicted.append(pred_2020)

# Step 14: Evaluate Metrics for 2020
print("Step 14: Evaluating metrics for 2020.")
rmse_dict_2020 = {var: [] for var in columns_to_include}
mae_dict_2020 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(valid_list[i][var], SIO_2020_predicted[i][var])
            mae_value = mae(valid_list[i][var], SIO_2020_predicted[i][var])
            rmse_dict_2020[var].append(rmse_value)
            mae_dict_2020[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2020 = {var: statistics.mean(rmse_dict_2020[var]) for var in columns_to_include}
mean_mae_2020 = {var: statistics.mean(mae_dict_2020[var]) for var in columns_to_include}

# Step 15: Save Predictions and Evaluation Metrics for 2019
print("Step 15: Saving predictions and evaluation metrics for 2019.")
np_arr_res_2019 = []
for x in SIO_2019_predicted:
    np_arr_res_2019.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2019, open("SIO_2019_RandomForest_predicted_final_nparray.bin", 'wb'))  # Changed filename
# pickle.dump(rmse_dict_2019, open("SIO_2019_RandomForest_rmse_final_dict.bin", 'wb'))            # Changed filename
# pickle.dump(mae_dict_2019, open("SIO_2019_RandomForest_mae_final_dict.bin", 'wb'))              # Changed filename

# Step 16: Save Predictions and Evaluation Metrics for 2020
print("Step 16: Saving predictions and evaluation metrics for 2020.")
np_arr_res_2020 = []
for x in SIO_2020_predicted:
    np_arr_res_2020.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2020, open("SIO_2020_RandomForest_predicted_final_nparray.bin", 'wb'))  # Changed filename
# pickle.dump(rmse_dict_2020, open("SIO_2020_RandomForest_rmse_final_dict.bin", 'wb'))            # Changed filename
# pickle.dump(mae_dict_2020, open("SIO_2020_RandomForest_mae_final_dict.bin", 'wb'))              # Changed filename

# Step 17: Construct the Final Dataset for 2019
print("Step 17: Constructing the final dataset for 2019.")
df_orig19 = df.loc[df['time'] <= '2018-12-31'].copy()
df_orig19['time'] = pd.to_datetime(df_orig19['time'])

pred_dfs19 = []
for idx, pred_series in enumerate(SIO_2019_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs19.append(pred_df)

df_pred19 = pd.concat(pred_dfs19, ignore_index=True)
df_final19 = pd.concat([df_orig19, df_pred19])
df_final19['time'] = pd.to_datetime(df_final19['time'])
df_final19 = df_final19.set_index(["time", "lat", "lon"])
df_final19['sst'] = df_final19['tmpsf']
#df_final19.to_csv("SIO_2019_RandomForest_predicted_final.csv")  # Changed filename

# Step 18: Construct the Final Dataset for 2020
print("Step 18: Constructing the final dataset for 2020.")
df_orig20 = df.loc[df['time'] <= '2019-12-31'].copy()
df_orig20['time'] = pd.to_datetime(df_orig20['time'])

pred_dfs20 = []
for idx, pred_series in enumerate(SIO_2020_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs20.append(pred_df)

df_pred20 = pd.concat(pred_dfs20, ignore_index=True)
df_final20 = pd.concat([df_orig20, df_pred20])
df_final20['time'] = pd.to_datetime(df_final20['time'])
df_final20 = df_final20.set_index(["time", "lat", "lon"])
df_final20['sst'] = df_final20['tmpsf']
#df_final20.to_csv("SIO_2020_RandomForest_predicted_final.csv")  # Changed filename

# Step 19: Convert Final Dataset for 2019 to xarray Dataset
print("Step 19: Converting final dataset for 2019 to xarray Dataset.")
ds19 = df_final19.to_xarray()
pred_df_2019 = ds19.sst.sel(time=slice("2019-01-01", "2019-12-31")).to_dataframe()
pred_df_2019.to_csv("SIO_2019_SST_Predictions_RandomForest.csv")  # Changed filename

# Step 20: Convert Final Dataset for 2020 to xarray Dataset
print("Step 20: Converting final dataset for 2020 to xarray Dataset.")
ds20 = df_final20.to_xarray()
pred_df_2020 = ds20.sst.sel(time=slice("2020-01-01", "2020-12-31")).to_dataframe()
pred_df_2020.to_csv("SIO_2020_SST_Predictions_RandomForest.csv")  # Changed filename

# Step 21: Save Mean RMSE and MAE Metrics to CSV
print("Step 21: Saving mean RMSE and MAE metrics to CSV.")
metrics_2019 = []
for var in columns_to_include:
    metrics_2019.append({
        'Year': 2019,
        'Variable': var,
        'Mean_RMSE': mean_rmse_2019[var],
        'Mean_MAE': mean_mae_2019[var]
    })

metrics_2020 = []
if 'mean_rmse_2020' in locals() and 'mean_mae_2020' in locals():
    for var in columns_to_include:
        metrics_2020.append({
            'Year': 2020,
            'Variable': var,
            'Mean_RMSE': mean_rmse_2020[var],
            'Mean_MAE': mean_mae_2020[var]
        })

all_metrics = metrics_2019 + metrics_2020
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("evaluation_metrics_summary_RandomForest.csv", index=False)  # Changed filename

















####################################################################
# NBEATS ALGORITHM

import pandas as pd
import numpy as np
import torch
from darts.models import NBEATSModel  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from darts import TimeSeries
from darts.metrics import rmse, mae
import pickle  
import statistics
from collections import ChainMap

# Set float32 matmul precision to 'medium'
torch.set_float32_matmul_precision('medium')  

torch.manual_seed(1)
np.random.seed(1)

# Step 1: Load the Dataset
print("Step 1: Loading the dataset.")
df = pd.read_csv('combined_output_1degree.csv')
df = df.fillna(method='ffill')
df['time'] = pd.to_datetime(df['time'])  

# Step 2: Apply Dynamic Data Types
print("Step 2: Applying dynamic data types.")
float_dtype = 'float32'
columns_to_convert =  ['tmpsf','qnet']
for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(float_dtype)
    else:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

# Step 3: Split the DataFrame
print("Step 3: Splitting the DataFrame.")
df_split = np.array_split(df, 252)

# Step 4: Create Multivariate TimeSeries
print("Step 4: Creating multivariate TimeSeries.")
series_list = []
columns_to_include = ['tmpsf','qnet']
for i in range(len(df_split)):
    series_list.append(TimeSeries.from_dataframe(df_split[i], 'time', columns_to_include))

# Step 5: Split into Training and Testing Sets
print("Step 5: Splitting into training and testing sets.")
train_list = []
test_list = []
valid_list = []
for i in range(len(df_split)):
    series = series_list[i]
    train_series = series.slice(pd.Timestamp('2008-01-01'), pd.Timestamp('2018-12-31'))
    test_series = series.slice(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    valid_series = series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    train_list.append(train_series)
    test_list.append(test_series)
    valid_list.append(valid_series) 

# Step 6: Initialize the Model
print("Step 6: Initializing the model.")
model = NBEATSModel(
    input_chunk_length=365,
    output_chunk_length=365,   
    n_epochs=53,
    batch_size=50, 
    random_state=1,    
  
    
)

# Step 7: Train the Model
print("Step 7: Training the model.")
model.fit(series=train_list, verbose=True)

# Step 8: Save the Trained Model
print("Step 8: Saving the trained model.")
pickle.dump(model, open('SIO_2020_model_NBEATS.pkl', 'wb'))

# Step 9: Load the Trained Model
print("Step 9: Loading the trained model.")
model = pickle.load(open('SIO_2020_model_NBEATS.pkl', 'rb'))

# Step 10: Define Prediction Function
print("Step 10: Defining the prediction function.")
def predict_wa(args):
    index, n = args
    try:
        val = model.predict(n=n, series=train_list[index])
        return {f'{index}': val}
    except Exception:
        return None  

# Step 11: Predict for 2019 (Testing Data)
print("Step 11: Predicting for 2019.")
results_2019 = []
all_args_2019 = [(index, 365) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2019}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2019.append(result)
        except Exception:
            pass 

new_res_dict_2019 = dict(ChainMap(*results_2019))
SIO_2019_predicted = []
for x in range(len(df_split)):
    SIO_2019_predicted.append(new_res_dict_2019[f'{x}'])  

# Step 12: Evaluate Metrics for 2019
print("Step 12: Evaluating metrics for 2019.")
rmse_dict_2019 = {var: [] for var in columns_to_include}
mae_dict_2019 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(test_list[i][var], SIO_2019_predicted[i][var])
            mae_value = mae(test_list[i][var], SIO_2019_predicted[i][var])
            rmse_dict_2019[var].append(rmse_value)
            mae_dict_2019[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2019 = {var: statistics.mean(rmse_dict_2019[var]) for var in columns_to_include}
mean_mae_2019 = {var: statistics.mean(mae_dict_2019[var]) for var in columns_to_include}

# Step 13: Predict for Unseen Data (2020)
print("Step 13: Predicting for 2020.")
results_2020 = []
all_args_2020 = [(index, 731) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2020}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2020.append(result)
        except Exception:
            pass

new_res_dict_2020 = dict(ChainMap(*results_2020))
SIO_2020_predicted = []
for x in range(len(df_split)):
    full_pred_series = new_res_dict_2020[f'{x}']
    pred_2020 = full_pred_series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    SIO_2020_predicted.append(pred_2020)

# Step 14: Evaluate Metrics for 2020
print("Step 14: Evaluating metrics for 2020.")
rmse_dict_2020 = {var: [] for var in columns_to_include}
mae_dict_2020 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(valid_list[i][var], SIO_2020_predicted[i][var])
            mae_value = mae(valid_list[i][var], SIO_2020_predicted[i][var])
            rmse_dict_2020[var].append(rmse_value)
            mae_dict_2020[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2020 = {var: statistics.mean(rmse_dict_2020[var]) for var in columns_to_include}
mean_mae_2020 = {var: statistics.mean(mae_dict_2020[var]) for var in columns_to_include}

# Step 15: Save Predictions and Evaluation Metrics for 2019
print("Step 15: Saving predictions and evaluation metrics for 2019.")
np_arr_res_2019 = []
for x in SIO_2019_predicted:
    np_arr_res_2019.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2019, open("SIO_2019_NBEATS_predicted_final_nparray.bin", 'wb'))
# pickle.dump(rmse_dict_2019, open("SIO_2019_NBEATS_rmse_final_dict.bin", 'wb'))
# pickle.dump(mae_dict_2019, open("SIO_2019_NBEATS_mae_final_dict.bin", 'wb'))

# Step 16: Save Predictions and Evaluation Metrics for 2020
print("Step 16: Saving predictions and evaluation metrics for 2020.")
np_arr_res_2020 = []
for x in SIO_2020_predicted:
    np_arr_res_2020.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2020, open("SIO_2020_NBEATS_predicted_final_nparray.bin", 'wb'))
# pickle.dump(rmse_dict_2020, open("SIO_2020_NBEATS_rmse_final_dict.bin", 'wb'))
# pickle.dump(mae_dict_2020, open("SIO_2020_NBEATS_mae_final_dict.bin", 'wb'))

# Step 17: Construct the Final Dataset for 2019
print("Step 17: Constructing the final dataset for 2019.")
df_orig19 = df.loc[df['time'] <= '2018-12-31'].copy()
df_orig19['time'] = pd.to_datetime(df_orig19['time'])

pred_dfs19 = []
for idx, pred_series in enumerate(SIO_2019_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs19.append(pred_df)

df_pred19 = pd.concat(pred_dfs19, ignore_index=True)
df_final19 = pd.concat([df_orig19, df_pred19])
df_final19['time'] = pd.to_datetime(df_final19['time'])
df_final19 = df_final19.set_index(["time", "lat", "lon"])
df_final19['sst'] = df_final19['tmpsf']
df_final19.to_csv("SIO_2019_NBEATS_predicted_final.csv")  

# Step 18: Construct the Final Dataset for 2020
print("Step 18: Constructing the final dataset for 2020.")
df_orig20 = df.loc[df['time'] <= '2019-12-31'].copy()
df_orig20['time'] = pd.to_datetime(df_orig20['time'])

pred_dfs20 = []
for idx, pred_series in enumerate(SIO_2020_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs20.append(pred_df)

df_pred20 = pd.concat(pred_dfs20, ignore_index=True)
df_final20 = pd.concat([df_orig20, df_pred20])
df_final20['time'] = pd.to_datetime(df_final20['time'])
df_final20 = df_final20.set_index(["time", "lat", "lon"])
df_final20['sst'] = df_final20['tmpsf']
df_final20.to_csv("SIO_2020_NBEATS_predicted_final.csv")  

# Step 19: Convert Final Dataset for 2019 to xarray Dataset
print("Step 19: Converting final dataset for 2019 to xarray Dataset.")
ds19 = df_final19.to_xarray()
pred_df_2019 = ds19.sst.sel(time=slice("2019-01-01", "2019-12-31")).to_dataframe()
pred_df_2019.to_csv("SIO_2019_SST_Predictions_NBEATS.csv")

# Step 20: Convert Final Dataset for 2020 to xarray Dataset
print("Step 20: Converting final dataset for 2020 to xarray Dataset.")
ds20 = df_final20.to_xarray()
pred_df_2020 = ds20.sst.sel(time=slice("2020-01-01", "2020-12-31")).to_dataframe()
pred_df_2020.to_csv("SIO_2020_SST_Predictions_NBEATS.csv")

# Step 21: Save Mean RMSE and MAE Metrics to CSV
print("Step 21: Saving mean RMSE and MAE metrics to CSV.")
metrics_2019 = []
for var in columns_to_include:
    metrics_2019.append({
        'Year': 2019,
        'Variable': var,
        'Mean_RMSE': mean_rmse_2019[var],
        'Mean_MAE': mean_mae_2019[var]
    })

metrics_2020 = []
if 'mean_rmse_2020' in locals() and 'mean_mae_2020' in locals():
    for var in columns_to_include:
        metrics_2020.append({
            'Year': 2020,
            'Variable': var,
            'Mean_RMSE': mean_rmse_2020[var],
            'Mean_MAE': mean_mae_2020[var]
        })

all_metrics = metrics_2019 + metrics_2020
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("evaluation_metrics_summary_NBEATS.csv", index=False)



#############################################################################


#'tmpsf','evapr','lhtfl','nlwrs','hum2m','qnet','shtfl','nswrs','tmp2m','wnd10','msl','rain','cloud'

# TFT ALGORITHM

import pandas as pd
import numpy as np
import torch
from darts.models import TFTModel  # Changed from NBEATSModel to TFTModel
from concurrent.futures import ThreadPoolExecutor, as_completed  
from darts import TimeSeries
from darts.metrics import rmse, mae
import pickle  
import statistics
from collections import ChainMap

# Set float32 matmul precision to 'medium'
torch.set_float32_matmul_precision('medium')  

torch.manual_seed(1)
np.random.seed(1)

# Step 1: Load the Dataset
print("Step 1: Loading the dataset.")
df = pd.read_csv('combined_output_1degree.csv')
df = df.fillna(method='ffill')
df['time'] = pd.to_datetime(df['time'])  

# Step 2: Apply Dynamic Data Types
print("Step 2: Applying dynamic data types.")
float_dtype = 'float32'
columns_to_convert = ['tmpsf','qnet','tmp2m']
for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(float_dtype)
    else:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

# Step 3: Split the DataFrame
print("Step 3: Splitting the DataFrame.")
df_split = np.array_split(df, 252)

# Step 4: Create Multivariate TimeSeries
print("Step 4: Creating multivariate TimeSeries.")
series_list = []
columns_to_include = ['tmpsf','qnet','tmp2m']
for i in range(len(df_split)):
    series_list.append(TimeSeries.from_dataframe(df_split[i], 'time', columns_to_include))

# Step 5: Split into Training and Testing Sets
print("Step 5: Splitting into training and testing sets.")
train_list = []
test_list = []
valid_list = []
for i in range(len(df_split)):
    series = series_list[i]
    train_series = series.slice(pd.Timestamp('1989-01-01'), pd.Timestamp('2018-12-31'))
    test_series = series.slice(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    valid_series = series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    train_list.append(train_series)
    test_list.append(test_series)
    valid_list.append(valid_series) 

# Step 6: Initialize the Model
print("Step 6: Initializing the model.")
model = TFTModel(
    input_chunk_length=365,
    output_chunk_length=365,
    n_epochs=30,                # Changed from 50 to 30
    batch_size=50, 
    random_state=1,                       
    add_relative_index=True,    # Added parameter
)

# Step 7: Train the Model
print("Step 7: Training the model.")
model.fit(series=train_list, verbose=True)

# Step 8: Save the Trained Model
print("Step 8: Saving the trained model.")
pickle.dump(model, open('SIO_2020_model_TFT.pkl', 'wb'))  # Changed filename

# Step 9: Load the Trained Model
print("Step 9: Loading the trained model.")
model = pickle.load(open('SIO_2020_model_TFT.pkl', 'rb'))  # Changed filename

# Step 10: Define Prediction Function
print("Step 10: Defining the prediction function.")
def predict_wa(args):
    index, n = args
    try:
        val = model.predict(n=n, series=train_list[index])
        return {f'{index}': val}
    except Exception:
        return None  

# Step 11: Predict for 2019 (Testing Data)
print("Step 11: Predicting for 2019.")
results_2019 = []
all_args_2019 = [(index, 365) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2019}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2019.append(result)
        except Exception:
            pass 

new_res_dict_2019 = dict(ChainMap(*results_2019))
SIO_2019_predicted = []
for x in range(len(df_split)):
    SIO_2019_predicted.append(new_res_dict_2019[f'{x}'])  

# Step 12: Evaluate Metrics for 2019
print("Step 12: Evaluating metrics for 2019.")
rmse_dict_2019 = {var: [] for var in columns_to_include}
mae_dict_2019 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(test_list[i][var], SIO_2019_predicted[i][var])
            mae_value = mae(test_list[i][var], SIO_2019_predicted[i][var])
            rmse_dict_2019[var].append(rmse_value)
            mae_dict_2019[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2019 = {var: statistics.mean(rmse_dict_2019[var]) for var in columns_to_include}
mean_mae_2019 = {var: statistics.mean(mae_dict_2019[var]) for var in columns_to_include}

# Step 13: Predict for Unseen Data (2020)
print("Step 13: Predicting for 2020.")
results_2020 = []
all_args_2020 = [(index, 731) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2020}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2020.append(result)
        except Exception:
            pass

new_res_dict_2020 = dict(ChainMap(*results_2020))
SIO_2020_predicted = []
for x in range(len(df_split)):
    full_pred_series = new_res_dict_2020[f'{x}']
    pred_2020 = full_pred_series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    SIO_2020_predicted.append(pred_2020)

# Step 14: Evaluate Metrics for 2020
print("Step 14: Evaluating metrics for 2020.")
rmse_dict_2020 = {var: [] for var in columns_to_include}
mae_dict_2020 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(valid_list[i][var], SIO_2020_predicted[i][var])
            mae_value = mae(valid_list[i][var], SIO_2020_predicted[i][var])
            rmse_dict_2020[var].append(rmse_value)
            mae_dict_2020[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2020 = {var: statistics.mean(rmse_dict_2020[var]) for var in columns_to_include}
mean_mae_2020 = {var: statistics.mean(mae_dict_2020[var]) for var in columns_to_include}

# Step 15: Save Predictions and Evaluation Metrics for 2019
print("Step 15: Saving predictions and evaluation metrics for 2019.")
np_arr_res_2019 = []
for x in SIO_2019_predicted:
    np_arr_res_2019.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2019, open("SIO_2019_TFT_predicted_final_nparray.bin", 'wb'))  # Changed filename
# pickle.dump(rmse_dict_2019, open("SIO_2019_TFT_rmse_final_dict.bin", 'wb'))            # Changed filename
# pickle.dump(mae_dict_2019, open("SIO_2019_TFT_mae_final_dict.bin", 'wb'))              # Changed filename

# Step 16: Save Predictions and Evaluation Metrics for 2020
print("Step 16: Saving predictions and evaluation metrics for 2020.")
np_arr_res_2020 = []
for x in SIO_2020_predicted:
    np_arr_res_2020.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2020, open("SIO_2020_TFT_predicted_final_nparray.bin", 'wb'))  # Changed filename
# pickle.dump(rmse_dict_2020, open("SIO_2020_TFT_rmse_final_dict.bin", 'wb'))            # Changed filename
# pickle.dump(mae_dict_2020, open("SIO_2020_TFT_mae_final_dict.bin", 'wb'))              # Changed filename

# Step 17: Construct the Final Dataset for 2019
print("Step 17: Constructing the final dataset for 2019.")
df_orig19 = df.loc[df['time'] <= '2018-12-31'].copy()
df_orig19['time'] = pd.to_datetime(df_orig19['time'])

pred_dfs19 = []
for idx, pred_series in enumerate(SIO_2019_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs19.append(pred_df)

df_pred19 = pd.concat(pred_dfs19, ignore_index=True)
df_final19 = pd.concat([df_orig19, df_pred19])
df_final19['time'] = pd.to_datetime(df_final19['time'])
df_final19 = df_final19.set_index(["time", "lat", "lon"])
df_final19['sst'] = df_final19['tmpsf']
df_final19.to_csv("SIO_2019_TFT_predicted_final.csv")  # Changed filename

# Step 18: Construct the Final Dataset for 2020
print("Step 18: Constructing the final dataset for 2020.")
df_orig20 = df.loc[df['time'] <= '2019-12-31'].copy()
df_orig20['time'] = pd.to_datetime(df_orig20['time'])

pred_dfs20 = []
for idx, pred_series in enumerate(SIO_2020_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs20.append(pred_df)

df_pred20 = pd.concat(pred_dfs20, ignore_index=True)
df_final20 = pd.concat([df_orig20, df_pred20])
df_final20['time'] = pd.to_datetime(df_final20['time'])
df_final20 = df_final20.set_index(["time", "lat", "lon"])
df_final20['sst'] = df_final20['tmpsf']
df_final20.to_csv("SIO_2020_TFT_predicted_final.csv")  # Changed filename

# Step 19: Convert Final Dataset for 2019 to xarray Dataset
print("Step 19: Converting final dataset for 2019 to xarray Dataset.")
ds19 = df_final19.to_xarray()
pred_df_2019 = ds19.sst.sel(time=slice("2019-01-01", "2019-12-31")).to_dataframe()
pred_df_2019.to_csv("SIO_2019_SST_Predictions_TFT.csv")  # Changed filename

# Step 20: Convert Final Dataset for 2020 to xarray Dataset
print("Step 20: Converting final dataset for 2020 to xarray Dataset.")
ds20 = df_final20.to_xarray()
pred_df_2020 = ds20.sst.sel(time=slice("2020-01-01", "2020-12-31")).to_dataframe()
pred_df_2020.to_csv("SIO_2020_SST_Predictions_TFT.csv")  # Changed filename

# Step 21: Save Mean RMSE and MAE Metrics to CSV
print("Step 21: Saving mean RMSE and MAE metrics to CSV.")
metrics_2019 = []
for var in columns_to_include:
    metrics_2019.append({
        'Year': 2019,
        'Variable': var,
        'Mean_RMSE': mean_rmse_2019[var],
        'Mean_MAE': mean_mae_2019[var]
    })

metrics_2020 = []
if 'mean_rmse_2020' in locals() and 'mean_mae_2020' in locals():
    for var in columns_to_include:
        metrics_2020.append({
            'Year': 2020,
            'Variable': var,
            'Mean_RMSE': mean_rmse_2020[var],
            'Mean_MAE': mean_mae_2020[var]
        })

all_metrics = metrics_2019 + metrics_2020
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("evaluation_metrics_summary_TFT.csv", index=False)  # Changed filename

#####################################################################################

#NBEATS with optuna for hyperparameters


import pandas as pd
import numpy as np
import torch
from darts.models import NBEATSModel  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from darts import TimeSeries
from darts.metrics import rmse, mae
import pickle  
import statistics
from collections import ChainMap
import optuna
from optuna.trial import Trial

# Set float32 matmul precision to 'medium'
torch.set_float32_matmul_precision('medium')  

torch.manual_seed(1)
np.random.seed(1)

# Step 1-5: Data Loading and Preprocessing (Same as original)
print("Steps 1-5: Loading and preprocessing data...")
df = pd.read_csv('combined_output_1degree.csv')
df = df.fillna(method='ffill')
df['time'] = pd.to_datetime(df['time'])  

float_dtype = 'float32'
columns_to_convert = ['tmpsf']
for col in columns_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(float_dtype)
    else:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")

df_split = np.array_split(df, 252)

series_list = []
columns_to_include = ['tmpsf']
for i in range(len(df_split)):
    series_list.append(TimeSeries.from_dataframe(df_split[i], 'time', columns_to_include))

train_list = []
test_list = []
valid_list = []
for i in range(len(df_split)):
    series = series_list[i]
    train_series = series.slice(pd.Timestamp('1989-01-01'), pd.Timestamp('2018-12-31'))
    test_series = series.slice(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'))
    valid_series = series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    train_list.append(train_series)
    test_list.append(test_series)
    valid_list.append(valid_series)

# Step 6: Define Optuna Objective Function
def objective(trial: Trial) -> float:
    # Define hyperparameter search space
    input_chunk_length = trial.suggest_int('input_chunk_length', 7, 365)
    output_chunk_length = trial.suggest_int('output_chunk_length', 180, 365)
    num_stacks = trial.suggest_int('num_stacks', 4, 12)
    num_blocks = trial.suggest_int('num_blocks', 1, 4)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    layer_widths = trial.suggest_int('layer_widths', 128, 512)
    batch_size = trial.suggest_int('batch_size', 32, 512)
    n_epochs = trial.suggest_int('n_epochs', 20, 150)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Initialize model with trial parameters
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        batch_size=batch_size,
        n_epochs=n_epochs,
        optimizer_kwargs={"lr": learning_rate},
        random_state=1
    )
    
    try:
        # Train on a subset of data for faster optimization
        validation_rmse = []
        for i in range(min(10, len(train_list))):  # Use first 10 series for optimization
            model.fit(series=train_list[i], verbose=False)
            pred = model.predict(n=365, series=train_list[i])
            validation_rmse.append(rmse(test_list[i], pred))
        
        mean_rmse = np.mean(validation_rmse)
        return mean_rmse
    
    except Exception as e:
        print(f"Error in trial: {e}")
        return float('inf')

# Step 7: Run Optuna Optimization
print("Step 7: Running hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Adjust n_trials based on your computational resources

# Print optimization results
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Step 8: Initialize Model with Best Parameters
print("Step 8: Initializing model with best parameters...")
best_params = study.best_params
model = NBEATSModel(
    input_chunk_length=best_params['input_chunk_length'],
    output_chunk_length=best_params['output_chunk_length'],
    num_stacks=best_params['num_stacks'],
    num_blocks=best_params['num_blocks'],
    num_layers=best_params['num_layers'],
    layer_widths=best_params['layer_widths'],
    batch_size=best_params['batch_size'],
    n_epochs=best_params['n_epochs'],
    optimizer_kwargs={"lr": best_params['learning_rate']},
    random_state=1
)

# Save optimization results
optimization_results = {
    'best_params': study.best_params,
    'best_value': study.best_value,
    'all_trials': study.trials_dataframe()
}
with open('optuna_optimization_results.pkl', 'wb') as f:
    pickle.dump(optimization_results, f)

# Steps 9-21: Continue with the original pipeline using optimized model
# Training
print("Step 9: Training the model with optimized parameters...")
model.fit(series=train_list, verbose=True)

# Save the trained model
print("Step 10: Saving the trained model...")
pickle.dump(model, open('SIO_2020_model_NBEATS_optimized.pkl', 'wb'))

# Continue with the rest of the original steps...
# [Previous prediction and evaluation code remains the same]
def predict_wa(args):
    index, n = args
    try:
        val = model.predict(n=n, series=train_list[index])
        return {f'{index}': val}
    except Exception:
        return None

# Steps 11-21: [Rest of the original code remains exactly the same]
# [Include all the prediction, evaluation, and saving steps from the original script]

# Step 11: Predict for 2019 (Testing Data)
print("Step 11: Predicting for 2019.")
results_2019 = []
all_args_2019 = [(index, 365) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2019}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2019.append(result)
        except Exception:
            pass 

new_res_dict_2019 = dict(ChainMap(*results_2019))
SIO_2019_predicted = []
for x in range(len(df_split)):
    SIO_2019_predicted.append(new_res_dict_2019[f'{x}'])  

# Step 12: Evaluate Metrics for 2019
print("Step 12: Evaluating metrics for 2019.")
rmse_dict_2019 = {var: [] for var in columns_to_include}
mae_dict_2019 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(test_list[i][var], SIO_2019_predicted[i][var])
            mae_value = mae(test_list[i][var], SIO_2019_predicted[i][var])
            rmse_dict_2019[var].append(rmse_value)
            mae_dict_2019[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2019 = {var: statistics.mean(rmse_dict_2019[var]) for var in columns_to_include}
mean_mae_2019 = {var: statistics.mean(mae_dict_2019[var]) for var in columns_to_include}

# Step 13: Predict for Unseen Data (2020)
print("Step 13: Predicting for 2020.")
results_2020 = []
all_args_2020 = [(index, 731) for index in range(len(df_split))]

with ThreadPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(predict_wa, args): args[0] for args in all_args_2020}
    for future in as_completed(futures):
        index = futures[future]
        try:
            result = future.result()
            if result is not None:
                results_2020.append(result)
        except Exception:
            pass

new_res_dict_2020 = dict(ChainMap(*results_2020))
SIO_2020_predicted = []
for x in range(len(df_split)):
    full_pred_series = new_res_dict_2020[f'{x}']
    pred_2020 = full_pred_series.slice(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-12-31'))
    SIO_2020_predicted.append(pred_2020)

# Step 14: Evaluate Metrics for 2020
print("Step 14: Evaluating metrics for 2020.")
rmse_dict_2020 = {var: [] for var in columns_to_include}
mae_dict_2020 = {var: [] for var in columns_to_include}

for i in range(len(df_split)):
    for var in columns_to_include:
        try:
            rmse_value = rmse(valid_list[i][var], SIO_2020_predicted[i][var])
            mae_value = mae(valid_list[i][var], SIO_2020_predicted[i][var])
            rmse_dict_2020[var].append(rmse_value)
            mae_dict_2020[var].append(mae_value)
        except Exception:
            pass

mean_rmse_2020 = {var: statistics.mean(rmse_dict_2020[var]) for var in columns_to_include}
mean_mae_2020 = {var: statistics.mean(mae_dict_2020[var]) for var in columns_to_include}

# Step 15: Save Predictions and Evaluation Metrics for 2019
print("Step 15: Saving predictions and evaluation metrics for 2019.")
np_arr_res_2019 = []
for x in SIO_2019_predicted:
    np_arr_res_2019.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2019, open("SIO_2019_NBEATS_predicted_final_nparray.bin", 'wb'))
# pickle.dump(rmse_dict_2019, open("SIO_2019_NBEATS_rmse_final_dict.bin", 'wb'))
# pickle.dump(mae_dict_2019, open("SIO_2019_NBEATS_mae_final_dict.bin", 'wb'))

# Step 16: Save Predictions and Evaluation Metrics for 2020
print("Step 16: Saving predictions and evaluation metrics for 2020.")
np_arr_res_2020 = []
for x in SIO_2020_predicted:
    np_arr_res_2020.extend(x.pd_dataframe().values)

# Uncomment the following lines to save the predictions and metrics
# pickle.dump(np_arr_res_2020, open("SIO_2020_NBEATS_predicted_final_nparray.bin", 'wb'))
# pickle.dump(rmse_dict_2020, open("SIO_2020_NBEATS_rmse_final_dict.bin", 'wb'))
# pickle.dump(mae_dict_2020, open("SIO_2020_NBEATS_mae_final_dict.bin", 'wb'))

# Step 17: Construct the Final Dataset for 2019
print("Step 17: Constructing the final dataset for 2019.")
df_orig19 = df.loc[df['time'] <= '2018-12-31'].copy()
df_orig19['time'] = pd.to_datetime(df_orig19['time'])

pred_dfs19 = []
for idx, pred_series in enumerate(SIO_2019_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs19.append(pred_df)

df_pred19 = pd.concat(pred_dfs19, ignore_index=True)
df_final19 = pd.concat([df_orig19, df_pred19])
df_final19['time'] = pd.to_datetime(df_final19['time'])
df_final19 = df_final19.set_index(["time", "lat", "lon"])
df_final19['sst'] = df_final19['tmpsf']
df_final19.to_csv("SIO_2019_NBEATS_predicted_final.csv")  

# Step 18: Construct the Final Dataset for 2020
print("Step 18: Constructing the final dataset for 2020.")
df_orig20 = df.loc[df['time'] <= '2019-12-31'].copy()
df_orig20['time'] = pd.to_datetime(df_orig20['time'])

pred_dfs20 = []
for idx, pred_series in enumerate(SIO_2020_predicted):
    lat = df_split[idx]['lat'].iloc[0]
    lon = df_split[idx]['lon'].iloc[0]
    pred_df = pred_series.pd_dataframe().reset_index()
    pred_df['lat'] = lat
    pred_df['lon'] = lon
    pred_df['time'] = pd.to_datetime(pred_df['time'])
    pred_dfs20.append(pred_df)

df_pred20 = pd.concat(pred_dfs20, ignore_index=True)
df_final20 = pd.concat([df_orig20, df_pred20])
df_final20['time'] = pd.to_datetime(df_final20['time'])
df_final20 = df_final20.set_index(["time", "lat", "lon"])
df_final20['sst'] = df_final20['tmpsf']
df_final20.to_csv("SIO_2020_NBEATS_predicted_final.csv")  

# Step 19: Convert Final Dataset for 2019 to xarray Dataset
print("Step 19: Converting final dataset for 2019 to xarray Dataset.")
ds19 = df_final19.to_xarray()
pred_df_2019 = ds19.sst.sel(time=slice("2019-01-01", "2019-12-31")).to_dataframe()
pred_df_2019.to_csv("SIO_2019_SST_Predictions_NBEATS.csv")

# Step 20: Convert Final Dataset for 2020 to xarray Dataset
print("Step 20: Converting final dataset for 2020 to xarray Dataset.")
ds20 = df_final20.to_xarray()
pred_df_2020 = ds20.sst.sel(time=slice("2020-01-01", "2020-12-31")).to_dataframe()
pred_df_2020.to_csv("SIO_2020_SST_Predictions_NBEATS.csv")

# Step 21: Save Mean RMSE and MAE Metrics to CSV
print("Step 21: Saving mean RMSE and MAE metrics to CSV.")
metrics_2019 = []
for var in columns_to_include:
    metrics_2019.append({
        'Year': 2019,
        'Variable': var,
        'Mean_RMSE': mean_rmse_2019[var],
        'Mean_MAE': mean_mae_2019[var]
    })

metrics_2020 = []
if 'mean_rmse_2020' in locals() and 'mean_mae_2020' in locals():
    for var in columns_to_include:
        metrics_2020.append({
            'Year': 2020,
            'Variable': var,
            'Mean_RMSE': mean_rmse_2020[var],
            'Mean_MAE': mean_mae_2020[var]
        })

all_metrics = metrics_2019 + metrics_2020
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("evaluation_metrics_summary_NBEATS.csv", index=False)














