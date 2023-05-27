import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn import linear_model


def linear_regression(X_train, X_test, y_train, y_test=None):
    # Fit model
    lr = linear_model.Ridge(alpha=1, copy_X=True, fit_intercept=True, max_iter=None, solver="auto", tol=1e-10) 

    lr.fit(X_train, y_train)
    
    # Predict and score
    y_pred = lr.predict(X_test)
    
    return lr, y_pred

train = pd.read_csv("data/train_processed.csv")
test = pd.read_csv("data/test_processed.csv")

stations_meta = pd.read_csv("data/bicikelj_metadata.csv", sep="\t")

# Get list of stations
stations = []
for i in range(1, 84):
    stations.append(train.columns[i])
    

# Generate station dataframes
train_station_dataframes = {}
test_station_dataframes = {}

for station in stations:
            
    for n in range(2):
        if n == 0:
            dataframe = train
        else:
            dataframe = test
        station_df = pd.DataFrame({
            "timestamp": dataframe["timestamp"].values,
            "n_bikes": dataframe[station].values,
            "month": dataframe["month"].values,
            "day": dataframe["day"].values,
            "hour": dataframe["hour"].values,
            "minute": dataframe["minute"].values,
            "day_of_week": dataframe["day_of_week"].values,
            "is_holiday": dataframe["is_holiday"].values,
            "is_weekend": dataframe["is_weekend"].values,
            "is_night": dataframe["is_night"].values,
            "school_holiday": dataframe["school_holiday"].values,
            "n_bikes_1hr": dataframe[f"{station}_1hr"].values,
            "n_bikes_2hr": dataframe[f"{station}_2hr"].values,
            "n_bikes_90min": dataframe[f"{station}_90min"].values,
            "rain": dataframe["rain"].values,
            "rain_condition": dataframe["rain_condition"].values,
            "feelslike": dataframe["feelslike"].values,
            "temp": dataframe["temp"].values
        })
        
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        for i, day in enumerate(days_of_week):
            station_df[day] = dataframe[day].values


        hours = [i for i in range(24)]
        for i in hours:
            station_df[f"hour_{i}"] = dataframe[f"hour_{i}"].values   
            
        if n == 0:    
            train_station_dataframes[station] = station_df
        else: 
            test_station_dataframes[station] = station_df
       


output = pd.read_csv("data/bicikelj_test.csv").copy()

# Perform linear regression on all stations
for station in stations:

    station_df = train_station_dataframes[station]
    
    X_train = station_df.drop(["timestamp", "n_bikes"], axis=1)
    y_train = station_df["n_bikes"]

    
    X_test = test_station_dataframes[station].drop(["timestamp", "n_bikes"], axis=1)
    
    X_train_1hr = X_train[X_train.index % 2 == 0]
    X_train_2hr = X_train[X_train.index % 2 == 1]
    y_train_1hr = y_train[y_train.index % 2 == 0]
    y_train_2hr = y_train[y_train.index % 2 == 1]
    X_test_1hr = X_test[X_test.index % 2 == 0]
    X_test_2hr = X_test[X_test.index % 2 == 1]
    
    # No data for 1hr back
    X_train_2hr = np.array(X_train_2hr.drop(["n_bikes_1hr", "n_bikes_90min"], axis=1))
    X_test_2hr = np.array(X_test_2hr.drop(["n_bikes_1hr", "n_bikes_90min"], axis=1))
    
    lr_1h, y_pred_1h = linear_regression(X_train_1hr, X_test_1hr, y_train_1hr)
    lr_2h, y_pred_2h = linear_regression(X_train_2hr, X_test_2hr, y_train_2hr)
    
    y_pred = np.empty(len(y_pred_1h) + len(y_pred_2h))
    y_pred[::2] = y_pred_1h
    y_pred[1::2] = y_pred_2h
    
    y_pred = pd.DataFrame({"n_bikes": y_pred})
    
    # Set values in y_pred to 0 if negative
    y_pred[y_pred < 0] = 0
    
    total_space = stations_meta.loc[stations_meta["postaja"] == station]["total_space"].values[0]
    y_pred[y_pred > total_space] = total_space
    
    
    output[station] = y_pred
    
output.to_csv("data/output.csv", index=False)

