# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# %%
train = pd.read_csv("data/bicikelj_train.csv")
train["timestamp"] = pd.to_datetime(train["timestamp"])

test = pd.read_csv("data/bicikelj_test.csv")
test["timestamp"] = pd.to_datetime(test["timestamp"])

# %%
# Extract new features
def get_time_based_features(timestamp):
    # From datetime64 get month, day, hour, dayOfWeek and isHoliday
    timestamp = np.datetime64(timestamp)
    dt_python = timestamp.astype(datetime.datetime)

    # Extract month, day, hour, and day_of_week
    month = dt_python.month
    day = dt_python.day
    hour = dt_python.hour
    minute = dt_python.minute
    day_of_week = dt_python.weekday()

    # Check for holidays
    is_holiday = 0
    if dt_python.month == 8 and dt_python.day in [15, 17]:
        is_holiday = 1
    elif dt_python.month == 9 and dt_python.day in [15, 23]:
        is_holiday = 1
   
    return month, day, hour, minute, day_of_week, is_holiday


# %%
def find_closest_smaller_row(df, timestamp, hour_offset):
    target_timestamp = pd.to_datetime(timestamp) - pd.DateOffset(hours=hour_offset)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
            
    smaller_rows = df[df["timestamp"] <= target_timestamp]
    if smaller_rows.empty:
        return df.iloc[0,:]  # No smaller rows found
    closest_index = np.abs(smaller_rows["timestamp"] - target_timestamp).idxmin()
    closest_row = df.loc[closest_index]
    return closest_row


# %%
# Get list of stations
stations = []
for i in range(1, 84):
    stations.append(train.columns[i])

# %%
# Add new features to train
# train['month'], train['day'], train['hour'], train['minute'], train['day_of_week'], train['is_holiday'] = zip(*train['timestamp'].map(get_time_based_features))

# %%
# One hot encode days of week and hours
# days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# for i, day in enumerate(days_of_week):
#     train[day] = (train["day_of_week"] == i).astype(int)


# hours = [i for i in range(24)]
# for i in hours:
#     train[f"hour_{i}"] = (train["hour"] == i).astype(int)

# %%
# for station in stations:
#     train[f"{station}_1hr"] = train["timestamp"].apply(lambda time: find_closest_smaller_row(train, time, 1)[station])
#     train[f"{station}_2hr"] = train["timestamp"].apply(lambda time: find_closest_smaller_row(train, time, 2)[station])

# %%
train_station_dataframes = {}
test_station_dataframes = {}
test = pd.read_csv("data/test_expanded.csv")
train = pd.read_csv("data/processed_data.csv")

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
            "n_bikes_1hr": dataframe[f"{station}_1hr"].values,
            "n_bikes_2hr": dataframe[f"{station}_2hr"].values
        })
        
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        for i, day in enumerate(days_of_week):
            station_df[day] = dataframe[day].values


        hours = [i for i in range(24)]
        for i in hours:
            station_df[f"hour_{i}"] = dataframe[f"hour_{i}"].values   
            
        if n == 0:    
            train_station_dataframes[station] = station_df
        else: test_station_dataframes[station] = station_df
    # station = station.replace("/", "_")
    # station_df.to_csv(f"data/stations/{station}.csv", index=False)

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# %%
def linear_regression(X_train, X_test, y_train, y_test=None):
        # Fit model
    lr = linear_model.Ridge() #Lasso()
    # lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    
    coefs = lr.coef_
    
    for i in range(len(coefs)):
        if coefs[i] == 0:
            print(f"Removed {X.columns[i]}")
        # else:
        #     print(f"{(X[i])}: {lr.coef_[i]}")
    
    # Predict and score
    y_pred = lr.predict(X_test)
    
    # plt.scatter(y_test, y_pred)
    
    # score = mean_squared_error(y_test, y_pred)
    return lr, y_pred

# %%
# test = pd.read_csv("data/test_expanded.csv")
output = pd.read_csv("data/bicikelj_test.csv").copy()

# Perform linear regression on all stations
for station in stations[:2]:
    station_df = train_station_dataframes[station]
    X_train = station_df.drop(["timestamp", "n_bikes"], axis=1)
    y_train = station_df["n_bikes"]
    
    # X_test = pd.read_csv(f"/data/stations/{station.replace('/', '_')}_test.csv").drop(["timestamp"], axis=1)
    X_test = test_station_dataframes[station].drop(["timestamp", "n_bikes"], axis=1)
    
    lr, y_pred = linear_regression(X_train, X_test, y_train)
    m = lr.coef_.argsort()[::-1]
    
    for i in range(len(m)):
        
        print(X_train.columns[m[i]], lr.coef_[m[i]])
    output[station] = y_pred
    
output.to_csv("data/output.csv", index=False)


