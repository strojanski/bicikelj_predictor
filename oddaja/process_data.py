import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


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
        
        
    is_weekend = 0
    if dt_python.weekday() in [5, 6]:
        is_weekend = 1
   
    is_night = 0
    if hour < 6 or hour > 22:
        is_night = 1
        
    school_holiday = 0
    if 6 <= month <= 8:
        school_holiday = 1
   
    return month, day, hour, minute, day_of_week, is_holiday, is_weekend, is_night, school_holiday


def closest(df, timestamp, minute_offset=60, thresh=15):
    target_ts = pd.to_datetime(timestamp) - pd.DateOffset(minutes=minute_offset) 
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data[(data["timestamp"] <= target_ts + pd.DateOffset(minutes=5)) & (data["timestamp"] > target_ts - pd.DateOffset(minutes=thresh))]
    if len(data) == 0: 
        return None
    
    # print(target_ts, "\n", data["timestamp"])
    closest_index = np.argmin(np.abs(target_ts - pd.to_datetime(data["timestamp"])))
    # print("Closest", data.iloc[closest_index]["timestamp"])
    return data.iloc[closest_index]


def get_bikes_offsets(df):
    train = df.copy()
    for i, timestamp in enumerate(train["timestamp"]):
        row_1hr = closest(train, timestamp, 60)
        row_90 = closest(train, timestamp, 90)
        row_2hr = closest(train, timestamp, 120)
        
            
        if row_1hr is not None:
            train.loc[i, [f"{station}_1hr" for station in stations]] = row_1hr[stations].values
        
        if row_90 is not None:
            train.loc[i, [f"{station}_150min" for station in stations]] = row_90[stations].values
        
        if row_2hr is not None:
            train.loc[i, [f"{station}_2hr" for station in stations]] = row_2hr[stations].values
    
    return train

train = pd.read_csv("data/bicikelj_train.csv")
train["timestamp"] = pd.to_datetime(train["timestamp"])

test = pd.read_csv("data/bicikelj_test.csv")
test["timestamp"] = pd.to_datetime(test["timestamp"])

stations_meta = pd.read_csv("data/bicikelj_metadata.csv", sep="\t")


# Get list of stations
stations = []
for i in range(1, 84):
    stations.append(train.columns[i])
 # offset_df = pd.DataFrame(columns=[f"{station}_1hr" for station in stations] + [f"{station}_2hr" for station in stations])



# Add new features to train
train['month'], train['day'], train['hour'], train['minute'], train['day_of_week'], train['is_holiday'], train['is_weekend'], train['is_night'], train['school_holiday'] = zip(*train['timestamp'].map(get_time_based_features))
test['month'], test['day'], test['hour'], test['minute'], test['day_of_week'], test['is_holiday'], test['is_weekend'], test['is_night'], test['school_holiday'] = zip(*test['timestamp'].map(get_time_based_features))

days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

for i, day in enumerate(days_of_week):
    train[day] = (train["day_of_week"] == i).astype(int) 
    test[day] = (test["day_of_week"] == i).astype(int)


hours = [i for i in range(24)]
for i in hours:
    train[f"hour_{i}"] = (train["hour"] == i).astype(int)
    test[f"hour_{i}"] = (test["hour"] == i).astype(int)
    
train = train.dropna()
