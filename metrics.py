import requests
import json
import pandas as pd
import os
import joblib
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
def get_fred_data(series_ids, fred_api_key, aqi_api_key, zipcode):
    # Dataframe
    df = pd.DataFrame(columns=["Series ID", "Value"])
    aggregation_method = "avg"
    file_type = "json"
    for series_id in series_ids:

        # URL
        if series_id == "FPCPITOTLZGUSA":
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FPCPITOTLZGUSA&realtime_start=2023-01-01&sort_order=desc&limit=1&api_key={fred_api_key}&file_type=json"
            response = requests.get(url)
        elif series_id == "ROWFDNA027N":
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=ROWFDNA027N&realtime_start=2023-01-01&sort_order=desc&limit=1&api_key={fred_api_key}&file_type=json"
            response = requests.get(url)
        else:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&sort_order=asc&observation_start=2023-01-01&aggregation_method={aggregation_method}&file_type={file_type}"
            response = requests.get(url)

        # Check the response status code
        if response.status_code == 200:
            # Get the JSON response
            data = json.loads(response.content)

            # Get the most recent value
        if data["observations"]:
            most_recent_value = float(data["observations"][0]["value"])
            
            # Add the value to the DataFrame
            df[series_id] = [most_recent_value]
        else:
            print("Error: No observations found for series ID " + series_id)

    # Get the AQI data
    avg_aqi = get_avg_aqi(zipcode, aqi_api_key)
    df['avg_aqi'] = [avg_aqi]

    return df



def get_current_aqi(zipcode, api_key):
    # Base URL for AirNow API
    base_url = "http://www.airnowapi.org/aq/observation/zipCode/current/"
    # Prepare query parameters
    params = {
        "format": "application/json",
        "zipCode": zipcode,
        "distance": "25",
        "API_KEY": api_key,
    }
    # Send request to AirNow API
    response = requests.get(base_url, params=params)
    # Check the response
    if response.status_code == 200:
        # If response is OK, parse the data
        data = json.loads(response.text)
        return data
    else:
        # If response is not OK, return the status code and message
        return f"Error: {response.status_code}, {response.text}"
    
def get_avg_aqi(zipcode, api_key):
    data = get_current_aqi(zipcode, api_key)
    if isinstance(data, list) and data:
        aqi_values = [item['AQI'] for item in data]
        avg_aqi = sum(aqi_values) / len(aqi_values)
        return avg_aqi
    return None

# print(get_fred_data(["FPCPITOTLZGUSA","CSCICP03USM665S","USSTHPI","CSUSHPINSA","Year", "MSACSR","EMVELECTGOVRN","ROWFDNA027N"], os.environ.get("FRED_API_KEY"), os.environ.get("AIRNOW_API_KEY"), "35806"))



df = get_fred_data(["FPCPITOTLZGUSA","CSCICP03USM665S","USSTHPI","CSUSHPINSA","Year", "MSACSR","EMVELECTGOVRN","ROWFDNA027N"], os.environ.get("FRED_API_KEY"), os.environ.get("AIRNOW_API_KEY"), "35806")

# Assume prev_year_price is the input previous year price
prev_year_price = input("Please enter the previous year price: ")

# Add it to the DataFrame
df['Prev_Year_Price'] = float(prev_year_price)

# Assume state is the input state
state = input("Please enter the state: ")

# Add it to the DataFrame
df['State'] = state

# Handling categorical variable 'State' by using one hot encoding
df = pd.get_dummies(df, columns=['State'])

# If the new data is missing some states that were present in the training data,
# you'll need to add those columns and fill with zeros. You'll need a list of all
# states that were in the training data. This is just a placeholder, replace with
# your actual list of states.
all_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

for state in all_states:
    if state not in df.columns:
        df[state] = 0


# Convert the DataFrame to a numpy array
array = df.values


# Define the PyTorch model


