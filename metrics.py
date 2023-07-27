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
    df = pd.DataFrame()
    
    
    for series_id in series_ids:
        aggregation_method = "avg"
        file_type = "json"
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
                df.loc[0, series_id] = most_recent_value
            else:
                print("Error: No observations found for series ID " + series_id)

    # Get the AQI data
    avg_aqi = get_avg_aqi(zipcode, aqi_api_key)
    df.loc[0, 'Average AQI'] = avg_aqi

    # Request user input for year, quarter, and state

    df = {
        'CSUSHPINSA': 123.4,
        'FPCPITOTLZGUSA': 123.4,
        'CSCICP03USM665S': 110.2, 
        'MSACSR': 102.7,
        'EMVELECTGOVRN': 10000,
        'USSTHPI': 280,
        'ROWFDNA027N': 123        
    }
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



df = get_fred_data(["FPCPITOTLZGUSA","CSCICP03USM665S","USSTHPI","CSUSHPINSA","Year", "MSACSR","EMVELECTGOVRN","ROWFDNA027N"], os.environ.get("FRED_API_KEY"), os.environ.get("AIRNOW_API_KEY"), "35749")
df.columns.append("State")
print(df)