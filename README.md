# Housing Price Prediction

This project builds a machine learning model to predict housing prices based on economic indicators. It uses a random forest regressor trained on historical housing price data.

## Files

- `metrics.py`: Gets input data from APIs and user, loads model, and makes prediction
- `model_engine.py`: Trains and saves random forest regression model

## Overview

The `metrics.py` script:

1. Retrieves latest economic indicator data from APIs (Fred, AirNow)
2. Requests user input for location, time period, and previous price
3. Loads pre-trained random forest model
4. Makes prediction on user input data 

The model is trained beforehand in `model_engine.py` using historical housing price data.

Key steps:

- Preprocesses data (scaling, one-hot encoding)
- Trains random forest regressor with hyperparameter tuning
- Saves best model to `best_model.pkl`

## Usage

To make a prediction:

1. Run `model_engine.py` to train and save model 
2. Run `metrics.py`, input requested info when prompted
3. See predicted housing price 

Required API keys:

- FRED_API_KEY 
- AIRNOW_API_KEY

## Model Training

The model is trained on 1991 housing price data. Features include economic indicators and location/time information.

Data preprocessing:

- Numeric features are standardized
- Categorical features are one-hot encoded

Hyperparameters tuned via randomized search cross validation:

- n_estimators
- max_depth
- max_features
- min_samples_split

Best model is saved to `best_model.pkl`

## Prediction

New data is passed through the same feature preprocessing transforms.

Location and time period are input by user. Economic indicators are retrieved from APIs.

Processed data is fed to trained model to generate predicted housing price.
