import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from joblib import load

# Load model
model = load('best_model.pkl') 

# Sample input data
input_data = {
  'CSUSHPINSA': 123.4,
  'FPCPITOTLZGUSA': 123.4,
  'CSCICP03USM665S': 110.2, 
  'MSACSR': 102.7,
  'EMVELECTGOVRN': 10000,
  'USSTHPI': 280,
  'ROWFDNA027N': 123,
  'Year': 1993,
  'Average AQI': 90,
  'State': 'CA',
  'Quarter': 3 
}

# Create dataframe 
df = pd.DataFrame(input_data, index=[0])

# One-hot encode state
ohe = OneHotEncoder()
df = pd.concat([df, pd.DataFrame(ohe.fit_transform(df[['State']]).toarray())], axis=1)

# Remove original state column
# df.drop(columns=['State'], inplace=True)

# Make prediction
prediction = model.predict(df)

print('Predicted Price:', prediction[0])
