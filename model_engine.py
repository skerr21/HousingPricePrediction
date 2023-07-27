import pandas as pd, joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('updated_data_copy.csv') 

# Define columns
feature_cols = ['CSUSHPINSA', 'FPCPITOTLZGUSA', 'CSCICP03USM665S', 'MSACSR', 'EMVELECTGOVRN', 'USSTHPI', 'ROWFDNA027N', 'Year', 'Average AQI', 'State']
feature_cols.append('Quarter')
target_col = 'Price'

# Split data
X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[target_col], test_size=0.2, stratify=data['Quarter'])

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', feature_cols[:-2]),
        ('cat', OneHotEncoder(), [feature_cols[-2], feature_cols[-1]])
    ])

# Define model 
model = RandomForestRegressor(random_state=42) 

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Grid search
# Simplify param_grid
param_grid = {
    'model__n_estimators': [180, 190, 200],
    'model__max_depth': [30, 32, 34], 
    'model__max_features': [0.8, 0.9, 1.0],
    'model__min_samples_split': [2, 5]  
}




# Add KFold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Search grid 
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=-1)

# Fit model
grid_search.fit(X_train, y_train)

# Get best model
# Get best model 
best_model = grid_search.best_estimator_

# Print best parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Evaluate model on test set
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE:", mse)
print("R-squared:", r2) 

# Save model
joblib.dump(best_model, 'best_model.pkl')

print("Model saved successfully.")
print(f"MSE: {mse}")
print(f"R-squared: {r2}")


