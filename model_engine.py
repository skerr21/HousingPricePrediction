import pandas as pd, joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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
param_grid = {
    'model__n_estimators': [10, 50, 100, 250, 500, 1000],
    'model__max_depth': [None, 5, 10, 25, 50, 75, 100], 
    'model__min_samples_split': [2, 5, 10, 20, 50],
    'model__min_samples_leaf': [1, 2, 4, 10, 20],
    'model__max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
    'model__criterion': ['mse', 'mae'],
    'model__max_leaf_nodes': [10, 50, 100, 500], 
    'model__bootstrap': [True, False, 'subsample'],
    'model__oob_score': [True, False]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate model on test set
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Evaluate model on test set 
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Save model

joblib.dump(best_model, 'best_model.pkl') 

