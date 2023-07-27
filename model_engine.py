import pandas as pd, joblib
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('1991_merged_csv.csv') # Define a function to shift the 'Price' column to get the previous year's price
encoder = LabelEncoder()


# Identify the numerical and categorical columns
numerical_cols = df.drop(['State', 'Quarter', 'Price'], axis=1).columns
categorical_cols = ['State', 'Quarter']

# Define preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Split data into features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a pipeline that preprocesses the data, then trains a RandomForestRegressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])

# Define the parameter grid for the random forest
param_grid = {
    'regressor__n_estimators': [180, 190, 205],
    'regressor__max_depth': [30, 32,  38], 
    'regressor__max_features': [0.8, 0.9, 1.0],
    'regressor__min_samples_split': [2, 5]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Set up the grid search
# Search grid 
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)
print("Iteration | MSE")
for i, params in enumerate(grid_search.cv_results_['params']):
    print(f"{i} | {-grid_search.cv_results_['mean_test_score'][i]:.3f}")

print("Best Parameters:")
print(grid_search.best_params_)
# Get best model
best_model = grid_search.best_estimator_

# Evaluate model on test set
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)



# Save model

joblib.dump(best_model, 'best_model2.pkl') 