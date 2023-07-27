import pandas as pd, joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('1991_merged_csv.csv') # Define a function to shift the 'Price' column to get the previous year's price
encoder = LabelEncoder()


# Identify the numerical and categorical columns
numerical_cols = df.drop(['State', 'Price'], axis=1).columns
categorical_cols = ['State']

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
grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1)  # n_jobs=-1 to use all CPU cores


# Train the model using the grid search estimator
# This will take the pipeline and find the best parameters
grid_search.fit(X_train, y_train)

# Print the best parameters it found
print("Best parameters:")
print(grid_search.best_params_)

# Evaluate the model
print("Model score on test data:")
print(grid_search.score(X_test, y_test))

# Save the model
joblib.dump(grid_search, 'model.pkl')
