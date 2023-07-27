import pandas as pd

# Load the data from CSV files
df1 = pd.read_csv('updated_data_copy.csv')
df2 = pd.read_csv('HPI_AT_state.csv', names=['State', 'Year', 'Quarter', 'Prev_Year_Price'])

# Shift the year by 1 to represent the previous year
df2['Year_Shifted'] = df2['Year'] + 1

# Group by state and the shifted year to calculate the median price
df2_grouped = df2.groupby(['State', 'Year_Shifted'])['Prev_Year_Price'].median().reset_index()

# Rename the columns to match the first DataFrame
df2_grouped = df2_grouped.rename(columns={'Year_Shifted': 'Year', 'Prev_Year_Price': 'Prev_Year_Median_Price'})

# Merge the first DataFrame with the grouped data from the second DataFrame
merged_df = pd.merge(df1, df2_grouped, on=['State', 'Year'], how='left')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('1991_merged_csv.csv', index=False)

# Print a success message
print("CSV file has been saved successfully!")
