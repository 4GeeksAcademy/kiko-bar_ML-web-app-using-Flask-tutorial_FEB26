import pandas as pd

# 1. Load the dataset
# Make sure the path to the 'adult-census-income.csv' is correct
df = pd.read_csv('adult-census-income.csv')

# 2. Cleanup: Remove rows with '?' (often used in this dataset for missing data)
df = df.replace('?', pd.NA).dropna()

# 3. Decision: We keep 'income' as text (<=50K / >50K) for the UI
# But we won't fit the model on ONLY high earners, per our previous decision.

# 4. Feature Selection: Keep the columns we actually want to use
columns_to_keep = ['age', 'education', 'marital.status', 'occupation', 'hours.per.week', 'income']
df = df[columns_to_keep]

print("Step 1 Complete: Data loaded and cleaned.")
print(df.head())