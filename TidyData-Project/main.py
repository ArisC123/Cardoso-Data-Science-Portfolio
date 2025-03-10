# Applying the data cleaning concepts 
# Output the original untidy data frame

import pandas as pd

print("Original (untidy) Federal R&D budget data frame ")
df = pd.read_csv("Data/fed_rd_year&gdp.csv")
print(df)


# Melt the DataFrame: convert subject columns of years into a single 'year' column,
# with their corresponding budgets in a new 'Budget' column


