import os
import pandas as pd
import numpy as np

# Load the dataset from a CSV file
csv_file = 'Plastic Waste Around the World.csv'  # Replace with your dataset file name
data = pd.read_csv(csv_file)

# Calculate the min and max values for each variable
total_plastic_min = data['Total_Plastic_Waste_MT'].min()
total_plastic_max = data['Total_Plastic_Waste_MT'].max()

recycling_rate_min = data['Recycling_Rate'].min()
recycling_rate_max = data['Recycling_Rate'].max()

per_capita_waste_min = data['Per_Capita_Waste_KG'].min()
per_capita_waste_max = data['Per_Capita_Waste_KG'].max()

# Function to categorize the variables into High, Medium, Low
def categorize_variable(value, min_value, max_value):
    range_value = max_value - min_value
    low = min_value + range_value * 0.33
    high = min_value + range_value * 0.66
    
    if value <= low:
        return 'Low'
    elif value <= high:
        return 'Medium'
    else:
        return 'High'

# Apply the categorization for Total_Plastic_Waste_MT
data['Total_Plastic_Waste_Category'] = data['Total_Plastic_Waste_MT'].apply(
    categorize_variable, args=(total_plastic_min, total_plastic_max))

# Apply the categorization for Recycling_Rate
data['Recycling_Rate_Category'] = data['Recycling_Rate'].apply(
    categorize_variable, args=(recycling_rate_min, recycling_rate_max))

# Apply the categorization for Per_Capita_Waste_KG
data['Per_Capita_Waste_Category'] = data['Per_Capita_Waste_KG'].apply(
    categorize_variable, args=(per_capita_waste_min, per_capita_waste_max))

# Count the number of high, medium, and low values for each category
category_counts = {
    'Total_Plastic_Waste_MT': data['Total_Plastic_Waste_Category'].value_counts(),
    'Recycling_Rate': data['Recycling_Rate_Category'].value_counts(),
    'Per_Capita_Waste_KG': data['Per_Capita_Waste_Category'].value_counts()
}

# Print the counts for each category
for variable, counts in category_counts.items():
    print(f"Counts for {variable}:")
    print(counts)
    print()
