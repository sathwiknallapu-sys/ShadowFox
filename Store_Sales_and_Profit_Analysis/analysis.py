import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')

# Data Cleaning
print("Cleaning data...")
# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])
# Extract Year and Month for time series analysis
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['YearMonth'] = df['Order Date'].dt.to_period('M')

# 1. Monthly Sales Trend
print("Generating Monthly Sales Trend...")
monthly_sales = df.groupby('YearMonth')['Sales'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

plt.figure(figsize=(15, 6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='Sales', marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Sales Trend')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig('plots/monthly_sales_trend.png')
plt.close()

# 2. Profit by Category
print("Generating Profit by Category...")
category_profit = df.groupby('Category')['Profit'].sum().reset_index().sort_values(by='Profit', ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(data=category_profit, x='Category', y='Profit', palette='viridis')
plt.title('Total Profit by Category')
plt.xlabel('Category')
plt.ylabel('Total Profit')
plt.tight_layout()
plt.savefig('plots/profit_by_category.png')
plt.close()

# 3. Top 10 Sub-Categories by Sales
print("Generating Top 10 Sub-Categories by Sales...")
subcat_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=subcat_sales, x='Sales', y='Sub-Category', palette='mako')
plt.title('Top 10 Sub-Categories by Sales')
plt.xlabel('Total Sales')
plt.ylabel('Sub-Category')
plt.tight_layout()
plt.savefig('plots/top_10_subcategories_by_sales.png')
plt.close()

# 4. Profit by Region
print("Generating Profit by Region...")
region_profit = df.groupby('Region')['Profit'].sum().reset_index().sort_values(by='Profit', ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(data=region_profit, x='Region', y='Profit', palette='magma')
plt.title('Total Profit by Region')
plt.xlabel('Region')
plt.ylabel('Total Profit')
plt.tight_layout()
plt.savefig('plots/profit_by_region.png')
plt.close()

# Analysis summary
print("\n--- Quick Insights ---")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Most Profitable Category: {category_profit.iloc[0]['Category']} (${category_profit.iloc[0]['Profit']:,.2f})")
print(f"Least Profitable Category: {category_profit.iloc[-1]['Category']} (${category_profit.iloc[-1]['Profit']:,.2f})")
print(f"Most Profitable Region: {region_profit.iloc[0]['Region']} (${region_profit.iloc[0]['Profit']:,.2f})")
print("Analysis complete! Visualizations have been saved to the 'plots' folder.")
