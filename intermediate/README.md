# Store Sales and Profit Analysis

## Overview
This project provides a comprehensive analysis of the sales and profit performance of a retail store using the "Sample - Superstore" dataset. The goal is to leverage data-driven insights to identify areas for improvement and drive revenue and growth. 

## Key Insights
Based on our exploratory data analysis, we uncovered several actionable insights:
1. **Total Performance**: The store has generated significant total sales with a solid overall profit margin.
2. **Category Performance**: 
   - **Technology** and **Office Supplies** are typically the most profitable categories.
   - **Furniture** often shows high sales but low profit margins (sometimes even operating at a loss in certain regions).
3. **Regional Disparities**: The East and West regions tend to outperform the Central and South regions in terms of overall profitability.
4. **Sub-Category Leaders**: Categories like Phones, Chairs, and Storage drive the bulk of the sales volume.

## Project Structure
- `Sample - Superstore.csv`: The dataset used for analysis.
- `analysis.py`: Python script containing data cleaning, aggregation, and visualization logic.
- `plots/`: Directory containing generated visualizations:
  - `monthly_sales_trend.png`: Time series of sales over months.
  - `profit_by_category.png`: Breakdown of profit across major product categories.
  - `top_10_subcategories_by_sales.png`: Top-selling sub-categories.
  - `profit_by_region.png`: Profitability mapped by region.

## How to Run the Analysis
1. Ensure you have Python installed on your system.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python analysis.py
   ```
5. Check the `plots` folder for the generated visual insights!
