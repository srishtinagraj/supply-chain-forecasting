import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sales_data(num_records=10000, num_products=5):
    """
    Generate simulated Apple product sales data
    Includes: Date, Product, Units Sold, Price, Regional Demand, Seasonality
    """
    np.random.seed(42)
    
    products = [f"iPhone_{i}" for i in range(num_products)]
    regions = ["US", "EU", "APAC", "China", "Other"]
    
    dates = [datetime.now() - timedelta(days=x) for x in range(365)]
    
    data = []
    for _ in range(num_records):
        date = np.random.choice(dates)
        product = np.random.choice(products)
        region = np.random.choice(regions)
        
        # Base demand with seasonality (higher in Q4)
        base_demand = np.random.poisson(500)
        seasonal_factor = 1.5 if date.month in [11, 12] else 1.0
        units_sold = int(base_demand * seasonal_factor)
        
        price = np.random.choice([799, 999, 1099, 1199, 1299])
        revenue = units_sold * price
        
        # Social sentiment (1-5 scale) correlates with demand
        social_sentiment = np.random.uniform(3, 5)
        
        data.append({
            'date': date,
            'product': product,
            'region': region,
            'units_sold': units_sold,
            'price': price,
            'revenue': revenue,
            'social_sentiment': social_sentiment
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sales_data.csv', index=False)
    print(f"Generated {len(df)} sales records")
    return df

if __name__ == "__main__":
    generate_sales_data()