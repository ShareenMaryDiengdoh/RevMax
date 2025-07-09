import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

products = [
    ('P001', 'Energy Drink', 'Beverages'),
    ('P002', 'Protein Bar', 'Snacks'),
    ('P003', 'Multivitamin', 'Health'),
    ('P004', 'Yoga Mat', 'Fitness'),
    ('P005', 'Running Shoes', 'Apparel'),
    ('P006', 'Sports Bottle', 'Accessories'),
    ('P007', 'Fitness Tracker', 'Electronics'),
    ('P008', 'Gym Gloves', 'Accessories'),
    ('P009', 'Whey Protein', 'Health'),
    ('P010', 'Massage Gun', 'Electronics'),
]

store_locations = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata']

data = []

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

for date in date_range:
    for prod in products:
        units_sold = np.random.poisson(lam=random.randint(2, 10))
        price = round(random.uniform(10, 200), 2)
        promotion = random.choice(['Yes', 'No'])
        location = random.choice(store_locations)
        revenue = round(price * units_sold, 2)
        data.append([
            prod[0], prod[1], prod[2], price, units_sold, date.strftime('%Y-%m-%d'),
            location, promotion, revenue
        ])

df = pd.DataFrame(data, columns=[
    'ProductID', 'ProductName', 'Category', 'Price',
    'UnitsSold', 'Date', 'StoreLocation', 'PromotionApplied', 'Revenue'
])

df.to_csv('data/sales_data.csv', index=False)
print("✅ Mock sales data generated.")