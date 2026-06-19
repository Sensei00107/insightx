import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
N_ORDERS      = 5000
N_CUSTOMERS   = 1200
START_DATE    = datetime(2023, 1, 1)
END_DATE      = datetime(2023, 12, 31)
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "data", "ecommerce_data.csv")

# ── REFERENCE DATA ────────────────────────────────────────────────────────────
CATEGORIES = {
    "Electronics":   {"products": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Phone Case", "USB Hub"],        "price_range": (15,  350)},
    "Clothing":      {"products": ["T-Shirt", "Jeans", "Hoodie", "Sneakers", "Jacket"],                                     "price_range": (12,  180)},
    "Home & Kitchen":{"products": ["Air Fryer", "Coffee Maker", "Blender", "Bed Sheets", "Scented Candle"],                 "price_range": (10,  220)},
    "Books":         {"products": ["Python Programming", "Data Science Handbook", "Self-Help Guide", "Novel", "Cook Book"], "price_range": (8,   55)},
    "Beauty":        {"products": ["Face Serum", "Lip Balm", "Hair Mask", "Sunscreen", "Moisturiser"],                      "price_range": (7,   90)},
}

REGIONS     = ["North", "South", "East", "West", "Central"]
CHANNELS    = ["Organic Search", "Paid Ad", "Social Media", "Email", "Direct"]
STATUSES    = ["Completed", "Completed", "Completed", "Returned", "Cancelled"]   # weighted
GENDERS     = ["Male", "Female", "Other"]
AGE_GROUPS  = ["18-24", "25-34", "35-44", "45-54", "55+"]

# ── CUSTOMERS ─────────────────────────────────────────────────────────────────
customer_ids   = [f"CUST{str(i).zfill(4)}" for i in range(1, N_CUSTOMERS + 1)]
customer_region = {cid: random.choice(REGIONS)    for cid in customer_ids}
customer_gender = {cid: random.choice(GENDERS)    for cid in customer_ids}
customer_age    = {cid: random.choice(AGE_GROUPS) for cid in customer_ids}

# ── ORDERS ────────────────────────────────────────────────────────────────────
records = []
date_range_days = (END_DATE - START_DATE).days

for i in range(N_ORDERS):
    order_date = START_DATE + timedelta(days=int(np.random.triangular(0, date_range_days * 0.6, date_range_days)))
    category   = random.choices(list(CATEGORIES.keys()), weights=[3, 2.5, 2, 1.5, 1])[0]
    cat_info   = CATEGORIES[category]
    product    = random.choice(cat_info["products"])
    lo, hi     = cat_info["price_range"]
    unit_price = round(random.uniform(lo, hi), 2)
    quantity   = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
    discount   = random.choices([0, 0.05, 0.10, 0.15, 0.20], weights=[50, 20, 15, 10, 5])[0]
    revenue    = round(unit_price * quantity * (1 - discount), 2)
    customer   = random.choice(customer_ids)

    records.append({
        "order_id":      f"ORD{str(i+1).zfill(5)}",
        "order_date":    order_date.strftime("%Y-%m-%d"),
        "customer_id":   customer,
        "gender":        customer_gender[customer],
        "age_group":     customer_age[customer],
        "region":        customer_region[customer],
        "category":      category,
        "product":       product,
        "quantity":      quantity,
        "unit_price":    unit_price,
        "discount":      discount,
        "revenue":       revenue,
        "channel":       random.choice(CHANNELS),
        "status":        random.choice(STATUSES),
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset saved → {OUTPUT_PATH}")
print(f"Shape: {df.shape}")
print(df.head(3))
