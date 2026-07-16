#sample program of how to generate data set using predefined rules.

import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

products = [
    ("Laptop", 800, 2000),
    ("Mouse", 10, 80),
    ("Keyboard", 20, 150),
    ("Monitor", 120, 600),
    ("Phone", 300, 1200)
]

countries = {
    "USA": 0.08,
    "Germany": 0.19,
    "India": 0.18,
    "Japan": 0.10
}


def random_date():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


records = []

for i in range(1000):

    product, low, high = random.choice(products)

    price = round(random.uniform(low, high), 2)

    quantity = random.randint(1, 5)

    country = random.choice(list(countries.keys()))

    tax_rate = countries[country]

    subtotal = round(price * quantity, 2)

    tax = round(subtotal * tax_rate, 2)

    total = round(subtotal + tax, 2)

    records.append({
        "transaction_id": i + 1,
        "date": random_date().strftime("%Y-%m-%d"),
        "country": country,
        "product": product,
        "unit_price": price,
        "quantity": quantity,
        "subtotal": subtotal,
        "tax": tax,
        "total": total
    })

df = pd.DataFrame(records)

print(df.head())

df.to_csv("synthetic_transactions.csv", index=False)

print("Saved", len(df), "records.")
