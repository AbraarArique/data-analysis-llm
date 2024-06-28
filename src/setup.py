from datasets import load_dataset
import pandas
from langchain_community.utilities import SQLDatabase

# Zillow home rentals dataset from Hugging Face
dataset = load_dataset("misikoff/zillow-viewer", "rentals")["train"]

# Get city, county, and state for US ZIP codes
zipData = pandas.read_csv(
    "https://raw.githubusercontent.com/scpike/us-state-county-zip/master/geo-data.csv"
)
zipHash = {}

# Create hash of ZIP codes for fast access
for row in zipData.iterrows():
    zipHash[row[1]["zipcode"]] = {
        "city": row[1]["city"],
        "county": row[1]["county"],
        "state": row[1]["state_abbr"],
    }

# Create SQLite database and table
db = SQLDatabase.from_uri("sqlite:///zillow.db")

db.run(
    """CREATE TABLE rentals (
id INTEGER PRIMARY KEY AUTOINCREMENT,
zip VARCHAR,
city VARCHAR,
county VARCHAR,
state VARCHAR,
home_type INTEGER,
date DATETIME,
rent FLOAT,
rent_adjusted FLOAT
)"""
)

# Insert 10k rows
row_count = 0
for i in range(len(dataset)):
    row = dataset[i]

    # Region Type == 0 is ZIP code data, ensure the rent column is not empty
    if row["Region Type"] != 0 or row["Rent (Smoothed)"] is None:
        continue

    # Add leading 0 to ZIP codes if missing
    zip = f"0{row['Region']}" if len(row["Region"]) < 5 else row["Region"]
    geo = zipHash.get(zip)

    # Ensure we have geo data
    if geo is None:
        continue

    db.run(
        "INSERT INTO rentals (zip, city, county, state, home_type, date, rent, rent_adjusted) VALUES (:zip, :city, :county, :state, :home_type, :date, :rent, :rent_adjusted)",
        parameters={
            "zip": zip,
            "city": geo["city"],
            "county": geo["county"],
            "state": geo["state"],
            "home_type": row["Home Type"],
            "date": row["Date"].isoformat(),
            "rent": row["Rent (Smoothed)"],
            "rent_adjusted": row["Rent (Smoothed) (Seasonally Adjusted)"],
        },
    )

    row_count += 1
    if row_count >= 10000:
        break
