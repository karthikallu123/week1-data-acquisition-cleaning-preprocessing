"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Virtual Data Science with Python Trainee

Dataset:
Kaggle Titanic - Machine Learning from Disaster
Direct reproducible CSV mirror:
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
LOCAL_FILE = Path("titanic.csv")

# 1. Data acquisition
try:
    df = pd.read_csv(DATA_URL)
except Exception:
    # If internet is unavailable, place titanic.csv beside this script.
    df = pd.read_csv(LOCAL_FILE)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe(include="all").T)

# 2. Normalize text fields
df["Sex"] = df["Sex"].str.strip().str.lower()
df["Embarked"] = df["Embarked"].str.strip().str.upper()

# 3. Preserve cabin availability as a useful indicator
df["CabinKnown"] = df["Cabin"].notna().astype(int)

# 4. Feature extraction
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")[0].str.strip()
common_titles = {"Mr", "Miss", "Mrs", "Master"}
df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 5. Missing-value treatment
# Median is used for Age because it is robust to skew/outliers.
df["Age"] = df["Age"].fillna(df["Age"].median())

# Mode is used for the two missing Embarked values.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin is more than 77% missing. Keep CabinKnown and remove raw Cabin.
df = df.drop(columns=["Cabin"])

# 6. Remove exact duplicate records
df = df.drop_duplicates().reset_index(drop=True)

# 7. Select analysis-ready columns
clean_cols = [
    "PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "CabinKnown", "Title", "FamilySize", "IsAlone"
]
clean = df[clean_cols]

# 8. Encode categorical features for downstream ML
model_ready = pd.get_dummies(
    clean.drop(columns=["PassengerId"]),
    columns=["Sex", "Embarked", "Title"],
    drop_first=False,
    dtype=int
)

# 9. Final quality checks
print("\nFinal shape:", clean.shape)
print("\nRemaining missing values:\n", clean.isna().sum())
print("\nFinal duplicate rows:", clean.duplicated().sum())

clean.to_csv("titanic_cleaned.csv", index=False)
model_ready.to_csv("titanic_model_ready.csv", index=False)

print("\nSaved: titanic_cleaned.csv")
print("Saved: titanic_model_ready.csv")
