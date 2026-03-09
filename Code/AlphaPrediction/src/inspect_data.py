import pandas as pd

DATA_PATH = "../Data/OptimalAlpha.xlsx"

df = pd.read_excel(DATA_PATH)

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isna().sum())

if "best_alpha" in df.columns:
    print("\n===== ALPHA DISTRIBUTION =====")
    print(df["best_alpha"].value_counts())
else:
    print("\n⚠️ Column 'best_alpha' not found — tell me the target column name.")
