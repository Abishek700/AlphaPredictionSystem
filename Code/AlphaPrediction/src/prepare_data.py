import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "OptimalAlpha.xlsx"
)

def prepare_data():
    df = pd.read_excel(DATA_PATH)

    # Encode research domain
    le = LabelEncoder()
    df["research_domain"] = le.fit_transform(df["source_project"])

    # Feature engineering
    df["statistical_power_proxy"] = df["sample_size"] * df["effect_size"]
    df["log_sample_size"] = np.log1p(df["sample_size"])
    df["effect_squared"] = df["effect_size"] ** 2

    features = [
        "sample_size",
        "effect_size",
        "research_domain",
        "replication_success",
        "statistical_power_proxy",
        "log_sample_size",
        "effect_squared"
    ]

    X = df[features].values
    y = df["best_alpha"].values.astype(float)

    return X, y
