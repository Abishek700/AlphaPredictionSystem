"""
@file errorhandling.py
@brief Handles error detection and validation for input data.
"""

import pandas as pd
import logging

def load_data(filepath):
    """
    @brief Loads and validates data from a CSV file.
    @param filepath Path to the CSV file.
    @return Validated DataFrame.
    """
    logging.info(f"Loading data from {filepath}.")
    try:
        data = pd.read_csv(filepath)
        if data.isnull().any().any():
            raise ValueError("Dataset contains missing values.")
        logging.info(f"Data loaded successfully with shape: {data.shape}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise