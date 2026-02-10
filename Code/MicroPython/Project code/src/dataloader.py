## @file dataloader.py
#  @brief Contains functions to load datasets for the machine learning pipeline.

import pandas as pd
import logging

## @brief Loads data from a CSV file into a pandas DataFrame.
#  @param filepath Path to the CSV file.
#  @return A pandas DataFrame containing the data from the file.
#  @exception Raises an exception if the file cannot be loaded.
def loadData(filepath):
    """Load data from a CSV file."""
    try:
        logging.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise