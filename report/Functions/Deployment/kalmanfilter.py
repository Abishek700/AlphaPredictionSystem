"""
@file kalmanfilter.py
@brief Applies Kalman Filter to smooth data columns.
"""

from pykalman import KalmanFilter
import logging

def apply_kalman_filter(data, columns):
    """
    @brief Smoothens specified data columns using Kalman Filter.
    @param data DataFrame with raw data.
    @param columns List of column names to smooth.
    @return DataFrame with smoothed data.
    """
    logging.info("Applying Kalman Filter.")
    try:
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )
        for column in columns:
            data[column], _ = kf.filter(data[column])
            logging.info(f"Column '{column}' smoothed successfully.")
        return data
    except Exception as e:
        logging.error(f"Error in Kalman Filter: {e}")
        raise