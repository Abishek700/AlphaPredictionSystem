## @file kalmanfilter.py
#  @brief Contains functions to apply Kalman Filter to smooth data columns.

from pykalman import KalmanFilter
import logging

## @brief Applies Kalman Filter to smooth specific columns in a dataset.
#  @param data The dataset (e.g., pandas DataFrame) containing the columns to be smoothed.
#  @param columns List of column names to which the Kalman Filter should be applied.
#  @param params Dictionary of Kalman Filter parameters, including:
#     - "kalmanTransitionMatrices": State transition matrix.
#     - "kalmanObservationMatrices": Observation matrix.
#     - "kalmanInitialStateMean": Initial state mean.
#     - "kalmanInitialStateCovariance": Initial state covariance.
#     - "kalmanObservationCovariance": Observation covariance.
#     - "kalmanTransitionCovariance": Transition covariance.
#  @return The dataset with smoothed columns.
#  @exception Raises an exception if a specified column is missing or an error occurs.
def applyKalmanFilter(data, columns, params):
    """Apply Kalman Filter to smooth data columns."""
    try:
        logging.info("Applying Kalman Filter.")
        kf = KalmanFilter(
            transition_matrices=params["kalmanTransitionMatrices"],
            observation_matrices=params["kalmanObservationMatrices"],
            initial_state_mean=params["kalmanInitialStateMean"],
            initial_state_covariance=params["kalmanInitialStateCovariance"],
            observation_covariance=params["kalmanObservationCovariance"],
            transition_covariance=params["kalmanTransitionCovariance"]
        )
        for column in columns:
            if column not in data:
                logging.error(f"Column '{column}' not found in data.")
                raise KeyError(f"Column '{column}' is missing.")
            data[column], _ = kf.filter(data[column])
            logging.info(f"Kalman Filter applied to column: {column}")
        return data
    except Exception as e:
        logging.error(f"Error applying Kalman Filter: {e}")
        raise