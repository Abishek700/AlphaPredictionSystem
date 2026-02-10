## @file visualize.py
#  @brief Contains functions to visualize predictions made by the CNN model.

import matplotlib.pyplot as plt
import logging

## @brief Visualizes the predictions made by the CNN model compared to the true labels.
#  @param model The trained CNN model.
#  @param features Input features used for predictions.
#  @param labels Ground truth labels for comparison.
#  @exception Raises an exception if visualization fails.
def visualizePredictions(model, features, labels):
    """Visualize the predictions vs true labels."""
    try:
        logging.info("Visualizing predictions.")
        predictions = model.predict(features)
        plt.figure(figsize=(10, 6))
        plt.plot(labels, 'b', label='True Labels')
        plt.plot(predictions, 'r', linestyle='--', label='Predictions')
        plt.title('True Labels vs Predictions')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing predictions: {e}")
        raise