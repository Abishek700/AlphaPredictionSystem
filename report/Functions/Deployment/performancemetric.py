import logging  # Library for logging messages

## @brief Evaluate the performance of a trained model.
#  @details This function evaluates the provided model's performance on the given features and labels.
#           It calculates the loss and accuracy metrics and logs the results.
#
#  @param model The trained TensorFlow/Keras model to be evaluated.
#  @param features The input features for evaluation, typically a NumPy array or Tensor.
#  @param labels The ground truth labels corresponding to the input features.
#  @return None
#
#  @section Libraries Used
#  - logging: For logging evaluation results and progress.
#  - TensorFlow/Keras: For model evaluation using `evaluate`.
def evaluate_model(model, features, labels):
    """
    Evaluate the performance of a trained model.

    Args:
        model: The trained TensorFlow/Keras model to be evaluated.
        features: Input features for evaluation.
        labels: Ground truth labels corresponding to the features.

    Returns:
        None. Logs the accuracy and loss to the console.
    """
    try:
        logging.info("Evaluating model performance.")  # Log start of evaluation
        loss, accuracy = model.evaluate(features, labels, verbose=0)  # Evaluate model
        logging.info(f"Model evaluation completed: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")  # Log results
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")  # Log any errors encountered
        raise