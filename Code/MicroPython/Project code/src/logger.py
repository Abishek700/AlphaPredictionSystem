## @file logger.py
#  @brief Provides centralized logging configuration for the project.

import logging

## @brief Configures logging settings for the project.
#  @details Sets the logging level, format, and timestamp for all log entries in the project.
def configureLogging():
    """Configure logging settings for the entire project."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging is configured.")