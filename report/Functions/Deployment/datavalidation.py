# Required Libraries
import pandas as pd  # For handling and manipulating data

## @brief Load and validate a dataset from a CSV file.
#  @param filepath Path to the CSV file to load.
#  @return A Pandas DataFrame containing the loaded data.
#  @exception ValueError Raised if the dataset contains missing values or missing columns.
#  @exception FileNotFoundError Raised if the file is not found.
#  @exception Exception Raised for any other error during data loading.
def load_data(filepath):
    """
    Load and validate a dataset from a CSV file.

    This function reads data from the specified CSV file and performs the following checks:
    - Ensures no missing values are present in the dataset.
    - Validates the presence of required columns: ['Step_length', 'Stride_length', 'Step_time'].

    :param filepath: Path to the CSV file.
    :type filepath: str
    :return: Loaded data as a Pandas DataFrame.
    :rtype: pd.DataFrame
    :raises ValueError: If the dataset contains missing values or lacks required columns.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises Exception: For any other errors during data loading.
    """
    try:
        # Load the CSV file into a Pandas DataFrame
        data = pd.read_csv(filepath)

        # Check for missing values
        if data.isnull().values.any():
            raise ValueError("Dataset contains missing values.")

        # Validate required columns
        required_columns = ['Step_length', 'Stride_length', 'Step_time']
        if not all(column in data.columns for column in required_columns):
            raise ValueError("Dataset is missing required columns.")

        # Return the validated data
        return data

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        raise

    except Exception as e:
        print(f"Error during data loading: {e}")
        raise