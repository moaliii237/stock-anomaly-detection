import pandas as pd


def get_sample_data(filename: str, num_rows: int):
    """
    Reads a CSV file and returns a DataFrame with a specified number of rows.

    Parameters:
    filename (str): The path to the CSV file.
    num_rows (int): The number of rows to read from the file.

    Returns:
    pd.DataFrame: A DataFrame containing the specified number of rows.
    """
    try:
        df = pd.read_csv(filename, nrows=num_rows)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return pd.DataFrame()


def save_to_csv(df: pd.DataFrame, filename: str):
    """
    Saves a DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    filename (str): The path where the CSV file will be saved.
    """
    try:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving the file: {e}")


if __name__ == "__main__":
    filename_in = "data/BKNG_engineering.csv"
    filename_out = "data/BKNG_engineering_sample.csv"

    num_rows = 100
    df_sample = get_sample_data(filename_in, num_rows)
    if not df_sample.empty:
        save_to_csv(df_sample, filename_out)
    else:
        print("No data to save.")


