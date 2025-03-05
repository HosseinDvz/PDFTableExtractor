import os
import shutil
import pandas as pd

class DataFrameSaver:
    def __init__(self, folder_name="csv_tables", clean_folder=True):
        """
        Initializes the DataFrameSaver class, ensures the target folder exists,
        and cleans any existing files in the folder.
        
        :param folder_name: Name of the folder where CSV files will be saved.
        """
        self.folder_name = folder_name
        if clean_folder:
            self.clean_folder()  # Clean the folder on initialization

    def clean_folder(self):
        """Deletes all existing files in the target folder."""
        if os.path.exists(self.folder_name):
            shutil.rmtree(self.folder_name)  # Remove the folder and all its contents
        os.makedirs(self.folder_name, exist_ok=True)  # Recreate the folder

    def save(self, df, filename):
        """
        Saves a given DataFrame as a CSV file in the specified folder.

        :param df: Pandas DataFrame to be saved.
        :param filename: Name of the file (without .csv extension).
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")
        
        file_path = os.path.join(self.folder_name, f"{filename}.csv")
        df.to_csv(file_path, index=True) 
        print(f"DataFrame saved successfully at: {file_path}")

# Example usage:
if __name__ == "__main__":
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  # Sample DataFrame
    saver = DataFrameSaver()
    saver.save(df, "example_table")
