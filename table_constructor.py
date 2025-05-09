import numpy as np
import pandas as pd
import re

import warnings
warnings.filterwarnings("ignore")

class TableConstruction:
    
    """
    A utility class to reconstruct structured tabular data from OCR-extracted text and bounding boxes.

    This class partitions OCR data into rows and columns based on spatial coordinates, cleans numeric values,
    aligns misaligned cells, and returns a clean DataFrame representing the original table layout.

    Attributes:
        df (pd.DataFrame): The Input of this process is a dataframe created from the OCR’s output with at 
        least 'bbox' and 'text' columns. 
    """
    
    def __init__(self, df):
        
        """
        Initializes the TableConstruction object and prepares internal representation.

        Args:
            df (pd.DataFrame): The OCR result containing bounding boxes and text.
        """
        
        
        self.df = df.copy()  # Avoid modifying the original dataframe
        self._row_col_coordinates()
        self.df['text'] = self.df['text'].apply(self._clean_numeric_value)
        
    
    def table_creator(self,row_column='row_pos',row_threshold=15, col_column='col_pos', col_threshold=30):
        
        """
        Constructs a DataFrame representing the detected table by grouping data into rows and columns.

        Args:
            row_column (str): Name of the column representing row positions. Default is 'row_pos'.
            row_threshold (int): Vertical pixel threshold to consider values in the same row.
                i.e. values that belong to the same row have close y coordinates
            col_column (str): Name of the column representing column positions. Default is 'col_pos'.
            col_threshold (int): Horizontal pixel threshold to consider values in the same column.
                i.e. values that belong to the same column have close x coordinates

        Returns:
            pd.DataFrame: A structured table with cleaned values and aligned headers/rows.
        """
        
        partitions = self._partition_dataframe(self.df,column=row_column,row_threshold=row_threshold)
        processed_dfs = self._process_dataframes(partitions,column=col_column, col_threshold=col_threshold)
        
        if processed_dfs:
            
            column_names = list(processed_dfs[0]['text'].values)  # Use the first partition for column names
            if column_names[0] == '-':
                column_names[0] = 'Sample_Name' 
            column_names = np.array(column_names)
            
            rows = []
            for processed_dfs in processed_dfs[1:]:
                row_values = list(processed_dfs['text'].values)

                row_values = [values.replace(',', '') for values in row_values]
                rows.append(np.array(row_values))

            
        
        final_df = pd.DataFrame(rows, columns=column_names)
        final_df.set_index(final_df.columns[0], inplace=True)
            

        return final_df
    

    def _row_col_coordinates(self):
        """Extracts row and column positions from bbox and removes bbox."""
        self.df['col_pos'] = self.df['bbox'].apply(lambda x: x[0][0])
        self.df['row_pos'] = self.df['bbox'].apply(lambda x: x[-1][-1])
        self.df.drop('bbox', axis=1, inplace=True)
    
    
    def _clean_numeric_value(self, value):
        if isinstance(value, str) and value.strip():  # Check for non-empty string
            if value.lstrip()[0].isdigit():  
                match = re.match(r'(-?\d+\.?\d*)', value)
                return match.group(0) if match else value
        return value  # Leave pure text and non-string values as they are
    
    
    # Partition logic
    def _partition_dataframe(self, df,  column='row_pos', row_threshold=15):
        """ It creates a list of dataframes from the output of OCR. Each dataframe is one row of original table
        It uses the text box coordinate found by EasyOCR to determine which values belong to a row"""
        partitions = []
        current_partition = [df.iloc[0]]  # Start with the first row
        current_value = self.df.iloc[0][column]  # Use the first row's value as the reference

        for i in range(1, len(df)):
            
            #print(current_value)
            # If the current row's value is within the threshold, add it to the current partition
            if abs(df.iloc[i][column] - current_value) <= row_threshold:
                current_partition.append(df.iloc[i])
            else:
                # Create a new partition
                partitions.append(pd.DataFrame(current_partition))
                current_partition = [df.iloc[i]]
                current_value = df.iloc[i][column]  # Update the reference value

        # Add the last partition
        if current_partition:
            partitions.append(pd.DataFrame(current_partition))

        return partitions


    def _align_and_merge(self, df_small, df_large, column='col_pos', col_tolerance=30):
        """
        Adjusts numeric values within a tolerance range to be identical, or assigns the nearest smaller value.
        the column coordinates (which determone the position of a value in a row) becomes identical to coordinate
        above (or below, whaterver is available).
        Then, it performs a left join to merge both DataFrames. If the two rows have a table have different length,
        sum NaN values will be inserted in the missing parts and both rows now have the same length. remember that
        from the previos function, each row is dataframe.

        :param df_small: Smaller DataFrame
        :param df_large: Larger DataFrame
        :param column: Numeric column to compare
        :param tolerance: Range within which numbers will be made equal
        :return: Merged DataFrame with missing rows added
        """

        df_small = df_small.copy()
        df_large = df_large.copy()

        # Sort df_large for efficient searching
        df_large = df_large.sort_values(by=column)

        for i in list(df_small.index):
            value = df_small.at[i, column]
            
            # Find values within the tolerance range
            close_values = df_large[(df_large[column] - value).abs() <= col_tolerance][column]

            if not close_values.empty:
                # If a close match is found, take the first one
                df_small.at[i, column] = close_values.iloc[0]
            else:
                # If no close match, find the nearest smaller value
                smaller_values = df_large[df_large[column] < value][column]
                if not smaller_values.empty:
                    df_small.at[i, column] = smaller_values.iloc[-1]  # Take the largest smaller value
        # Perform left join
        merged_df = df_large.merge(df_small, on=column, how='outer', suffixes=('_large', ''))
        final_df = merged_df[['text', 'col_pos', 'row_pos']]
        final_df['row_pos'] = final_df['row_pos'].ffill()
        final_df['row_pos'] = final_df['row_pos'].bfill()
        final_df['text'] = final_df['text'].fillna('-')
        df_grouped = final_df.groupby("col_pos", as_index=False).agg({
        "text": " ".join,  # Concatenate text values
        "row_pos": "first"  # Keep the first row_pos value
        })
        return df_grouped

    
    def _process_dataframes(self, partitions, column='col_pos', col_threshold=30):
        """
        Finds the longest DataFrame and applies the alignment function to all smaller ones.

        :param dataframes: List of DataFrames
        :param column: Numeric column to compare
        :param tolerance: Range within which numbers will be made equal
        :return: List of adjusted DataFrames
        """
        # Find the longest DataFrame (based on row count)
        df_large = max(partitions[1:], key=len)

        # Apply the alignment function to all smaller DataFrames
        processed_dataframes = [self._align_and_merge(df, df_large, column, col_threshold) if len(df) != len(df_large) else df for df in partitions]

        return processed_dataframes

    
    
    
    
if __name__ == '__main__':
    
    from ocr import Ocr
    
    image_path = "/Users/hosseindavarzanisani/AC/Semester_2/AI Project_2/AAFC/Codes/table_images/page_2_T_1.jpg"  # Replace with your image path
    ocr_instance = Ocr(image_path)  # Create an instance of the Ocr class
    
    # Apply OCR with preprocessing (resize and 300 DPI adjustment)
    df = ocr_instance.apply_easyocr(dpi_adjustment=False, dilation=True, resize_float=0.4)

    table = TableConstruction(df)
    table = table.table_creator()
    
    print(table)