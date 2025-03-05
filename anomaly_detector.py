import pandas as pd
import numpy as np

class AnomalyDetector:
    def __init__(self, threshold=10):
        self.threshold = threshold  # Set threshold for anomalies

    def find_large_anomalies(self, df):
        anomalies_dict = {}

        for column in df.columns:
            column_values = df[column].astype(str).values  # Ensure all values are strings
            if isinstance(column_values[0], list) :
                column_values = [item for sublist in column_values for item in sublist]
                
            elif isinstance(column_values[0], np.ndarray):
                arr_list_converted = [arr.tolist() for arr in column_values]
                column_values = [str(item )for sublist in arr_list_converted for item in sublist]
            else:
                column_values = [str(val) for val in column_values]
            
            #column_values = column_values[1:]
            numeric_values = []
            anomalies = []

            # Convert values to float where possible
            for value in column_values:
                try:
                    num_value = float(value)  # Convert to float
                    numeric_values.append(num_value)
                except ValueError:
                    if value != '-':
                        anomalies.append(value)  # If conversion fails, it's an anomaly

            # Detect numeric anomalies using Median Absolute Deviation (MAD)
            if numeric_values:
                median = np.median(numeric_values)
                #mad = np.median(np.abs(numeric_values - median))  # Compute MAD

                # Define threshold for anomalies (values threshold times larger/smaller than median)
                for value in numeric_values:
                    if value > median * self.threshold or value < median / self.threshold:
                        anomalies.append(str(value))  # Store as string for consistency

            if anomalies:
                anomalies_dict[column] = anomalies  # Store anomalies for this column
        
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in anomalies_dict.items()]))

        return df