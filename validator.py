import io
import time
import json
import pandas as pd
import google.generativeai as genai


# gemini-1.5-flash-001
# gemini-2.0-flash-exp
class TableDataValidator:
    
    """
    A class that uses Gemini AI to validate and correct table data extracted from images by 
    sending the output of TableConstruction class and the corresponding image to Gemeni.
    Gemini will compare the image with csv file and possibly corrects misread digits, misaligned headers, misplaced decimal points.
    After getting the response from Gemini, removes unwanted rows or column and produces final results

    Attributes:
        api_key (str): API key for Gemini access.
        model_name (str): Name of the Gemini model to use.
        generation_config (dict): Configuration parameters for the generative model.
        table_data_validator_model (GenerativeModel): Initialized Gemini generative model for validation tasks.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-001"):
        
        """
        Initializes the TableDataValidator with Gemini configuration and system instructions.

        Args:
            api_key (str): The API key to authenticate with Gemini.
            model_name (str): The name of the Gemini model to use. Default is "gemini-2.0-flash-001".
        """
        
        self.api_key = api_key
        self.model_name = model_name
        self.generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Configure API Key for Gemini
        genai.configure(api_key=self.api_key)

        # Load AI model with system instructions
        self.table_data_validator_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction="""
            - You are an AI that compares an image of a table with a CSV file that represents the extracted table data.
            - Identify and correct any discrepancies in the CSV file while preserving the original table's meaning.
            - Ignore superscripts in the columns or row headers names
            """,
        )
    def _rm_casein_untreated(self, df):
        """
        Removes any rows or columns containing the keywords 'casein' or 'untreated'.

        Args:
            df (pd.DataFrame): The DataFrame to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        
        # Remove rows where the first column contains "casein" or "untreated"
        df = df.loc[~df.iloc[:, 0].str.contains("casein|untreated", case=False, na=False)]
    
        # Remove columns that contain "casein" or "untreated"
        df = df.loc[:, ~df.columns.str.contains("casein|untreated", case=False)]
    
        return df
    
    @staticmethod
    def upload_to_gemini(file_path: str, mime_type: str):
        """
        Uploads a file to Gemini and returns the file object.
        """
        return genai.upload_file(file_path, mime_type=mime_type)

    @staticmethod
    def wait_for_files_active(files):
        """
        Waits until the uploaded files are processed and active.
        """
        for name in (file.name for file in files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING":
                time.sleep(5)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")

    def validate_table_data(self, image_path: str, csv_path: str):
        """
        Compares a table image with an extracted CSV file to identify and correct discrepancies.

        Args:
            image_path (str): The path to the table image.
            csv_path (str): The path to the extracted CSV file.

        Returns:
            pd.DataFrame: The corrected table data.
        """
        # Upload files to Gemini
        files = [
            self.upload_to_gemini(image_path, "image/png"),
            self.upload_to_gemini(csv_path, "text/csv"),
        ]
        self.wait_for_files_active(files)

        # Start chat session with Gemini
        chat_session = self.table_data_validator_model.start_chat(
            history=[{"role": "user", "parts": files}]
        )

        # Ask Gemini to compare the image and CSV
        response = chat_session.send_message(
            """
            Compare the numbers in the extracted CSV with the table in the provided image.
            Identify any incorrect values caused by OCR errors (e.g., misread digits, misplaced decimal points, missing digits, misaligned columns or rows) and correct them.
            do not remove sample names (e.g. Chickpea) and it values from rows or columns.
            ignore the rows which may only contain units like gram
            ignore the superscripts which may come with words and numbers
            Return the corrected CSV as a JSON file.

            """
        )

        # Parse response
        corrected_data = json.loads(response.text)
        #print(f'this is after first round of validataion++++++++++s: \n {corrected_data}')

        # Convert JSON output to DataFrame
        try:
            df_corrected = pd.DataFrame(corrected_data)
        except Exception:
            # Convert the CSV string into a DataFrame
            # print(f'from validator: {corrected_data}')
            csv_data = io.StringIO(corrected_data["corrected_csv"])
            df_corrected = pd.read_csv(csv_data)

        # Post-processing: Ensure empty cells remain empty
        for col in df_corrected.columns:
            df_corrected[col] = df_corrected[col].apply(
                lambda x: "-" if (x == "" or x is None) else str(x).strip()
            )
        df_corrected = self._rm_casein_untreated(df_corrected)
        return df_corrected


# Example usage
if __name__ == "__main__":
    import os
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print(api_key)
    from csv_creator import DataFrameSaver
    validator = TableDataValidator(api_key)

    image_path = "table_images/page_5_T_1.jpg"
    csv_path = "csv_tables/Table_0.csv"

    corrected_df = validator.validate_table_data(image_path, csv_path)
    saver = DataFrameSaver(folder_name='validated_tables', clean_folder=False)
    saver.save(corrected_df, 'crrected_amino')
    
    print(corrected_df)
