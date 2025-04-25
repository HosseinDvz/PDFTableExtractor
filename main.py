from pdf_to_image import PdfToImage
from image_processor import ImageProcessor
from ocr import Ocr
from table_transformer import TableTransformer
from table_constructor import TableConstruction
from csv_creator import DataFrameSaver
#from anomaly_detector import AnomalyDetector
import numpy as np
from validator import TableDataValidator
import os
import dotenv


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

import os

# reading pdf and converting to image
PdfToImage(pdf_path='pdfs/foods-11-00289.pdf').pdf_to_png()

dirname, _, pages = list(os.walk('page_images'))[0]
pages_with_path = [os.path.join(dirname, page) for page in pages] #  getting list of pdf page images' path
    
image_processor = ImageProcessor()  # Create the ImageProcessor instance

table_transformer = TableTransformer(image_processor=image_processor)
cropped_tables = table_transformer.detect_and_crop_tables(pages_with_path) # save the table images in a folder

dirname, _, images = list(os.walk('table_images'))[0]
#print(images)
images_with_path = [os.path.join(dirname, image) for image in images] # getting list of images in the table_image forlder

saver_1= DataFrameSaver()

validator = TableDataValidator(api_key)
saver_2 = DataFrameSaver(folder_name='validated_tables', clean_folder=True)

for i, image in enumerate(images_with_path):
    print(image)
    try:
        ocr_instance = Ocr(image)  # Create an instance of the Ocr class
    except Exception: # except some sys files in the image folders
        continue
    
    
    #creating final table
    resize_float = 0.35
    while True:
        try: 
            # Apply OCR with preprocessing
            df = ocr_instance.apply_easyocr(dpi_adjustment=False,expansion=True,  dilation=False, resize_float=resize_float)
            table = TableConstruction(df)
            final_table = table.table_creator()
            saver_1.save(final_table, f'Table_{i}')
            break 
        except Exception:
            resize_float = np.random.uniform(0.29, 0.34)
    
    corrected_df = validator.validate_table_data(image, f'csv_tables/Table_{i}.csv')
    saver_2.save(corrected_df, f'table_{i}_validated')  
        
    print(corrected_df)
    #detector = AnomalyDetector(threshold=10)
    print('='*40)
    
    

'''
'/pdfs/1-s2.0-S0308814617312839-Lentils.pdf' 
apply_easyocr(dpi_adjustment=False,expansion=False,  dilation=True, resize_float=0.3)

acs.jafc.7b00697 Buckwheat pinto.pdf'
(dpi_adjustment=False,expansion=False,  dilation=True, resize_float=0.3)

easyocr(dpi_adjustment=False,expansion=True,  dilation=True, resize_float=0.3)
/Users/hosseindavarzanisani/AC/Semester_2/AI Project_2/AAFC/Codes/pdfs/1-s2.0-S0963996916304306-main.pdf (3,3), image_extender(img, num_white_rows=7)


/kaggle/input/aafc-pdfs/foods-11-00289.pdf - (dpi_adjustment=False,expansion=False,  dilation=False, resize_float=0.35)

'''