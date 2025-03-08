from PIL import Image
from image_processor import ImageProcessor
import easyocr
import pytesseract
import pandas as pd

class Ocr():
    
    def __init__(self, path_to_image):
        self.path_to_image = path_to_image
        self.image = Image.open(self.path_to_image)
        
        self.image_processor = ImageProcessor()
    
    def apply_pytesseract(self, resize_float=0.6, custom_config=r'--psm 6'):
        #custom_config = r'--psm 6 -c preserve_interword_spaces=1'
        processed_image = ImageProcessor.ocr_resize(self.image,resize_float=resize_float)
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        return text
    
    def apply_easyocr(self, dpi_adjustment=False, expansion=False, dilation=False, resize_float=0.4):
        """Process the image using static methods from ImageProcessor class before applying OCR."""
        
        # Start with the original image
        processed_image = self.image

        # Apply DPI adjustment if enabled (no resizing in this case)
        if dpi_adjustment:
            processed_image = ImageProcessor.ensure_300dpi(processed_image)  # Convert to 300 DPI

        else:
            # Apply dilation if enabled
            if dilation:
                _, processed_image = ImageProcessor.dilation(processed_image)
            
            # Resize the image after dilation or if neither dpi_adjustment nor dilation was applied
            processed_image = ImageProcessor.ocr_resize(processed_image, resize_float=resize_float)
            
            if expansion:
                processed_image = ImageProcessor.image_extender(processed_image)
            

        # Now processed_image is ready for OCR
        ocr_result = self._init_easy_ocr(processed_image)
        df = pd.DataFrame(ocr_result, columns=['bbox','text','conf'])
        
        return df

    
    def _init_easy_ocr(self, image):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image)
        df = pd.DataFrame(result, columns=['bbox','text','conf'])
        return df
    
    
    

if __name__ == "__main__":
    # Test the Ocr class
    image_path = "/Users/hosseindavarzanisani/gitrep_pdfTable/TableExtraction/csv_tables/Table_2.csv"  # Replace with your image path
    ocr_instance = Ocr(image_path)  # Create an instance of the Ocr class
    
    # Apply OCR with preprocessing (resize and 300 DPI adjustment)
    result = ocr_instance.apply_easyocr(dpi_adjustment=False, expansion=False, dilation=True, resize_float=0.4)
    
    print(result.tail(30))
    