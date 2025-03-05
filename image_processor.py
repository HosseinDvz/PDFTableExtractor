from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


class ImageProcessor:
    """
    A class for processing images for AI/ML tasks.

    Note:
        - All image inputs to methods in this class should be PIL images unless stated otherwise.
        - The `imge_extender` method is an exception and expects an OpenCV (NumPy array) image.
        _ you may still pass PIL image to image_exgender

    Attributes:
        max_size (int): Maximum size for resizing images.
        device (str): Device to run tensor operations (e.g., 'cpu' or 'cuda').
    """
    
    def __init__(self, max_size=1000, device='cpu'):
        self.max_size = max_size
        self.device = device

    def resize(self, image):
        """Resize image such that the longest dimension is within max_size.
        this resizing is for modeling"""
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

    def model_input(self, image):
        detection_transform = transforms.Compose([
            self.resize,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   
        ])
        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)
        return pixel_values

    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=10):
        """
        Process bounding boxes into cropped images and tokens.

        :param img: PIL image
        :param tokens: Detected tokens
        :param objects: Detected tables
        :param class_thresholds: Confidence thresholds for each class
        :param padding: Extra padding around tables
        :return: List of cropped tables with tokens
        """
        def iob(boxA, boxB):
            """Intersection over Box (IOB) calculation."""
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            return interArea / boxAArea if boxAArea > 0 else 0

        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds.get(obj['label'], 0):  # Handle missing thresholds
                continue

            bbox = obj['bbox']
            bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
            cropped_img = img.crop(bbox)

            # Get tokens within bounding box
            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]

            # Adjust token bounding boxes
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0] - bbox[0],
                                 token['bbox'][1] - bbox[1],
                                 token['bbox'][2] - bbox[0],
                                 token['bbox'][3] - bbox[1]]

            # Handle rotated tables
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0] - bbox[3] - 1,
                            bbox[0],
                            cropped_img.size[0] - bbox[1] - 1,
                            bbox[2]]
                    token['bbox'] = bbox

            table_crops.append({'image': cropped_img, 'tokens': table_tokens})

        return table_crops


    
    @staticmethod
    def image_extender(img, num_white_rows=7) -> np.array :
        """
        Processes an image by converting it to grayscale and inserting horizontal patterns (white rows) into white row spaces.
        When we have a very dense table image, better to apply this function before OCR
        
        :param img: Input image (OpenCV format)
        :return: Processed image (numpy array)
        """
        img = ImageProcessor._check_image_type(img)
        
        rows, columns = img.shape
        #print(f"Original Image Shape: {img.shape}")
        img_processed = img.copy()

        # Create the horizontal pattern to insert (for rows)
        arr_to_insert_rows = np.full((num_white_rows, columns), 255, dtype=np.uint8)
        
        # Process rows
        skip_mode_row = False
        counter = 0
        
        for row in range(rows):
            if (img[row] > 200).all():  # Current row is white
                if skip_mode_row:
                    continue  # Skip all white rows after an insertion
                row_adj = counter * num_white_rows 
                img_processed = np.insert(img_processed, row + row_adj, arr_to_insert_rows, axis=0)
                counter += 1
                skip_mode_row = True  # Activate skip mode after insertion
            else:  # Reset skip mode when encountering a black row
                skip_mode_row = False
            
        pil_image = Image.fromarray(img_processed)
        return np.array(pil_image)
    
    
    @staticmethod
    def dilation(image, kernel_size=(2,2), iterations=1):
        """
        dilate the black pixels by eroding since we are interested in black areas of image
        Expands black text in an image using dilation with OpenCV.
        :param kernel_size: Size of the dilation kernel
        :param iterations: Number of dilation iterations
        :return: Processed OpenCV images (original dilation and resized)
        :return: PIL image
        
        """
        
        image = ImageProcessor._check_image_type(image)
        
        # Define a kernel for dilation
        kernel = np.ones(kernel_size, np.uint8)

        # Apply dilation (expands black text by erosion) 
        dilated_cv2 = cv2.erode(image, kernel, iterations=iterations) # it is a numpy array
        # Convert back to PIL image
        dilated_pil= Image.fromarray(dilated_cv2)

        return dilated_cv2, dilated_pil

    @staticmethod
    def ocr_resize(image, resize_float=0.4):
        
        img = ImageProcessor._check_image_type(image)
        
        new_size = (int(img.shape[1] * resize_float), int(img.shape[0] * resize_float))
        """Resize the image. This one is for OCR"""
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return resized_img
    
    
    @staticmethod
    def _check_image_type(image):
        """ checks the image type (PIL or cv2 and produces a grayscale image)"""
        if isinstance(image, Image.Image):  # If it's a PIL image
            image = np.array(image)  # Convert to NumPy array

        # Now, 'image' is always a NumPy array
        if image.ndim == 3:  # Color image (RGB/BGR)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[-1] == 3 else image
        return image
    
    
    @staticmethod
    def ensure_300dpi(image):
        """Ensure the image is 300 DPI by resizing it if needed and convert it to grayscale.
        
        The image is resized to 300 DPI if needed and returned as a NumPy array.
        """
        # Convert the image to grayscale
        gray_image = image.convert("L")

        # Get the DPI from the image metadata (default value if not present)
        dpi = image.info.get('dpi', (72, 72))  # Default DPI is usually 72

        # Print current DPI
        print(f"Current DPI: {dpi[0]} DPI")

        # If DPI is not 300, adjust (resize the image to match 300 DPI)
        if dpi[0] != 300:
            print(f"Resizing image to 300 DPI...")
            
            # Get current image size (in inches)
            width_inch = image.width / dpi[0]
            height_inch = image.height / dpi[1]
            
            # Set target DPI to 300
            new_width = int(width_inch * 300)
            new_height = int(height_inch * 300)

            # Resize the image to the new DPI using the LANCZOS filter
            gray_image = gray_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert grayscale image to a NumPy array
        image_array = np.array(gray_image)

        # Return the image as a NumPy array
        return image_array
    
    

 