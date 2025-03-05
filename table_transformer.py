import torch
from transformers import AutoModelForObjectDetection
from PIL import Image
import os
import glob


class TableTransformer:
    
    def __init__(self, image_processor, path='table_images' ,device='cpu'):
        # Initialize the model and device
        self.device = device
        self.path = path
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        self.model.to(self.device)
        
        # Update id2label to include "no object"
        self.id2label = self.model.config.id2label
        self.id2label[len(self.id2label)] = "no object"
        
        # Initialize the image processor
        self.image_processor = image_processor

    def forward(self, pixel_values):
        # Perform inference with no gradient calculation
        with torch.no_grad():
            outputs = self.model(pixel_values.to(self.device))
        
        return outputs
    def detect_and_crop_tables(self, pages: list, detection_class_thresholds : dict =None ):
        """ save the detected tables in table_images folder"""
        # Set default detection_class_thresholds if not passed
        if detection_class_thresholds is None:
            detection_class_thresholds = {
                "table": 0.95,
                "table rotated": 0.7,
                "no object": 10
            }
        
        cropped_tables = []
        
        self._clear_folder(self.path)
        
        for filename in pages:
            try:
                image = Image.open(filename).convert('RGB')
            except Exception:
                continue
            
            pixel_values = self.image_processor.model_input(image)  # Use model_input from ImageProcessor
            outputs = self.forward(pixel_values)
            objects = self._outputs_to_objects(outputs, image.size)

            if len(objects) == 0: 
                continue
            
            #print(f'Page number: {filename}')
            tokens = []
            
            tables_crops = self.image_processor.objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)  # Use objects_to_crops from ImageProcessor
            #print(f'Number of tables in that page: {len(tables_crops)}')
            
            
            for i in range(len(tables_crops)):
                
                cropped_table = tables_crops[i]['image'].convert("RGB")
                cropped_tables.append(cropped_table)  # Collect the cropped tables

                # Save the cropped table image
                # Extract only the filename (without path)
                base_name = os.path.basename(filename)  
                name, _ = os.path.splitext(base_name) #dropping the extensions(.png) from filenames
                
                #output_folder = self.path
                os.makedirs(self.path, exist_ok=True)  # Ensure directory exists
                save_path = os.path.join(self.path, f"{name}_T_{i+1}.jpg") 
                cropped_table.save(save_path)

        return cropped_tables
    
    def _clear_folder(self, path):
        """Deletes all files in the specified folder."""
        if not os.path.exists(path):
            print(f"Folder {path} does not exist.")
            return
        
        files = glob.glob(os.path.join(path, "*"))  # Get all files in the folder
        for file in files:
            try:
                os.remove(file)  # Delete each file
            except Exception as e:
                print(f"Error deleting {file}: {e}")
                continue
    
    # Private methods for output bounding box post-processing
    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.51 * w), (y_c - 0.51 * h), (x_c + 0.51 * w), (y_c + 0.51 * h)]
        return torch.stack(b, dim=1)

    def _rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self._box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def _outputs_to_objects(self, outputs, img_size):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self._rescale_bboxes(pred_bboxes, img_size)]

        objects = []

        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = self.id2label[int(label)]

            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects

    



if __name__ == '__main__':
    
    from image_processor import ImageProcessor
    from pdf_to_image import PdfToImage
    import os
    
    
    PdfToImage(pdf_path='pdfs/1-s2.0-S0308814617312839-Lentils.pdf').pdf_to_png()
    dirname, _, pages = list(os.walk('/Users/hosseindavarzanisani/AC/Semester_2/AI Project_2/AAFC/Codes/page_images'))[0]
    pages_with_path = [os.path.join(dirname, page) for page in pages]
    
    image_processor = ImageProcessor()  # Create the ImageProcessor instance
    table_transformer = TableTransformer(image_processor=image_processor)

    #pages = ["images/page_5.png", "images/page_7.png"]
    cropped_tables = table_transformer.detect_and_crop_tables(pages_with_path)  # Default thresholds will be used
    print(len(cropped_tables))

    # Do something with cropped_tables
    '''for table in cropped_tables:
        table.show()'''
    