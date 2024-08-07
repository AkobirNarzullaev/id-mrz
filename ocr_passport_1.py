import cv2
import torch
import pytesseract
from fastmrz import FastMRZ
import os
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Path to your trained YOLOv5 model
MODEL_PATH = 'yolov5/runs/train/exp4/weights/best.pt'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

class MRZ:
    def __init__(self):
        # Initialize Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust this path as needed

    def detect_mrz(self, image_path):
        # Read the image
        img = cv2.imread(image_path)

        # Check if the image is loaded properly
        if img is None:
            raise ValueError(f"Image at path {image_path} could not be loaded. Check the file path and try again.")

        # Inference
        results = model(img)

        # Extract bounding boxes
        boxes = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class

        # Filter MRZ regions based on class and confidence threshold
        mrz_regions = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if cls == 0 and conf > 0.5:  # Adjust confidence threshold as needed
                mrz_regions.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        return img, mrz_regions

    def extract_text_from_mrz(self, image, regions):
        texts = []
        for (x1, y1, x2, y2) in regions:
            # Crop MRZ region
            crop_img = image[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(gray, config='--psm 6', lang='ocrb')
            texts.append(text.strip())
        
        return texts

    def extract_mrz_from_image(self, image_path):
        image, mrz_regions = self.detect_mrz(image_path)

        if not mrz_regions:
            # Fallback to fastmrz if yolov5 did not work
            fast_mrz = FastMRZ()
            passport_mrz = fast_mrz.get_raw_mrz(image_path)
            
            if passport_mrz:
                print(passport_mrz)
            else:
                print("No MRZ regions found in the image.")
            return

        mrz_texts = self.extract_text_from_mrz(image, mrz_regions)
        for i, text in enumerate(mrz_texts):
            if len(text) >= 88:
                card_num = text[44:54]
                pnfl = text[-16:-2]
                print(f'MRZ Text {i+1}:')
                print(text)
                print('---')
                print(card_num, pnfl)
            else:
                print(f'MRZ Text {i+1} is too short. Possibly incomplete OCR result.')
                print(text)

if __name__ == '__main__':
    mrz_instance = MRZ()
    mrz_instance.extract_mrz_from_image('images/photo_2024-06-21_08-57-45.jpg')
