import torch
import cv2
from paddleocr import PaddleOCR
import logging
import numpy as np
import requests

# Suppress PaddleOCR logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Load your YOLOv5 MRZ detection model
mrz_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/mrz.pt')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_model_dir = 'paddle_models/en_PP-OCRv3_rec_infer', rec_char_type='en')

# def detect_and_crop_mrz(model, image, mrz_class_idx=0):
#     """
#     Detect the MRZ region using YOLOv5 and crop the detected region from the image.
#     Manually filter for the specified class after inference.
#     """
#     results = model(image)
#
#     # Move detections to CPU and convert to NumPy for easier processing
#     detections = results.xyxy[0].cpu().numpy()
#
#     # Loop through detections and only consider the specified class
#     for detection in detections:
#         x1, y1, x2, y2, confidence, class_id = map(int, detection[:6])
#
#         if class_id == mrz_class_idx:  # Only process if it's the MRZ class
#             # Crop the detected MRZ region
#             cropped_mrz = image[y1:y2, x1:x2]
#             return cropped_mrz
#
#     return None  # Return None if no MRZ was detected
#
# def correct_country_code(text):
#     """
#     Correct the country code part of the MRZ to 'UZB'.
#     The country code is located in positions [-34:-31] of the second line.
#     """
#     lines = text.strip().split('\n')
#
#     if len(lines) == 2 and len(lines[1]) == 44:
#         # Replace the country code at [-34:-31] of the second line with 'UZB'
#         lines[1] = lines[1][:10] + 'UZB' + lines[1][13:]
#
#     # Rejoin the two lines to form the corrected MRZ
#     corrected_text = '\n'.join(lines)
#
#     return corrected_text
#
# def extract_mrz_text(mrz_image):
#     """
#     Extract text from the MRZ region using PaddleOCR.
#     """
#     mrz_image_rgb = cv2.cvtColor(mrz_image, cv2.COLOR_BGR2RGB)
#     result = ocr.ocr(mrz_image_rgb)
#
#     mrz_text = ''
#     for line in result:
#         for item in line:
#             if isinstance(item, list) and len(item) >= 2:
#                 # Extract the recognized text (first element of the tuple)
#                 text = item[1][0] if isinstance(item[1], tuple) else item[1]
#                 mrz_text += text + '\n'  # Add the extracted text followed by a newline
#
#     # Correct the country code to 'UZB' in the [-34:-31] position
#     corrected_mrz_text = correct_country_code(mrz_text)
#     return corrected_mrz_text
#
# # Step 1: Load the input image
# image_path = 'images/photo_2024-07-01_15-21-28.jpg'
# image = cv2.imread(image_path)
#
# # YOLOv5 expects RGB images, so convert BGR to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Step 2: Detect and crop the MRZ region using only the MRZ class (class index 1)
# mrz_cropped = detect_and_crop_mrz(mrz_model, image_rgb, mrz_class_idx=0)
#
# if mrz_cropped is not None:
#     # Step 3: Extract the MRZ text using PaddleOCR and correct the country code
#     mrz_text = extract_mrz_text(mrz_cropped)
#     print("MRZ:", mrz_text)
# else:
#     print("No MRZ detected.")


class ExtractMRZ:
    def __init__(self, image_url: str):
        self.image_url = image_url
        self.ocr = ocr
        self.mrz_model = mrz_model

    def get_image_from_url(self):
        response = requests.get(self.image_url)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def detect_and_crop_mrz(self, mrz_class_idx=0):
        """
        Detect the MRZ region using YOLOv5 and crop the detected region from the image.
        Manually filter for the specified class after inference.
        """
        image = self.get_image_from_url()

        # YOLOv5 expects RGB images, so convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mrz_model(image_rgb)

        # Move detections to CPU and convert to NumPy for easier processing
        detections = results.xyxy[0].cpu().numpy()

        # Loop through detections and only consider the specified class
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = map(int, detection[:6])

            if class_id == mrz_class_idx:  # Only process if it's the MRZ class
                # Crop the detected MRZ region
                cropped_mrz = image[y1:y2, x1:x2]
                return cropped_mrz

        return None  # Return None if no MRZ was detected

    def correct_country_code(self, text):
        """
        Correct the country code part of the MRZ to 'UZB'.
        The country code is located in positions [-34:-31] of the second line.
        """
        lines = text.strip().split('\n')

        if len(lines) == 2 and len(lines[1]) == 44:
            # Replace the country code at [-34:-31] of the second line with 'UZB'
            lines[1] = lines[1][:10] + 'UZB' + lines[1][13:]

        # Rejoin the two lines to form the corrected MRZ
        corrected_text = '\n'.join(lines)

        return corrected_text

    def extract_mrz_text(self, mrz_image):
        """
        Extract text from the MRZ region using PaddleOCR.
        """
        mrz_image_rgb = cv2.cvtColor(mrz_image, cv2.COLOR_BGR2RGB)
        result = ocr.ocr(mrz_image_rgb)

        mrz_text = ''
        for line in result:
            for item in line:
                if isinstance(item, list) and len(item) >= 2:
                    # Extract the recognized text (first element of the tuple)
                    text = item[1][0] if isinstance(item[1], tuple) else item[1]
                    mrz_text += text + '\n'  # Add the extracted text followed by a newline

        # Correct the country code to 'UZB' in the [-34:-31] position
        corrected_mrz_text = self.correct_country_code(mrz_text)
        return corrected_mrz_text

    def check_mrz(self, mrz: str):
        mrz = mrz.replace(" ", "").replace("\n", "")
        if len(mrz) == 88:
            return {"passport": mrz[44:53], "pinfl": mrz[72:86], "mrz": mrz, "type": "6"}
        return None

    def finish(self):
        mrz_cropped = self.detect_and_crop_mrz()
        if mrz_cropped is not None:
            # Step 3: Extract the MRZ text using PaddleOCR and correct the country code
            mrz_text = self.extract_mrz_text(mrz_cropped)
        else:
            return None
        mrz_data = self.check_mrz(mrz_text)
        return mrz_data
