import os
import cv2
from qreader import QReader
import re
import time


def scan_qr_code(image_path):
    qreader = QReader()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    decoded_text = qreader.detect_and_decode(image=image)
    return decoded_text


def monitor_directory(directory):
    scanned_filenames = set() 
    while True:
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                if filename not in scanned_filenames:
                    image_path = os.path.join(directory, filename)
                    decoded_text = scan_qr_code(image_path)
                    pattern = r'\b[a-zA-Z]+\b'
                    text = decoded_text[0]
                    text_2 = text[61:90]
                    words = re.findall(pattern, text_2)
                    #name = words[1] Ism, Familiya shart emas ekan
                    #second_name = words[0]
                    card_num = text[5:14]
                    pnfl = text[15:29]
                    print(f'Passport raqami: {card_num}, PNFL: {pnfl}')
                    scanned_filenames.add(filename) 
        time.sleep(1)

if __name__ == "__main__":
    directory = "id/"
    monitor_directory(directory)