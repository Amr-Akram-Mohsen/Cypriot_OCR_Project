# inference_demo.py
import torch
from ultralytics import YOLO

model = YOLO("YOLO_Model/weights/cypriot_yolo_finetuned_best.pt")

def detect_and_build_words(image_path):
    results = model.predict(source=image_path, conf=0.25)[0]
    detections = []
    
    # 1. استخراج الإحداثيات واسم الحرف
    for box in results.boxes:
        cls_id = int(box.cls[0])
        char_name = model.names[cls_id]
        x_min = box.xyxy[0][0].item() # موقع الحرف أفوقياً
        detections.append((x_min, char_name))
    
    # 2. ترتيب الحروف أفقياً لبناء الكلمة
    detections.sort(key=lambda x: x[0]) 
    extracted_word = "".join([item[1] for item in detections])
    
    return extracted_word