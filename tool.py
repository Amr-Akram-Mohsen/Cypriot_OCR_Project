from ultralytics import YOLO

# Load fine-tuned weights
model = YOLO(r"C:/Users/ammar\Desktop/un_4/Cypriot_OCR_Project/YOLO_Model/weights/cypriot_yolo_finetuned_best.pt")
# Export to TFLite format for mobile execution
model.export(format="tflite")
# Creates: cypriot_yolo_finetuned_best_saved_model/cypriot_yolo_finetuned_best_float32.tflite