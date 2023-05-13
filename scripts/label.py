import os
import cv2
import torch
import numpy as np

weights_path = "runs/train/mw2_model/weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

images_folder = "outputs"
labels_folder = "outputslabels"
classes = model.names

os.makedirs(labels_folder, exist_ok=True)

for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        img = cv2.imread(image_path)
        results = model(img)
        
        label_path = os.path.join(labels_folder, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2 = detection[:4].tolist()
                class_id = int(detection[5])
                class_name = classes[class_id]
                label = f"{class_id} {x1} {y1} {x2} {y2}\n"
                f.write(label)