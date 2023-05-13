import torch
import cv2
import time
from torchvision.ops import nms
import mss
import numpy as np


# Load the YOLOv5 model
detection_threshold = 0.1  # confidence threshold for object detection
colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]  # colors for the bounding boxes

sct = mss.mss()
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

weights_path = 'valorantmodels/Yolov5/YOLOv5s/valorant-5.pt'
model = load_model(weights_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the screenshot capture
screenshot_interval = 0.5  # interval between screenshots in seconds
last_screenshot_time = 0
screenshot_counter = 0
# Set up the image labeling
classes = ['head', 'enemy', 'teammate']  # classes for the objects

# Set up the output folder for the labeled images
output_folder = 'valorant_images/'
# Set up the screenshot capture

sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}

def grab_screenshot(sctArea):
    img = sct.grab(sctArea)

    img = np.array(img)

    frame = img
    return frame

def detectx (frame, model):
    frame = [frame]
    results = model(frame)
    return results


def get_boxes(results):
    rows = []
    n = len(results.xyxy[0])
    if results is not None and n > 0:
        for i in range(n):
            row = results.xyxy[0][i].numpy()
            if row[4] >= detection_threshold:
                print(f"Detection: {row}")
                rows.append(row)
    else:
        print("No results")
    return rows



while True:
    # Take a screenshot if the interval has elapsed
    if time.time() - last_screenshot_time > screenshot_interval:
        frame = grab_screenshot(sctArea)
        if frame is not None and frame.size>0:
            last_screenshot_time = time.time()
            screenshot_counter += 1
            print(f'Taking screenshot {screenshot_counter}')
            results = detectx(frame, model)
            rows = get_boxes(results)

            if rows:
                cv2.imwrite(f'{output_folder}{screenshot_counter}.jpg', frame)
                with open(f'{output_folder}{screenshot_counter}.txt', 'w') as f:
                    for row in rows:
                        f.write(f'{int(row[5])} {row[0]} {row[1]} {row[2]} {row[3]} \n')