import cv2
import numpy as np
from mss import mss
import pyautogui
import torch

def capture_screenshot():
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model


def distance(x1, y1, x2, y2):
    return ((x1-x2) ** 2 + (y1-y2)**2) **5

def detect_and_draw_boxes(model, img, conf_threshold=0.7, move_speed=5):
    results = model(img, size=640)
    results = results.pandas().xyxy[0]

    valid_boxes = []

    for _, row in results.iterrows():
        conf, x1, y1, x2, y2 = row['confidence'], row['xmin'], row['ymin'], row['xmax'], row['ymax']
        if conf > conf_threshold:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            x_center = int((x1+x2) /2)
            y_center = int((y1+y2)/2)

            valid_boxes.append((x_center, y_center))
    target_x, target_y = None, None
    if(len(valid_boxes) == 1):
        target_x, target_y = valid_boxes[0]
    elif len(valid_boxes) > 1:
        mouse_x, mouse_y = pyautogui.position()
        target_x, target_y = min(valid_boxes, key=lambda box_center: distance(mouse_x, mouse_y, box_center[0], box_center[1]))

    if target_x and target_y:
        # Smoothly move the mouse to the target position
        mouse_x, mouse_y = pyautogui.position()
        diff_x, diff_y = target_x - mouse_x, target_y - mouse_y
        move_x = diff_x / move_speed
        move_y = diff_y / move_speed
        pyautogui.move(move_x, move_y, duration=0.1)

    print(results)
    return img

def display_output(img):
    cv2.imshow('Output', img)
    cv2.waitKey(1)

def main():
    weights_path = 'runs/train/mw2_model/weights/best.pt'
    model = load_model(weights_path)

    while True:
        img = capture_screenshot()
        img = detect_and_draw_boxes(model, img)
        display_output(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()