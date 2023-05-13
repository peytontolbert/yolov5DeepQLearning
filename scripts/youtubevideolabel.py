import torch
import cv2
import time
import youtube_dl
from torchvision.ops import nms
import numpy as np
# Load the YOLOv5 model
detection_threshold = 0.1  # confidence threshold for object detection
colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]  # colors for the bounding boxes

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

weights_path = 'runs/train/mw2_model/weights/best.pt'
model = load_model(weights_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set up the image labeling
classes = ['head', 'enemy', 'teammate']  # classes for the objects

# Set up the output folder for the labeled images
output_folder = 'labeled_youtube/'

def detectx(frame, model):
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

def get_youtube_video_url(video_url):
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict['title']

def process_video(video_url):
    video_stream_url = get_youtube_video_url(video_url)
    capture = cv2.VideoCapture(video_stream_url)

    frame_counter = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_counter += 1
        print(f'Processing frame {frame_counter}')

        results = detectx(frame, model)
        rows = get_boxes(results)

        if rows:
            cv2.imwrite(f'{output_folder}{frame_counter}.jpg', frame)
            with open(f'{output_folder}{frame_counter}.txt', 'w') as f:
                for row in rows:
                    f.write(f'{int(row[5])} {row[0]} {row[1]} {row[2]} {row[3]} \n')

    capture.release()

if __name__ == '__main__':
    video_url = 'http://www.youtube.com/watch?v=CO0-VubUsRc'
    process_video(video_url)