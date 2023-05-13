import torch
import cv2
import time
from torchvision.ops import nms
import mss
import numpy as np
from queue import Queue
from threading import Thread
from multiprocessing import Process, Queue

# Load the YOLOv5 model

def grab_screenshot(sctArea):
    with mss.mss() as sct:
        img = display_sct.grab(sctArea)
        img = np.array(img)
        return img

def screenshot_process(frame_queue, sctArea, display_sct):
        while True:
            frame = grab_screenshot(sctArea, display_sct)
            frame_queue.put(frame)

def display_process(frame_queue, results_queue):
    while True:
        if frame_queue is not None:
            frame = frame_queue.get()
            if not results_queue.empty():
                last_results = results_queue.get()
            if last_results is not None:
                results = last_results
                for c in results.pred[0][:, -1].unique():
                    if c >= 0:
                        for box in results.pred[0][results.pred[0][:, -1] == c]:
                            x1, y1, x2, y2, conf, cls = box
                            label = f"{classes[int(cls)]} ({conf * 100:.1f}%)"
                            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            frame = cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            cv2.imshow("Live View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

def detection_process (frame_queue, results_queue):
    while True:
         if frame_queue is not None and frame_queue.size > 0:
              results = model(frame_queue)
              results_queue.put(results)

if __name__ == '__main__':
    frame_queue = Queue()
    results_queue = Queue()
    display_sct = mss.mss()
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]  # colors for the bounding boxes
    sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}
    cv2.namedWindow("Live View", cv2.WINDOW_NORMAL)
    def load_model(weights_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        return model

    weights_path = 'runs/train/mw2_model/weights/best.pt'
    model = load_model(weights_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    #detection_thread = Thread(target=detection_process, args=(results_queue, detection_sct))
    screenshot_process = Process(target=screenshot_process, args=(frame_queue, sctArea, display_sct))
    display_process = Process(target=display_process, args=(frame_queue,results_queue))
    detection_process = Process(target=detection_process, args=(frame_queue,))

    screenshot_process.start()
    display_process.start()

    screenshot_process.join()
    display_process.join()


 #   display_thread = Thread(target=display_process, args=(results_queue, display_sct))



    #detection_thread.start()
    #display_thread.start()

    #detection_thread.join()
    #display_thread.join()