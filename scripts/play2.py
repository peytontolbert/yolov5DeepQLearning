import cv2
import gc
import numpy as np
import os
from json.encoder import INFINITY
import mss
import pyautogui
import pygetwindow
import win32api, win32con
import torch
import PySimpleGUI as sg
import time
import keyboard
from math import sqrt
import random
import collections
from aimbot_agent import AimbotAgent
import torch
import torch.nn as nn
from gym import spaces
import pickle
import mouse
import sys
from PIL import Image
from models.experimental import attempt_load
from utils.torch_utils import select_device, smart_inference_mode

from models.yolo import Model


sct = mss.mss()
aimbot = True # Enables aimbot if True

# Autoaim mouse movement amplifier
aaMovementAmp = .8
screenShotWidth = 1920 # Width of the detection box
screenShotHeight = 1080 # Height of the detection box

headshot_mode = False # Pulls aim up towards head if True
no_headshot_multiplier = 0.2 # Amount multiplier aim pulls up if headshot mode is false
headshot_multiplier = 0.35 # Amount multiplier aim pulls up if headshot mode is true

detection_threshold = 0.2 # Cutoff enemy certainty percentage for aiming

# Set to True if you want to get the visuals
visuals = True
lockKey = 0x14

# Add the state and action spaces
state_size = 2  # Adjust this to match the size of the state representation
action_size = 4  # In this example, the agent can choose between two actions: left click, aim and no-aim

# Initialize the agent
agent = AimbotAgent(state_size, action_size)
# Define the action space and observation space
#action_space = spaces.Discrete(4) # Up, down, left, right
action_space = [0,1,2,3]
observation_space = spaces.Box(low=0, high=255, shape=(screenShotHeight, screenShotWidth, 3), dtype=np.uint8)

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def move_mouse_smoothly(target_x, target_y, duration):
    start_x, start_y = win32api.GetCursorPos()
    start_time = time.time()
    elapsed_time = 0
    
    while elapsed_time < duration:
        elapsed_time = time.time() - start_time
        progress = min(elapsed_time / duration, 1.0)
        
        x = start_x + (target_x - start_x) * progress
        y = start_y + (target_y - start_y) * progress
        print(f"x: {x}")
        win32api.SetCursorPos((int(x), int(y)))
        time.sleep(0.01)

def step(action, bbox_center):
    move_distance = 10
    move_duration = 0.1
    if action == 0 and bbox_center is not None:  # move
        x, y = bbox_center
        move_mouse_smoothly(x, y, move_duration)
    elif action == 1:  # left click
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif action == 2:  # right click down
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    elif action == 3:  # right click release
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    else:
        print(f"Invalid action: {action}")

# Load the agent's memory from a file
def load_agent_memory(agent, filename='agent_memory.pickle'):
    try:
        with open(filename, 'rb') as f:
            loaded_memory = pickle.load(f)
        agent.memory = loaded_memory
    except FileNotFoundError:
        print("No saved memory file found. Starting with an empty memory.")

def get_active_window_titles():
    windows = pygetwindow.getAllTitles()
    active_windows = [window for window in windows if window != '']
    return active_windows

def grab_screenshot(sctArea):
    img = sct.grab(sctArea)
    img = np.array(img)
    frame = img
    return frame

def detectx (frame, model):
    frame = [frame]
    results = model(frame)
    return results

def drawbox(bbox_coords, frame):
    print(bbox_coords)
    if bbox_coords:
        for coords in bbox_coords:
            x1, y1, x2, y2 = coords
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2) ## BBox
    else:
        ("no bbox coords to draw")
    return frame

def get_closest_box_center(results):
    screen_width = 1920
    screen_height = 1080
    center_x = screen_width // 2
    center_y = screen_height // 2
    closest_distance = None
    closest_box_center = None
    labels = None
    bbox_coords = []
    n = len(results.xyxy[0])
    if results is not None and n > 0:
        closest_distance = float('inf')
        for i in range(n):
            row = results.xyxy[0][i].numpy()
            if row[4] >= detection_threshold:
                print(f"Detection: {row}")
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                box_center_x = (x1+x2)/2
                box_center_y = (y1+y2)/2
                bbox_coords.append([int(x1),int(y1),int(x2),int(y2)])
                distance = np.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2 )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box_center = (box_center_x,box_center_y)
                    labels = row[4]
    else:
        print("No results")
    print(bbox_coords)
    return bbox_coords, closest_box_center

def extract_state(results):
    screen_width = 1920
    screen_height = 1080
    bbox_center = None
    center_x = screen_width // 2
    center_y = screen_height // 2
    state = torch.zeros(state_size, dtype=torch.float32)
    # print(f"results: {results.xyxy}")
    # print(f"State tensor {state}")
    # Extract information about the detected objects
    if results is not None:
        bbox_coords, closest_box_center = get_closest_box_center(results)
        if closest_box_center is not None:
            bbox_center = closest_box_center
            print(f"closest box: {bbox_coords}")
            state[:len(bbox_center)] = torch.tensor(closest_box_center, dtype=torch.float32)
            print("Updated state with closest box:", state)
        else:
            print("No closest box found")
        # Extract the coordinates of the bounding boxes for each detected object
        print(f"state: {state}")
        # Store the extracted information in the state tensor
    else:
        print("No results or empty results")    
    # Extract information about the mouse coordinates
    print("Final state: ", state)
    
    return state, bbox_center, bbox_coords


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def aim_towards_center(action, bbox_center):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    if aimbot and win32api.GetKeyState(lockKey) and bbox_center is not None:
        if action is not None:
            print("performing action")
            step(action, bbox_center)
    else:
        print("no action or bbox_center")
    print("finished step function")
    return action

def execute_action(done, action, bbox_center):
    print(f"action: {action}")
    aim_towards_center(action, bbox_center)
    done = True
    return done

def display_output(frame):
    cv2.imshow('Output', frame)
    cv2.waitKey(1)

def main(run_loop=False, gameWindow=None):
    weights_path = 'valorantmodels/Yolov5/YOLOv5s/valorant-2.pt'
    model = load_model(weights_path)
    #model = Model(cfg='models/yolov5s.yaml')
    #model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    print(f"[INFO] Loading model... ")
    closest_distance = None
    done = False
    if run_loop==True:
        sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}

        agent = AimbotAgent(state_size, action_size)
        load_agent_memory(agent)
        count = 0

        print("Program Working")

        while True:

            frame = grab_screenshot(sctArea)
            #print(f"[INFO] Working with frame {frame_no}Call of Duty® HQ ")
            if (gameWindow == "Call of Duty® HQ"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = detectx(frame, model)
            state, bbox_center, bbox_coords = extract_state(results)
            frame = drawbox(bbox_coords, frame) #draw box after extracting state information
            action = agent.choose_action(state)
            done = execute_action(done, action, bbox_center)    #execute action and reward   
            
            display_output(frame)    
            #cv2.imshow("vid", frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break
            if keyboard.is_pressed('esc'):
                print(f"[INFO] Exiting. . . ")   
                break

        print(f"[INFO] Cleaning up. . . ")
        ## closing all windows
    exit()  


def selectSettings():
    
    active_windows = get_active_window_titles()

    layout = [
    [
        [sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )],
        [sg.Text('Game Window', size=(15, 1)), sg.Combo(active_windows, key="gw1", default_value=active_windows[0] if active_windows else "")],
        [sg.Button('Start'), sg.Button('Exit')]
    ],
]

    window = sg.Window("Proton Client", layout)

    while True:
        event, values = window.read()

        if event == 'Start':
            if values['gw1'] != "":
                gw = values['gw1']
            else:
                gw = "Counter"
            break
        elif event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()
    print("Game Window: ", str(gw))

    return gw

gw = selectSettings()

if __name__ == "__main__":
    
    prev_distance = None
    q_table_path = 'q_table.pk1'
    if os.path.exists(q_table_path):
        agent.load(q_table_path)
        print("Q-table loaded.")
    main(run_loop=True, gameWindow="Call of Duty® HQ")