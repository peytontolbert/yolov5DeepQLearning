import cv2
import gc
import pyautogui
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
state_size = 4  # Adjust this to match the size of the state representation
action_size = 1  # In this example, the agent can choose between two actions: left click, aim and no-aim

# Initialize the agent
agent = AimbotAgent(state_size, action_size)


# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the action space and observation space
#action_space = spaces.Discrete(4) # Up, down, left, right
action_space = [0]
observation_space = spaces.Box(low=0, high=255, shape=(screenShotHeight, screenShotWidth, 3), dtype=np.uint8)

def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model


def move_mouse_smoothly(target_x, target_y, duration):
    #start_x, start_y = win32api.GetCursorPos()
    start_x, start_y = pyautogui.position()
    start_time = time.time()
    elapsed_time = 0
    
    while elapsed_time < duration:
        elapsed_time = time.time() - start_time
        progress = min(elapsed_time / duration, 1.0)
        
        x = start_x + (target_x - start_x) * progress
        y = start_y + (target_y - start_y) * progress
        print(f"x: {x} y: {y}")
        #win32api.SetCursorPos((int(x), int(y)))
        pyautogui.moveTo(x, y, duration=0)
        time.sleep(0.01)

def step(action, bbox_center):
    if action == 0:  # move
        aim_towards_center(action, bbox_center)
    elif action == 1:  # left click
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
#    elif action == 2:  # right click down
#        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
#    elif action == 3:  # right click release
#        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    else:
        print(f"Invalid action: {action}")

# Save the agent's memory to a file
def save_agent_memory(agent, filename='agent_memory.pickle'):
    with open(filename, 'wb') as f:
        pickle.dump(agent.memory, f)

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

def get_new_observation(frame2, model):
    frame = frame2
    results = detectx(frame, model)
    new_box_center_distance = None
    closest_box_center = None
    closest_distance = None
    if results is not None:
        bbox_coords, labels, new_box_center_distance = get_boxes_coords(results)
        drawbox(bbox_coords, frame)
    display_output(frame)
    return bbox_coords, labels, new_box_center_distance


def get_boxes_coords(results):
    screen_width = 1920
    screen_height = 1080
    center_x = screen_width // 2
    center_y = screen_height // 2
    labels = []
    bbox_coords = []
    new_box_center_distance = None
    n = len(results.xyxy[0])
    if results is not None and n > 0:
        closest_distance = float('inf')
        for i in range(n):
            row = results.xyxy[0][i].numpy()
            
            if row[4] >= detection_threshold:
                print(f"Detection: {row}")
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                bbox_coords.append([x1, y1, x2, y2])
                box_center_x = (x1+x2)/2
                box_center_y = (y1+y2)/2
                distance = np.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2 )
                labels = row[4]
                if distance < closest_distance:
                    new_box_center_distance = distance
    else:
        print("No results")
    print(new_box_center_distance)
    return bbox_coords, labels, new_box_center_distance



def detectx (frame, model):
    frame = [frame]
    #print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])
    return results

   # return labels, cordinates

def drawbox(bbox_coords, frame):
    print("box coords::")
    print(bbox_coords)
    if bbox_coords:
        for coords in bbox_coords:
            x1, y1, x2, y2 = int(coords[0]),int(coords[1]),int(coords[2]),int(coords[3])
            frame = cv2.rectangle(frame, (int(x1),y1), (x2,y2), (0, 255, 0), 2) ## BBox
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
    labels = []
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
                distance = np.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2 )
                if distance < closest_distance:
                    closest_distance = int(distance)
                    closest_box_center = (box_center_x,box_center_y)
                    labels = row[4]
    else:
        print("No results")
    return closest_box_center, closest_distance, labels


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
        closest_box_center, closest_distance, labels  = get_closest_box_center(results)
        if closest_box_center is not None and labels:
            bbox_center = closest_box_center
            print(f"closest box: {closest_box_center}")
            state[:len(bbox_center)] = torch.tensor(bbox_center, dtype=torch.float32)
            state[len(bbox_center):] = torch.tensor(labels, dtype=torch.float32)
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
    
    return state, bbox_center, closest_distance






### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def aim_towards_center(action, bbox_center):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    if aimbot and win32api.GetKeyState(lockKey) and bbox_center is not None:
        if action == 0:
            print("performing move action")
            x,y = bbox_center
            x = int(x)
            y = int(y)
            mouse_duration = 0.1
            move_mouse_smoothly(x,y,mouse_duration)
            step(action, bbox_center)
        else:
            print("invalid aiming without action 0")
    else:
        print("no action or bbox_center")
    print("finished step function")
    return action

def execute_action(model, done, reward, action, prev_distance, closest_distance, bbox_center):
    print(f"action: {action}")
    step(action, bbox_center)
    sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}
    frame2 = grab_screenshot(sctArea)
    bbox_coords, labels, new_box_center_distance = get_new_observation(frame2, model)
    cWidth = 1920/2
    cHeight = 1080/2
    center_coords = [cWidth, cHeight]
    reward, done, prev_distance = compute_reward(action, closest_distance, new_box_center_distance, prev_distance)  # Implement this function to compute the reward for the chosen action
    return reward, done, prev_distance

def compute_reward(action, closest_distance, new_box_center_distance, prev_distance):
    print("starting compute reward")
    reward=0
    print(f"took action: {action}")
    
    if action is not None and closest_distance is not None and new_box_center_distance is not None:
        if prev_distance is None:
            prev_distance = closest_distance

        if action == 0:  # Move closer
            print(f"prev distance: {prev_distance} new_box_distance {new_box_center_distance}")
            if new_box_center_distance < prev_distance:
                reward = .5
                print("rewarded for moving closer to target")
            else:
                reward = -0.2
                print("penalty for moving away from target")
        elif action == 1:  # Shoot
            if new_box_center_distance <= 10:  # If within 10 pixels of the target
                reward = 1
                print("rewarded 1 for shooting on target")
            else:
                reward = -0.5
                print("penalty for shooting far from target")

        prev_distance = new_box_center_distance

    done = True
    print("")
    return reward, done, prev_distance

def display_output(frame):
    cv2.imshow('Output', frame)
    cv2.waitKey(1)

def main(run_loop=False, gameWindow=None):
    count=0
    weights_path = 'runs/train/mw2_modelimproved/weights/best.pt'
    model = load_model(weights_path)
    #model = Model(cfg='models/yolov5s.yaml')
    #model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    print(f"[INFO] Loading model... ")
    classes = model.names
    reward=0
    prev_distance = None  # Initialize prev_distance as infinity
    closest_distance = None
    closest_box = None
    save_interval = 100
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
            state, bbox_center, closest_distance = extract_state(results)
            #frame = drawbox(bbox_coords, frame) #draw box after extracting state information
            action = agent.choose_action(state)
            reward, done, prev_distance = execute_action(model, done, reward, action, prev_distance, closest_distance, bbox_center)    #execute action and reward   
            
            #display_output(frame)    
            #cv2.imshow("vid", frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break
            if keyboard.is_pressed('esc'):
                print(f"[INFO] Exiting. . . ")   
                break

            # Forced garbage cleanup every second
            count += 1
            if count % save_interval == 0:
                save_agent_memory(agent)

        print(f"[INFO] Cleaning up. . . ")
        ## closing all windows
    
    save_agent_memory(agent)
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