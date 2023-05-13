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
from time import time
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
from models.experimental import attempt_load
from utils.torch_utils import select_device, smart_inference_mode


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
action_size = 3  # In this example, the agent can choose between two actions: left click, aim and no-aim

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
action_space = [0,1,2,3,4,5,6,7,8]
observation_space = spaces.Box(low=0, high=255, shape=(screenShotHeight, screenShotWidth, 3), dtype=np.uint8)

def step(action):
    if action == 0:  # move up
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, -1, 0, 0)
    elif action == 1:  # move down
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, 1, 0, 0)
    elif action == 2:  # move left
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -1, 0, 0, 0)
    elif action == 3:  # move right
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1, 0, 0, 0)
    elif action == 4:  # move up-left
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -1, -1, 0, 0)
    elif action == 5:  # move up-right
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1, -1, 0, 0)
    elif action == 6:  # move down-left
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -1, 1, 0, 0)
    elif action == 7:  # move down-right
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1, 1, 0, 0)
    elif action == 8:  # left-click
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
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
    results = model(frame)
    new_box_center_distance = None
    print(results)
    closest_box_center = None
    closest_distance = None
    if results is not None:
        closest_box_center, closest_distance, labels = get_closest_box_center(results)
    if closest_box_center is not None and closest_distance is not None:
            new_box_center_distance = closest_distance
            print(f"closest box: {new_box_center_distance}")
    return new_box_center_distance

def detectx (frame, model):
    frame = [frame]
    #print(f"[INFO] Detecting. . . ")
    results = model(frame)
    print(results)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])
    return results
  #  labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

   # return labels, cordinates

def drawbox(bbox_coords, frame):
    if bbox_coords is not None:
        print(f"box cords: {bbox_coords[0]}, {bbox_coords[1]}")
        x1, y1, x2, y2 = int(bbox_coords[0]), int(bbox_coords[1]), int(bbox_coords[2]), int(bbox_coords[3]),
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2) ## BBox
        print(x1,x2,y1,y2)
        return frame,x1,x2,y1,y2
    else:
        return frame

def get_closest_box_center(results):
    screen_width = 1920
    screen_height = 1080
    center_x = screen_width // 2
    center_y = screen_height // 2
    closest_distance = None
    closest_box_center = None
    labels = None
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
                    closest_distance = distance
                    closest_box_center = (box_center_x,box_center_y)
                    labels = row[4]
    else:
        print("No results")
    print(closest_box_center, closest_distance)
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
        closest_box_center, closest_distance, labels = get_closest_box_center(results)
        if closest_box_center is not None:
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
def aim_towards_center(action=None):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    if aimbot and win32api.GetKeyState(lockKey):
        if action is not None:
            print("performing action")
            step(action)
    
    print("finished step function")
    return action

def execute_action(done, reward, action, prev_distance, model, closest_distance):
    print(f"action: {action}")
    aim_towards_center(action)
    sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}
    frame2 = grab_screenshot(sctArea)
    new_box_center_distance = get_new_observation(frame2)
    cWidth = 1920/2
    cHeight = 1080/2
    center_coords = [cWidth, cHeight]
    reward, done, prev_distance = compute_reward(action, closest_distance, new_box_center_distance)  # Implement this function to compute the reward for the chosen action
    return reward, done, prev_distance

def compute_reward(action, closest_distance, new_box_center_distance):
    reward = 0
    print("starting compute reward")
    prev_distance=None
    if action is not None and closest_distance is not None and new_box_center_distance is not None:
        min_distance = float("inf")

        if new_box_center_distance < min_distance:
            min_distance = new_box_center_distance

        if closest_distance is not None:
            if prev_distance is not None:
                if min_distance < prev_distance:
                    reward = 1
                else:
                    reward = -0.2
            prev_distance = min_distance
    done = True
    return reward, done, prev_distance

def display_output(img):
    cv2.imshow('Output', img)
    cv2.waitKey(1)

def main(run_loop=False, gameWindow=None):
        
    weights_path = 'runs/train/mw2_model2/weights/best.pt'
    device = torch.device('cpu')
    model = Model(cfg='models/mw2.yaml')
    print(f"[INFO] Loading model... ")
    classes = model.names
    prev_distance = None
    closest_distance = None
    closest_box = None
    reward = 0
    q_table_path = 'q_table.pk1'
    if os.path.exists(q_table_path):
        agent.load(q_table_path)
        print("Q-table loaded.")


    done = False
    if run_loop==True:
        sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}

        agent = AimbotAgent(state_size, action_size)
        load_agent_memory(agent)
        count = 0
        sTime = time()
        batch_size = 32
        episode_count = 0

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
            reward, done, prev_distance = execute_action(done, reward, action, prev_distance, closest_distance, model)    #execute action and reward   
            display_output(frame)    
            #cv2.imshow("vid", frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

            if keyboard.is_pressed('esc'):
                print(f"[INFO] Exiting. . . ")               
                break

            # Forced garbage cleanup every second
            count += 1
            if (time() - sTime) > 1:
                #print("CPS: {}".format(count))
                count = 0
                sTime = time()
                
                save_agent_memory(agent)
                gc.collect(generation=0)

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
            window.close()
            exit()

    window.close()
    print("Game Window: ", str(gw))

    return gw

gw = selectSettings()

if __name__ == "__main__":
    
    prev_distance = None
    main(run_loop=True, gameWindow=gw)