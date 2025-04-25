# I Spy by Edward Ng
# User picks an object and says its colour. The computer searches all objects of that colour in view
# and guesses which object was chosen.
# 2/18/2023

import os
import random
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# GLOBAL variables
obj_list = []
frame = None
processed_frame = None
current_guess_index = 0

def map_colours(colour, frame):
    '''Converts colour string to cv colour range.'''
    print(f"Mapping colour: {colour}")
    if colour == "red":
        lower_red1 = (0, 50, 50)
        upper_red1 = (10, 255, 255)
        lower_red2 = (170, 50, 50)
        upper_red2 = (180, 255, 255)
        mask1 = cv2.inRange(frame, lower_red1, upper_red1)
        mask2 = cv2.inRange(frame, lower_red2, upper_red2)
        return cv2.bitwise_or(mask1, mask2)
    elif colour == "orange":
        return cv2.inRange(frame, (10, 100, 100), (25, 255, 255))
    elif colour == "yellow":
        return cv2.inRange(frame, (20, 100, 100), (35, 255, 255))
    elif colour == "green":
        return cv2.inRange(frame, (35, 40, 40), (85, 255, 255))
    elif colour == "blue":
        return cv2.inRange(frame, (90, 50, 50), (130, 255, 255))
    elif colour == "pink":
        return cv2.inRange(frame, (145, 50, 50), (170, 255, 255))
    elif colour == "brown":
        return cv2.inRange(frame, (10, 50, 20), (20, 200, 150))
    elif colour == "black":
        return cv2.inRange(frame, (0, 0, 0), (180, 70, 70))
    elif colour == "grey":
        return cv2.inRange(frame, (0, 0, 50), (180, 50, 200))
    elif colour == "white":
        return cv2.inRange(frame, (0, 0, 200), (180, 50, 255))
    else:
        print(f"Color '{colour}' is not recognized. Please choose a valid color.")
        return None

def process_image(colour):
    '''Processes the image based on the selected colour.'''
    global frame, processed_frame, obj_list

    print("Processing image...")
    if frame is None:
        print("No frame loaded.")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = map_colours(colour, hsv)

    if mask is None or np.count_nonzero(mask) == 0:
        print("I cannot find your object :(")
        return

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("I cannot find your object :(")
        return

    print(f"Found {len(contours)} contours.")
    obj_list.clear()

    # Calculate the total image area
    image_area = frame.shape[0] * frame.shape[1]
    min_area = 20  # Minimum area set to approximately 20 pixels
    max_area = 0.50 * image_area  # 50% of the image area

    for contour in contours:
        area = cv2.contourArea(contour)
        print(f"Contour area: {area}")

        # Check if the contour area is within the allowed range
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = frame[y:y+h, x:x+w]
            classIds, confs, bbox = net.detect(cropped_image, confThreshold=0.5)

            # Check if any objects are detected
            if classIds is not None and len(classIds) > 0:
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, classNames[classId - 1].upper(), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    if classNames[classId - 1] not in obj_list:
                        obj_list.append(classNames[classId - 1])
            else:
                print("No objects detected in the cropped region.")
        else:
            print(f"Contour area {area} is out of bounds (min: {min_area}, max: {max_area}).")

    processed_frame = frame.copy()
    update_image()
    start_guessing()

def update_image():
    '''Updates the displayed image in the GUI.'''
    global processed_frame
    if processed_frame is None:
        return

    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

def start_guessing():
    '''Starts guessing objects in the GUI.'''
    global current_guess_index
    current_guess_index = 0
    if len(obj_list) == 0:
        print("No objects to guess.")
        return

    print("Starting guessing...")
    color_input.config(state=tk.DISABLED)
    submit_button.config(state=tk.DISABLED)
    guess_next_object()

def guess_next_object():
    '''Guesses the next object in the list.'''
    global current_guess_index
    if current_guess_index >= len(obj_list):
        print("I cannot guess your object :(")
        return

    guess_label.config(text=f"Is your object a {obj_list[current_guess_index]}?")
    yes_button.config(state=tk.NORMAL)
    no_button.config(state=tk.NORMAL)

def on_yes():
    '''Handles the "Yes" button click.'''
    print(f"Guessed correctly: {obj_list[current_guess_index]}")
    guess_label.config(text="I have guessed your object!")
    yes_button.config(state=tk.DISABLED)
    no_button.config(state=tk.DISABLED)

def on_no():
    '''Handles the "No" button click.'''
    global current_guess_index
    print(f"Guessed incorrectly: {obj_list[current_guess_index]}")
    current_guess_index += 1
    if current_guess_index < len(obj_list):
        guess_next_object()
    else:
        guess_label.config(text="I cannot guess your object :(")
        yes_button.config(state=tk.DISABLED)
        no_button.config(state=tk.DISABLED)

def on_submit():
    '''Handles the color input submission.'''
    colour = color_input.get().strip().lower()
    process_image(colour)

def reset_game():
    '''Resets the game to its initial state and loads a new image.'''
    global frame, processed_frame, obj_list, current_guess_index

    print("Starting a new game...")
    # Reset global variables
    obj_list.clear()
    current_guess_index = 0

    # Load a new random image
    random_image_path = os.path.join(scenes_folder, random.choice(image_files))
    print(f"Loaded new image: {random_image_path}")
    frame = cv2.imread(random_image_path)
    frame = cv2.resize(frame, (800, 600))
    processed_frame = frame.copy()

    # Update the GUI
    update_image()
    guess_label.config(text="")
    color_input.config(state=tk.NORMAL)
    submit_button.config(state=tk.NORMAL)
    yes_button.config(state=tk.DISABLED)
    no_button.config(state=tk.DISABLED)

# Initialize the GUI
root = tk.Tk()
root.title("I Spy")

# Load the image
scenes_folder = "scenes"
image_files = [f for f in os.listdir(scenes_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("No images found in the scenes folder.")
    exit()

random_image_path = os.path.join(scenes_folder, random.choice(image_files))
frame = cv2.imread(random_image_path)
frame = cv2.resize(frame, (800, 600))
processed_frame = frame.copy()

# Load the COCO class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the detection model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph_coco.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create the canvas for displaying the image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# Create the input field and submit button
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

color_label = ttk.Label(input_frame, text="Enter a color:")
color_label.pack(side=tk.LEFT, padx=5)

color_input = ttk.Entry(input_frame)
color_input.pack(side=tk.LEFT, padx=5)

submit_button = ttk.Button(input_frame, text="Submit", command=on_submit)
submit_button.pack(side=tk.LEFT, padx=5)

# Create the guessing section
guess_frame = ttk.Frame(root)
guess_frame.pack(pady=10)

guess_label = ttk.Label(guess_frame, text="")
guess_label.pack()

yes_button = ttk.Button(guess_frame, text="Yes", command=on_yes, state=tk.DISABLED)
yes_button.pack(side=tk.LEFT, padx=5)

no_button = ttk.Button(guess_frame, text="No", command=on_no, state=tk.DISABLED)
no_button.pack(side=tk.LEFT, padx=5)

# Create the "New Game" button
new_game_button = ttk.Button(root, text="New Game", command=reset_game)
new_game_button.pack(pady=10)

# Display the initial image
update_image()

# Start the GUI event loop
root.mainloop()

