import cv2
import os

# Set the path to your video file
video_path = 'video.flv'

# Set the interval for taking screenshots (in seconds)
interval = 0.5

# Set the path to the output directory for the screenshots
output_dir = 'outputs'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize the frame count and time counter
frame_count = 0
time_counter = 0

# Loop through the video frames
while True:
    # Read the next frame from the video file
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame count and time counter
    frame_count += 1
    time_counter += 1 / cap.get(cv2.CAP_PROP_FPS)

    # Take a screenshot at the specified interval
    if time_counter >= interval:
        # Reset the time counter
        time_counter = 0

        # Set the output file name and path
        output_path = os.path.join(output_dir, f'{frame_count}.png')

        # Save the screenshot to disk
        cv2.imwrite(output_path, frame)

# Release the video file
cap.release()