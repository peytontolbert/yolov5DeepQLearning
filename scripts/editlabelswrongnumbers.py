import os

# Define the directory containing the text files
dir_path = 'outputs/labels'

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):  # Only process text files
        file_path = os.path.join(dir_path, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        # Replace the numbers using Python's string replace method
        text = text.replace('15', '0').replace('16', '1').replace('17', '2')
        # Write the modified text back to the file
        with open(file_path, 'w') as file:
            file.write(text)