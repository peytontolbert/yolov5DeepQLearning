import os

directory = 'outputs/imagesrenamed'

# Get all PNG files in the directory
png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

# Sort the files alphabetically
png_files.sort()

# Rename the files sequentially
for i, old_name in enumerate(png_files):
    new_name = str(i+116) + '.png'
    os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))