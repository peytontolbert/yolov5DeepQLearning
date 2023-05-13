import os

directory = 'outputs/labelsrenamed'

# Get all PNG files in the directory
txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

# Sort the files alphabetically
txt_files.sort()

# Rename the files sequentially
for i, old_name in enumerate(txt_files):
    new_name = str(i+116) + '.txt'
    os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))