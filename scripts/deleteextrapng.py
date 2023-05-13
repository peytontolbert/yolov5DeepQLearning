import os

folder1 = 'outputslabels'
folder2 = 'outputs'

# Get file names in folder1 without extension
txt_files = [os.path.splitext(f)[0] for f in os.listdir(folder1) if f.endswith('.txt')]

# Get file names in folder2 without extension
png_files = [os.path.splitext(f)[0] for f in os.listdir(folder2) if f.endswith('.png')]

# Delete files in folder2 that don't have corresponding file names in folder1
for png_file in png_files:
    if png_file not in txt_files:
        os.remove(os.path.join(folder2, png_file + '.png'))
        print(f"Deleted {png_file}.png from {folder2}.")