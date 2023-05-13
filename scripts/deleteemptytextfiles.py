import os

folder_path = 'outputslabels' # replace with the path to your folder

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.endswith('.txt') and os.stat(file_path).st_size == 0:
        os.remove(file_path)
        print(f'Deleted {filename}')