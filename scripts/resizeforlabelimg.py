import os

# define paths for input and output directories
input_dir = 'labeled_images/textfiles/'
output_dir = 'labeled_images/newtextfiles/'

# define image dimensions
img_width = 1920
img_height = 1080

# loop through all text files in input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        with open(input_dir + filename, 'r') as f:
            contents = f.read().split()

            # get class label
            class_label = contents[0]

            # calculate normalized bounding box coordinates
            x_min = float(contents[1]) / img_width
            y_min = float(contents[2]) / img_height
            x_max = float(contents[3]) / img_width
            y_max = float(contents[4]) / img_height

            # write normalized bounding box coordinates to new file in output directory
            with open(output_dir + filename, 'w') as f_out:
                f_out.write(f'{class_label} {x_min:.6f} {y_min:.6f} {x_max - x_min:.6f} {y_max - y_min:.6f}\n')