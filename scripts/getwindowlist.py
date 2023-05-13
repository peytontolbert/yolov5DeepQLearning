import pygetwindow as gw
import re

# This function removes non-breaking space characters and other unwanted characters from the window title
def clean_window_title(title):
    cleaned_title = re.sub(r'\u200b', '', title)  # Remove non-breaking space characters
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # Remove extra spaces
    return cleaned_title.strip()

all_windows = gw.getAllTitles()

cleaned_all_windows = [clean_window_title(title) for title in all_windows]

print(cleaned_all_windows)