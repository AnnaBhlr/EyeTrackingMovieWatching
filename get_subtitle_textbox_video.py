import csv
import cv2
import easyocr
import numpy as np
import os

def save_as_csv(data, filename, headers):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(headers)

        # write the data
        writer.writerows(data)

def find_ribbon_top(img, threshold=30):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get the height of the image
    height = img.shape[0]
    # Initialize the previous row's average color
    prev_avg = np.average(gray[height-1])

    # Scan the image from the bottom up
    for y in range(height-2, -1, -1):
        # Calculate the average color of the current row
        curr_avg = np.average(gray[y])

        # If the difference in average color is greater than the threshold, return the current y-coordinate
        if abs(curr_avg - prev_avg) > threshold:
            return y + 1

        # Update the previous row's average color
        prev_avg = curr_avg

    # If no significant color change was found, return 0
    return 0

video_directory = 'PATH/TO/VIDEOFOLDER'

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

# Get a list of all .mp4 files in the directory
mp4_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

# Loop through the list of .mp4 files
for mp4_file in mp4_files:

    # Open the video
    video = cv2.VideoCapture(os.path.join(video_directory, mp4_file))

    # Initialize a counter for the frames
    frame_counter = 0

    # List to store all bounding box coordinates
    all_boxes = []


    while True:
        # Read a frame from the video file
        ret, frame = video.read()
        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break
        #print(frame_counter)
        # Find the top of the ribbon
        ribbon_top = find_ribbon_top(frame)

        if ribbon_top == 0:
            coordinates = [frame_counter, np.nan, np.nan, np.nan, np.nan]
            all_boxes.append(coordinates)
            frame_counter += 1
            continue

        # Crop the image to the ribbon
        ribbon = frame[ribbon_top:]
        

        # perform OCR on cropped frame
        results = reader.readtext(ribbon)

        # List to store all bounding box coordinates
        coords = []

        if results:
            for result in results:
                # each result is a tuple of the bounding box and the recognized text
                bounding_box, _ , _ = result
                bounding_box = [(x, y + ribbon_top) for x, y in bounding_box]
                coords.extend(bounding_box)

            # Convert to numpy array for easier manipulation
            coords = np.array(coords)

            # Get minimum and maximum coordinates
            min_coord = coords.min(axis=0)
            max_coord = coords.max(axis=0)

            # Define the amount of space to add in pixels according to requirements
            space = 10  # in pixel


            # Subtract space from minimum coordinates and add space to maximum coordinates
            min_coord -= space
            max_coord += space

            # Create a rectangle patch for the merged bounding box
            top_left = tuple(map(int, min_coord))
            bottom_right = tuple(map(int, max_coord))
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]

            coordinates = [frame_counter, top_left[0], top_left[1], width, height]

        else:
            coordinates = [frame_counter, np.nan, np.nan, np.nan, np.nan]

        all_boxes.append(coordinates)


        # Increment the frame counter
        frame_counter += 1

    # Release the video file
    video.release()

    headers = ["frame_number", "top_left_x", "top_left_y", "width", 'height']
    output_filename = mp4_file.split('.')[0] + '.csv'
    save_as_csv(all_boxes, os.path.join(video_directory,output_filename), headers)

    print(mp4_file + ' done')
