# Get box around Subtitles in videos

This Python script automates the process of extracting and recognizing subtitle text from video frames.
 It uses OpenCV for video processing and easyOCR for Optical Character Recognition (OCR).

## Features
- Scans video frames to find text ribbon locations based on color differences.
- Performs OCR on detected ribbons to extract english text (OCR supports heaps of other languages)
- Outputs the results to CSV files, with details on text location and frame number.

## Prerequisites
Before you can run this script, you'll need to have Python installed, along with the following libraries:
- OpenCV
- easyOCR
- NumPy

## Usage
- Place MP4 videos in a designated folder.
- Edit the script to specify the path to your video folder by changing the video_directory variable.
