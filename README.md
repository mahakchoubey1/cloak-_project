🧙‍♂️ Red Cloak Invisibility using OpenCV

This project implements a Harry Potter–style invisibility cloak effect using Python and OpenCV.
When a person wears a red-colored cloak, the cloak gets replaced with the background, making them appear invisible.


*************************************************************************************************************************************************************************************
📌 Features

Detects red color in real-time video.

Masks the detected region and replaces it with a pre-captured background.

Works with any webcam.

Simple and efficient OpenCV HSV color masking.



*************************************************************************************************************************************************************************************
📷 Demo

[https://www.linkedin.com/in/mahak-choubey-a38b90289/](https://www.linkedin.com/posts/mahak-choubey-a38b90289_opencv-python-computervision-activity-7362187576801202176-gEC_?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEYesswBRGQsJleTVM6ZowW1NNSUKv-dQJo)

*************************************************************************************************************************************************************************************
🛠️ Requirements

Install dependencies:

pip install opencv-python numpy



*************************************************************************************************************************************************************************************
📄 Code Overview

HSV Color Ranges for Red:
import numpy as np

# Red color ranges in HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])


*************************************************************************************************************************************************************************************
🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/red-cloak-invisibility.git
cd red-cloak-invisibility


Run the script:

python cloak.py


Wear a red cloak and see the magic happen.



*************************************************************************************************************************************************************************************


⚙️ Working Principle

Capture the background frame.

Convert each video frame from BGR to HSV color space.

Create masks for the red color.

Replace masked areas with background pixels.

Display the final frame.



*************************************************************************************************************************************************************************************

📌 Notes

Ensure you have good lighting for better results.

The cloak should be pure red for best detection.

You can tweak the HSV values to improve detection.

















    
