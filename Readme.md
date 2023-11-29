# Drowsiness Detection System

## Introduction

This Python script utilizes computer vision techniques to detect drowsiness based on facial landmarks. It plays an alert sound if the person's eyes are closed for an extended period, indicating potential drowsiness.

## Libraries Used

```python
from scipy.spatial import distance  # Library for calculating distances between points
from imutils import face_utils  # Utility functions for working with facial landmarks
from pygame import mixer  # Library for playing audio
import imutils  # Utility functions for working with images and video
import dlib  # Library for facial landmark detection
import cv2  # OpenCV library for computer vision tasks
```

The above code imports the necessary libraries for the Driver Drowsiness Detection project. Here's a brief explanation of each library:

- **scipy.spatial.distance:** This library provides functions for calculating distances between points. It will be used to calculate the eye aspect ratio (EAR) for detecting drowsiness.

- **imutils.face_utils:** This module contains utility functions for working with facial landmarks. It will be used to extract the coordinates of the eyes from the facial landmarks.

- **pygame.mixer:** This library provides functions for playing audio. It will be used to play an alarm sound when drowsiness is detected.

- **imutils:** This module contains utility functions for working with images and video. It will be used for resizing and rotating images.

- **dlib:** This library is used for facial landmark detection. It will be used to detect the facial landmarks, including the eyes.

- **cv2:** OpenCV library for computer vision tasks.

Make sure to install these libraries before running the code.

## Initialization

```python
mixer.init()
mixer.music.load("music.wav")
```

Initializes the Pygame mixer for audio and loads the alert sound.

## Eye Aspect Ratio (EAR) Calculation

```python
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
```

Calculates the Eye Aspect Ratio (EAR) using facial landmarks.

```python
thresh = 0.25
frame_check = 20
```

Sets parameters for drowsiness detection. `thresh` is the EAR threshold, and `frame_check` is the number of consecutive frames to trigger an alert.

## Face Detection Setup

```python
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
```

Initializes the face detector, shape predictor, and defines indices for left and right eyes in facial landmarks.

## Video Capture Setup

```python
cap = cv2.VideoCapture(0)
```

Opens the video capture device (webcam).

# Main Loop

The main loop in the script is the central part responsible for continuously monitoring the video feed, detecting facial landmarks, calculating the Eye Aspect Ratio (EAR), and triggering an alert for drowsiness when necessary. Let's break down the main loop step by step:

```python
flag = 0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
```

Let's break down the key components of the main loop:

1. **Frame Acquisition:**
   ```python
   ret, frame = cap.read()
   ```
   - `cap.read()` captures a frame from the video feed.
   - `ret` is a boolean indicating whether the frame was successfully captured.
   - `frame` contains the captured frame.

2. **Frame Processing:**
   ```python
   frame = imutils.resize(frame, width=450)
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   ```
   - `imutils.resize()` resizes the frame to a width of 450 pixels for faster processing.
   - `cv2.cvtColor()` converts the resized frame to grayscale for face detection.

3. **Face Detection:**
   ```python
   subjects = detect(gray, 0)
   ```
   - `detect` is the face detection function provided by the `dlib` library.
   - `gray` is the grayscale image.
   - `subjects` contains the detected faces.

4. **Facial Landmark Detection and EAR Calculation:**
   ```python
   for subject in subjects:
       shape = predict(gray, subject)
       shape = face_utils.shape_to_np(shape)
       leftEye = shape[lStart:lEnd]
       rightEye = shape[rStart:rEnd]
       leftEAR = eye_aspect_ratio(leftEye)
       rightEAR = eye_aspect_ratio(rightEye)
       ear = (leftEAR + rightEAR) / 2.0
   ```
   - For each detected face (`subject`), facial landmarks are predicted using the `predict` function.
   - The coordinates of the facial landmarks are converted to a NumPy array for easier manipulation (`face_utils.shape_to_np`).
   - The coordinates of the left and right eyes are extracted.
   - The Eye Aspect Ratio (EAR) is calculated for both eyes, and the average is taken.

5. **Drowsiness Alert:**
   ```python
   if ear < thresh:
       flag += 1
       print(flag)
       if flag >= frame_check:
           cv2.putText(frame, "****************ALERT!****************", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           cv2.putText(frame, "****************ALERT!****************", (10, 325),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
           mixer.music.play()
   else:
       flag = 0
   ```
   - If the calculated EAR is below the threshold (`thresh`), it indicates potential drowsiness.
   - The `flag` variable keeps track of consecutive frames with drowsiness.
   - If the number of consecutive frames with drowsiness exceeds a threshold (`frame_check`), an alert is triggered.
   - The alert includes printing the flag count and playing an alarm sound.

6. **Overlay and Display:**
   ```python
   cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
   cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
   cv2.imshow("Frame", frame)
   ```
   - Contours of the convex hulls around the eyes are drawn on the frame for visual representation.
   - The frame with overlays is displayed using `cv2.imshow`.

7. **Key Check and Exit:**
   ```python
   key = cv2.waitKey(1) & 0xFF
   if key == ord("q"):
       break
   ```
   - The loop is exited if the 'q' key is pressed.

This main loop effectively integrates face detection, facial landmark detection, EAR calculation, and drowsiness alerting in real-time, providing a comprehensive solution for driver drowsiness detection.

Captures video frames, detects faces, calculates EAR, and triggers an alert for drowsiness. Press 'q' to exit.

## Cleanup

```python
cv2.destroyAllWindows()
cap.release()
```

Releases resources when the loop is terminated.

This script continuously monitors for drowsiness, providing an alert when detected.