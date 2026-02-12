"""
Cricket Batting Performance Analysis
------------------------------------
This script analyzes a cricket batting video and computes:
1. Average Bat Speed
2. Average Bat Angle
3. Footwork Stability
4. Bat-Ball Connection Score

Author: Dinesh Kumar
"""

import cv2
import numpy as np
import math
import mediapipe as mp
import matplotlib.pyplot as plt


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error: Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    bat_speeds = []
    bat_angles = []
    foot_movements = []
    contact_frames = 0

    prev_bat_center = None
    prev_foot_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # -------- BAT DETECTION (Color Based) --------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Adjust HSV range if needed
        lower_color = np.array([10, 40, 40])
        upper_color = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            bat = max(contours, key=cv2.contourArea)

            if cv2.contourArea(bat) > 700:
                rect = cv2.minAreaRect(bat)
                (cx, cy), (bw, bh), angle = rect

                if bw < bh:
                    angle += 90

                bat_angles.append(abs(angle))

                bat_center = (int(cx), int(cy))

                if prev_bat_center:
                    speed = math.dist(bat_center, prev_bat_center) * fps
                    bat_speeds.append(speed)

                prev_bat_center = bat_center

        # -------- FOOTWORK USING MEDIAPIPE --------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            left_ankle = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]
            right_ankle =_
