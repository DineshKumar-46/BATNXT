"""
AI Cricket Performance Analyzer
Description:
Analyzes cricket batting video and calculates:
- Bat Angle
- Bat Speed
- Footwork Movement
- Ball Contact Quality
- Generates analysis graph + output video + CSV report
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# ==============================================================
# CONFIGURATION
# ==============================================================

VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO = "output_analysis.mp4"
CSV_REPORT = "performance_report.csv"
GRAPH_OUTPUT = "performance_graph.png"

# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def calculate_distance(p1, p2):
    """Calculate Euclidean distance"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_speed(prev_point, curr_point, fps):
    """Calculate speed in pixels per second"""
    if prev_point is None:
        return 0
    distance = calculate_distance(prev_point, curr_point)
    return distance * fps


def calculate_angle(rect):
    """Extract angle from rotated rectangle"""
    (cx, cy), (w, h), angle = rect

    if w < h:
        angle = 90 + angle

    return abs(angle)


def contact_quality(speed, angle):
    """Estimate ball contact quality"""
    if speed > 400 and 30 < angle < 70:
        return "Excellent"
    elif speed > 250:
        return "Good"
    elif speed > 100:
        return "Average"
    else:
        return "Poor"

# ==============================================================
# ANALYZER CLASS
# ==============================================================

class CricketAnalyzer:

    def __init__(self, video_path):
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception("Video file not found")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out = cv2.VideoWriter(
            OUTPUT_VIDEO,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.bat_prev_center = None
        self.speeds = []
        self.angles = []
        self.foot_movements = []

    # ----------------------------------------------------------
    # BAT DETECTION
    # ----------------------------------------------------------

    def detect_bat(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([5, 50, 50])
        upper = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            bat = max(contours, key=cv2.contourArea)

            if cv2.contourArea(bat) > 800:
                rect = cv2.minAreaRect(bat)
                box = cv2.boxPoints(rect)
                box = box.astype(int)

                center = (int(rect[0][0]), int(rect[0][1]))
                angle = calculate_angle(rect)

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                return center, angle

        return None, None

    # ----------------------------------------------------------
    # FOOTWORK DETECTION USING MEDIAPIPE
    # ----------------------------------------------------------

    def analyze_footwork(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            left_ankle = results.pose_landmarks.landmark[27]
            right_ankle = results.pose_landmarks.landmark[28]

            lx = int(left_ankle.x * self.width)
            ly = int(left_ankle.y * self.height)
            rx = int(right_ankle.x * self.width)
            ry = int(right_ankle.y * self.height)

            movement = calculate_distance((lx, ly), (rx, ry))

            self.foot_movements.append(movement)

            cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)
            cv2.circle(frame, (rx, ry), 5, (255, 0, 0), -1)

    # ----------------------------------------------------------
    # PROCESS VIDEO
    # ----------------------------------------------------------

    def process(self):
        print("Processing video...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            center, angle = self.detect_bat(frame)

            if center:
                speed = calculate_speed(self.bat_prev_center,
                                        center,
                                        self.fps)

                self.speeds.append(speed)
                self.angles.append(angle)

                quality = contact_quality(speed, angle)

                cv2.putText(frame, f"Speed: {int(speed)}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

                cv2.putText(frame, f"Angle: {int(angle)}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

                cv2.putText(frame, f"Contact: {quality}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                self.bat_prev_center = center

            self.analyze_footwork(frame)

            self.out.write(frame)

        self.cap.release()
        self.out.release()

        print("Video processing completed!")

    # ----------------------------------------------------------
    # GENERATE REPORT
    # ----------------------------------------------------------

    def generate_report(self):

        avg_speed = np.mean(self.speeds) if self.speeds else 0
        avg_angle = np.mean(self.angles) if self.angles else 0
        avg_foot = np.mean(self.foot_movements) if self.foot_movements else 0

        performance_score = (
            (avg_speed / 500) * 40 +
            (avg_angle / 90) * 30 +
            (avg_foot / 200) * 30
        )

        performance_score = min(100, performance_score)

        data = {
            "Average Bat Speed": [avg_speed],
            "Average Bat Angle": [avg_angle],
            "Footwork Movement": [avg_foot],
            "Performance Score": [performance_score]
        }

        df = pd.DataFrame(data)
        df.to_csv(CSV_REPORT, index=False)

        print("\n========= PERFORMANCE SUMMARY =========")
        print(df)

        return avg_speed, avg_angle, avg_foot, performance_score

    # ----------------------------------------------------------
    # PLOT GRAPH
    # ----------------------------------------------------------

    def plot_graph(self, avg_speed, avg_angle,
                   avg_foot, performance_score):

        labels = ["Bat Speed", "Bat Angle",
                  "Footwork", "Performance"]

        values = [avg_speed, avg_angle,
                  avg_foot, performance_score]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Cricket Performance Analysis")
        plt.ylabel("Values")
        plt.savefig(GRAPH_OUTPUT)
        plt.close()

        print("Graph saved!")

# ==============================================================
# MAIN FUNCTION
# ==============================================================

def main():

    analyzer = CricketAnalyzer(VIDEO_PATH)

    analyzer.process()

    avg_speed, avg_angle, avg_foot, score = analyzer.generate_report()

    analyzer.plot_graph(avg_speed,
                        avg_angle,
                        avg_foot,
                        score)

    print("\nAll outputs generated successfully!")
    print("Output Video:", OUTPUT_VIDEO)
    print("CSV Report:", CSV_REPORT)
    print("Graph:", GRAPH_OUTPUT)


if __name__ == "__main__":
    main()


