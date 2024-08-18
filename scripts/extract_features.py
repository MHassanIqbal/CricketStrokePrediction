import os
import numpy as np
import mediapipe as mp
import cv2
import joblib

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_pose(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(99)  # 33 landmarks * 3 coordinates (x, y, z)

def extract_features(frame_dir):
    features = []
    labels = []
    for stroke in os.listdir(frame_dir):
        stroke_dir = os.path.join(frame_dir, stroke)
        for frame_file in os.listdir(stroke_dir):
            frame_path = os.path.join(stroke_dir, frame_file)
            features.append(detect_pose(frame_path))
            labels.append(stroke)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    frame_dir = "frames"
    X, y = extract_features(frame_dir)
    joblib.dump((X, y), "data/features_labels.pkl")
