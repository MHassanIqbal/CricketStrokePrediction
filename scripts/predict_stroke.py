import cv2
import joblib
import numpy as np
from extract_features import detect_pose

def predict_stroke(frame, model):
    features = detect_pose(frame)
    features = features.reshape(1, -1)  # Reshape for prediction
    return model.predict(features)[0]

def predict_stroke_in_video(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    model = joblib.load(model_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        stroke = predict_stroke(frame, model)
        cv2.putText(frame, stroke, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "path_to_new_video.mp4"
    model_path = "models/random_forest_model.pkl"
    predict_stroke_in_video(video_path, model_path)
