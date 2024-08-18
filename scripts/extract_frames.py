import cv2
import os

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    cv2.destroyAllWindows()

def process_videos(input_dir, output_dir):
    for video_file in os.listdir(input_dir):
        if video_file.endswith('.mp4'):
            stroke_name = os.path.splitext(video_file)[0]
            stroke_output_dir = os.path.join(output_dir, stroke_name)
            os.makedirs(stroke_output_dir, exist_ok=True)
            video_path = os.path.join(input_dir, video_file)
            extract_frames(video_path, stroke_output_dir)

if __name__ == "__main__":
    input_dir = "data/videos"
    output_dir = "frames"
    process_videos(input_dir, output_dir)
