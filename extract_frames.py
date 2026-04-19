import cv2
import os

video_path = "app/static/store-theft.mp4"
artifact_dir = "C:/Users/shaik/.gemini/antigravity/brain/461cf9e3-54c9-49b4-9835-cde166f75d1f"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

times = [1, 3, 5, 7, 9, 11]

for t in times:
    frame_id = int(fps * t)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(artifact_dir, f"frame_{t}.jpg"), frame)

cap.release()
print("Extracted frames!")
