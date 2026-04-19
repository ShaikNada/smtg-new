import os
import cv2
import time as time_module
from ultralytics import YOLO

def test_tracker():
    vid_path = "app/static/uploads/test_video_98f2eb88.mp4"
    cap = cv2.VideoCapture(vid_path)
    model = YOLO("yolo11n.pt")
    frame_index = 0
    stride = 4

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        frame_index += 1
        if frame_index % stride != 0: continue
        
        try:
            results = model.track(
                source=frame,
                conf=0.35,
                imgsz=640,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
            res = results[0]
            if res.boxes is not None:
                print(f"Frame {frame_index}: boxes length = {len(res.boxes)}. ids = {res.boxes.id}")
            else:
                print(f"Frame {frame_index}: no boxes")
        except Exception as e:
            print("ERROR", e)
            break
            
    cap.release()

if __name__ == "__main__":
    test_tracker()
