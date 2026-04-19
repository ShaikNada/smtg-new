import cv2
from ultralytics import YOLO

vid_path = r"app/static/uploads/test_video_98f2eb88.mp4"
cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total_frames}, FPS: {fps}")

model = YOLO("yolo11n.pt")
ok, frame = cap.read()
if ok:
    print(f"Frame shape: {frame.shape}")
    res = model.predict(frame, conf=0.35, imgsz=640)
    print("Boxes detected in first frame:", len(res[0].boxes) if res[0].boxes else 0)
    if res[0].boxes:
        for b in res[0].boxes:
            print(f"Class: {b.cls}, Conf: {b.conf}")
else:
    print("ERROR: Could not read first frame.")
cap.release()
