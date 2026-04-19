import cv2
from ultralytics import YOLO

def test():
    vid_path = "app/static/uploads/test_video_98f2eb88.mp4"
    cap = cv2.VideoCapture(vid_path)
    model = YOLO("yolo11n.pt")
    
    ok, frame = cap.read()
    results = model.track(frame, conf=0.35, imgsz=640, persist=True)
    res = results[0]
    for box in res.boxes:
        print(f"box.id: {box.id}")

if __name__ == "__main__":
    test()
