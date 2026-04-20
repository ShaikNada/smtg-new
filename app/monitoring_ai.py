import json
import os
import shutil
import threading
import time as time_module
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MONITORING_STATIC_DIR = STATIC_DIR / "monitoring"
MODELS_DIR = BASE_DIR / "models"

DETECT_MODEL_NAME = os.getenv("CIRIS_DETECT_MODEL", "yolov8s.pt")
POSE_MODEL_NAME = os.getenv("CIRIS_POSE_MODEL", "yolov8n-pose.pt")
TRACKER_NAME = os.getenv("CIRIS_TRACKER", "")
EVENT_MODEL_PATH = os.getenv("CIRIS_EVENT_MODEL_PATH", str(MODELS_DIR / "crime_event_classifier.ts"))
EVENT_LABELS_PATH = os.getenv("CIRIS_EVENT_LABELS_PATH", str(MODELS_DIR / "crime_event_labels.json"))
DEVICE = os.getenv("CIRIS_MONITOR_DEVICE", "cpu")
FRAME_STRIDE = int(os.getenv("CIRIS_FRAME_STRIDE", "4"))
POSE_EVERY_N = int(os.getenv("CIRIS_POSE_EVERY_N", "2"))
CLIP_LEN = int(os.getenv("CIRIS_EVENT_CLIP_LEN", "16"))
CONF_THRESHOLD = float(os.getenv("CIRIS_DETECT_CONF", "0.15"))
IOU_THRESHOLD = float(os.getenv("CIRIS_DETECT_IOU", "0.45"))
NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("CIRIS_NOTIFICATION_COOLDOWN", "120"))
INCIDENT_CLEAR_FRAMES = int(os.getenv("CIRIS_INCIDENT_CLEAR_FRAMES", "60"))
WS_SEND_INTERVAL_SECONDS = float(os.getenv("CIRIS_WS_INTERVAL", "0.35"))

LOGICAL_CAMERAS = {
    "cam-01": {"label": "CAM-01", "area": "Main Road Junction - Vehicle Accident Video", "feed_id": "feed1"},
    "cam-02": {"label": "CAM-02", "area": "Public Square - Fighting Video", "feed_id": "feed2"},
    "cam-03": {"label": "CAM-03", "area": "Market Area - Chain Snatching Video", "feed_id": "feed3"},
    "cam-04": {"label": "CAM-04", "area": "Bank Zone - Robbery/Weapon Video", "feed_id": "feed4"},
    "cam-05": {"label": "CAM-05", "area": "Main Road Junction - Vehicle Accident Video", "feed_id": "feed1"},
    "cam-06": {"label": "CAM-06", "area": "Public Square - Fighting Video", "feed_id": "feed2"},
    "cam-07": {"label": "CAM-07", "area": "Market Area - Chain Snatching Video", "feed_id": "feed3"},
    "cam-08": {"label": "CAM-08", "area": "Bank Zone - Robbery/Weapon Video", "feed_id": "feed4"},
}

CAMERA_GROUPS = {
    "group1": ["cam-01", "cam-02", "cam-03", "cam-04"],
    "group2": ["cam-05", "cam-06", "cam-07", "cam-08"],
}

PHYSICAL_FEEDS = {
    "feed1": {
        "source_camera": "cam-01",
        "display_name": "SOURCE-01 (Vehicle Accident)",
        "video_candidates": ["cam1.mp4", "cam1.mp4"],
        "logical_cameras": ["cam-01", "cam-05"],
    },
    "feed2": {
        "source_camera": "cam-02",
        "display_name": "SOURCE-02 (Fighting)",
        "video_candidates": ["cam2.mp4", "cam2.mp4"],
        "logical_cameras": ["cam-02", "cam-06"],
    },
    "feed3": {
        "source_camera": "cam-03",
        "display_name": "SOURCE-03 (Chain Snatching)",
        "video_candidates": ["cam3.mp4", "cam3.mp4"],
        "logical_cameras": ["cam-03", "cam-07"],
    },
    "feed4": {
        "source_camera": "cam-04",
        "display_name": "SOURCE-04 (Robbery/Weapon)",
        "video_candidates": ["cam4.mp4", "cam4.mp4"],
        "logical_cameras": ["cam-04", "cam-08"],
    },
}

ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "knife", "gun", "cell phone",
}
WEAPON_CLASSES = {"knife", "gun"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}

COLOR_MAP = {
    "person": "#3b82f6",
    "bicycle": "#06b6d4",
    "car": "#10b981",
    "motorcycle": "#f59e0b",
    "bus": "#8b5cf6",
    "truck": "#22c55e",
    "backpack": "#ec4899",
    "handbag": "#f97316",
    "suitcase": "#eab308",
    "knife": "#ef4444",
    "gun": "#dc2626",
}

EVENT_LABELS_DEFAULT = ["normal", "accident", "fighting", "chain_snatching", "robbery", "weapon_use"]
EVENT_THRESHOLDS = {
    "accident": 0.35,
    "fighting": 0.30,
    "chain_snatching": 0.30,
    "robbery": 0.35,
    "weapon_use": 0.25,
}

HEURISTIC_CONFIRM_FRAMES = {
    "ACCIDENT SUSPECTED": 2,
    "FIGHTING DETECTED": 2,
    "CHAIN SNATCHING / THEFT SUSPECTED": 2,
    "ARMED ROBBERY SUSPECTED": 1,
    "WEAPON DETECTED": 1,
}

SEVERITY_ORDER = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}


def distance_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return float((dx * dx + dy * dy) ** 0.5)


def average_step_distance(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    steps = [distance_points(points[i - 1], points[i]) for i in range(1, len(points))]
    return float(sum(steps) / max(len(steps), 1))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def iou_boxes(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a["w"] * a["h"]) + (b["w"] * b["h"]) - inter
    return inter / union if union > 0 else 0.0


def shrink_box(box: Dict[str, Any], scale_w: float = 0.88, scale_h: float = 0.90) -> Dict[str, Any]:
    cx = box["x"] + (box["w"] / 2.0)
    cy = box["y"] + (box["h"] / 2.0)
    nw = max(0.02, min(1.0, box["w"] * scale_w))
    nh = max(0.03, min(1.0, box["h"] * scale_h))
    nx = max(0.0, min(1.0 - nw, cx - nw / 2.0))
    ny = max(0.0, min(1.0 - nh, cy - nh / 2.0))
    data = dict(box)
    data.update({"x": nx, "y": ny, "w": nw, "h": nh})
    return data


def event_to_severity(label: str) -> Tuple[float, str]:
    upper = (label or "").upper()
    if "WEAPON" in upper or "ROBBERY" in upper:
        return 0.95, "CRITICAL"
    if "ACCIDENT" in upper:
        return 0.85, "HIGH"
    if "FIGHT" in upper or "CHAIN" in upper or "THEFT" in upper:
        return 0.70, "MODERATE"
    return 0.35, "LOW"


def as_notification_signature(feed_id: str, label: str) -> str:
    return f"{feed_id}|{label.strip().upper()}"


class MonitoringManager:
    def __init__(self) -> None:
        self.state_lock = threading.Lock()
        self.notifications_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started = False
        self.notifications: deque = deque(maxlen=400)
        self.last_notification_at: Dict[str, float] = {}
        self.workers: Dict[str, threading.Thread] = {}
        self.detect_model = None
        self.pose_model = None
        self.event_model = None
        self.event_labels: List[str] = list(EVENT_LABELS_DEFAULT)
        self.frontend_config: Dict[str, Any] = {}
        self.feed_states: Dict[str, Dict[str, Any]] = {}
        self._build_frontend_config()
        self._init_feed_states()

    @property
    def model_stack_label(self) -> str:
        event_model_label = "TorchScript Event Model" if self.event_model is not None else "YOLO + temporal heuristics"
        return f"{DETECT_MODEL_NAME} + {POSE_MODEL_NAME} + {event_model_label}"

    def _build_frontend_config(self) -> None:
        self.frontend_config = {
            "cameraGroups": CAMERA_GROUPS,
            "logicalCameras": LOGICAL_CAMERAS,
            "physicalFeeds": {},
        }
        for feed_id, spec in PHYSICAL_FEEDS.items():
            video_url = None
            video_path = self._resolve_feed_path(feed_id)
            if video_path:
                video_url = self._to_static_url(video_path)
            self.frontend_config["physicalFeeds"][feed_id] = {
                "display_name": spec["display_name"],
                "source_camera": spec["source_camera"],
                "video_url": video_url,
                "logical_cameras": list(spec["logical_cameras"]),
            }

    def _init_feed_states(self) -> None:
        self.feed_states = {}
        for feed_id, spec in PHYSICAL_FEEDS.items():
            self.feed_states[feed_id] = self._make_initial_feed_state(feed_id, spec)

    def _make_initial_feed_state(self, feed_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feed_id": feed_id,
            "source_camera": spec["source_camera"],
            "display_name": spec["display_name"],
            "video_path": None,
            "video_url": None,
            "status": "idle",
            "running": False,
            "action": "SCANNING",
            "summary": "Waiting for source video.",
            "confidence": 0.0,
            "severity": 0.0,
            "severity_label": "LOW",
            "alert": False,
            "counts": {},
            "events": [],
            "boxes": [],
            "telemetry": {
                "frame_index": 0,
                "total_frames": 0,
                "source_fps": 0.0,
                "processing_fps": 0.0,
                "latency_ms": 0.0,
                "loop_count": 0,
            },
            "error": None,
            "active_incidents": {},
            "last_video_time_sec": 0.0,
            "latest_notification_id": None,
        }

    def ensure_static_videos(self) -> None:
        MONITORING_STATIC_DIR.mkdir(parents=True, exist_ok=True)
        fallback_locations = [
            Path("/mnt/data"),
            BASE_DIR,
            BASE_DIR.parent,
        ]
        for feed_id, spec in PHYSICAL_FEEDS.items():
            target_path = MONITORING_STATIC_DIR / f"{spec['source_camera']}.mp4"
            if target_path.exists():
                self.frontend_config["physicalFeeds"][feed_id]["video_url"] = self._to_static_url(target_path)
                continue
            copied = False
            for candidate in spec["video_candidates"]:
                for root in fallback_locations:
                    source_path = root / candidate
                    if source_path.exists():
                        shutil.copy2(str(source_path), str(target_path))
                        copied = True
                        break
                if copied:
                    break
            if target_path.exists():
                self.frontend_config["physicalFeeds"][feed_id]["video_url"] = self._to_static_url(target_path)

    def _to_static_url(self, path: Path) -> str:
        rel = path.relative_to(STATIC_DIR)
        return "/static/" + quote(str(rel).replace(os.sep, "/"))

    def _resolve_feed_path(self, feed_id: str) -> Optional[Path]:
        spec = PHYSICAL_FEEDS[feed_id]
        preferred = MONITORING_STATIC_DIR / f"{spec['source_camera']}.mp4"
        if preferred.exists():
            return preferred
        for candidate in spec["video_candidates"]:
            p = MONITORING_STATIC_DIR / candidate
            if p.exists():
                return p
        return None

    def start(self) -> None:
        with self.start_lock:
            if self.started:
                return
            self.ensure_static_videos()
            self._load_models_if_needed()
            self._build_frontend_config()
            self.stop_event.clear()
            for feed_id in PHYSICAL_FEEDS.keys():
                thread = threading.Thread(target=self._run_feed_worker, args=(feed_id,), daemon=True)
                self.workers[feed_id] = thread
                thread.start()
            self.started = True

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.workers.values():
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.workers.clear()
        self.started = False

    def restart(self) -> None:
        self.stop()
        self._init_feed_states()
        self.start()

    def _load_models_if_needed(self) -> None:
        with self.model_lock:
            if YOLO is None:
                self._broadcast_model_error("Ultralytics is not installed. Run: pip install ultralytics")
                return
            if self.detect_model is None:
                try:
                    self.detect_model = YOLO(DETECT_MODEL_NAME)
                except Exception as e:
                    self._broadcast_model_error(f"Failed to load detection model: {e}")
                    return
            # Skip pose model for now to focus on basic detection
            # if self.pose_model is None:
            #     try:
            #         self.pose_model = YOLO(POSE_MODEL_NAME)
            #     except Exception as e:
            #         print(f"Failed to load pose model: {e}")
            #         self.pose_model = None
            if torch is not None and self.event_model is None and Path(EVENT_MODEL_PATH).exists():
                try:
                    self.event_model = torch.jit.load(EVENT_MODEL_PATH, map_location=DEVICE)
                    self.event_model.eval()
                except Exception:
                    self.event_model = None
            if Path(EVENT_LABELS_PATH).exists():
                try:
                    self.event_labels = json.loads(Path(EVENT_LABELS_PATH).read_text(encoding="utf-8"))
                except Exception:
                    self.event_labels = list(EVENT_LABELS_DEFAULT)

    def _broadcast_model_error(self, message: str) -> None:
        with self.state_lock:
            for state in self.feed_states.values():
                state["status"] = "error"
                state["error"] = message
                state["summary"] = message

    def get_frontend_context(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_stack_label,
            "monitoring_config_json": json.dumps(self.frontend_config),
        }

    def get_notifications_snapshot(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.notifications_lock:
            return [dict(item) for item in list(self.notifications)[:limit]]

    def get_websocket_snapshot(self) -> Dict[str, Any]:
        with self.state_lock:
            feeds = {}
            for feed_id, state in self.feed_states.items():
                feeds[feed_id] = {
                    "feed_id": feed_id,
                    "source_camera": state["source_camera"],
                    "display_name": state["display_name"],
                    "video_url": state["video_url"],
                    "status": state["status"],
                    "running": state["running"],
                    "action": state["action"],
                    "summary": state["summary"],
                    "confidence": state["confidence"],
                    "severity": state["severity"],
                    "severity_label": state["severity_label"],
                    "alert": state["alert"],
                    "counts": dict(state["counts"]),
                    "events": [dict(e) for e in state["events"]],
                    "boxes": [dict(b) for b in state["boxes"]],
                    "telemetry": dict(state["telemetry"]),
                    "error": state["error"],
                    "last_video_time_sec": state["last_video_time_sec"],
                }
        return {
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "running": any(feed["running"] for feed in feeds.values()),
            "model_stack": self.model_stack_label,
            "feeds": feeds,
            "notifications": self.get_notifications_snapshot(limit=100),
            "camera_groups": CAMERA_GROUPS,
            "logical_cameras": LOGICAL_CAMERAS,
        }

    def get_monitoring_stats(self) -> Dict[str, Any]:
        with self.state_lock:
            return {
                feed_id: {
                    "status": state["status"],
                    "telemetry": dict(state["telemetry"]),
                    "action": state["action"],
                    "severity_label": state["severity_label"],
                }
                for feed_id, state in self.feed_states.items()
            }

    def _push_notification(self, feed_id: str, state: Dict[str, Any], label: str, severity: str, confidence: float, frame_index: int) -> str:
        signature = as_notification_signature(feed_id, label)
        now_ts = time_module.time()
        with self.notifications_lock:
            last_at = self.last_notification_at.get(signature, 0.0)
            if now_ts - last_at < NOTIFICATION_COOLDOWN_SECONDS:
                return ""
            self.last_notification_at[signature] = now_ts
            notification = {
                "id": uuid.uuid4().hex[:12],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "camera": LOGICAL_CAMERAS[state["source_camera"]]["label"],
                "location": LOGICAL_CAMERAS[state["source_camera"]]["area"],
                "crime": label,
                "severity": severity,
                "confidence": round(float(confidence), 2),
                "summary": state["summary"],
                "frame_index": int(frame_index),
                "video_time_sec": round(float(state.get("last_video_time_sec", 0.0)), 1),
            }
            self.notifications.appendleft(notification)
            return notification["id"]

    def _run_feed_worker(self, feed_id: str) -> None:
        spec = PHYSICAL_FEEDS[feed_id]
        state = self.feed_states[feed_id]
        source_camera = spec["source_camera"]
        video_path = self._resolve_feed_path(feed_id)
        if video_path is None:
            with self.state_lock:
                state["status"] = "error"
                state["running"] = False
                state["summary"] = f"Place {source_camera}.mp4 in app/static/monitoring/."
                state["error"] = state["summary"]
            return

        with self.state_lock:
            state["video_path"] = str(video_path)
            state["video_url"] = self._to_static_url(video_path)
            state["status"] = "loading"
            state["summary"] = "Loading source video and AI models."
            state["error"] = None
            self.frontend_config["physicalFeeds"][feed_id]["video_url"] = state["video_url"]

        if self.detect_model is None or self.pose_model is None:
            with self.state_lock:
                state["status"] = "error"
                state["summary"] = "Detection models are unavailable. Install weights and ultralytics."
                state["error"] = state["summary"]
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            with self.state_lock:
                state["status"] = "error"
                state["summary"] = f"Could not open {video_path.name}."
                state["error"] = "OpenCV failed to open the video source."
            return

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wall_start = time_module.perf_counter()
        processed_frames = 0
        frame_index = 0
        loop_count = 0
        track_histories: Dict[int, Dict[str, Any]] = {}
        vehicle_histories: Dict[int, Dict[str, Any]] = {}
        detection_buffer: Dict[str, int] = {}
        miss_buffer: Dict[str, int] = {}
        clip_buffer: deque = deque(maxlen=max(CLIP_LEN, 16))
        pose_cache: List[Dict[str, Any]] = []
        pose_frame_counter = 0

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                loop_count += 1
                track_histories.clear()
                vehicle_histories.clear()
                clip_buffer.clear()
                pose_cache.clear()
                continue

            frame_index += 1
            video_time_sec = frame_index / source_fps if source_fps > 0 else 0.0
            frame_h, frame_w = frame.shape[:2]

            clip_frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            clip_buffer.append(clip_frame)

            if frame_index % FRAME_STRIDE != 0:
                desired = 1.0 / max(source_fps, 1.0)
                time_module.sleep(min(0.01, desired))
                continue

            t0 = time_module.perf_counter()
            try:
                result = self.detect_model.predict(
                    source=frame,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=640,
                    verbose=False,
                    device=DEVICE,
                )[0]
                infer_ms = (time_module.perf_counter() - t0) * 1000.0
            except Exception as e:
                print(f"Detection error: {e}")
                with self.state_lock:
                    state["status"] = "error"
                    state["error"] = f"Detection failed: {e}"
                result = None
            processed_frames += 1
            pose_frame_counter += 1

            counts: Dict[str, int] = {}
            person_boxes: Dict[int, Dict[str, Any]] = {}
            person_centers: Dict[int, Tuple[float, float]] = {}
            weapon_boxes: List[Dict[str, Any]] = []
            bag_boxes: List[Dict[str, Any]] = []
            vehicle_boxes: Dict[int, Dict[str, Any]] = {}
            active_person_ids = set()
            active_vehicle_ids = set()
            max_conf = 0.0

            if result is not None and result.boxes is not None:
                names = result.names
                for box in result.boxes:
                    cls_raw = box.cls.tolist()
                    conf_raw = box.conf.tolist()
                    xyxy = box.xyxy.tolist()[0]
                    track_id = None
                    cls_id = int(cls_raw[0] if isinstance(cls_raw, list) else cls_raw)
                    conf = float(conf_raw[0] if isinstance(conf_raw, list) else conf_raw)
                    if box.id is not None:
                        id_raw = box.id.tolist()
                        track_id = int(id_raw[0] if isinstance(id_raw, list) else id_raw)
                    class_name = (names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id])
                    if class_name not in ALLOWED_CLASSES:
                        continue
                    x1, y1, x2, y2 = xyxy
                    nx = clamp01(x1 / frame_w)
                    ny = clamp01(y1 / frame_h)
                    nw = clamp01((x2 - x1) / frame_w)
                    nh = clamp01((y2 - y1) / frame_h)
                    cx = nx + (nw / 2.0)
                    cy = ny + (nh / 2.0)
                    counts[class_name] = counts.get(class_name, 0) + 1
                    max_conf = max(max_conf, conf)
                    payload = {
                        "x": nx,
                        "y": ny,
                        "w": nw,
                        "h": nh,
                        "label": class_name.upper() if track_id is None else f"{class_name.upper()} #{track_id}",
                        "confidence": conf,
                        "track_id": track_id,
                        "class_name": class_name,
                        "center": (cx, cy),
                        "color": COLOR_MAP.get(class_name, "#3b82f6"),
                    }
                    if class_name == "person" and track_id is not None:
                        active_person_ids.add(track_id)
                        person_boxes[track_id] = payload
                        person_centers[track_id] = payload["center"]
                        hist = track_histories.setdefault(track_id, {"centers": deque(maxlen=32), "seen_count": 0, "last_seen": frame_index})
                        hist["centers"].append(payload["center"])
                        hist["seen_count"] += 1
                        hist["last_seen"] = frame_index
                    elif class_name in VEHICLE_CLASSES and track_id is not None:
                        active_vehicle_ids.add(track_id)
                        vehicle_boxes[track_id] = payload
                        hist = vehicle_histories.setdefault(track_id, {"centers": deque(maxlen=20), "areas": deque(maxlen=20), "last_seen": frame_index})
                        hist["centers"].append(payload["center"])
                        hist["areas"].append(nw * nh)
                        hist["last_seen"] = frame_index
                    if class_name in WEAPON_CLASSES:
                        weapon_boxes.append(payload)
                    if class_name in BAG_CLASSES:
                        bag_boxes.append(payload)

            stale_persons = [tid for tid, h in track_histories.items() if frame_index - h["last_seen"] > (FRAME_STRIDE * 12)]
            for tid in stale_persons:
                track_histories.pop(tid, None)
            stale_vehicles = [tid for tid, h in vehicle_histories.items() if frame_index - h["last_seen"] > (FRAME_STRIDE * 12)]
            for tid in stale_vehicles:
                vehicle_histories.pop(tid, None)

            # Skip pose for now to focus on basic detection
            # if person_boxes and (pose_frame_counter % POSE_EVERY_N == 0):
            #     pose_cache = self._run_pose(frame, person_boxes)

            event_scores = self._predict_event_scores(list(clip_buffer))
            events, incident_boxes = self._detect_events(
                feed_id=feed_id,
                frame_index=frame_index,
                event_scores=event_scores,
                person_boxes=person_boxes,
                person_centers=person_centers,
                vehicle_boxes=vehicle_boxes,
                weapon_boxes=weapon_boxes,
                bag_boxes=bag_boxes,
                track_histories=track_histories,
                vehicle_histories=vehicle_histories,
                pose_cache=[],  # Empty pose cache for now
            )

            # Add all detected objects to boxes (not just crime events)
            all_detected_boxes = []
            for tid, pb in person_boxes.items():
                box = shrink_box(pb)
                box.update({"label": f"PERSON", "color": COLOR_MAP.get("person", "#3b82f6")})
                all_detected_boxes.append(box)
            for tid, vb in vehicle_boxes.items():
                box = shrink_box(vb)
                box.update({"label": vb.get("class_name", "VEHICLE").upper(), "color": COLOR_MAP.get(vb.get("class_name", "car"), "#10b981")})
                all_detected_boxes.append(box)
            for wb in weapon_boxes:
                wb_copy = dict(wb)
                wb_copy.update({"label": wb.get("class_name", "WEAPON").upper(), "color": "#dc2626"})
                all_detected_boxes.append(wb_copy)
            for bb in bag_boxes:
                bb_copy = dict(bb)
                bb_copy.update({"label": bb.get("class_name", "BAG").upper(), "color": "#ec4899"})
                all_detected_boxes.append(bb_copy)
            
            # Add crime incident boxes on top
            for ib in incident_boxes:
                all_detected_boxes.append(ib)

            primary = self._choose_primary_event(events)
            action = primary["label"] if primary else "SCANNING"
            severity = primary["severity"] if primary else 0.0
            severity_label = primary["severity_label"] if primary else "LOW"
            alert = bool(primary)
            summary = self._build_summary(counts, events)

            with self.state_lock:
                state["status"] = "running"
                state["running"] = True
                state["action"] = action
                state["summary"] = summary
                state["confidence"] = round(primary["confidence"], 3) if primary else round(max_conf, 3)
                state["severity"] = severity
                state["severity_label"] = severity_label
                state["alert"] = alert
                state["counts"] = counts
                state["events"] = events
                state["boxes"] = all_detected_boxes
                state["error"] = None
                state["last_video_time_sec"] = video_time_sec
                state["telemetry"] = {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "source_fps": round(source_fps, 2),
                    "processing_fps": round(processed_frames / max(time_module.perf_counter() - wall_start, 1e-6), 2),
                    "latency_ms": round(infer_ms, 1),
                    "loop_count": loop_count,
                }

            seen_now = {event["label"] for event in events}
            active_incidents = state["active_incidents"]
            for label in list(miss_buffer.keys()):
                if label not in seen_now:
                    miss_buffer[label] = miss_buffer.get(label, 0) + 1
                else:
                    miss_buffer[label] = 0
                if miss_buffer.get(label, 0) >= INCIDENT_CLEAR_FRAMES:
                    active_incidents.pop(label, None)
                    miss_buffer.pop(label, None)
                    detection_buffer.pop(label, None)

            for event in events:
                label = event["label"]
                detection_buffer[label] = detection_buffer.get(label, 0) + 1
                miss_buffer[label] = 0
                needed = HEURISTIC_CONFIRM_FRAMES.get(label, 3)
                incident_state = active_incidents.get(label)
                if detection_buffer[label] >= needed and incident_state is None:
                    notif_id = self._push_notification(
                        feed_id=feed_id,
                        state=state,
                        label=label,
                        severity=event["severity_label"],
                        confidence=event["confidence"],
                        frame_index=frame_index,
                    )
                    active_incidents[label] = {
                        "notified_at": time_module.time(),
                        "notification_id": notif_id,
                        "last_seen_frame": frame_index,
                    }
                    if notif_id:
                        with self.state_lock:
                            state["latest_notification_id"] = notif_id
                elif incident_state is not None:
                    incident_state["last_seen_frame"] = frame_index

            desired_interval = FRAME_STRIDE / source_fps if source_fps > 0 else 0.14
            spent = time_module.perf_counter() - t0
            if desired_interval > spent:
                time_module.sleep(desired_interval - spent)

        cap.release()
        with self.state_lock:
            state["running"] = False
            state["status"] = "stopped"
            state["summary"] = "Monitoring stopped."

    def _run_pose(self, frame: np.ndarray, person_boxes: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not person_boxes:
            return []
        try:
            pose_result = self.pose_model.predict(source=frame, conf=0.25, imgsz=640, verbose=False, device=DEVICE)[0]
        except Exception:
            return []
        matched: List[Dict[str, Any]] = []
        if pose_result.keypoints is None or pose_result.boxes is None:
            return matched
        keypoints = pose_result.keypoints.xy.cpu().numpy() if hasattr(pose_result.keypoints.xy, "cpu") else pose_result.keypoints.xy.numpy()
        pose_boxes = pose_result.boxes.xyxy.cpu().numpy() if hasattr(pose_result.boxes.xyxy, "cpu") else pose_result.boxes.xyxy.numpy()
        frame_h, frame_w = frame.shape[:2]
        person_items = list(person_boxes.items())
        for idx, pose_box in enumerate(pose_boxes):
            x1, y1, x2, y2 = pose_box.tolist()
            px = clamp01(x1 / frame_w)
            py = clamp01(y1 / frame_h)
            pw = clamp01((x2 - x1) / frame_w)
            ph = clamp01((y2 - y1) / frame_h)
            norm_pose_box = {"x": px, "y": py, "w": pw, "h": ph}
            best_tid = None
            best_iou = 0.0
            for tid, pb in person_items:
                score = iou_boxes(norm_pose_box, pb)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is None or best_iou < 0.05:
                continue
            matched.append({
                "track_id": best_tid,
                "keypoints": (keypoints[idx] / np.array([[frame_w, frame_h]])).tolist(),
                "box": norm_pose_box,
            })
        return matched

    def _predict_event_scores(self, clip_frames: List[np.ndarray]) -> Dict[str, float]:
        if self.event_model is None or torch is None or len(clip_frames) < CLIP_LEN:
            return {}
        frames = np.stack(clip_frames[-CLIP_LEN:], axis=0).astype(np.float32) / 255.0
        frames = (frames - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
        try:
            with torch.no_grad():
                output = self.event_model(tensor)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0].tolist()
            return {label: float(prob) for label, prob in zip(self.event_labels, probs)}
        except Exception:
            return {}

    def _detect_events(
        self,
        *,
        feed_id: str,
        frame_index: int,
        event_scores: Dict[str, float],
        person_boxes: Dict[int, Dict[str, Any]],
        person_centers: Dict[int, Tuple[float, float]],
        vehicle_boxes: Dict[int, Dict[str, Any]],
        weapon_boxes: List[Dict[str, Any]],
        bag_boxes: List[Dict[str, Any]],
        track_histories: Dict[int, Dict[str, Any]],
        vehicle_histories: Dict[int, Dict[str, Any]],
        pose_cache: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        boxes: List[Dict[str, Any]] = []
        
        # Fighting detection - two or more persons close together
        if len(person_boxes) >= 2:
            person_ids = list(person_boxes.keys())
            fighting_found = False
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    tid1, tid2 = person_ids[i], person_ids[j]
                    dist = distance_points(person_centers[tid1], person_centers[tid2])
                    if dist < 0.25:  # Close proximity
                        fighting_found = True
                        break
                if fighting_found:
                    break
            if fighting_found:
                events.append({"label": "FIGHTING DETECTED", "confidence": 0.85, "severity": 0.75, "severity_label": "HIGH"})
                for tid in person_ids[:2]:
                    if tid in person_boxes:
                        box = shrink_box(person_boxes[tid])
                        box.update({"label": "FIGHT", "color": "#ef4444", "is_crime_box": True})
                        boxes.append(box)

        # Weapon detection
        if weapon_boxes:
            conf = max(max((box.get("confidence", 0.0) for box in weapon_boxes), default=0.0), 0.75)
            sev, sev_label = event_to_severity("WEAPON DETECTED")
            events.append({"label": "WEAPON DETECTED", "confidence": conf, "severity": sev, "severity_label": sev_label})
            for wbox in weapon_boxes:
                wb = dict(wbox)
                wb.update({"label": wbox.get("class_name", "WEAPON").upper(), "color": "#dc2626", "is_crime_box": True})
                boxes.append(wb)

        # Robbery detection - weapon near person
        if weapon_boxes and person_boxes:
            for wbox in weapon_boxes:
                for tid, center in person_centers.items():
                    if distance_points(center, wbox["center"]) < 0.25:
                        conf = 0.80
                        sev, sev_label = event_to_severity("ROBBERY DETECTED")
                        events.append({"label": "ROBBERY DETECTED", "confidence": conf, "severity": sev, "severity_label": sev_label})
                        # Add red boxes around both
                        if tid in person_boxes:
                            pbox = shrink_box(person_boxes[tid])
                            pbox.update({"label": "SUSPECT", "color": "#dc2626", "is_crime_box": True})
                            boxes.append(pbox)
                        wb = dict(wbox)
                        wb.update({"label": "WEAPON", "color": "#dc2626", "is_crime_box": True})
                        boxes.append(wb)
                        break

        # Accident detection - vehicles close together or stopped abruptly
        if len(vehicle_boxes) >= 2:
            vehicle_ids = list(vehicle_boxes.keys())
            accident_found = False
            for i in range(len(vehicle_ids)):
                for j in range(i + 1, len(vehicle_ids)):
                    v1, v2 = vehicle_boxes[vehicle_ids[i]], vehicle_boxes[vehicle_ids[j]]
                    dist = distance_points(v1["center"], v2["center"])
                    if dist < 0.15:  # Very close
                        accident_found = True
                        break
                if accident_found:
                    break
            if accident_found:
                events.append({"label": "ACCIDENT DETECTED", "confidence": 0.80, "severity": 0.85, "severity_label": "HIGH"})
                for vid in list(vehicle_boxes.keys())[:2]:
                    vb = dict(vehicle_boxes[vid])
                    vb.update({"label": "CRASH", "color": "#ef4444", "is_crime_box": True})
                    boxes.append(vb)

        # Chain snatching - person near bag that's moving fast or vehicle involved
        if bag_boxes and person_boxes and vehicle_boxes:
            for pb in person_boxes.values():
                for bb in bag_boxes:
                    dist = distance_points(pb["center"], bb["center"])
                    if dist < 0.20:
                        events.append({"label": "THEFT DETECTED", "confidence": 0.75, "severity": 0.70, "severity_label": "MODERATE"})
                        # Add boxes around person and bag
                        pbox = shrink_box(pb)
                        pbox.update({"label": "SUSPECT", "color": "#f59e0b", "is_crime_box": True})
                        boxes.append(pbox)
                        bb_copy = dict(bb)
                        bb_copy.update({"label": "ITEM", "color": "#f59e0b", "is_crime_box": True})
                        boxes.append(bb_copy)
                        break

        # Event model scores can add additional events
        for label_key, display in [
            ("accident", "ACCIDENT DETECTED"),
            ("fighting", "FIGHTING DETECTED"),
            ("chain_snatching", "THEFT DETECTED"),
            ("robbery", "ROBBERY DETECTED"),
            ("weapon_use", "WEAPON DETECTED"),
        ]:
            score = event_scores.get(label_key, 0.0)
            if score >= EVENT_THRESHOLDS[label_key] and display not in {e["label"] for e in events}:
                sev, sev_label = event_to_severity(display)
                events.append({"label": display, "confidence": score, "severity": sev, "severity_label": sev_label})

        return events, boxes

    def _choose_primary_event(self, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not events:
            return None
        return sorted(events, key=lambda e: (SEVERITY_ORDER.get(e["severity_label"], 0), e["confidence"]), reverse=True)[0]

    def _build_summary(self, counts: Dict[str, int], events: List[Dict[str, Any]]) -> str:
        if events:
            names = ", ".join(event["label"] for event in events[:2])
            return f"AI monitoring flagged: {names}."
        people = counts.get("person", 0)
        vehicles = sum(counts.get(key, 0) for key in ["car", "bus", "truck", "motorcycle", "bicycle"])
        bags = sum(counts.get(key, 0) for key in ["backpack", "handbag", "suitcase"])
        parts = []
        if people:
            parts.append(f"{people} person(s)")
        if vehicles:
            parts.append(f"{vehicles} vehicle(s)")
        if bags:
            parts.append(f"{bags} bag object(s)")
        return "Scanning live feed." if not parts else "Detected " + ", ".join(parts) + "."


monitoring_manager = MonitoringManager()
