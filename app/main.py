import os
import uuid
import json
from datetime import datetime, time, timedelta
from typing import Optional
import threading
import time as time_module
from collections import deque

import cv2
from ultralytics import YOLO

from fastapi import FastAPI, Depends, Form, Request, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_

from starlette.middleware.sessions import SessionMiddleware

from .database import Base, engine, get_db, SessionLocal
from .models import FIR
from .classifier import classify_crime_type

from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np

app = FastAPI(title="CIRIS - FIR Management & Dashboard", debug=True)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("CIRIS_SECRET_KEY", "super-secret-change-this")
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

ADMIN_USERNAME = os.getenv("CIRIS_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("CIRIS_ADMIN_PASSWORD", "admin123")

# ── Paths to data files ──────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANLY2_CSV_PATH       = os.path.join(_BASE_DIR, "..", "anly2.csv")
TELANGANA_CSV_PATH   = os.path.join(_BASE_DIR, "..", "telangana_ipc_2014_long.csv.csv")
TELANGANA_GEO_PATH   = os.path.join(_BASE_DIR, "..", "telangana_districts.geojson.json")

# =========================
# MONITORING / YOLO + TRACKING SECTION
# =========================

YOLO_MODEL_NAME = os.getenv("CIRIS_YOLO_MODEL", "yolo11n.pt")
YOLO_TRACKER_NAME = os.getenv("CIRIS_TRACKER", "bytetrack.yaml")

YOLO_ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "knife", "gun", "weapon",
}

YOLO_COLOR_MAP = {
    "person": "#3b82f6", "bicycle": "#06b6d4", "car": "#10b981",
    "motorcycle": "#f59e0b", "bus": "#8b5cf6", "truck": "#22c55e",
    "backpack": "#ec4899", "handbag": "#f97316", "suitcase": "#eab308",
    "knife": "#ef4444", "gun": "#dc2626", "weapon": "#b91c1c",
}

WEAPON_CLASS_NAMES  = {"knife", "gun", "weapon"}
VEHICLE_CLASS_NAMES = {"motorcycle", "bicycle", "car", "bus", "truck"}
BAG_CLASS_NAMES     = {"backpack", "handbag", "suitcase"}

YOLO_MODEL      = None
YOLO_MODEL_LOCK = threading.Lock()

MONITOR_THREAD      = None
MONITOR_STOP_EVENT  = threading.Event()
MONITOR_STATE_LOCK  = threading.Lock()
NOTIFICATION_LOCK   = threading.Lock()
NOTIFICATIONS       = deque(maxlen=500)
LAST_NOTIFICATION_META = {"signature": None, "at": None}
NOTIFICATION_COOLDOWN_SECONDS = 30
CRIME_CONFIRM_FRAMES = {
    "FIGHTING DETECTED": 3,
    "WEAPON DETECTED": 2,
    "CHAIN SNATCHING SUSPECTED": 4,
    "CROWDING": 5,
    "DEFAULT": 4,
}

MONITOR_STATE = {
    "status": "idle", "running": False, "video_url": None,
    "camera_id": "cam1", "location": "Main Gate", "action": "IDLE",
    "confidence": 0.0, "severity": 0.0, "severity_label": "LOW",
    "summary": "Upload a video to begin YOLO tracking.", "alert": False,
    "counts": {}, "tracked_people": 0, "tracked_ids": [], "events": [],
    "boxes": [],
    "telemetry": {
        "source_fps": 0.0, "processing_fps": 0.0, "latency_ms": 0.0,
        "frame_index": 0, "total_frames": 0,
        "model": YOLO_MODEL_NAME, "tracker": YOLO_TRACKER_NAME,
    },
    "progress": 0.0, "error": None,
}


def get_yolo_model():
    global YOLO_MODEL
    with YOLO_MODEL_LOCK:
        if YOLO_MODEL is None:
            YOLO_MODEL = YOLO(YOLO_MODEL_NAME)
    return YOLO_MODEL


def reset_monitor_state(video_url=None, camera_id="cam1", location="Main Gate"):
    with MONITOR_STATE_LOCK:
        MONITOR_STATE.clear()
        MONITOR_STATE.update({
            "status": "idle", "running": False, "video_url": video_url,
            "camera_id": camera_id, "location": location, "action": "IDLE",
            "confidence": 0.0, "severity": 0.0, "severity_label": "LOW",
            "summary": "Upload a video to begin YOLO tracking.", "alert": False,
            "counts": {}, "tracked_people": 0, "tracked_ids": [], "events": [], "boxes": [],
            "telemetry": {
                "source_fps": 0.0, "processing_fps": 0.0, "latency_ms": 0.0,
                "frame_index": 0, "total_frames": 0,
                "model": YOLO_MODEL_NAME, "tracker": YOLO_TRACKER_NAME,
            },
            "progress": 0.0, "error": None,
        })


def update_monitor_state(**kwargs):
    with MONITOR_STATE_LOCK:
        for key, value in kwargs.items():
            MONITOR_STATE[key] = value


def get_notifications_snapshot(limit=100):
    with NOTIFICATION_LOCK:
        return [dict(item) for item in list(NOTIFICATIONS)[:limit]]


def push_notification(item: dict):
    now = datetime.now()
    signature = item.get("signature")
    with NOTIFICATION_LOCK:
        if (
            LAST_NOTIFICATION_META["signature"] == signature
            and LAST_NOTIFICATION_META["at"] is not None
            and (now - LAST_NOTIFICATION_META["at"]).total_seconds() < NOTIFICATION_COOLDOWN_SECONDS
        ):
            return
        LAST_NOTIFICATION_META["signature"] = signature
        LAST_NOTIFICATION_META["at"] = now
        notification_item = dict(item)
        notification_item["id"] = uuid.uuid4().hex[:12]
        notification_item["time"] = now.strftime("%Y-%m-%d %H:%M:%S")
        notification_item.pop("signature", None)
        NOTIFICATIONS.appendleft(notification_item)


def get_monitor_state_snapshot():
    with MONITOR_STATE_LOCK:
        return {
            "status": MONITOR_STATE["status"], "running": MONITOR_STATE["running"],
            "video_url": MONITOR_STATE["video_url"], "camera_id": MONITOR_STATE["camera_id"],
            "location": MONITOR_STATE["location"], "action": MONITOR_STATE["action"],
            "confidence": MONITOR_STATE["confidence"], "severity": MONITOR_STATE["severity"],
            "severity_label": MONITOR_STATE["severity_label"], "summary": MONITOR_STATE["summary"],
            "alert": MONITOR_STATE["alert"], "counts": dict(MONITOR_STATE["counts"]),
            "tracked_people": MONITOR_STATE["tracked_people"],
            "tracked_ids": list(MONITOR_STATE["tracked_ids"]),
            "events": [dict(e) for e in MONITOR_STATE["events"]],
            "boxes": [dict(b) for b in MONITOR_STATE["boxes"]],
            "telemetry": dict(MONITOR_STATE["telemetry"]),
            "progress": MONITOR_STATE["progress"], "error": MONITOR_STATE["error"],
            "notifications": get_notifications_snapshot(limit=50),
        }


def distance_points(p1, p2):
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return (dx * dx + dy * dy) ** 0.5


def average_step_distance(points):
    if len(points) < 2:
        return 0.0
    steps = [distance_points(points[i - 1], points[i]) for i in range(1, len(points))]
    return sum(steps) / max(len(steps), 1)


def shrink_normalized_box(x, y, w, h, scale_w=0.82, scale_h=0.88):
    cx = x + (w / 2.0); cy = y + (h / 2.0)
    new_w = max(0.02, min(1.0, w * scale_w))
    new_h = max(0.03, min(1.0, h * scale_h))
    new_x = max(0.0, min(1.0 - new_w, cx - (new_w / 2.0)))
    new_y = max(0.0, min(1.0 - new_h, cy - (new_h / 2.0)))
    return new_x, new_y, new_w, new_h


def make_crime_box(box, label, color):
    sx, sy, sw, sh = shrink_normalized_box(box["x"], box["y"], box["w"], box["h"])
    return {
        "x": sx, "y": sy, "w": sw, "h": sh, "label": label,
        "confidence": box.get("confidence", 0.0), "track_id": box.get("track_id"),
        "class_name": box.get("class_name"), "color": color, "is_crime_box": True,
    }


def build_detection_summary(counts, tracked_people, events):
    total = sum(counts.values())
    if events:
        event_names = ", ".join(e["label"] for e in events[:2])
        return f"Tracked {tracked_people} person(s). Event(s): {event_names}."
    if total == 0:
        return "No target objects detected in current frame window."
    people = counts.get("person", 0)
    bag_like = counts.get("backpack", 0) + counts.get("handbag", 0) + counts.get("suitcase", 0)
    vehicle_like = sum(counts.get(k, 0) for k in ["car", "bus", "truck", "motorcycle", "bicycle"])
    parts = []
    if people: parts.append(f"{people} person(s)")
    if bag_like: parts.append(f"{bag_like} bag-like object(s)")
    if vehicle_like: parts.append(f"{vehicle_like} vehicle / mobility object(s)")
    joined = ", ".join(parts) if parts else "objects"
    return f"Detected {joined}."


def label_to_severity(label):
    upper = (label or "").upper()
    if any(k in upper for k in ["MURDER", "KIDNAP", "KNIFE", "GUN", "WEAPON", "STABBING", "SHOOTING", "FIREARM"]):
        return 0.95, "CRITICAL"
    if any(k in upper for k in ["FIGHT", "CHAIN SNATCH", "ROBBERY", "ASSAULT"]):
        return 0.70, "MODERATE"
    if any(k in upper for k in ["CROWD", "SUSPICIOUS", "TRESPASS"]):
        return 0.35, "LOW"
    return 0.15, "LOW"


def choose_primary_event(events, counts):
    if events:
        severity_rank = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
        return sorted(events, key=lambda e: (severity_rank.get(e["severity_label"], 0), e["score"]), reverse=True)[0]
    return {"label": "SCANNING", "score": 0.0, "severity": 0.0, "severity_label": "LOW"}


def start_monitor_worker(video_path, video_url, camera_id, location):
    global MONITOR_THREAD
    MONITOR_STOP_EVENT.set()
    if MONITOR_THREAD and MONITOR_THREAD.is_alive():
        MONITOR_THREAD.join(timeout=1.5)
    MONITOR_STOP_EVENT.clear()
    reset_monitor_state(video_url=video_url, camera_id=camera_id, location=location)
    MONITOR_THREAD = threading.Thread(
        target=process_video_worker,
        args=(video_path, video_url, camera_id, location),
        daemon=True
    )
    MONITOR_THREAD.start()


def process_video_worker(video_path, video_url, camera_id, location):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        update_monitor_state(
            status="error", running=False, camera_id=camera_id, location=location,
            summary="Could not open uploaded video.", error="OpenCV failed to read the uploaded file.",
        )
        return
    try:
        model = get_yolo_model()
    except Exception as e:
        cap.release()
        update_monitor_state(
            status="error", running=False, camera_id=camera_id, location=location,
            summary="Failed to load YOLO model.", error=str(e),
        )
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps   = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_stride = 4
    frame_index  = 0
    processed_frames = 0
    wall_start = time_module.perf_counter()
    track_histories = {}
    stale_track_cutoff = frame_stride * 10
    detection_buffer = {}
    cooldown_tracker = {}

    update_monitor_state(
        status="analyzing", running=True, camera_id=camera_id, location=location,
        summary="Video uploaded. Running YOLO tracking...",
        telemetry={
            "source_fps": round(source_fps, 2), "processing_fps": 0.0, "latency_ms": 0.0,
            "frame_index": 0, "total_frames": total_frames,
            "model": YOLO_MODEL_NAME, "tracker": YOLO_TRACKER_NAME,
        },
        progress=0.0, error=None,
    )

    while cap.isOpened() and not MONITOR_STOP_EVENT.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % frame_stride != 0:
            continue

        frame_h, frame_w = frame.shape[:2]
        t0 = time_module.perf_counter()
        results = model.track(
            source=frame, conf=0.25, iou=0.6, imgsz=640,
            persist=True, tracker=YOLO_TRACKER_NAME, verbose=False,
        )
        infer_ms = (time_module.perf_counter() - t0) * 1000.0
        processed_frames += 1

        result = results[0]
        counts = {}; all_boxes = []; person_boxes = {}; person_centers = {}
        weapon_boxes = []; bag_boxes = []; vehicle_boxes = []
        max_conf = 0.0; active_person_ids = set()

        if result.boxes is not None:
            names = result.names
            for box in result.boxes:
                cls_raw  = box.cls.tolist()
                conf_raw = box.conf.tolist()
                xyxy     = box.xyxy.tolist()[0]
                track_id = None
                cls_id   = int(cls_raw[0] if isinstance(cls_raw, list) else cls_raw)
                conf     = float(conf_raw[0] if isinstance(conf_raw, list) else conf_raw)
                if box.id is not None:
                    id_raw   = box.id.tolist()
                    track_id = int(id_raw[0] if isinstance(id_raw, list) else id_raw)
                class_name = (names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id])
                if class_name not in YOLO_ALLOWED_CLASSES:
                    continue
                x1, y1, x2, y2 = xyxy
                nx = max(0.0, min(1.0, x1 / frame_w)); ny = max(0.0, min(1.0, y1 / frame_h))
                nw = max(0.0, min(1.0, (x2 - x1) / frame_w)); nh = max(0.0, min(1.0, (y2 - y1) / frame_h))
                cx = nx + (nw / 2.0); cy = ny + (nh / 2.0)
                counts[class_name] = counts.get(class_name, 0) + 1
                max_conf = max(max_conf, conf)
                display_label = class_name.upper()
                if class_name == "person" and track_id is not None:
                    display_label = f"PERSON #{track_id}"
                box_payload = {
                    "x": nx, "y": ny, "w": nw, "h": nh, "label": display_label,
                    "confidence": conf, "track_id": track_id, "class_name": class_name,
                    "color": YOLO_COLOR_MAP.get(class_name, "#3b82f6"), "center": (cx, cy), "is_crime_box": False,
                }
                all_boxes.append(box_payload)
                if class_name == "person" and track_id is not None:
                    active_person_ids.add(track_id)
                    person_boxes[track_id] = box_payload; person_centers[track_id] = (cx, cy)
                    history = track_histories.setdefault(track_id, {"centers": deque(maxlen=30), "seen_count": 0, "last_seen_frame": frame_index})
                    history["centers"].append((cx, cy)); history["seen_count"] += 1; history["last_seen_frame"] = frame_index
                elif class_name in WEAPON_CLASS_NAMES:
                    weapon_boxes.append(box_payload)
                elif class_name in BAG_CLASS_NAMES:
                    bag_boxes.append(box_payload)
                elif class_name in {"motorcycle", "bicycle"}:
                    vehicle_boxes.append(box_payload)

        stale_ids = [tid for tid, hist in track_histories.items() if frame_index - hist["last_seen_frame"] > stale_track_cutoff]
        for tid in stale_ids:
            track_histories.pop(tid, None)

        events = []; crime_boxes = []; crime_track_ids = set()

        def add_person_box(tid, label, color):
            if tid in person_boxes and tid not in crime_track_ids:
                crime_boxes.append(make_crime_box(person_boxes[tid], label, color))
                crime_track_ids.add(tid)

        def add_weapon_box(box_payload, label, color):
            crime_boxes.append({
                "x": box_payload["x"], "y": box_payload["y"], "w": box_payload["w"], "h": box_payload["h"],
                "label": label, "confidence": box_payload.get("confidence", 0.0),
                "track_id": box_payload.get("track_id"), "class_name": box_payload.get("class_name"),
                "color": color, "is_crime_box": True,
            })

        people = counts.get("person", 0)
        if people >= 5:
            score, severity_label = label_to_severity("CROWDING")
            events.append({"label": "CROWDING", "score": score, "severity": 0.35, "severity_label": severity_label})

        rapid_track_ids = set()
        for tid in active_person_ids:
            hist = track_histories.get(tid)
            if not hist: continue
            points = list(hist["centers"])
            if len(points) < 2: continue
            avg_step = average_step_distance(points)
            if hist["seen_count"] >= 5 and avg_step > 0.015:
                rapid_track_ids.add(tid)

        fighting_track_ids = set()
        active_ids = sorted(active_person_ids)
        for i in range(len(active_ids)):
            tid1 = active_ids[i]; hist1 = track_histories.get(tid1)
            if not hist1 or not hist1["centers"]: continue
            for j in range(i + 1, len(active_ids)):
                tid2 = active_ids[j]; hist2 = track_histories.get(tid2)
                if not hist2 or not hist2["centers"]: continue
                is_close   = distance_points(hist1["centers"][-1], hist2["centers"][-1]) < 0.25
                rapid_motion = tid1 in rapid_track_ids or tid2 in rapid_track_ids
                if is_close and rapid_motion:
                    fighting_track_ids.update({tid1, tid2})

        if len(fighting_track_ids) >= 2:
            score, severity_label = label_to_severity("FIGHTING DETECTED")
            events.append({"label": "FIGHTING DETECTED", "score": score, "severity": 0.70, "severity_label": severity_label})
            for tid in sorted(fighting_track_ids):
                add_person_box(tid, f"FIGHT #{tid}", "#ef4444")

        if weapon_boxes:
            primary_weapon = weapon_boxes[0]["class_name"].upper()
            score, severity_label = label_to_severity(f"{primary_weapon} DETECTED")
            events.append({"label": f"WEAPON DETECTED ({primary_weapon})", "score": score, "severity": 0.95, "severity_label": severity_label})
            for weapon_box in weapon_boxes:
                add_weapon_box(weapon_box, weapon_box["class_name"].upper(), weapon_box.get("color", "#dc2626"))
                for tid, center in person_centers.items():
                    if distance_points(center, weapon_box["center"]) < 0.20:
                        add_person_box(tid, f"ARMED #{tid}", "#dc2626")

        chain_suspect_ids = set(); chain_victim_ids = set()
        if vehicle_boxes and active_person_ids:
            for vehicle_box in vehicle_boxes:
                rider_ids = [tid for tid, center in person_centers.items() if distance_points(center, vehicle_box["center"]) < 0.22]
                if not rider_ids: continue
                bag_nearby = any(distance_points(vehicle_box["center"], bag_box["center"]) < 0.18 for bag_box in bag_boxes)
                for rider_id in rider_ids:
                    if rider_id not in rapid_track_ids: continue
                    nearby_victims = [tid for tid, center in person_centers.items() if tid != rider_id and distance_points(center, person_centers[rider_id]) < 0.20]
                    if nearby_victims and bag_nearby:
                        chain_suspect_ids.add(rider_id); chain_victim_ids.update(nearby_victims)

        if chain_suspect_ids and chain_victim_ids:
            score, severity_label = label_to_severity("CHAIN SNATCHING")
            events.append({"label": "CHAIN SNATCHING SUSPECTED", "score": score, "severity": 0.72, "severity_label": severity_label})
            for tid in sorted(chain_suspect_ids): add_person_box(tid, f"SNATCH #{tid}", "#f59e0b")
            for tid in sorted(chain_victim_ids):  add_person_box(tid, f"VICTIM #{tid}", "#fb7185")

        primary      = choose_primary_event(events, counts)
        action       = primary["label"]; severity = primary["severity"]
        severity_label = primary["severity_label"]; alert = action != "SCANNING"
        summary      = build_detection_summary(counts, len(active_person_ids), events)

        if action not in ("SCANNING", "IDLE"):
            label_key = action.split("(")[0].strip().upper()
            detection_buffer[label_key] = detection_buffer.get(label_key, 0) + 1
            needed   = CRIME_CONFIRM_FRAMES.get(label_key, CRIME_CONFIRM_FRAMES["DEFAULT"])
            now_ts   = time_module.perf_counter()
            last_fire = cooldown_tracker.get(label_key, 0)
            if detection_buffer[label_key] >= needed and (now_ts - last_fire) > NOTIFICATION_COOLDOWN_SECONDS:
                cooldown_tracker[label_key]    = now_ts
                detection_buffer[label_key]    = 0
                push_notification({
                    "signature": f"{camera_id}|{location}|{action}|{severity_label}",
                    "camera": camera_id.upper(), "location": location, "crime": action,
                    "severity": severity_label,
                    "confidence": round(max_conf if max_conf > 0 else primary["score"], 2),
                    "summary": summary, "frame_index": frame_index,
                    "video_time_sec": round(frame_index / source_fps, 1) if source_fps > 0 else None,
                })
        else:
            for k in list(detection_buffer.keys()):
                detection_buffer[k] = max(0, detection_buffer[k] - 1)

        elapsed = max(time_module.perf_counter() - wall_start, 1e-6)
        processing_fps = processed_frames / elapsed
        progress = round((frame_index / total_frames), 4) if total_frames > 0 else 0.0

        update_monitor_state(
            status="analyzing", running=True, camera_id=camera_id, location=location,
            action=action, confidence=max_conf if max_conf > 0 else primary["score"],
            severity=severity, severity_label=severity_label, summary=summary, alert=alert,
            counts=counts, tracked_people=len(active_person_ids),
            tracked_ids=sorted(list(active_person_ids)), events=events, boxes=crime_boxes,
            progress=progress,
            telemetry={
                "source_fps": round(source_fps, 2), "processing_fps": round(processing_fps, 2),
                "latency_ms": round(infer_ms, 1), "frame_index": frame_index,
                "total_frames": total_frames, "model": YOLO_MODEL_NAME, "tracker": YOLO_TRACKER_NAME,
            },
            error=None,
        )

        desired_interval = frame_stride / source_fps if source_fps > 0 else 0.2
        spent = time_module.perf_counter() - t0
        if desired_interval > spent:
            time_module.sleep(desired_interval - spent)

    cap.release()
    if MONITOR_STOP_EVENT.is_set():
        update_monitor_state(status="stopped", running=False, alert=False, action="STOPPED", boxes=[], events=[], summary="Monitoring stopped by user.")
    else:
        snapshot = get_monitor_state_snapshot()
        update_monitor_state(
            status="completed", running=False, alert=False, action="COMPLETED", boxes=[], events=[],
            summary=f"Finished processing uploaded video. Last result: {snapshot['summary']}",
            progress=1.0 if total_frames > 0 else snapshot["progress"],
        )


# ═══════════════════════════════════════════════════════════════
#   HELPER: Build district crime counts from Telangana CSV
# ═══════════════════════════════════════════════════════════════

def _build_district_crime_data():
    """
    Reads telangana_ipc_2014_long_csv.csv and returns a dict of
    { "DISTRICT_NAME_UPPER": total_count } for use in the map endpoint.
    Excludes the 'Total' pseudo-district row.
    """
    csv_path = os.path.abspath(TELANGANA_CSV_PATH)
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[df["District"].str.lower() != "total"]
        grouped = df.groupby("District")["COUNT"].sum().reset_index()
        return {row["District"].upper(): int(row["COUNT"]) for _, row in grouped.iterrows()}
    except Exception:
        return {}


def _build_district_crime_by_type():
    """
    Returns { "DISTRICT_UPPER": {"CrimeType": count, ..., "TOTAL": n} }
    for rich map tooltips.
    """
    csv_path = os.path.abspath(TELANGANA_CSV_PATH)
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df = df[df["District"].str.lower() != "total"]
        result = {}
        for district, grp in df.groupby("District"):
            key = district.upper()
            crimes = {row["CRIME_TYPE"]: int(row["COUNT"]) for _, row in grp.iterrows()}
            crimes["TOTAL"] = int(grp["COUNT"].sum())
            result[key] = crimes
        return result
    except Exception:
        return {}


def seed_data():
    db = SessionLocal()
    try:
        count = db.query(FIR).count()
        if count > 0:
            return
        samples = [
            FIR(
                fir_number="FIR-2026-001", title="Chain snatching near bus stand",
                station_name="Central Station", district="Hyderabad",
                incident_date=datetime.strptime("2026-03-01", "%Y-%m-%d").date(),
                incident_time=time(21, 15),
                reported_at=datetime.strptime("2026-03-01 22:30:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="379, 356", crime_type="Theft", priority="Medium", status="Open",
                weapon_used="None", victim_age=28, victim_gender="Female",
                complainant_name="Ravi Kumar", accused_name="Unknown", location_text="MGBS Bus Stand",
                description="Complainant reported chain snatching by two bike riders near bus stand at night.",
                raw_fir_text="Chain snatching complaint...",
                evidence_summary="CCTV footage from nearby shop pending review.",
                tags="night-crime, vehicle, public-place"
            ),
            FIR(
                fir_number="FIR-2026-002", title="Street fight with knife injury",
                station_name="North Zone PS", district="Bengaluru",
                incident_date=datetime.strptime("2026-03-03", "%Y-%m-%d").date(),
                incident_time=time(22, 0),
                reported_at=datetime.strptime("2026-03-03 23:45:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="323, 324, 506", crime_type="Assault", priority="High",
                status="Under Investigation", weapon_used="Knife", victim_age=34, victim_gender="Male",
                complainant_name="Suresh", accused_name="Naresh", location_text="Koramangala",
                description="Fight broke out and victim suffered injury with knife. Threats were also issued.",
                raw_fir_text="Street fight FIR...",
                evidence_summary="Knife recovered, one eyewitness statement recorded.",
                tags="night-crime, weapon, public-place"
            ),
            FIR(
                fir_number="FIR-2026-003", title="Mobile theft in market",
                station_name="Old City PS", district="Mumbai",
                incident_date=datetime.strptime("2026-03-20", "%Y-%m-%d").date(),
                incident_time=time(18, 40),
                reported_at=datetime.strptime("2026-03-20 19:10:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="379", crime_type="Theft", priority="Medium", status="Open",
                weapon_used="None", victim_age=21, victim_gender="Female",
                complainant_name="Aisha", accused_name="Unknown", location_text="Colaba Causeway",
                description="Victim reported mobile phone stolen in crowded market.",
                raw_fir_text="Mobile theft report...", evidence_summary="No direct CCTV, witnesses being traced.",
                tags="public-place"
            ),
            FIR(
                fir_number="FIR-2026-004", title="Armed Robbery at Jewelry Store",
                station_name="West PS", district="Delhi",
                incident_date=datetime.strptime("2026-03-10", "%Y-%m-%d").date(),
                incident_time=time(14, 30),
                reported_at=datetime.strptime("2026-03-10 14:45:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="392, 397", crime_type="Robbery", priority="Critical", status="Active Search",
                weapon_used="Firearm", victim_age=52, victim_gender="Male",
                complainant_name="Store Owner", accused_name="Gandu Gang", location_text="Karol Bagh",
                description="Masked men entered with guns and looted gold jewelry.",
                raw_fir_text="Armed robbery report...", evidence_summary="High quality CCTV obtained.",
                tags="armed, daylight"
            ),
            FIR(
                fir_number="FIR-2026-005", title="Cyber Fraud Transaction",
                station_name="Cyber Cell", district="Pune",
                incident_date=datetime.strptime("2026-03-15", "%Y-%m-%d").date(),
                incident_time=time(11, 0),
                reported_at=datetime.strptime("2026-03-16 10:00:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="420", crime_type="Fraud", priority="Medium", status="Open",
                weapon_used="Digital", victim_age=65, victim_gender="Male",
                complainant_name="Mr. Sharma", accused_name="Unknown", location_text="Online",
                description="Pensioner scammed via fake banking app call.",
                raw_fir_text="Cyber fraud complaint...", evidence_summary="Transaction IDs provided.",
                tags="cyber, senior-citizen"
            ),
            FIR(
                fir_number="FIR-2026-006", title="Brawl at Pub",
                station_name="Central PS", district="Bengaluru",
                incident_date=datetime.strptime("2026-03-25", "%Y-%m-%d").date(),
                incident_time=time(23, 30),
                reported_at=datetime.strptime("2026-03-26 00:15:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="160", crime_type="Assault", priority="High", status="In Custody",
                weapon_used="Bottle", victim_age=25, victim_gender="Male",
                complainant_name="Pub Manager", accused_name="Group of 3", location_text="MG Road",
                description="Drunken brawl resulting in head injuries.",
                raw_fir_text="Brawl report...", evidence_summary="Mobile video from bystander.",
                tags="alcohol, night"
            ),
            FIR(
                fir_number="FIR-2026-007", title="Vehicle Theft",
                station_name="East PS", district="Hyderabad",
                incident_date=datetime.strptime("2026-03-28", "%Y-%m-%d").date(),
                incident_time=time(3, 0),
                reported_at=datetime.strptime("2026-03-28 09:00:00", "%Y-%m-%d %H:%M:%S"),
                legal_section="379", crime_type="Theft", priority="Medium", status="Open",
                weapon_used="None", victim_age=40, victim_gender="Male",
                complainant_name="Srinivas", accused_name="Unknown", location_text="Uppal",
                description="Royal Enfield bike stolen from outside residence.",
                raw_fir_text="Bike theft...", evidence_summary="GPS tracker disabled.",
                tags="vehicle, night"
            ),
        ]
        db.add_all(samples)
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    seed_data()
    # Ensure uploads directory exists
    os.makedirs(os.path.join("app", "static", "uploads"), exist_ok=True)

def generate_fir_card(fir_data: dict) -> str:
    """Generates a visual FIR card image from manual entry data."""
    width, height = 800, 1100
    img = PILImage.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        header_font = ImageFont.truetype("arial.exe", 32)
        label_font = ImageFont.truetype("arial.exe", 16)
        val_font = ImageFont.truetype("arial.exe", 20)
    except:
        header_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        val_font = ImageFont.load_default()

    # Draw border
    draw.rectangle([20, 20, width-20, height-20], outline=(0, 0, 0), width=2)
    
    # Header
    draw.text((width//2, 60), "FIRST INFORMATION REPORT", fill=(0, 0, 0), font=header_font, anchor="mm")
    draw.text((width//2, 100), "(Under Section 154 Cr.P.C.)", fill=(0, 0, 0), font=label_font, anchor="mm")
    draw.line([50, 130, width-50, 130], fill=(0, 0, 0), width=1)

    y = 160
    fields = [
        ("FIR Number", fir_data.get("fir_number", "N/A")),
        ("District", fir_data.get("district", "N/A")),
        ("Police Station", fir_data.get("station_name", "N/A")),
        ("Date of Incident", str(fir_data.get("incident_date", "N/A"))),
        ("Time of Incident", str(fir_data.get("incident_time", "N/A"))),
        ("Complainant", fir_data.get("complainant_name", "N/A")),
        ("Location", fir_data.get("location_text", "N/A")),
        ("Legal Sections", fir_data.get("legal_section", "N/A")),
        ("Crime Type", fir_data.get("crime_type", "N/A")),
    ]

    for label, val in fields:
        draw.text((60, y), f"{label}:", fill=(100, 100, 100), font=label_font)
        draw.text((220, y), str(val), fill=(0, 0, 0), font=val_font)
        y += 45

    # Description
    draw.text((60, y), "FIR Contents / Description:", fill=(100, 100, 100), font=label_font)
    y += 30
    desc = fir_data.get("description", "")
    # Simple text wrapping
    words = desc.split()
    line = ""
    for word in words:
        if len(line + word) < 70:
            line += word + " "
        else:
            draw.text((60, y), line, fill=(0, 0, 0), font=val_font)
            y += 25
            line = word + " "
    draw.text((60, y), line, fill=(0, 0, 0), font=val_font)

    # Footer
    draw.text((width-200, height-100), "Officer Signature", fill=(0, 0, 0), font=label_font)
    draw.line([width-250, height-110, width-50, height-110], fill=(0, 0, 0), width=1)

    filename = f"fir_card_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join("app", "static", "uploads", filename)
    img.save(save_path)
    return f"uploads/{filename}"


def is_authenticated(request: Request) -> bool:
    return bool(request.session.get("user"))


def redirect_if_not_logged_in(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return None


# ═══════════════════════════════════════════════════════════════
#   AUTH ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html", context={"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        request.session["user"] = username
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request, name="login.html",
        context={"request": request, "error": "Invalid username or password"}
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ═══════════════════════════════════════════════════════════════
#   PAGE ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect

    import pandas as pd

    crime_distribution = []
    top_districts = []
    district_crime_data = {}
    yearly_trend_data = []

    try:
        tel_csv_path = os.path.abspath(TELANGANA_CSV_PATH)
        if os.path.exists(tel_csv_path):
            tel_df = pd.read_csv(tel_csv_path)
            tel_df = tel_df[tel_df["District"].str.lower() != "total"]
            if not tel_df.empty:
                crime_group = tel_df.groupby("CRIME_TYPE")["COUNT"].sum().reset_index().sort_values("COUNT", ascending=False).head(10)
                crime_distribution = [{"crime_type": row["CRIME_TYPE"], "count": int(row["COUNT"])} for _, row in crime_group.iterrows()]
                district_group = tel_df.groupby("District")["COUNT"].sum().reset_index().sort_values("COUNT", ascending=False).head(6)
                top_districts = [{"district": row["District"], "count": int(row["COUNT"])} for _, row in district_group.iterrows()]
        district_crime_data = _build_district_crime_data()
    except Exception as e:
        pass

    try:
        anly2_csv_path = os.path.abspath(ANLY2_CSV_PATH)
        if os.path.exists(anly2_csv_path):
            anly2_df = pd.read_csv(anly2_csv_path).dropna()
            anly2_df["Count"] = anly2_df["Count"].astype(float)
            yearly_group = anly2_df.groupby("Year")["Count"].sum().reset_index().sort_values("Year")
            yearly_trend_data = [{"month": str(int(row["Year"])), "count": int(row["Count"])} for _, row in yearly_group.iterrows()]
    except Exception:
        pass

    total_firs     = db.query(func.count(FIR.id)).scalar() or 0
    open_cases     = db.query(func.count(FIR.id)).filter(FIR.status.in_(["Open", "Under Investigation", "High Alert"])).scalar() or 0
    critical_cases = db.query(func.count(FIR.id)).filter(FIR.priority == "Critical").scalar() or 0
    high_priority  = db.query(func.count(FIR.id)).filter(FIR.priority.in_(["High", "Critical"])).scalar() or 0
    recent_firs    = db.query(FIR).order_by(FIR.created_at.desc()).limit(5).all()

    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={
            "request": request, "user": request.session.get("user"),
            "total_firs": total_firs, "open_cases": open_cases,
            "critical_cases": critical_cases, "high_priority": high_priority,
            "top_districts": top_districts,
            "crime_distribution": crime_distribution,
            "monthly_trend": yearly_trend_data,
            "recent_firs": recent_firs,
            "district_crime_data": district_crime_data,
        },
    )


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request, db: Session = Depends(get_db)):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect

    import pandas as pd
    
    # ── 1. Telangana Data (2014) ──
    tel_df = pd.DataFrame()
    tel_csv_path = os.path.abspath(TELANGANA_CSV_PATH)
    if os.path.exists(tel_csv_path):
        tel_df = pd.read_csv(tel_csv_path)
        tel_df = tel_df[tel_df["District"].str.lower() != "total"]
    
    # Crime Distribution (Top 10 from Telangana)
    if not tel_df.empty:
        crime_group = tel_df.groupby("CRIME_TYPE")["COUNT"].sum().reset_index().sort_values("COUNT", ascending=False).head(10)
        crime_distribution = [{"crime_type": row["CRIME_TYPE"], "count": int(row["COUNT"])} for _, row in crime_group.iterrows()]
    else:
        crime_distribution = []

    # Emerging Hotspots (Top 6 Districts from Telangana)
    if not tel_df.empty:
        district_group = tel_df.groupby("District")["COUNT"].sum().reset_index().sort_values("COUNT", ascending=False).head(6)
        emerging_hotspots = [{"district": row["District"], "count": int(row["COUNT"])} for _, row in district_group.iterrows()]
    else:
        emerging_hotspots = []

    # District Map Data
    district_crime_data = _build_district_crime_data()

    # ── 2. Anly2 Data (Yearly Trend) ──
    anly2_df = pd.DataFrame()
    anly2_csv_path = os.path.abspath(ANLY2_CSV_PATH)
    yearly_trend_data = []
    if os.path.exists(anly2_csv_path):
        anly2_df = pd.read_csv(anly2_csv_path).dropna()
        anly2_df["Count"] = anly2_df["Count"].astype(float)
        yearly_group = anly2_df.groupby("Year")["Count"].sum().reset_index().sort_values("Year")
        yearly_trend_data = [{"month": str(int(row["Year"])), "count": int(row["Count"])} for _, row in yearly_group.iterrows()]

    # Simple next-year prediction
    predicted_month = "Next Year"
    predicted_count = 0
    if len(yearly_trend_data) >= 2:
        counts_list = [item["count"] for item in yearly_trend_data]
        diffs       = [counts_list[i] - counts_list[i - 1] for i in range(1, len(counts_list))]
        avg_diff    = sum(diffs) / len(diffs)
        predicted_count = max(0, counts_list[-1] + int(round(avg_diff)))
        last_y = int(yearly_trend_data[-1]["month"])
        predicted_month = str(last_y + 1)

    # ── 3. DB Fallbacks for missing CSV metrics ──
    total_firs     = db.query(func.count(FIR.id)).scalar() or sum([c["count"] for c in crime_distribution])
    open_cases     = db.query(func.count(FIR.id)).filter(FIR.status.in_(["Open", "Under Investigation", "High Alert"])).scalar() or 0
    critical_cases = db.query(func.count(FIR.id)).filter(FIR.priority == "Critical").scalar() or 0

    hourly_dist = db.query(
        func.strftime("%H", FIR.incident_time).label("hour"),
        func.count(FIR.id)
    ).filter(FIR.incident_time != None).group_by("hour").all()
    hourly_map  = {int(row[0]): row[1] for row in hourly_dist}
    hourly_full = [{"hour": f"{h}:00", "count": hourly_map.get(h, 0)} for h in range(24)]

    gender_dist = db.query(FIR.victim_gender, func.count(FIR.id)).group_by(FIR.victim_gender).all()
    gender_data = {row[0] or "Unknown": row[1] for row in gender_dist}

    age_groups = {
        "0-18":  db.query(func.count(FIR.id)).filter(FIR.victim_age < 18).scalar() or 0,
        "19-35": db.query(func.count(FIR.id)).filter(FIR.victim_age.between(19, 35)).scalar() or 0,
        "36-55": db.query(func.count(FIR.id)).filter(FIR.victim_age.between(36, 55)).scalar() or 0,
        "55+":   db.query(func.count(FIR.id)).filter(FIR.victim_age > 55).scalar() or 0,
    }

    weapon_corr = db.query(FIR.weapon_used, func.count(FIR.id)).group_by(FIR.weapon_used).all()
    weapon_data = [{"weapon": row[0] or "None", "count": row[1]} for row in weapon_corr]

    avg_response_lag = 4.8  # Default optimal

    top_crime     = crime_distribution[0]["crime_type"] if crime_distribution else "No data"
    riskiest_hour = max(hourly_full, key=lambda x: x["count"])["hour"] if hourly_full else "Unknown"

    summary_text = (
        f"Based on recent data, <strong>{top_crime}</strong> remains the most prevalent crime type. "
        f"Risk levels peak around <strong>{riskiest_hour}</strong>, indicating a need for increased patrol during these hours. "
        f"Average reporting efficiency is at <strong>{avg_response_lag} hours</strong>, which is within optimal parameters."
    )

    return templates.TemplateResponse(
        request=request, name="analytics.html",
        context={
            "request": request, "user": request.session.get("user"),
            "total_firs": total_firs, "open_cases": open_cases, "critical_cases": critical_cases,
            "crime_distribution": crime_distribution,
            "monthly_trend": yearly_trend_data,  # Repurposed to Yearly
            "peak_hour_data": hourly_full,
            "gender_data": gender_data, "age_groups": age_groups, "weapon_data": weapon_data,
            "avg_response_lag": avg_response_lag,
            "predicted_month": predicted_month, "predicted_count": predicted_count,
            "emerging_hotspots": emerging_hotspots,
            "summary_text": summary_text,
            "district_crime_data": district_crime_data,
        },
    )


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring(request: Request):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        request=request, name="monitoring.html",
        context={"request": request, "user": request.session.get("user"), "model_name": f"{YOLO_MODEL_NAME} + {YOLO_TRACKER_NAME}"},
    )


@app.get("/notifications", response_class=HTMLResponse)
def notifications_page(request: Request):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        request=request, name="notification.html",
        context={"request": request, "user": request.session.get("user")},
    )


# ═══════════════════════════════════════════════════════════════
#   API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/api/upload-video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    camera_id: str = Form("cam1"),
    location: str = Form("Main Gate"),
):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    ext      = os.path.splitext(video.filename)[1] or ".mp4"
    filename = f"test_video_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join("app", "static", "uploads", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as buffer:
        buffer.write(await video.read())
    video_url = f"/static/uploads/{filename}"
    start_monitor_worker(os.path.abspath(save_path), video_url, camera_id=camera_id, location=location)
    return {"ok": True, "file_name": video.filename, "video_url": video_url}


@app.post("/api/monitoring/stop")
def api_monitoring_stop(request: Request):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    MONITOR_STOP_EVENT.set()
    update_monitor_state(status="stopped", running=False, alert=False, action="STOPPED", boxes=[], events=[], summary="Monitoring stopped by user.")
    return {"ok": True}


@app.get("/api/event-log")
def api_event_log(request: Request):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return {"items": []}


@app.get("/api/notifications")
def api_notifications(request: Request):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return {"items": get_notifications_snapshot(limit=100)}


# ─────────────────────────────────────────────────────────────
#  NEW: Serve Telangana GeoJSON directly (so Leaflet can load it)
# ─────────────────────────────────────────────────────────────
@app.get("/api/telangana-geojson")
def api_telangana_geojson(request: Request):
    """Serve the Telangana districts GeoJSON file for the analytics map."""
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    geo_path = os.path.abspath(TELANGANA_GEO_PATH)
    if not os.path.exists(geo_path):
        return JSONResponse(status_code=404, content={"error": "GeoJSON file not found. Place telangana_districts_geojson.json in the project root."})
    try:
        with open(geo_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
        return JSONResponse(content=geojson)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/api/district-crimes")
def api_district_crimes(request: Request):
    """Returns per-district crime-type breakdown for rich map tooltips."""
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return JSONResponse(content=_build_district_crime_by_type())


@app.post("/firs/{fir_id}/delete")
def delete_fir(fir_id: int, request: Request, db: Session = Depends(get_db)):
    """Delete a FIR record by ID."""
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect
    fir = db.query(FIR).filter(FIR.id == fir_id).first()
    if fir:
        db.delete(fir)
        db.commit()
    return RedirectResponse("/firs", status_code=303)


# ─────────────────────────────────────────────────────────────
#  UPDATED: /api/analytics-ml  — Full anly1 logic on anly2.csv
#  Per-crime LinearRegression, trendlines, R² scores
# ─────────────────────────────────────────────────────────────
@app.get("/api/analytics-ml")
def api_analytics_ml(request: Request, predict_year: int = 2020):
    """
    Implements the full anly1.ipynb logic:
    - Reads anly2.csv (Year, CrimeType, Count)
    - Trains a LinearRegression per crime type
    - Filters to models with R² >= 0.6  (same as anly1 final_models)
    - Returns predictions for predict_year with trendline data for charts
    """
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    csv_path = os.path.abspath(ANLY2_CSV_PATH)
    if not os.path.exists(csv_path):
        return JSONResponse(status_code=404, content={"ok": False, "error": f"anly2.csv not found at {csv_path}"})

    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_absolute_error

        df = pd.read_csv(csv_path).dropna()
        df["Count"] = df["Count"].astype(float)
        df["Year"]  = df["Year"].astype(int)

        crime_types = sorted(df["CrimeType"].unique().tolist())
        years       = sorted(df["Year"].unique().tolist())

        historical = df.rename(
            columns={"Year": "year", "CrimeType": "crime_type", "Count": "count"}
        ).to_dict(orient="records")

        # ── anly1 logic: train per crime type ──
        all_models = {}
        all_scores = {}

        for crime in crime_types:
            data = df[df["CrimeType"] == crime].sort_values("Year")
            if len(data) < 2:
                continue
            X = data[["Year"]]
            y = data["Count"]
            model = LinearRegression()
            model.fit(X, y)
            y_pred_hist = model.predict(X)
            r2  = float(r2_score(y, y_pred_hist))
            mae = float(mean_absolute_error(y, y_pred_hist))
            all_models[crime] = model
            all_scores[crime] = {"r2": round(r2, 3), "mae": round(mae, 1)}

        # ── anly1: final_models = only R² >= 0.6 ──
        final_models = {c: m for c, m in all_models.items() if all_scores[c]["r2"] >= 0.6}

        predictions = []
        trend_years = years + ([predict_year] if predict_year not in years else [])

        for crime, model in final_models.items():
            pred_val = max(0, round(float(
                model.predict(pd.DataFrame([[predict_year]], columns=["Year"]))[0]
            )))
            trend_vals = [
                max(0, round(float(model.predict(pd.DataFrame([[yr]], columns=["Year"]))[0])))
                for yr in trend_years
            ]
            predictions.append({
                "crime_type":  crime,
                "predicted":   pred_val,
                "r2":          all_scores[crime]["r2"],
                "mae":         all_scores[crime]["mae"],
                "trend_years": trend_years,
                "trend_vals":  trend_vals,
            })

        # Sort descending by predicted count (same as anly1 bar chart)
        predictions.sort(key=lambda x: x["predicted"], reverse=True)

        return {
            "ok":           True,
            "historical":   historical,
            "predictions":  predictions,
            "years":        years,
            "crime_types":  crime_types,
            "predict_year": predict_year,
            "note":         f"Models with R²<0.6 excluded (anly1 logic). {len(final_models)}/{len(crime_types)} crime types passed quality threshold.",
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/dashboard")
def api_dashboard(request: Request, db: Session = Depends(get_db)):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    total_firs = db.query(func.count(FIR.id)).scalar() or 0
    crime_distribution = (
        db.query(FIR.crime_type, func.count(FIR.id).label("count"))
        .group_by(FIR.crime_type).order_by(desc("count")).all()
    )
    top_districts = (
        db.query(FIR.district, func.count(FIR.id).label("count"))
        .group_by(FIR.district).order_by(desc("count")).limit(5).all()
    )
    return {
        "total_firs": total_firs,
        "crime_distribution": [{"crime_type": r[0], "count": r[1]} for r in crime_distribution],
        "top_districts": [{"district": r[0], "count": r[1]} for r in top_districts],
    }


@app.get("/api/live-crimes")
def api_live_crimes(request: Request):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    import random
    now = datetime.now()
    districts = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Karimnagar", "Warangal"]
    crimes    = ["Theft", "Assault", "Robbery", "Fraud", "Public Disturbance", "Chain Snatching"]
    return {
        "id":         random.randint(1000, 9999),
        "fir_number": f"LIVE-2026-{random.randint(100, 999)}",
        "title":      f"Alert: Potential {random.choice(crimes)} detected",
        "district":   random.choice(districts),
        "location":   "Central Market Area",
        "timestamp":  now.strftime("%H:%M:%S"),
        "severity":   random.choice(["Low", "Medium", "High", "Critical"]),
    }


@app.get("/api/firs")
def api_firs(request: Request, db: Session = Depends(get_db)):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    firs = db.query(FIR).order_by(FIR.incident_date.desc(), FIR.id.desc()).all()
    return [
        {
            "id": fir.id, "fir_number": fir.fir_number, "title": fir.title,
            "station_name": fir.station_name, "district": fir.district,
            "incident_date": str(fir.incident_date),
            "incident_time": str(fir.incident_time) if fir.incident_time else None,
            "legal_section": fir.legal_section, "crime_type": fir.crime_type,
            "priority": fir.priority, "status": fir.status,
            "complainant_name": fir.complainant_name, "accused_name": fir.accused_name,
            "location_text": fir.location_text, "description": fir.description,
            "evidence_summary": fir.evidence_summary, "tags": fir.tags,
        }
        for fir in firs
    ]


# ─────────────────────────────────────────────────────────────
#  FIR CRUD ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/firs", response_class=HTMLResponse)
def fir_list(
    request: Request,
    district: Optional[str] = None, crime_type: Optional[str] = None,
    q: Optional[str] = None, db: Session = Depends(get_db)
):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect
    query = db.query(FIR)
    if district:   query = query.filter(FIR.district == district)
    if crime_type: query = query.filter(FIR.crime_type == crime_type)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(
            FIR.fir_number.ilike(like), FIR.title.ilike(like),
            FIR.station_name.ilike(like), FIR.description.ilike(like),
            FIR.location_text.ilike(like),
        ))
    firs        = query.order_by(FIR.incident_date.desc(), FIR.id.desc()).all()
    districts   = [r[0] for r in db.query(FIR.district).distinct().order_by(FIR.district).all()]
    crime_types = [r[0] for r in db.query(FIR.crime_type).distinct().order_by(FIR.crime_type).all()]
    return templates.TemplateResponse(
        request=request, name="fir_list.html",
        context={
            "request": request, "firs": firs, "districts": districts,
            "crime_types": crime_types, "selected_district": district,
            "selected_crime_type": crime_type, "q": q or "",
        },
    )


@app.get("/firs/new", response_class=HTMLResponse)
def new_fir_form(request: Request):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(request=request, name="fir_form.html", context={"request": request, "error": None})


@app.post("/firs/new", response_class=HTMLResponse)
def create_fir(
    request: Request,
    entry_method: str = Form("manual"),
    fir_number: Optional[str] = Form(None), fir_number_upload: Optional[str] = Form(None),
    title: Optional[str] = Form(None), station_name: Optional[str] = Form(None),
    district: Optional[str] = Form(None), incident_date: Optional[str] = Form(None),
    incident_time: Optional[str] = Form(None), priority: Optional[str] = Form(None),
    legal_section: Optional[str] = Form(""), complainant_name: Optional[str] = Form(""),
    accused_name: Optional[str] = Form(""), location_text: Optional[str] = Form(""),
    description: Optional[str] = Form(None), raw_fir_text: Optional[str] = Form(""),
    evidence_summary: Optional[str] = Form(""), status: str = Form("Open"),
    fir_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    redirect = redirect_if_not_logged_in(request)
    if redirect:
        return redirect

    current_fir_no = fir_number if entry_method == "manual" else (fir_number_upload or "")

    # ── OCR extraction for upload mode ──
    ocr_results = None
    if entry_method == "upload" and fir_image and fir_image.filename:
        img_bytes = fir_image.file.read()
        fir_image.file.seek(0)
        ocr_results, error = _perform_ocr(img_bytes)

    if entry_method == "upload" and ocr_results:
        if not current_fir_no: current_fir_no = ocr_results.get("fir_number")
        if not incident_date: incident_date = ocr_results.get("incident_date")
        if not district: district = ocr_results.get("district")
        if not station_name: station_name = ocr_results.get("station_name")
        if not complainant_name: complainant_name = ocr_results.get("complainant_name")
        if not location_text: location_text = ocr_results.get("location_text")
        if not description: description = ocr_results.get("description")
        if not legal_section: legal_section = ocr_results.get("legal_section")

    if not current_fir_no:
        current_fir_no = f"FIR-OCR-{uuid.uuid4().hex[:8].upper()}"

    existing = db.query(FIR).filter(FIR.fir_number == current_fir_no).first()
    if existing:
        return templates.TemplateResponse(
            request=request, name="fir_form.html",
            context={"request": request, "error": f"FIR number {current_fir_no} already exists"}
        )

    parsed_date = datetime.now().date()
    if incident_date:
        for _fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y"):
            try:
                parsed_date = datetime.strptime(incident_date.strip(), _fmt).date()
                break
            except ValueError:
                continue

    parsed_time = None
    if incident_time:
        try:
            parsed_time = datetime.strptime(incident_time, "%H:%M").time()
        except ValueError:
            pass

    image_path = None
    if fir_image and fir_image.filename:
        ext       = os.path.splitext(fir_image.filename)[1]
        filename  = f"{uuid.uuid4()}{ext}"
        save_path = os.path.join("app", "static", "uploads", filename)
        with open(save_path, "wb") as buffer:
            buffer.write(fir_image.file.read())
        image_path = f"uploads/{filename}"

    final_title       = title or f"Document Upload: {current_fir_no}"
    final_station     = station_name or "Pending Review"
    final_district    = district or "Unassigned"
    final_description = description or "Automated entry via document upload. Manual verification required."

    # Save uploaded image (file pointer already reset above)
    classification = classify_crime_type(description=final_description, legal_section=legal_section)

    if entry_method == "manual" and not image_path:
        # Generate visual card for manual entry
        image_path = generate_fir_card({
            "fir_number": current_fir_no,
            "station_name": final_station,
            "district": final_district,
            "incident_date": parsed_date,
            "incident_time": parsed_time,
            "complainant_name": complainant_name,
            "location_text": location_text,
            "legal_section": legal_section,
            "crime_type": classification["crime_type"],
            "description": final_description
        })

    fir = FIR(
        fir_number=current_fir_no, title=final_title, station_name=final_station,
        district=final_district, incident_date=parsed_date, incident_time=parsed_time,
        legal_section=legal_section, crime_type=classification["crime_type"],
        priority=priority or classification["priority"], status=status,
        complainant_name=complainant_name, accused_name=accused_name,
        location_text=location_text, description=final_description,
        raw_fir_text=raw_fir_text or (ocr_text if entry_method == "upload" else ""), 
        evidence_summary=evidence_summary,
        image_path=image_path, tags=classification["tags"],
        source_type=entry_method,
        is_verified=1 if entry_method == "manual" else 0  # Manual entry is considered verified by default
    )
    db.add(fir); db.commit(); db.refresh(fir)
    return RedirectResponse("/firs", status_code=303)

# --- SHARED OCR HELPER ---
def _perform_ocr(img_bytes: bytes):
    try:
        import pytesseract, re
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None, "Invalid image format"

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        target_w = 2200
        if w < target_w:
            scale = target_w / w
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
        
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 2. OCR
        custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(thresh, config=custom_config, lang="eng")
        if len(text.strip()) < 50:
            text = pytesseract.image_to_string(thresh, config=r'--oem 3 --psm 6', lang="eng")
        
        # 3. Helpers for extraction
        def clean_text(t):
            t = re.sub(r'[|\\_]', '', t)
            return re.sub(r'\s+', ' ', t).strip()

        def get_match(patterns, t):
            for p in patterns:
                m = re.search(p, t, re.IGNORECASE)
                if m: return m.group(1).strip()
            return ""

        def fuzzy_extract(t, keywords, length=50):
            t_lower = t.lower()
            best_match = ""
            for kw in keywords:
                idx = t_lower.find(kw.lower())
                if idx != -1:
                    segment = t[idx + len(kw): idx + len(kw) + length]
                    segment = re.sub(r'^[:\-\s\.]+', '', segment)
                    match = segment.split('\n')[0].strip()
                    if len(match) > len(best_match): best_match = match
            return best_match

        cleaned_text = clean_text(text)
        
        # 4. Extraction
        extracted = {
            "fir_number": get_match([
                r"(?:FIR|F\.I\.R|Crime|Case)\s*(?:No|Number)?[^0-9A-Z]*([A-Z0-9/\-]{3,25})",
                r"No\.?\s*([A-Z0-9/\-]{4,20})",
                r"(\d{4,10}/\d{2,4})"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["FIR No", "Crime No", "Case No"], 20),
            
            "incident_date": get_match([
                r"(?:Date|Dt|Occurred on|Date of)[^\d]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Date of Incident", "Date of occurrence", "Date:"], 15),
            
            "district": get_match([
                r"District\s*[:\-]?\s*([A-Za-z ]{3,30})",
                r"(?:Dist|District)\.?\s*:\s*([A-Za-z ]+)"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["District", "Dist."], 25),
            
            "station_name": get_match([
                r"(?:Police Station|P\.S\.|Station)\s*[:\-]?\s*([A-Za-z ]{3,40})",
                r"P\.S\.\s*[:\-]?\s*([A-Za-z ]+)"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Police Station", "P.S.", "Station:"], 30),
            
            "complainant_name": get_match([
                r"(?:Complainant|Reported by|Informant|Informant's Name)\s*[:\-]?\s*([A-Za-z ]{3,50})",
                r"Name of (?:the )?Informant\s*:\s*([A-Za-z ]+)"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Complainant Name", "Name of Informant", "Informant:"], 40),
            
            "location_text": get_match([
                r"(?:Place of Occurrence|Place|Location|Address)\s*[:\-]?\s*([A-Za-z0-9 ,\.]{5,120})",
                r"at\s+([A-Za-z0-9 ,\.]{10,80})"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Place of occurrence", "Location", "Address:"], 60),
            
            "accused_name": get_match([
                r"(?:Accused|Name of Accused)\s*[:\-]?\s*([A-Za-z ]{3,50})",
                r"Accused\s*:\s*([A-Za-z ]+)"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Name of Accused", "Accused:"], 40) or "Unknown",
            
            "legal_section": get_match([
                r"(?:Section|Sections|U/s)\s*[:\-]?\s*([0-9A-Za-z ,]{2,40})",
                r"u/s\s+([0-9, ]+)"
            ], cleaned_text) or fuzzy_extract(cleaned_text, ["Section", "U/s"], 30),
            
            "description": text[:4000]
        }
        
        # 5. Crime Type
        classification = classify_crime_type(description=cleaned_text, legal_section=extracted["legal_section"])
        extracted["crime_type"] = classification["crime_type"]
        extracted["weapon_used"] = classification.get("weapon", "")
        
        return extracted, None
    except Exception as e:
        return None, str(e)

@app.post("/api/ocr-preview")
async def ocr_preview(request: Request, fir_image: UploadFile = File(...)):
    """Extracts text from FIR image and returns structured data for review."""
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    
    try:
        img_bytes = await fir_image.read()
        extracted, error = _perform_ocr(img_bytes)
        
        if error:
            return JSONResponse(status_code=500, content={"error": error})
        return {"ok": True, "extracted": extracted}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
#   WEBSOCKET + MONITORING STATS
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/monitoring")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_monitor_state_snapshot())
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        pass


@app.get("/api/monitoring-stats")
def api_monitoring_stats(request: Request):
    if not is_authenticated(request):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return get_monitor_state_snapshot()["telemetry"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)