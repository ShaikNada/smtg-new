# Monitoring AI setup

## What changes are required

You need three code files:
- `app/main.py` → replace with `main_updated.py`
- `app/templates/monitoring.html` → replace with `monitoring.html`
- `app/monitoring_ai.py` → add this new file

## Where to place videos

Put the 4 source videos here:
- `app/static/monitoring/cam1.mp4`
- `app/static/monitoring/cam2.mp4`
- `app/static/monitoring/cam3.mp4`
- `app/static/monitoring/cam4.mp4`

The helper also accepts the original names with spaces if they are copied into the same folder.

## Models actually used by the code

### Runs immediately with:
- `yolo11s.pt` for detection/tracking
- `yolo11n-pose.pt` for pose estimation
- temporal heuristics for accident / fight / snatching / robbery confirmation

### Better option for proper event recognition
Add a custom TorchScript video classifier here:
- `app/models/crime_event_classifier.ts`
- `app/models/crime_event_labels.json`

Expected labels order in `crime_event_labels.json`:
```json
["normal", "accident", "fighting", "chain_snatching", "robbery", "weapon_use"]
```

Expected model input for `crime_event_classifier.ts`:
- shape: `[1, 3, T, 224, 224]`
- normalized RGB clip
- `T = 16` by default

## Install requirements

```bash
pip install -r monitoring_requirements.txt
```

## Important honesty note

Without a custom-trained event classifier and a custom weapon detector, this system is improved but still not guaranteed to classify every robbery / accident / snatching clip correctly. The code is ready for a proper AI event model, but the weights themselves are separate from the code.
