# SalatVision

**Real-time Islamic prayer posture recognition and rakaat counting using MediaPipe BlazePose.**

SalatVision detects four prayer postures — Qiyam (standing), Ruku (bowing), Sujud (prostration), and Tashahhud (sitting) — and automatically counts completed prayer cycles (rakaat) from a standard camera feed. The system is designed to work with prayer garments worn by Muslim women, including full telekung, which causes facial landmark collapse during Sujud — a condition unaddressed by all prior works.

---

## Demo

```
python main.py --prayer Fajr --debug
```

The dashboard shows the current posture, next expected posture, hold progress, rakaat counter, and live joint angle values.

---

## Features

- **Real-time posture detection** at 20–25 FPS on a standard laptop
- **Automatic rakaat counting** via sequence-enforced Finite State Machine
- **Telekung-robust Sujud detection** — shoulder-based Path B activates when facial landmarks are occluded by prayer garments
- **Front-view Ruku detection** without spine angle — uses depth-independent sh_hip_y_diff and ear_sh_y_diff measurements
- **Landmark smoothing** — median filter per joint x/y coordinate across 5-frame window
- **TransitionSmoother** — prevents brief intermediate postures from corrupting the sequence counter
- **Sujud trajectory prediction** — predicts Sujud early during landmark collapse window
- **Audio feedback** — instant playback via pre-generated WAV files (pygame)
- **IP Webcam support** — use an Android phone as a camera over WiFi
- **Session logging** — saves posture events and per-frame predictions to JSON

---

## Supported Prayer Types

| Prayer | Rakaat |
|--------|--------|
| Fajr | 2 |
| Maghrib | 3 |
| Zuhr | 4 |
| Asr | 4 |
| Isha | 4 |
| Custom | Unlimited |

---

## Requirements

- Python 3.9 or higher
- Webcam or Android phone running [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)

Install dependencies:

```bash
pip install mediapipe opencv-python numpy pyttsx3 pygame
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/SalatVision.git
cd SalatVision
python -m venv venv
venv\Scripts\activate        # Windows
pip install mediapipe opencv-python numpy pyttsx3 pygame
```

---

## Usage

**Basic — choose prayer type from menu:**
```bash
python main.py
```

**Specify prayer directly:**
```bash
python main.py --prayer Fajr
```

**With debug panel (shows live joint angles):**
```bash
python main.py --prayer Fajr --debug
```

**Using Android phone as webcam (IP Webcam app):**
```bash
python main.py --prayer Fajr --camera "http://192.168.x.x:8080/video"
```

**Rotate feed for vertical phone orientation:**
```bash
python main.py --prayer Fajr --camera "http://192.168.x.x:8080/video" --rotate 90
```

**Disable audio:**
```bash
python main.py --prayer Fajr --no-audio
```

**Find your camera index or test IP Webcam connection:**
```bash
python find_camera.py
python find_camera.py --ip 192.168.x.x
```

### Keyboard controls during session

| Key | Action |
|-----|--------|
| `Q` | Quit and save session |
| `R` | Reset — choose new prayer |
| `D` | Toggle debug panel |
| `S` | Toggle summary screen |
| `A` | Toggle audio on/off |

---

## System Architecture

```
Camera
  └─ PoseDetector          landmark detection + median filter (5-frame window)
       └─ SmoothedClassifier
            ├─ get_joint_angles()    hip, knee, spine, elbow, shoulder_y, etc.
            ├─ _detect_view()        avg_spread > 0.18 = side / ≤ 0.18 = front
            ├─ _classify_side()      spine-angle based (Paths A + B)
            ├─ _classify_front()     sh_hip_y_diff based (Paths A + B)
            └─ TransitionSmoother    3-frame confirmation before committing
                 └─ FramePreprocessor
                      ├─ noise detection + interpolation
                      └─ _predict_sujud()  trajectory prediction
                           └─ RakaatFSM   sequence enforcement + 0.5s hold time
                                └─ Dashboard + AudioEngine
```

---

## File Structure

```
SalatVision/
├── main.py                  Entry point
├── config.py                All thresholds and settings
├── pose_detector.py         MediaPipe wrapper + LandmarkSmoother
├── posture_classifier.py    View detection + side/front classifiers + TransitionSmoother
├── frame_preprocessor.py    Noise removal + interpolation + Sujud trajectory prediction
├── rakaat_fsm.py            Finite State Machine for rakaat counting
├── dashboard.py             OpenCV UI overlay
├── audio_engine.py          Audio feedback via pygame + pyttsx3
├── find_camera.py           Camera detection utility
├── evaluate.py              Confusion matrix + F1 score evaluation script
├── sessions/                Saved session JSON files (auto-created)
└── sounds/                  Pre-generated WAV files (auto-created on first run)
```

---

## Evaluation

Run the evaluation script against a recorded session:

```bash
python evaluate.py --session sessions/Fajr_20260522.json --ground ground_truth.csv
```

The ground truth CSV format:
```
time_sec,true_posture
0,QIYAM
1,QIYAM
5,RUKU
...
```

Valid posture labels: `QIYAM`, `RUKU`, `SUJUD`, `TASHAHHUD`

### Results

| Session | Prayer | Garment | Accuracy | Qiyam F1 | Ruku F1 | Sujud F1 | Tashahhud F1 |
|---------|--------|---------|----------|----------|---------|----------|--------------|
| 1 | Fajr | Hijab | 78.7% | 0.870 | 0.667 | 0.800 | 0.625 |
| 2 | Fajr | Hijab | 88.1% | 0.949 | 0.727 | 0.800 | 0.870 |
| 3 | Fajr | Hijab | 92.4% | 1.000 | 0.933 | 0.762 | 0.909 |
| 4 | Zuhr | Hijab | 82.1% | 0.937 | 0.700 | 0.689 | 0.789 |
| 5 | Maghrib | Hijab | 81.6% | 0.937 | 0.667 | 0.552 | 0.815 |
| 6 | Fajr | **Telekung** | 85.5% | 0.933 | **1.000** | 0.783 | 0.690 |
| **Avg** | — | Hijab | **84.6%** | **0.939** | **0.739** | **0.721** | **0.802** |

Rakaat counting accuracy: **100%** across all complete sessions.

---

## Camera Placement

- Place the camera **directly in front** of the prayer mat at approximately **knee-to-chest height**
- Ensure there is **contrast between the subject and background** — avoid white clothing against a white wall
- A dark backdrop behind the prayer mat significantly improves landmark detection

---

## Known Limitations

- **White-on-white contrast** — white telekung against a white wall causes complete landmark detection failure. Use a dark backdrop.
- **Side-view Ruku** — horizontal body spread collapses during Ruku at 45°, routing to the front classifier. All evaluation sessions used front-view placement.
- **Sujud–Tashahhud boundary** — both postures share bent knees and low hip position. This is the most common classification error.
- **Single-subject evaluation** — all sessions were performed by one female subject.

---

## Configuration

All thresholds and settings are in `config.py`. Key values:

| Setting | Value | Description |
|---------|-------|-------------|
| `POSTURE_HOLD_TIME` | 0.5s | Minimum hold time before FSM confirms a posture |
| `LANDMARK_SMOOTH_WINDOW` | 5 frames | Median filter window per joint |
| `TRANSITION_SMOOTH_FRAMES` | 3 frames | Consecutive frames required before committing to new posture |
| `VISIBILITY_THRESHOLD` | 0.45 | Minimum landmark confidence to trust a joint |
| `MEDIAPIPE_MODEL_COMPLEXITY` | 1 | BlazePose model complexity (0/1/2) |

---

## References

- Rahman et al., "Monitoring and Alarming Activity of Islamic Prayer (Salat) Posture Using Image Processing," ICCCE 2021
- Koubaa et al., "Activity Monitoring of Islamic Prayer (Salat) Postures using Deep Learning," arXiv:1911.04102, 2019
- Alfarizal et al., "Moslem Prayer Monitoring System Based on Image Processing," AHE vol. 14, 2023
- Bazarevsky et al., "BlazePose: On-device Real-time Body Pose tracking," arXiv:2006.10204, 2020

---

## Acknowledgements

Final Year Project — School of Computer Science and Engineering, Kyung Hee University, 2026.  
Supervised by Professor [Name].
