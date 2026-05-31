# ============================================================
# main.py - Salat Tracker Entry Point
# Usage: python main.py [--prayer Fajr] [--debug] [--no-audio]
#        python main.py --camera "http://192.168.x.x:8080/video"
#
# Changes from previous version:
#   - FramePreprocessor wired in between classifier and FSM
#   - TRANSITION label shown on dashboard, not passed to FSM
#   - --camera now accepts both int index and URL string
#   - SmoothedClassifier.reset() called on prayer reset
# ============================================================

import cv2
import argparse
import json
import os
import time as _time
from datetime import datetime

from pose_detector       import PoseDetector
from posture_classifier  import SmoothedClassifier
from frame_preprocessor  import FramePreprocessor
from rakaat_fsm          import RakaatFSM
from dashboard           import draw_dashboard, draw_summary_screen
from audio_engine        import AudioEngine
from config import (
    WINDOW_NAME, CAMERA_INDEX,
    FRAME_WIDTH, FRAME_HEIGHT,
    PRAYER_TYPES,
)


def camera_source(value):
    """Accept either an integer index or a URL string for --camera."""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(description="Salat Tracker")
    parser.add_argument("--prayer",   default="Custom",
                        choices=list(PRAYER_TYPES.keys()))
    parser.add_argument("--debug",    action="store_true",
                        help="Show live joint angles on screen")
    parser.add_argument("--camera",   type=camera_source, default=CAMERA_INDEX,
                        help="Camera index (0,1..) or IP Webcam URL")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio announcements")
    parser.add_argument("--rotate",   type=int, default=0,
                        choices=[0, 90, 180, 270],
                        help="Rotate camera feed: 0 (none), 90, 180, 270 degrees")
    return parser.parse_args()


def save_session(summary):
    os.makedirs("sessions", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "sessions/{}_{}.json".format(summary["prayer_name"], ts)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Session saved: {}".format(path))


def print_summary(summary):
    print("\n" + "="*50)
    print("         PRAYER SESSION SUMMARY")
    print("="*50)
    print("  Prayer   : {}".format(summary["prayer_name"]))
    print("  Rakaat   : {} / {}".format(summary["rakaat_done"], summary["target"]))
    print("  Duration : {}".format(summary["duration"]))
    status = "Complete" if summary["completed"] else "Incomplete"
    print("  Status   : {}".format(status))
    print("="*50)


def select_prayer_menu():
    print("\n" + "="*40)
    print("       SALAT TRACKER")
    print("="*40)
    options = list(PRAYER_TYPES.items())
    for i, (name, rak) in enumerate(options):
        target = str(rak) if rak < 99 else "unlimited"
        print("  [{}] {:<10} ({} rakaat)".format(i+1, name, target))
    print("="*40)
    while True:
        try:
            choice = input("Select prayer [1-6] or Enter for Custom: ").strip()
            if choice == "":
                return "Custom", 99
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                name, rak = options[idx]
                return name, rak
        except (ValueError, KeyboardInterrupt):
            pass
        print("Invalid, try again.")


def main():
    args = parse_args()

    if args.prayer == "Custom":
        prayer_name, total_rakaat = select_prayer_menu()
    else:
        prayer_name  = args.prayer
        total_rakaat = PRAYER_TYPES[args.prayer]

    print("Starting {} prayer ({} rakaat)...".format(prayer_name, total_rakaat))
    print("Controls: [Q] quit  [R] reset  [D] debug  [S] summary  [A] toggle audio\n")

    # ── Initialise modules ───────────────────────────────────
    audio = AudioEngine(
        rate    = 120,
        volume  = 0.75,
        enabled = not args.no_audio,
    )
    audio.announce_start(prayer_name, total_rakaat)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detector     = PoseDetector()
    classifier   = SmoothedClassifier()
    preprocessor = FramePreprocessor()       # NEW — directive #3
    fsm          = RakaatFSM(prayer_name, total_rakaat)

    debug_mode   = args.debug
    show_summary = False
    fps_time     = cv2.getTickCount()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        frame = cv2.flip(frame, 1) if isinstance(args.camera, int) else frame

        # Rotate frame if phone is held vertically
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        debug_angles  = None
        display_label = "UNKNOWN"   # what the dashboard shows
        landmarks, results = detector.process(frame)

        if landmarks:
            # Step 1: classify (includes transition smoother — directive #2)
            smoothed = classifier.update(landmarks)

            # Step 2: pre-process (noise removal + interpolation — directive #3)
            clean = preprocessor.process(smoothed, landmarks)

            # Step 3: only pass real postures to FSM — not TRANSITION/UNKNOWN
            # TRANSITION means "moving between postures, hold on"
            # The FSM simply waits; no state is corrupted
            if clean not in ("TRANSITION", "UNKNOWN"):
                rakaat_completed = fsm.update(clean)
            else:
                rakaat_completed = False

            display_label = clean
            
            # ── Evaluation logging (one per second) ──────────
            if hasattr(fsm, 'frame_log') and clean not in ("TRANSITION", "UNKNOWN"):
                current_sec = int(_time.time() - fsm.session_start)
                if not fsm.frame_log or fsm.frame_log[-1]["time"] < current_sec:
                    fsm.frame_log.append({
                        "time": current_sec,
                        "predicted": clean
                    })
            frame = detector.draw_skeleton(frame, results)

            if debug_mode:
                debug_angles = classifier.get_debug_angles(landmarks)

            if rakaat_completed:
                state = fsm.get_state()
                if state["is_complete"]:
                    audio.announce_complete(prayer_name)
                else:
                    audio.announce_rakaat(state["rakaat"], total_rakaat)

        # Build state dict for dashboard — inject display_label
        state = fsm.get_state()
        state["current_posture"] = display_label

        if show_summary:
            frame = draw_summary_screen(frame, fsm.get_summary())
        else:
            frame = draw_dashboard(frame, state, debug_angles)

        # FPS counter
        fps_now  = cv2.getTickCount()
        fps      = cv2.getTickFrequency() / (fps_now - fps_time)
        fps_time = fps_now
        cv2.putText(frame, "FPS: {:.0f}".format(fps),
                    (FRAME_WIDTH - 110, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        audio_label = "AUDIO: ON" if audio.enabled else "AUDIO: OFF"
        audio_color = (0, 200, 100) if audio.enabled else (60, 60, 200)
        cv2.putText(frame, audio_label,
                    (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, audio_color, 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            prayer_name, total_rakaat = select_prayer_menu()
            fsm.reset(prayer_name, total_rakaat)
            classifier.reset()
            preprocessor.reset()
            detector.reset_smoother()      # clear coordinate buffer on reset
            show_summary = False
            audio.announce_start(prayer_name, total_rakaat)
        elif key == ord("d"):
            debug_mode = not debug_mode
            print("Debug: {}".format("ON" if debug_mode else "OFF"))
        elif key == ord("s"):
            show_summary = not show_summary
        elif key == ord("a"):
            audio.set_enabled(not audio.enabled)

    # ── Shutdown ─────────────────────────────────────────────
    audio.shutdown()
    summary = fsm.get_summary()
    print_summary(summary)
    save_session(summary)
    detector.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()