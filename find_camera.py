"""
find_camera.py — Detects available cameras and tests IP Webcam URL
Usage:
    python find_camera.py                        # scan local camera indices
    python find_camera.py --192.168.45.207       # test IP Webcam
    python find_camera.py --192.168.45.207 --port 8080
"""

import cv2
import argparse
import sys


def scan_local_cameras(max_index=5):
    print("\n── Scanning local camera indices 0–{} ──".format(max_index))
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW faster on Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print("  [✓] Camera index {} — {}x{}".format(i, w, h))
                found.append(i)
            else:
                print("  [~] Camera index {} — opened but no frame".format(i))
            cap.release()
        else:
            print("  [✗] Camera index {} — not available".format(i))

    if found:
        print("\nUse one of these with:  python main.py --camera <index>")
        print("Recommended: --camera {}".format(found[-1]))  # last = most likely phone
    else:
        print("\nNo cameras found.")
    return found


def test_ip_webcam(ip, port=8080):
    url = "http://{}:{}/video".format(ip, port)
    print("\n── Testing IP Webcam at {} ──".format(url))
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("  [✗] Could not connect. Check:")
        print("      • Phone and PC on same WiFi network")
        print("      • IP Webcam app is running (tap 'Start server')")
        print("      • Correct IP/port")
        return False

    ret, frame = cap.read()
    if not ret:
        print("  [✗] Connected but could not read frame.")
        cap.release()
        return False

    h, w = frame.shape[:2]
    print("  [✓] Connected! Resolution: {}x{}".format(w, h))
    print("\nTo use this camera, edit main.py and replace:")
    print('  cap = cv2.VideoCapture(args.camera)')
    print('with:')
    print('  cap = cv2.VideoCapture("http://{}:{}/video")'.format(ip, port))

    # Show a preview
    print("\nShowing preview — press Q to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("IP Webcam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


def main():
    parser = argparse.ArgumentParser(description="Camera finder for Salat Tracker")
    parser.add_argument("--ip",   default=None, help="Phone IP for IP Webcam")
    parser.add_argument("--port", type=int, default=8080, help="IP Webcam port (default 8080)")
    args = parser.parse_args()

    if args.ip:
        test_ip_webcam(args.ip, args.port)
    else:
        scan_local_cameras()
        print("\nFor IP Webcam, run:  python find_camera.py --ip 192.168.45.207")


if __name__ == "__main__":
    main()