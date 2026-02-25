"""
Camera Detection Utility
Finds all available cameras on your system
"""

import cv2

def check_cameras():
    print("=" * 50)
    print("CAMERA DETECTION UTILITY")
    print("=" * 50)
    print()
    
    available = []
    
    print("Checking camera indices 0-9...")
    print()
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  ✅ Camera {i}: FOUND - {w}x{h} @ {fps:.0f} FPS")
                available.append(i)
            cap.release()
    
    print()
    print("=" * 50)
    
    if available:
        print(f"Available cameras: {available}")
        print(f"Your laptop camera is most likely index: {available[0]}")
    else:
        print("No cameras found!")
        print("Make sure your webcam is connected and not in use by another app.")
    
    print("=" * 50)
    
    return available


def test_droidcam(ip="10.110.213.111", port="4747"):
    print()
    print("=" * 50)
    print("DROIDCAM CONNECTION TEST")
    print("=" * 50)
    print()
    
    url = f"http://{ip}:{port}/video"
    print(f"Attempting to connect to: {url}")
    print()
    
    cap = cv2.VideoCapture(url)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  ✅ DroidCam connected successfully!")
            print(f"  Resolution: {w}x{h}")
        cap.release()
    else:
        print("  ❌ Could not connect to DroidCam")
        print()
        print("  Please check:")
        print("  1. DroidCam app is running on your phone")
        print("  2. Phone and PC are on the same Tailscale network")
        print(f"  3. IP address is correct: {ip}")
        print(f"  4. Port is correct: {port}")
    
    print()
    print("=" * 50)


if __name__ == "__main__":
    check_cameras()
    test_droidcam()