#!/usr/bin/env python3
import cv2
import numpy as np
import time
from gpiozero import PWMOutputDevice

# ============================================================
#    HARDWARE SETUP (Pi 5 Compatible)
# ============================================================
left_f  = PWMOutputDevice(16, frequency=100)
left_b  = PWMOutputDevice(12, frequency=100)
right_f = PWMOutputDevice(21, frequency=100)
right_b = PWMOutputDevice(20, frequency=100)

# ============================================================
#    HELPER FUNCTIONS
# ============================================================
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, int(x)))

def set_motor(left, right):
    l_val = clamp(abs(left)) / 100.0
    r_val = clamp(abs(right)) / 100.0
    if left >= 0:
        left_f.value = l_val; left_b.value = 0
    else:
        left_f.value = 0; left_b.value = l_val
    if right >= 0:
        right_f.value = r_val; right_b.value = 0
    else:
        right_f.value = 0; right_b.value = r_val

def stop_motor():
    left_f.value = 0; left_b.value = 0
    right_f.value = 0; right_b.value = 0

def timed_turn(direction):
    TURN_DURATION = 0.75  
    TURN_SPEED = 35
    print(f"Executing {direction} Turn...")
    if direction == "LEFT":
        set_motor(-TURN_SPEED, TURN_SPEED)
    else:
        set_motor(TURN_SPEED, -TURN_SPEED)
    time.sleep(TURN_DURATION)
    stop_motor()
    time.sleep(0.5)

# ============================================================
#    VISION & STEERING PARAMS
# ============================================================
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
BASE_SPEED   = 20    
STEER_GAIN   = 0.22  # Adjusted for full-frame sensitivity
NWINDOWS     = 10    # Breaks the frame into 10 horizontal parts
MARGIN       = 40    # Width of the search window
MINPIX       = 50    # Minimum pixels to re-center window

lower_red1 = np.array([0, 110, 70]);  upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70]); upper_red2 = np.array([180, 255, 255])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Bird's Eye View Warp Points
WARP_SRC = np.float32([[40, 95], [280, 95], [0, FRAME_HEIGHT-8], [FRAME_WIDTH-1, FRAME_HEIGHT-8]])
WARP_DST = np.float32([[60, 0], [FRAME_WIDTH-60, 0], [60, FRAME_HEIGHT], [FRAME_WIDTH-60, FRAME_HEIGHT]])
M_warp = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
Minv   = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)

# ArUco Setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
try:
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    OLD_ARUCO = False
except AttributeError:
    OLD_ARUCO = True

TAG_MAP = {1: "RIGHT", 4: "LEFT", 8: "RIGHT", 7: "LEFT", 6: "RIGHT", 10: "LEFT", 9: "STOP"}
last_turn_time = 0
TURN_COOLDOWN = 4.0  

# ============================================================
#    MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(1.0)

print("Robot Online. Tracking FULL FRAME for better cornering...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. ARUCO DETECTION
        action = None
        if (time.time() - last_turn_time) > TURN_COOLDOWN:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray) if not OLD_ARUCO else cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
            if ids is not None:
                tid = ids[0][0]
                if tid in TAG_MAP:
                    action = TAG_MAP[tid]

        if action:
            print(f"Tag Detected: {action}")
            if action == "STOP":
                stop_motor(); time.sleep(2.0)
            else:
                set_motor(BASE_SPEED, BASE_SPEED); time.sleep(0.4)
                timed_turn(action)
            last_turn_time = time.time()
        
        else:
            # 2. FULL-FRAME LANE FOLLOWING (Sliding Windows)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                                  cv2.inRange(hsv, lower_red2, upper_red2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            warped = cv2.warpPerspective(mask, M_warp, (FRAME_WIDTH, FRAME_HEIGHT))

            # Find starting point at the bottom
            bottom_hist = np.sum(warped[int(FRAME_HEIGHT*0.7):, :], axis=0)
            
            if np.max(bottom_hist) < 100:
                set_motor(BASE_SPEED * 0.7, BASE_SPEED * 0.7)
            else:
                current_x = np.argmax(bottom_hist)
                nonzero = warped.nonzero()
                nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])
                
                lane_inds = []
                window_height = FRAME_HEIGHT // NWINDOWS

                # Loop through horizontal segments from bottom to top
                for w in range(NWINDOWS):
                    win_y_low = FRAME_HEIGHT - (w+1) * window_height
                    win_y_high = FRAME_HEIGHT - w * window_height
                    win_x_left, win_x_right = current_x - MARGIN, current_x + MARGIN
                    
                    # Get pixels in this horizontal part
                    good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                 (nonzerox >= win_x_left) & (nonzerox < win_x_right)).nonzero()[0]
                    
                    lane_inds.append(good_inds)
                    if len(good_inds) > MINPIX:
                        current_x = int(np.mean(nonzerox[good_inds]))
                        # Draw visual rectangle on frame for debug view
                        cv2.rectangle(frame, (win_x_left, win_y_low), (win_x_right, win_y_high), (0,255,0), 1)

                lane_inds = np.concatenate(lane_inds)
                
                if len(lane_inds) > 0:
                    lane_center_warp = int(np.mean(nonzerox[lane_inds]))
                    error = lane_center_warp - (FRAME_WIDTH // 2)
                    steer = STEER_GAIN * error
                    set_motor(BASE_SPEED + steer, BASE_SPEED - steer)

        # SSH Debugging: Save image to check what the windows are doing
        #cv2.imwrite("/home/pi/debug.jpg", frame)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stop_motor()
    cap.release()