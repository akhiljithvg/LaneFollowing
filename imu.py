#!/usr/bin/env python3
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force X11 for monitor
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import board
import busio
import adafruit_mpu6050

# ============================================================
#   HARDWARE SETUP
# ============================================================
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(i2c)
    print("MPU6050 Initialized.")
except Exception as e:
    print(f"Error initializing MPU6050: {e}")
    exit(1)

MOTOR_LEFT_FORWARD  = 16
MOTOR_LEFT_BACKWARD = 12
MOTOR_RIGHT_FORWARD = 21
MOTOR_RIGHT_BACKWARD= 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for p in [MOTOR_LEFT_FORWARD, MOTOR_LEFT_BACKWARD, MOTOR_RIGHT_FORWARD, MOTOR_RIGHT_BACKWARD]:
    GPIO.setup(p, GPIO.OUT)

PWM_FREQ = 100
left_f  = GPIO.PWM(MOTOR_LEFT_FORWARD,  PWM_FREQ)
left_b  = GPIO.PWM(MOTOR_LEFT_BACKWARD, PWM_FREQ)
right_f = GPIO.PWM(MOTOR_RIGHT_FORWARD, PWM_FREQ)
right_b = GPIO.PWM(MOTOR_RIGHT_BACKWARD,PWM_FREQ)

for pwm in (left_f, left_b, right_f, right_b):
    pwm.start(0)

# ============================================================
#   HELPER FUNCTIONS
# ============================================================
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, int(x)))

def set_motor(left, right):
    if left >= 0:
        left_f.ChangeDutyCycle(clamp(left)); left_b.ChangeDutyCycle(0)
    else:
        left_f.ChangeDutyCycle(0); left_b.ChangeDutyCycle(clamp(-left))
    if right >= 0:
        right_f.ChangeDutyCycle(clamp(right)); right_b.ChangeDutyCycle(0)
    else:
        right_f.ChangeDutyCycle(0); right_b.ChangeDutyCycle(clamp(-right))

def stop_motor():
    set_motor(0, 0)

def calibrate_gyro(samples=100):
    print("Calibrating Gyro... Keep robot still!")
    stop_motor()
    offset = 0.0
    for _ in range(samples):
        offset += mpu.gyro[2] 
        time.sleep(0.01)
    return offset / samples

def turn_robot(target_angle, turn_speed=45):
    """Turns robot exactly X degrees using Gyro integration"""
    print(f"--- Gyro Turn: {target_angle} deg ---")
    current_angle = 0.0
    last_time = time.time()
    
    # Start turning
    if target_angle > 0:
        set_motor(-turn_speed, turn_speed) # Left
    else:
        set_motor(turn_speed, -turn_speed) # Right
        
    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        
        # Integrate Gyro Z
        gyro_z = (mpu.gyro[2] - GYRO_OFFSET) * 57.2958
        current_angle += gyro_z * dt
        
        # Check completion
        if target_angle > 0 and current_angle >= target_angle: break
        if target_angle < 0 and current_angle <= target_angle: break
        
    stop_motor()
    time.sleep(0.2) # Settle

# ============================================================
#   VISION PARAMS (Restored from First Script)
# ============================================================
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240

# Tuning Params
MAX_SPEED    = 35
BASE_SPEED   = int(MAX_SPEED * 0.6)
STEER_GAIN   = 0.12  # Sensitivity
NWINDOWS     = 10    # Number of sliding windows
MARGIN       = 40    # Width of sliding window
MINPIX       = 40    # Min pixels to recenter window

# Color Ranges (Red)
lower_red1 = np.array([0, 110, 70])
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70])
upper_red2 = np.array([180, 255, 255])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Perspective Transform
WARP_SRC = np.float32([[40, 95], [280, 95], [0, FRAME_HEIGHT-8], [FRAME_WIDTH-1, FRAME_HEIGHT-8]])
WARP_DST = np.float32([[60, 0], [FRAME_WIDTH-60, 0], [60, FRAME_HEIGHT], [FRAME_WIDTH-60, FRAME_HEIGHT]])
M_warp = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
Minv   = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)

# ArUco
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# NOTE: If you are on new OpenCV, uncomment the line below:
# aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

TAG_MAP = {1: "STOP", 2: "RIGHT", 3: "LEFT"}

# ============================================================
#   MAIN LOOP
# ============================================================

GYRO_OFFSET = calibrate_gyro()
print(f"Calibration Done. Offset: {GYRO_OFFSET:.4f}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.3)

print("Starting Loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # 1. CHECK ARUCO
        # If using old OpenCV (likely on Pi):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
        # If using new OpenCV 4.7+, use: corners, ids, _ = aruco_detector.detectMarkers(frame)

        action = None
        if ids is not None:
            tid = ids[0][0]
            if tid in TAG_MAP:
                action = TAG_MAP[tid]
                print(f"Junction: {action}")

        if action == "STOP":
            stop_motor()
            time.sleep(2.0)
            set_motor(BASE_SPEED, BASE_SPEED) # Drive past tag
            time.sleep(0.5)

        elif action == "LEFT":
            set_motor(BASE_SPEED, BASE_SPEED) # Drive into center
            time.sleep(0.4) 
            stop_motor()
            turn_robot(90) # Gyro Turn

        elif action == "RIGHT":
            set_motor(BASE_SPEED, BASE_SPEED)
            time.sleep(0.4)
            stop_motor()
            turn_robot(-90) # Gyro Turn

        else:
            # 2. LANE FOLLOWING (SLIDING WINDOW RESTORED)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                                  cv2.inRange(hsv, lower_red2, upper_red2))
            # Remove dark noise
            mask[hsv[:,:,2] < 40] = 0 
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            warped = cv2.warpPerspective(mask, M_warp, (FRAME_WIDTH, FRAME_HEIGHT))

            # Histogram to find start of lane
            bottom_half = warped[FRAME_HEIGHT//2:, :]
            col_counts = np.sum(bottom_half // 255, axis=0)
            max_val = np.max(col_counts)

            if max_val < 5:
                # LINE LOST -> Go slow / Straight (or use Gyro heading hold)
                set_motor(int(BASE_SPEED*0.8), int(BASE_SPEED*0.8))
            else:
                # Find lane base
                base_x = np.argmax(col_counts)
                
                # Sliding Window Logic
                nonzero = warped.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                
                current_x = base_x
                window_height = warped.shape[0] // NWINDOWS
                
                lane_inds = []
                
                for w in range(NWINDOWS):
                    win_y_low = warped.shape[0] - (w+1) * window_height
                    win_y_high = warped.shape[0] - w * window_height
                    win_x_left = max(0, current_x - MARGIN)
                    win_x_right = min(FRAME_WIDTH, current_x + MARGIN)
                    
                    good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                 (nonzerox >= win_x_left) & (nonzerox < win_x_right)).nonzero()[0]
                    
                    lane_inds.append(good_inds)
                    
                    if len(good_inds) > MINPIX:
                        current_x = int(np.mean(nonzerox[good_inds]))
                        
                lane_inds = np.concatenate(lane_inds)
                
                if len(lane_inds) > 0:
                    # Calculate center of lane pixels
                    lane_center_warp = int(np.mean(nonzerox[lane_inds]))
                    
                    # Convert back to real screen coordinates
                    pts = np.array([[[lane_center_warp, FRAME_HEIGHT-10]]], dtype=np.float32)
                    pts_orig = cv2.perspectiveTransform(pts, Minv)
                    lane_x_real = int(pts_orig[0,0,0])
                    
                    # Steering Logic
                    target_x = FRAME_WIDTH // 2
                    error = lane_x_real - target_x
                    
                    # Proportional Steering
                    steer = STEER_GAIN * error
                    
                    # Optional: Gyro Damping (Prevent oscillation)
                    gyro_z = (mpu.gyro[2] - GYRO_OFFSET) * 57.3
                    steer -= (0.02 * gyro_z) 
                    
                    set_motor(BASE_SPEED - steer, BASE_SPEED + steer)

        # DEBUG DISPLAY (Uncomment if monitor is connected)
        cv2.imshow("Frame", frame)
        cv2.imshow("Warp", warped)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    stop_motor()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()