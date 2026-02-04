#!/usr/bin/env python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ============================================================
#    HARDWARE SETUP
# ============================================================
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
#    HELPER FUNCTIONS
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

def timed_turn(direction):
    """Pivots the robot for a fixed duration to achieve ~90 degrees."""
    TURN_DURATION = 0.75  # <--- ADJUST THIS based on your floor/battery
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
#    VISION PARAMS
# ============================================================
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240
BASE_SPEED   = 18    # Slightly faster for responsiveness
STEER_GAIN   = 0.18  # Sensitivity to red color position

# HSV Red Ranges
lower_red1 = np.array([0, 110, 70])
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70])
upper_red2 = np.array([180, 255, 255])

# ArUco Setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
try:
    aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    OLD_ARUCO = False
except AttributeError:
    OLD_ARUCO = True

TAG_MAP = {1: "STOP", 6: "LEFT", 8: "RIGHT"}
last_turn_time = 0
TURN_COOLDOWN = 3.0  # Seconds to ignore tags after a turn

# ============================================================
#    MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.5)

print("Robot Online. Tracking Red + ArUco...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- 1. ARUCO DETECTION ---
        action = None
        if (time.time() - last_turn_time) > TURN_COOLDOWN:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if OLD_ARUCO:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
            else:
                corners, ids, _ = aruco_detector.detectMarkers(gray)
            
            if ids is not None:
                tid = ids[0][0] # Focus on the first detected tag
                if tid in TAG_MAP:
                    action = TAG_MAP[tid]
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # --- 2. EXECUTE ARUCO ACTION ---
        if action == "STOP":
            print("Tag: STOP")
            stop_motor()
            time.sleep(2.0)
            last_turn_time = time.time() # Trigger cooldown so it doesn't stop again immediately

        elif action == "LEFT":
            # Drive forward slightly to get into the intersection
            set_motor(BASE_SPEED, BASE_SPEED)
            time.sleep(0.4)
            timed_turn("LEFT")
            last_turn_time = time.time()

        elif action == "RIGHT":
            set_motor(BASE_SPEED, BASE_SPEED)
            time.sleep(0.4)
            timed_turn("RIGHT")
            last_turn_time = time.time()

        # --- 3. RED LINE FOLLOWING (Default Action) ---
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                                  cv2.inRange(hsv, lower_red2, upper_red2))
            
            # Look at the lower half only to stay focused on the track
            roi_mask = mask[FRAME_HEIGHT//2:, :]
            M = cv2.moments(roi_mask)
            
            if M["m00"] > 500:
                cX = int(M["m10"] / M["m00"])
                error = cX - (FRAME_WIDTH // 2)
                
                steer = STEER_GAIN * error
                # Steer toward the red blob
                set_motor(BASE_SPEED + steer, BASE_SPEED - steer)
                
                # Visual Indicator
                cv2.circle(frame, (cX, FRAME_HEIGHT - 30), 10, (0, 255, 0), -1)
            else:
                # No red? Drive slowly forward searching
                set_motor(BASE_SPEED * 0.7, BASE_SPEED * 0.7)
                cv2.putText(frame, "SEARCHING FOR RED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Show the camera feed
        cv2.imshow("Robot View", frame)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("\nUser Stopped Program.")
finally:
    stop_motor()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
