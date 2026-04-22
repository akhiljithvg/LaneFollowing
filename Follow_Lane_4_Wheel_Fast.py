#!/usr/bin/env python3
import cv2
import numpy as np
from gpiozero import PWMOutputDevice
import time

# ============================================================
#    HARDWARE SETUP
# ============================================================
left_f  = PWMOutputDevice(12, frequency=100)
left_b  = PWMOutputDevice(16, frequency=100)
right_f = PWMOutputDevice(20, frequency=100)
right_b = PWMOutputDevice(21, frequency=100)

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, int(x)))

def set_motor(left, right):
    # Mapping logic: Swapping as per physical wiring
    actual_left = right
    actual_right = left
    l_val = clamp(abs(actual_left)) / 100.0
    r_val = clamp(abs(actual_right)) / 100.0
    
    left_f.value = l_val if actual_left >= 0 else 0
    left_b.value = 0 if actual_left >= 0 else l_val
    right_f.value = r_val if actual_right >= 0 else 0
    right_b.value = 0 if actual_right >= 0 else r_val

def stop_motor():
    left_f.value = 0; left_b.value = 0
    right_f.value = 0; right_b.value = 0

# ============================================================
#    CONTROL PARAMS (Tuned for Full Frame)
# ============================================================
BASE_SPEED = 100       # Reduced speed: Full frame processing is noisier
STEER_P = 0.95        # Increased P to react faster to drifts
STEER_D = 0.85        # D term to stop the "wobble"
last_error = 0
last_search_direction = 1

# Red Mask
lower_red1 = np.array([0, 130, 70]);  upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 130, 70]); upper_red2 = np.array([180, 255, 255])

# ============================================================
#    MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(3, 320); cap.set(4, 240)

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), 
                              cv2.inRange(hsv, lower_red2, upper_red2))

        # Using FULL MASK (No ROI slicing)
        M = cv2.moments(mask)
        
        if M["m00"] > 500:
            # LINE IS VISIBLE
            cX = int(M["m10"] / M["m00"])
            # cY can tell us if the line is close or far
            cY = int(M["m01"] / M["m00"]) 
            
            error = cX - (160) # 160 is half of 320 width

            # --- DRIFT PROTECTION ---
            # If the line is lost on the left, we search right
            if error < -20: last_search_direction = -1
            elif error > 20: last_search_direction = 1

            # PD Controller
            derivative = error - last_error
            steer = (STEER_P * error) + (STEER_D * derivative)
            last_error = error
            
            # Speed Control: If the line is "high" in the frame (cY is small),
            # it means the line is far away. We should slow down to approach carefully.
            speed_factor = 1.0 if cY > 120 else 0.7
            current_speed = max(25, (BASE_SPEED - abs(steer) * 0.4) * speed_factor)
            
            set_motor(current_speed + steer, current_speed - steer)
            
            # Visual Debug (Optional)
            cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)
            
        else:
            # LINE LOST: Pivot Right if line was on left
            search_speed = 70
            set_motor(search_speed * last_search_direction, -search_speed * last_search_direction)
            time.sleep(0.05)
            stop_motor()

except KeyboardInterrupt:
    stop_motor()
finally:
    cap.release()