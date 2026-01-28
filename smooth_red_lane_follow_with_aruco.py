#!/usr/bin/env python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ============================================================
#   MOTOR SETUP
# ============================================================
MOTOR_LEFT_FORWARD  = 16
MOTOR_LEFT_BACKWARD = 12
MOTOR_RIGHT_FORWARD = 21
MOTOR_RIGHT_BACKWARD= 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [MOTOR_LEFT_FORWARD, MOTOR_LEFT_BACKWARD,
            MOTOR_RIGHT_FORWARD, MOTOR_RIGHT_BACKWARD]:
    GPIO.setup(pin, GPIO.OUT)

PWM_FREQ = 100
left_f  = GPIO.PWM(MOTOR_LEFT_FORWARD,  PWM_FREQ)
left_b  = GPIO.PWM(MOTOR_LEFT_BACKWARD, PWM_FREQ)
right_f = GPIO.PWM(MOTOR_RIGHT_FORWARD, PWM_FREQ)
right_b = GPIO.PWM(MOTOR_RIGHT_BACKWARD,PWM_FREQ)

for pwm in (left_f, left_b, right_f, right_b):
    pwm.start(0)

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

# ============================================================
#   PARAMETERS
# ============================================================
FRAME_WIDTH  = 320
FRAME_HEIGHT = 240

TARGET_X = int(FRAME_WIDTH * 2/3)

MAX_SPEED   = 40
BASE_SPEED  = int(MAX_SPEED * 0.6)
STEER_GAIN  = 0.1
LOOKAHEAD_Y = FRAME_HEIGHT - 40

NWINDOWS = 6
MARGIN   = 40
MINPIX   = 40

# ============================================================
#   PERSPECTIVE TRANSFORM
# ============================================================
WARP_SRC = np.float32([
    [40, 95],
    [280, 95],
    [0, FRAME_HEIGHT-8],
    [FRAME_WIDTH-1, FRAME_HEIGHT-8]
])

WARP_DST = np.float32([
    [60, 0],
    [FRAME_WIDTH-60, 0],
    [60, FRAME_HEIGHT],
    [FRAME_WIDTH-60, FRAME_HEIGHT]
])

M_warp = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
Minv   = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)

# ============================================================
#   HSV FOR RED LANE
# ============================================================
lower_red1 = np.array([0, 110, 70])
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70])
upper_red2 = np.array([180, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# ============================================================
#   ARUCO (DICT_4X4_50)
# ============================================================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Simple ID mapping
TAG_MAP = {
    1: "STOP",
    2: "RIGHT",
    3: "LEFT"
}

mode = "FOLLOW"          # FOLLOW, TURN_LEFT, TURN_RIGHT, STOP
turn_start = 0
TURN_TIMEOUT = 2.5       # seconds

# ============================================================
#   CAMERA INIT
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.3)

print("\n--- Red-Lane Follower + ArUco Control (ID 1 STOP, 2 RIGHT, 3 LEFT) ---\n")

# ============================================================
#   MAIN LOOP
# ============================================================
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera error.")
            break

        frame = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ---------------- ARUCO DETECTION -------------------
        corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None and mode == "FOLLOW":
            for tid in ids.flatten():
                if tid in TAG_MAP:
                    action = TAG_MAP[tid]
                    print(f"Aruco Tag {tid} Detected → {action}")

                    if action == "STOP":
                        mode = "STOP"
                        stop_motor()
                        time.sleep(1.5)     # real stop
                        mode = "FOLLOW"      # resume moving
                        continue

                    elif action == "LEFT":
                        mode = "TURN_LEFT"
                        turn_start = time.time()

                    elif action == "RIGHT":
                        mode = "TURN_RIGHT"
                        turn_start = time.time()

        # ---------------- TURNING STATES ----------------------
        if mode == "TURN_LEFT":
            # smooth arc: slow left wheel, fast right wheel
            set_motor(int(BASE_SPEED*0.35), BASE_SPEED)

            # check for lane reappearing after turn
            lane_check = cv2.warpPerspective(hsv[:,:,2], M_warp, (FRAME_WIDTH, FRAME_HEIGHT))
            if np.sum(lane_check[160:,:] > 50) > 200:
                print("Lane Reacquired → FOLLOW")
                mode = "FOLLOW"

            # emergency exit
            if time.time() - turn_start > TURN_TIMEOUT:
                print("Left Turn Timeout → FOLLOW")
                mode = "FOLLOW"

        elif mode == "TURN_RIGHT":
            set_motor(BASE_SPEED, int(BASE_SPEED*0.35))

            lane_check = cv2.warpPerspective(hsv[:,:,2], M_warp, (FRAME_WIDTH, FRAME_HEIGHT))
            if np.sum(lane_check[160:,:] > 50) > 200:
                print("Lane Reacquired → FOLLOW")
                mode = "FOLLOW"

            if time.time() - turn_start > TURN_TIMEOUT:
                print("Right Turn Timeout → FOLLOW")
                mode = "FOLLOW"

        # ------------------------------------------------------
        #   NORMAL LANE FOLLOWING
        # ------------------------------------------------------
        if mode == "FOLLOW":
            # ---- Red mask ----
            mask = cv2.bitwise_or(
                       cv2.inRange(hsv, lower_red1, upper_red1),
                       cv2.inRange(hsv, lower_red2, upper_red2)
                   )
            mask[hsv[:,:,2] < 40] = 0

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            warped = cv2.warpPerspective(mask, M_warp, (FRAME_WIDTH, FRAME_HEIGHT))

            bottom = warped[FRAME_HEIGHT//2:, :]
            col_count = np.sum(bottom//255, axis=0)

            if col_count.max() < 1:
                set_motor(int(BASE_SPEED*0.8), int(BASE_SPEED*0.8))
            else:
                base_x = int(np.argmax(col_count))

                nz = warped.nonzero()
                nonzeroy = np.array(nz[0])
                nonzerox = np.array(nz[1])

                window_h = int(FRAME_HEIGHT / NWINDOWS)
                cur_x = base_x
                px, py = [], []

                for w in range(NWINDOWS):
                    wy_low  = FRAME_HEIGHT - (w+1)*window_h
                    wy_high = FRAME_HEIGHT - w*window_h
                    if wy_low < 0: wy_low = 0

                    wx_l = max(0, cur_x - MARGIN)
                    wx_r = min(FRAME_WIDTH-1, cur_x + MARGIN)

                    good = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                            (nonzerox >= wx_l) & (nonzerox < wx_r)).nonzero()[0]

                    cy = int((wy_low+wy_high)/2)

                    if len(good) > MINPIX:
                        cur_x = int(np.mean(nonzerox[good]))

                    px.append(cur_x)
                    py.append(cy)

                px = np.array(px, np.float32)
                py = np.array(py, np.float32)

                poly = np.polyfit(py, px, 2)

                predicted_x = int(
                    poly[0]*LOOKAHEAD_Y**2 +
                    poly[1]*LOOKAHEAD_Y +
                    poly[2]
                )

                pt = np.array([[[predicted_x, LOOKAHEAD_Y]]], dtype=np.float32)
                pt_orig = cv2.perspectiveTransform(pt, Minv)
                lane_x = int(pt_orig[0,0,0])

                deviation = lane_x - TARGET_X
                steer = STEER_GAIN * deviation

                left_speed  = clamp(BASE_SPEED - steer, 0, 100)
                right_speed = clamp(BASE_SPEED + steer, 0, 100)
                set_motor(left_speed, right_speed)

        # ---------------- DISPLAY -------------------------
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted")

finally:
    stop_motor()
    left_f.stop(); left_b.stop()
    right_f.stop(); right_b.stop()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Clean exit.")
