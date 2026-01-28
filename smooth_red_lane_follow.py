#!/usr/bin/env python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ---------------- GPIO motor pins (adjust to your wiring) ----------------
MOTOR_LEFT_FORWARD  = 16
MOTOR_LEFT_BACKWARD = 12
MOTOR_RIGHT_FORWARD = 21
MOTOR_RIGHT_BACKWARD= 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [MOTOR_LEFT_FORWARD, MOTOR_LEFT_BACKWARD, MOTOR_RIGHT_FORWARD, MOTOR_RIGHT_BACKWARD]
for p in motor_pins:
    GPIO.setup(p, GPIO.OUT)

# PWM setup (each direction gets its own PWM so we can run forward/back independently)
PWM_FREQ = 100
left_f_pwm  = GPIO.PWM(MOTOR_LEFT_FORWARD, PWM_FREQ)
left_b_pwm  = GPIO.PWM(MOTOR_LEFT_BACKWARD, PWM_FREQ)
right_f_pwm = GPIO.PWM(MOTOR_RIGHT_FORWARD, PWM_FREQ)
right_b_pwm = GPIO.PWM(MOTOR_RIGHT_BACKWARD, PWM_FREQ)

for pwm in (left_f_pwm, left_b_pwm, right_f_pwm, right_b_pwm):
    pwm.start(0)

def clamp(x, a=0, b=100):
    return max(a, min(b, int(x)))

def set_motor(left_speed, right_speed):
    """
    left_speed, right_speed in range -100..100
    positive => forward, negative => backward
    """
    # left motor
    if left_speed >= 0:
        left_f_pwm.ChangeDutyCycle(clamp(left_speed))
        left_b_pwm.ChangeDutyCycle(0)
    else:
        left_f_pwm.ChangeDutyCycle(0)
        left_b_pwm.ChangeDutyCycle(clamp(-left_speed))
    # right motor
    if right_speed >= 0:
        right_f_pwm.ChangeDutyCycle(clamp(right_speed))
        right_b_pwm.ChangeDutyCycle(0)
    else:
        right_f_pwm.ChangeDutyCycle(0)
        right_b_pwm.ChangeDutyCycle(clamp(-right_speed))

def stop():
    set_motor(0, 0)

# ---------------- Vision & control params ----------------
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Perspective transform source points (tuned starting values for your camera)
WARP_SRC = np.float32([
    [40, 95],    # top-left
    [280, 95],   # top-right
    [0, FRAME_HEIGHT-8],   # bottom-left
    [FRAME_WIDTH-1, FRAME_HEIGHT-8]  # bottom-right
])
WARP_DST = np.float32([
    [60, 0],
    [FRAME_WIDTH-60, 0],
    [60, FRAME_HEIGHT],
    [FRAME_WIDTH-60, FRAME_HEIGHT]
])

M_warp = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
Minv = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)

# Sliding window params
NWINDOWS = 200
MARGIN = 40
MINPIX = 40

# Steering & speed params
CENTER_TOLERANCE = 20   # pixels near center considered aligned
MAX_SPEED = 30          # base max speed (tune down for safety)
BASE_SPEED = int(MAX_SPEED * 0.6)  # forward base speed when centered
STEER_GAIN = 0.1        # proportional gain: larger -> more aggressive steering (tune)

# HSV red ranges (two ranges for wrapping hue)
lower_red1 = np.array([0, 110, 70])
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70])
upper_red2 = np.array([180, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.3)

print("Starting smooth red-lane birdseye sliding-window tracker.")
print("Keep robot lifted for first test. Press 'q' to quit, 's' to save debug images.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- red mask ---
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # optional: remove very dark pixels to avoid dark noise being counted as red
        v_channel = hsv[:,:,2]
        mask_red[v_channel < 40] = 0

        # morphological cleanup
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- perspective warp to bird's-eye ---
        warped = cv2.warpPerspective(mask_red, M_warp, (FRAME_WIDTH, FRAME_HEIGHT), flags=cv2.INTER_LINEAR)

        # --- histogram on bottom half to find base x ---
        bottom_half = warped[FRAME_HEIGHT//2:, :]
        col_counts = np.sum(bottom_half // 255, axis=0)  # number of red pixels per column
        max_count = int(col_counts.max())

        if max_count < 1:
            # no red found on warped -> slow forward/search
            print("No red found in warped image -> slow forward/search")
            # very gentle forward to reposition and search; still smooth (no reverse)
            set_motor(int(BASE_SPEED * 0.9), int(BASE_SPEED * 0.9))
            cv2.imshow("Frame", frame)
            cv2.imshow("Warped", warped)
            cv2.imshow("Mask Red", mask_red)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # base x in warped coordinates
        base_x = int(np.argmax(col_counts))

        # --- sliding windows on warped image ---
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        window_height = int(np.ceil(warped.shape[0] / NWINDOWS))
        current_x = base_x
        centers = []

        for w in range(NWINDOWS):
            win_y_low = warped.shape[0] - (w+1) * window_height
            win_y_high = warped.shape[0] - w * window_height
            if win_y_low < 0:
                win_y_low = 0
            win_x_left = max(0, current_x - MARGIN)
            win_x_right = min(FRAME_WIDTH-1, current_x + MARGIN)

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_left) & (nonzerox < win_x_right)).nonzero()[0]

            if len(good_inds) > MINPIX:
                current_x = int(np.mean(nonzerox[good_inds]))
                centers.append(current_x)
            else:
                centers.append(current_x)

        # weighted average of centers (lower windows weigh more)
        weights = np.linspace(1.5, 1.0, num=len(centers))
        centers_arr = np.array(centers)
        lane_center_warp = int(np.average(centers_arr, weights=weights))

        # map lane_center_warp back to original image coordinates
        pts = np.array([[[lane_center_warp, FRAME_HEIGHT - 10]]], dtype=np.float32)
        pts_orig = cv2.perspectiveTransform(pts, Minv)
        lane_center_orig_x = int(pts_orig[0,0,0])

        # debug visuals
        vis_warp = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        for i, cx in enumerate(centers):
            y = warped.shape[0] - (i * window_height) - window_height//2
            cv2.circle(vis_warp, (int(cx), int(y)), 3, (0,255,255), -1)
        cv2.line(vis_warp, (lane_center_warp, 0), (lane_center_warp, warped.shape[0]-1), (0,0,255), 2)

        # draw lane center on original frame
        y_vis = FRAME_HEIGHT - 12
        cv2.circle(frame, (lane_center_orig_x, y_vis), 8, (0,255,255), -1)
        cv2.line(frame, (FRAME_WIDTH//2, 0), (FRAME_WIDTH//2, FRAME_HEIGHT), (255,0,0), 1)

        # compute deviation and smooth proportional steering
        deviation = lane_center_orig_x - (FRAME_WIDTH // 2)
        print(f"lane_center={lane_center_orig_x} dev={deviation}")

        # proportional steering: compute steer offset
        steer = STEER_GAIN * deviation  # positive -> lane on right -> steer left
        # compute differential speeds (no reverse)
        left_speed  = BASE_SPEED - steer
        right_speed = BASE_SPEED + steer

        # clamp into allowed motor range 0..100 (we don't go negative here)
        # if steer is very large it can try to set one motor negative; instead cap and reduce base speed
        left_speed_clamped = clamp(left_speed, 0, 100)
        right_speed_clamped = clamp(right_speed, 0, 100)

        # If deviation small, go forward at BASE_SPEED; else use differential speeds
        if abs(deviation) <= CENTER_TOLERANCE:
            set_motor(BASE_SPEED, BASE_SPEED)
            action = "Forward"
        else:
            # gentle turns; ensure we don't reverse by keeping >0
            set_motor(left_speed_clamped, right_speed_clamped)
            action = "Turning L" if deviation > 0 else "Turning R"

        print(f"{action}: left={left_speed_clamped} right={right_speed_clamped}")

        # show windows
        cv2.imshow("Frame", frame)
        cv2.imshow("Warped", vis_warp)
     #   cv2.imshow("Mask Red", mask_red)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("debug_frame.png", frame)
            cv2.imwrite("debug_warped.png", vis_warp)
           # cv2.imwrite("debug_mask_red.png", mask_red)
            print("Saved debug images.")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("Stopping motors and cleaning up")
    stop()
    left_f_pwm.stop()
    left_b_pwm.stop()
    right_f_pwm.stop()
    right_b_pwm.stop()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
