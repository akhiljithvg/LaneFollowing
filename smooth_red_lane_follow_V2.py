#!/usr/bin/env python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ============================================================
# MOTOR SETUP
# ============================================================
MOTOR_LEFT_FORWARD = 16
MOTOR_LEFT_BACKWARD = 12
MOTOR_RIGHT_FORWARD = 21
MOTOR_RIGHT_BACKWARD = 20

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [MOTOR_LEFT_FORWARD, MOTOR_LEFT_BACKWARD, MOTOR_RIGHT_FORWARD, MOTOR_RIGHT_BACKWARD]:
	GPIO.setup(pin, GPIO.OUT)

PWM_FREQ = 100
left_f = GPIO.PWM(MOTOR_LEFT_FORWARD, PWM_FREQ)
left_b = GPIO.PWM(MOTOR_LEFT_BACKWARD, PWM_FREQ)
right_f = GPIO.PWM(MOTOR_RIGHT_FORWARD, PWM_FREQ)
right_b = GPIO.PWM(MOTOR_RIGHT_BACKWARD, PWM_FREQ)

for pwm in (left_f, left_b, right_f, right_b):
	pwm.start(0)


def clamp(x, lo=0, hi=100):
	return max(lo, min(hi, int(x)))


def set_motor(left, right):
	if left >= 0:
		left_f.ChangeDutyCycle(clamp(left))
		left_b.ChangeDutyCycle(0)
	else:
		left_f.ChangeDutyCycle(0)
		left_b.ChangeDutyCycle(clamp(-left))

	if right >= 0:
		right_f.ChangeDutyCycle(clamp(right))
		right_b.ChangeDutyCycle(0)
	else:
		right_f.ChangeDutyCycle(0)
		right_b.ChangeDutyCycle(clamp(-right))


def stop():
	set_motor(0, 0)


# ============================================================
# PARAMETERS
# ============================================================
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Align lane to right 2/3 vertical boundary:
TARGET_X = int(FRAME_WIDTH * 2 / 4)

# Steering:
MAX_SPEED = 40
BASE_SPEED = int(MAX_SPEED * 0.7)
STEER_GAIN = 0.2  # tune this

LOOKAHEAD_Y = FRAME_HEIGHT - 40  # predicted point location in warped image

# Sliding windows:
NWINDOWS = 10
MARGIN = 20
MINPIX = 20

# ============================================================
# PERSPECTIVE TRANSFORM
# ============================================================
WARP_SRC = np.float32([
	[40, 95],
	[280, 95],
	[0, FRAME_HEIGHT - 8],
	[FRAME_WIDTH - 1, FRAME_HEIGHT - 8]
])
WARP_DST = np.float32([
	[60, 0],
	[FRAME_WIDTH - 60, 0],
	[60, FRAME_HEIGHT],
	[FRAME_WIDTH - 60, FRAME_HEIGHT]
])
M_warp = cv2.getPerspectiveTransform(WARP_SRC, WARP_DST)
Minv = cv2.getPerspectiveTransform(WARP_DST, WARP_SRC)

# ============================================================
# HSV FOR RED LANE
# ============================================================
lower_red1 = np.array([0, 110, 70])
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([165, 110, 70])
upper_red2 = np.array([180, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ============================================================
# CAMERA INIT
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
time.sleep(0.3)

print("\n--- Predicted-Lane Follower Running ---\n")

# ============================================================
# MAIN LOOP
# ============================================================
try:
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			print("Camera error.")
			break

		frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# ---- Red mask ----
		mask = cv2.bitwise_or(
			cv2.inRange(hsv, lower_red1, upper_red1),
			cv2.inRange(hsv, lower_red2, upper_red2)
		)
		mask[hsv[:, :, 2] < 40] = 0
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

		# ---- Warp ----
		warped = cv2.warpPerspective(mask, M_warp, (FRAME_WIDTH, FRAME_HEIGHT))

		# ---- Histogram ----
		bottom = warped[FRAME_HEIGHT // 2:, :]
		col_count = np.sum(bottom // 255, axis=0)
		maxcount = int(col_count.max()) if col_count.size > 0 else 0

		# ====================================================
		# NO RED FOUND -> just slow forward
		# ====================================================
		if maxcount < 1:
			print("Lane lost -> slow forward")
			set_motor(int(BASE_SPEED * 0.8), int(BASE_SPEED * 0.9))
			cv2.imshow("Frame", frame)
			cv2.imshow("Warped", warped)
			cv2.imshow("Mask", mask)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		# ====================================================
		# SLIDING WINDOWS
		# ====================================================
		base_x = int(np.argmax(col_count))
		nz = warped.nonzero()
		nonzeroy = np.array(nz[0])
		nonzerox = np.array(nz[1])

		window_h = int(FRAME_HEIGHT / NWINDOWS)
		cur_x = base_x
		lane_points_x = []
		lane_points_y = []

		for w in range(NWINDOWS):
			wy_low = FRAME_HEIGHT - (w + 1) * window_h
			wy_high = FRAME_HEIGHT - w * window_h
			if wy_low < 0:
				wy_low = 0
			wx_l = max(0, cur_x - MARGIN)
			wx_r = min(FRAME_WIDTH - 1, cur_x + MARGIN)

			good = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
					(nonzerox >= wx_l) & (nonzerox < wx_r)).nonzero()[0]
			cy = int((wy_low + wy_high) / 2)
			if len(good) > MINPIX:
				cur_x = int(np.mean(nonzerox[good]))
				lane_points_x.append(cur_x)
				lane_points_y.append(cy)

		lane_points_x = np.array(lane_points_x, dtype=np.float32)
		lane_points_y = np.array(lane_points_y, dtype=np.float32)

		# ====================================================
		# POLYNOMIAL FITTING FOR LANE CURVE PREDICTION
		# ====================================================
		if lane_points_x.size < 3:
			# Not enough points to fit a degree-2 polynomial: slow forward
			print("Insufficient lane points -> slow forward")
			set_motor(int(BASE_SPEED * 0.8), int(BASE_SPEED * 0.9))
			cv2.imshow("Frame", frame)
			cv2.imshow("Warped", warped)
			cv2.imshow("Mask", mask)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			continue

		poly = np.polyfit(lane_points_y, lane_points_x, 2)
		predicted_x = int(poly[0] * LOOKAHEAD_Y ** 2 + poly[1] * LOOKAHEAD_Y + poly[2])

		# ---- Map predicted point back to original image ----
		pt = np.array([[[predicted_x, LOOKAHEAD_Y]]], dtype=np.float32)
		pt_orig = cv2.perspectiveTransform(pt, Minv)
		lane_x = int(pt_orig[0, 0, 0])

		# ====================================================
		# STEERING
		# ====================================================
		deviation = lane_x - TARGET_X
		steer = STEER_GAIN * deviation
		left_speed = clamp(BASE_SPEED - steer, 0, 140)
		right_speed = clamp(BASE_SPEED + steer, 0, 140)
		set_motor(left_speed, right_speed)

		# ====================================================
		# DEBUG VISUALS
		# ====================================================
		cv2.line(frame, (TARGET_X, 0), (TARGET_X, FRAME_HEIGHT), (0, 255, 0), 2)
		cv2.circle(frame, (lane_x, FRAME_HEIGHT - 20), 6, (255, 255, 0), -1)
		cv2.imshow("Frame", frame)
		cv2.imshow("Warped", warped)
		cv2.imshow("Mask", mask)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

except KeyboardInterrupt:
	print("Interrupted.")
finally:
	stop()
	left_f.stop()
	left_b.stop()
	right_f.stop()
	right_b.stop()
	cap.release()
	cv2.destroyAllWindows()
	GPIO.cleanup()
	print("Clean exit.")